import os
import json
import datetime
from flask import Flask, request
from google.cloud import storage, bigquery
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

app = Flask(__name__)

# CONFIG
PROJECT_ID = "pdf-etl-479411"
DATASET = "etl_reports"
vertexai.init(project=PROJECT_ID, location="us-central1")

# 1. Define the "Brain" and the expected types
# You can expand this list as you grow!
model = GenerativeModel("gemini-1.5-flash")

# ==========================
# SMART EXTRACTION (Gemini)
# ==========================
def extract_with_gemini(pdf_bytes):
    # This prompt tells Gemini exactly how to behave
    prompt = """
    Extract all key information (KPIs) from this document. 
    Return the data in a clean JSON format.
    - If a value is a price or number, return it as a float (e.g., 1500.50).
    - If a value is a date, return it in YYYY-MM-DD format.
    - Use lowercase keys with underscores (e.g., total_amount, invoice_date).
    """
    
    pdf_part = Part.from_data(data=pdf_bytes, mime_type="application/pdf")
    
    # We force Gemini to give us JSON only
    response = model.generate_content(
        [prompt, pdf_part],
        generation_config=GenerationConfig(
            response_mime_type="application/json",
        ),
    )
    
    return json.loads(response.text)

# ==========================
# DYNAMIC TABLE LOGIC
# ==========================
def get_or_create_table(client_id, data_sample):
    bq_client = bigquery.Client()
    table_id = f"{PROJECT_ID}.{DATASET}.{client_id.lower()}"

    try:
        bq_client.get_table(table_id)
        return table_id
    except:
        schema = [
            bigquery.SchemaField("row_id", "STRING"),
            bigquery.SchemaField("file_name", "STRING"),
            bigquery.SchemaField("uploaded_at", "TIMESTAMP"),
        ]
        
        # Vertex AI already set the types for us!
        for key, value in data_sample.items():
            if isinstance(value, (int, float)):
                field_type = "FLOAT"
            elif isinstance(value, str) and len(value) == 10 and value.count("-") == 2:
                field_type = "DATE"
            else:
                field_type = "STRING"
            
            schema.append(bigquery.SchemaField(f"kpi_{key}", field_type))
            
        table = bigquery.Table(table_id, schema=schema)
        bq_client.create_table(table)
        return table_id

@app.post("/")
def handle_event():
    payload = request.get_json(silent=True) or {}
    data = payload.get("data", payload)
    file_path = data.get("name")
    bucket_name = data.get("bucket")

    if not file_path or "incoming/" not in file_path:
        return ("Skip", 200)

    # 1. Get Client ID
    client_id = file_path.split("/")[1]

    # 2. Extract with Vertex AI
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    content = blob.download_as_bytes()
    
    # THE MAGIC HAPPENS HERE
    extracted_data = extract_with_gemini(content)

    # 3. Insert to BigQuery
    table_full_id = get_or_create_table(client_id, extracted_data)
    
    row = {
        "row_id": str(datetime.datetime.now().timestamp()),
        "file_name": file_path,
        "uploaded_at": datetime.datetime.utcnow().isoformat()
    }
    
    for k, v in extracted_data.items():
        row[f"kpi_{k}"] = v

    bq_client = bigquery.Client()
    bq_client.insert_rows_json(table_full_id, [row])
    
    # 4. Cleanup
    new_path = file_path.replace("incoming/", "processed/")
    bucket.copy_blob(blob, bucket, new_path)
    blob.delete()
    
    return ("Success", 200)
