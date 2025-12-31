import os
import json
import datetime
import re
from flask import Flask, request
from google.cloud import storage, bigquery
from google import genai
from google.genai import types

app = Flask(__name__)

# ==========================
# CONFIG
# ==========================
PROJECT_ID = "pdf-etl-479411"
DATASET = "etl_reports"
LOCATION = "us-central1"

# Initialize the Gen AI Client
client = genai.Client(
    vertexai=True, 
    project=PROJECT_ID, 
    location=LOCATION
)

# ==========================
# INTELLIGENT EXTRACTION
# ==========================
def extract_with_gemini(pdf_bytes):
    prompt = """
    Extract all key information (KPIs) from this document. 
    Return the data in a clean JSON format.
    - If a value is a price or number, return it as a float (e.g., 1500.50).
    - If a value is a date, return it in YYYY-MM-DD format.
    - Use lowercase keys with underscores (e.g., total_amount, invoice_date).
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    
    data = json.loads(response.text)

    # FIX: Handle cases where Gemini returns a list instead of a dictionary
    if isinstance(data, list) and len(data) > 0:
        return data[0]
    
    return data

# ==========================
# DYNAMIC TABLE LOGIC
# ==========================
def get_or_create_table(client_id, data_sample):
    bq_client = bigquery.Client()
    # Clean client name for BigQuery (no spaces or dashes)
    clean_client = re.sub(r'[^a-zA-Z0-9_]', '_', client_id).lower()
    table_id = f"{PROJECT_ID}.{DATASET}.{clean_client}"

    try:
        bq_client.get_table(table_id)
        return table_id
    except:
        print(f"✨ Creating smart table for: {clean_client}")
        schema = [
            bigquery.SchemaField("row_id", "STRING"),
            bigquery.SchemaField("file_name", "STRING"),
            bigquery.SchemaField("uploaded_at", "TIMESTAMP"),
        ]
        
        for key, value in data_sample.items():
            # Clean column names
            col_name = f"kpi_{re.sub(r'[^a-zA-Z0-9_]', '_', key).lower()}"
            
            if isinstance(value, (int, float)):
                field_type = "FLOAT"
            elif isinstance(value, str) and re.match(r'\d{4}-\d{2}-\d{2}', value):
                field_type = "DATE"
            else:
                field_type = "STRING"
            
            schema.append(bigquery.SchemaField(col_name, field_type))
            
        table = bigquery.Table(table_id, schema=schema)
        bq_client.create_table(table)
        return table_id

# ==========================
# MAIN EVENT HANDLER
# ==========================
@app.post("/")
def handle_event():
    payload = request.get_json(silent=True) or {}
    data = payload.get("data", payload)
    file_path = data.get("name")
    bucket_name = data.get("bucket")

    if not file_path or "incoming/" not in file_path:
        return ("Ignored", 200)

    try:
        # 1. Get Client Folder
        parts = file_path.split("/")
        client_id = parts[1] if len(parts) > 1 else "unknown"

        # 2. Download PDF
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        content = blob.download_as_bytes()
        
        # 3. Extract with Gemini
        extracted_data = extract_with_gemini(content)

        # 4. Insert to BigQuery
        table_full_id = get_or_create_table(client_id, extracted_data)
        
        row = {
            "row_id": str(datetime.datetime.now().timestamp()),
            "file_name": file_path,
            "uploaded_at": datetime.datetime.utcnow().isoformat()
        }
        
        # Add the extracted KPIs to the row
        for k, v in extracted_data.items():
            col_name = f"kpi_{re.sub(r'[^a-zA-Z0-9_]', '_', k).lower()}"
            row[col_name] = v

        bq_client = bigquery.Client()
        errors = bq_client.insert_rows_json(table_full_id, [row])
        
        if not errors:
            # 5. Move to processed/
            new_path = file_path.replace("incoming/", "processed/")
            bucket.copy_blob(blob, bucket, new_path)
            blob.delete()
            print(f"✅ Success: {file_path}")
        else:
            print(f"❌ BigQuery Errors: {errors}")

    except Exception as e:
        print(f"🔥 Critical Error: {str(e)}")

    return ("ok", 200)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
