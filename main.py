import os
import json
import datetime
import re
import time
from flask import Flask, request
from google.cloud import storage, bigquery
from google import genai
from google.genai import types

app = Flask(__name__)

# CONFIG
PROJECT_ID = "pdf-etl-479411"
DATASET = "etl_reports"
LOCATION = "us-central1"

# Initialize Client
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# ==========================================
# 1. THE BRAIN: Wait for Gemini to finish
# ==========================================
def extract_with_gemini(pdf_bytes):
    print("🧠 Gemini is analyzing the PDF... waiting for full extraction.")
    prompt = """
    Analyze this PDF and extract EVERY single KPI or data point you see. 
    Do not skip anything. Return a FLAT JSON object.
    - Numbers: float (1500.0)
    - Dates: YYYY-MM-DD
    - Keys: lowercase_underscores
    """
    
    # generate_content is a blocking call; it WAITS until Gemini is done.
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=[types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    
    # Repair and Parse
    clean_json = re.sub(r',\s*([\]}])', r'\1', response.text)
    data = json.loads(clean_json)
    
    # Ensure we have a dictionary
    extracted = data[0] if isinstance(data, list) else data
    print(f"✅ Gemini finished! Found {len(extracted)} KPIs.")
    return extracted

# ==========================================
# 2. THE BUILDER: Sync Table with Gemini's Data
# ==========================================
def sync_bigquery_schema(client_id, extracted_data):
    bq_client = bigquery.Client()
    clean_client = re.sub(r'[^a-zA-Z0-9_]', '_', client_id).lower()
    table_id = f"{PROJECT_ID}.{DATASET}.{clean_client}"
    
    try:
        table = bq_client.get_table(table_id)
        print(f"📊 Table '{clean_client}' found. Checking for new columns...")
    except Exception:
        print(f"✨ Table '{clean_client}' not found. Creating it now...")
        # Start with standard system columns
        schema = [
            bigquery.SchemaField("row_id", "STRING"),
            bigquery.SchemaField("file_name", "STRING"),
            bigquery.SchemaField("uploaded_at", "TIMESTAMP"),
        ]
        table = bq_client.create_table(bigquery.Table(table_id, schema=schema))
        time.sleep(5) # CRITICAL: Wait for BQ to register the new table globally
        table = bq_client.get_table(table_id)

    # Now, compare Gemini's findings with BQ's current columns
    existing_columns = {field.name for field in table.schema}
    new_fields = []

    for key, value in extracted_data.items():
        col_name = f"kpi_{re.sub(r'[^a-zA-Z0-9_]', '_', key).lower()}"
        if col_name not in existing_columns:
            print(f"🆕 Gemini found a new KPI: '{col_name}'. Adding to BigQuery.")
            dtype = "FLOAT" if isinstance(value, (int, float)) else "STRING"
            new_fields.append(bigquery.SchemaField(col_name, dtype))

    if new_fields:
        table.schema += new_fields
        bq_client.update_table(table, ["schema"])
        print("🛠️ Schema updated successfully.")
        time.sleep(2) # Wait for schema to propagate
    
    return table_id

# ==========================================
# 3. THE HANDLER
# ==========================================
@app.post("/")
def handle_event():
    payload = request.get_json(silent=True) or {}
    data = payload.get("data", payload)
    file_path = data.get("name", "")

    if "incoming/" not in file_path: return ("OK", 200)

    try:
        client_id = file_path.split("/")[1]
        
        # Download
        storage_client = storage.Client()
        bucket = storage_client.bucket(data.get("bucket"))
        blob = bucket.blob(file_path)
        pdf_content = blob.download_as_bytes()
        
        # STEP 1: Gemini processes EVERYTHING first
        kpis = extract_with_gemini(pdf_content)

        # STEP 2: BigQuery prepares the "Box" based on EXACTLY what Gemini found
        target_table = sync_bigquery_schema(client_id, kpis)

        # STEP 3: Insert the data
        row = {
            "row_id": str(datetime.datetime.now().timestamp()),
            "file_name": file_path,
            "uploaded_at": datetime.datetime.utcnow().isoformat()
        }
        for k, v in kpis.items():
            clean_key = f"kpi_{re.sub(r'[^a-zA-Z0-9_]', '_', k).lower()}"
            row[clean_key] = v

        bq_client = bigquery.Client()
        insert_errors = bq_client.insert_rows_json(target_table, [row])
        
        if not insert_errors:
            # Move to processed/
            new_path = file_path.replace("incoming/", "processed/")
            bucket.copy_blob(blob, bucket, new_path)
            blob.delete()
            print(f"🚀 SUCCESS: {file_path} is now in BigQuery.")
        else:
            print(f"❌ BigQuery Insert Errors: {insert_errors}")

    except Exception as e:
        print(f"🔥 System Error: {str(e)}")

    return ("ok", 200)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
