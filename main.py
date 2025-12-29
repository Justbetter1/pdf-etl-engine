import os
import json
import datetime
import re
from flask import Flask, request
from google.cloud import storage, bigquery, documentai_v1 as documentai

app = Flask(__name__)

# ==========================
# CONFIG
# ==========================
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "pdf-etl-479411")
DATASET = "etl_reports"
LOCATION = "us" # Ensure your DocAI processor is in this region
PROCESSOR_ID = os.environ.get("DOC_AI_PROCESSOR_ID") # Set in Cloud Run Env Vars

# ==========================
# DYNAMIC TABLE LOGIC
# ==========================
def get_or_create_client_table(client_id, extracted_fields):
    bq_client = bigquery.Client()
    table_id = f"{PROJECT_ID}.{DATASET}.{client_id.lower()}"

    try:
        bq_client.get_table(table_id)
        return table_id
    except Exception:
        print(f"✨ Creating new table for client: {client_id}")
        schema = [
            bigquery.SchemaField("row_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("file_name", "STRING"),
            bigquery.SchemaField("client_id", "STRING"),
            bigquery.SchemaField("uploaded_at", "TIMESTAMP"),
            bigquery.SchemaField("raw_text", "STRING"),
        ]

        # Add dynamic KPI columns found by the AI
        for key in extracted_fields.keys():
            clean_col = re.sub(r'[^a-zA-Z0-9_]', '_', key).lower()
            if not clean_col.startswith("kpi_"):
                clean_col = f"kpi_{clean_col}"
            schema.append(bigquery.SchemaField(clean_col, "STRING"))

        table = bigquery.Table(table_id, schema=schema)
        bq_client.create_table(table)
        return table_id

# ==========================
# DOCUMENT AI ENGINE
# ==========================
def run_doc_ai(content):
    client = documentai.DocumentProcessorServiceClient(
        client_options={"api_endpoint": f"{LOCATION}-documentai.googleapis.com"}
    )
    name = client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)
    raw_doc = documentai.RawDocument(content=content, mime_type="application/pdf")
    request = documentai.ProcessRequest(name=name, raw_document=raw_doc)
    
    result = client.process_document(request=request)
    doc = result.document

    fields = {}
    for page in doc.pages:
        for field in page.form_fields:
            k = field.field_name.text_anchor.content.strip().replace("\n", " ")
            v = field.field_value.text_anchor.content.strip().replace("\n", " ")
            if k: fields[k] = v
    
    return fields, doc.text

# ==========================
# MAIN HANDLER
# ==========================
@app.post("/")
def event_handler():
    payload = request.get_json(silent=True) or {}
    headers = request.headers
    
    # Unique ID to stop loops/duplicates
    event_id = headers.get("ce-id") or headers.get("X-Goog-Message-Id") or str(datetime.datetime.now().timestamp())
    
    # Universal Parser for GitHub-based triggers
    data = payload.get("data", payload)
    file_path = data.get("name")
    bucket_name = data.get("bucket")

    if not file_path or not bucket_name or not file_path.startswith("incoming/"):
        return ("Ignored", 200)

    try:
        # 1. Identify Client (incoming/C1/file.pdf)
        parts = file_path.split("/")
        client_id = parts[1] if len(parts) > 1 else "unknown"

        # 2. Download and Process
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        content = blob.download_as_bytes()

        fields, raw_text = run_doc_ai(content)

        # 3. Dynamic BigQuery Insert
        table_full_id = get_or_create_client_table(client_id, fields)
        
        row = {
            "row_id": event_id,
            "file_name": file_path,
            "client_id": client_id,
            "uploaded_at": datetime.datetime.utcnow().isoformat(),
            "raw_text": raw_text
        }
        # Map fields to their kpi_ columns
        for k, v in fields.items():
            clean_col = re.sub(r'[^a-zA-Z0-9_]', '_', k).lower()
            row[f"kpi_{clean_col}"] = v

        bq_client = bigquery.Client()
        errors = bq_client.insert_rows_json(table_full_id, [row])
        
        if not errors:
            # 4. Success -> Move file
            new_path = file_path.replace("incoming/", "processed/")
            bucket.copy_blob(blob, bucket, new_path)
            blob.delete()
            print(f"✅ Success: {client_id} -> {table_full_id}")

    except Exception as e:
        print(f"🔥 Error: {e}")

    return ("ok", 200)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
