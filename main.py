import json
import os
import datetime
import pdfplumber
from flask import Flask, request
from google.cloud import storage, bigquery
from vertexai import init
from vertexai.generative_models import GenerativeModel

app = Flask(__name__)

BUCKET_NAME = "pdf_platform_main"
DATASET = "etl_reports"
TABLE = "documents"


###########################################################
# MOVE FILE TO /processed/
###########################################################
def move_to_processed(bucket, file_path):
    new_path = file_path.replace("incoming/", "processed/")
    src = bucket.blob(file_path)

    if not src.exists():
        print("⛔ File missing — skipping move")
        return None

    bucket.copy_blob(src, bucket, new_path)
    src.delete()

    print(f"📦 File moved to {new_path}")
    return new_path


###########################################################
# EXTRACT TEXT FROM PDF
###########################################################
def extract_text(local_path):
    try:
        with pdfplumber.open(local_path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)
    except Exception as e:
        print("❌ PDF extraction failed:", e)
        return ""


###########################################################
# GEMINI CALL
###########################################################
def call_gemini(text):
    init(project=os.environ["GOOGLE_CLOUD_PROJECT"], location="us-central1")
    model = GenerativeModel("gemini-2.5-flash")

    prompt = f"""
Extract structured information and return ONLY JSON:

PDF TEXT:
{text}
"""

    response = model.generate_content(prompt)

    try:
        return json.loads(response.text)
    except Exception:
        print("⚠️ Gemini returned non-JSON output")
        return {
            "summary": "",
            "key_points": [],
            "extracted_fields": {},
            "tables": [],
            "raw_text": text,
            "error": response.text,
        }


###########################################################
# INSERT INTO BIGQUERY
###########################################################
def insert_bigquery(client_id, filename, parsed):
    table = f"{os.environ['GOOGLE_CLOUD_PROJECT']}.{DATASET}.{TABLE}"
    client = bigquery.Client()

    row = {
        "client_id": client_id,
        "file_name": filename,
        "uploaded_at": datetime.datetime.utcnow().isoformat(),
        "summary": parsed.get("summary", ""),
        "key_points": json.dumps(parsed.get("key_points", [])),
        "extracted_fields": json.dumps(parsed.get("extracted_fields", {})),
        "tables": json.dumps(parsed.get("tables", [])),
        "raw_text": parsed.get("raw_text", ""),
        "raw_error": parsed.get("error"),
    }

    errors = client.insert_rows_json(table, [row])
    if errors:
        print("❌ BigQuery insert error:", errors)
    else:
        print("✅ Inserted into BigQuery")


###########################################################
# MAIN ENTRY — Eventarc → Cloud Run → POST /
###########################################################
@app.post("/")
def handler():
    envelope = request.get_json(silent=True)

    if not envelope:
        print("⚠️ No Eventarc envelope")
        return ("Ignored", 200)

    # Eventarc Cloud Storage event format
    event = envelope.get("data", {})
    file_path = event.get("name")

    if not file_path:
        print("⚠️ Not a valid GCS event")
        return ("Ignored", 200)

    print("=======================")
    print(f"📄 Triggered: {file_path}")
    print("=======================")

    # Ignore folder events
    if file_path.endswith("/"):
        print("⛔ Folder event — ignored")
        return ("OK", 200)

    # Ignore processed files
    if "processed/" in file_path:
        print("⛔ Already processed — ignored")
        return ("OK", 200)

    # Must be incoming/
    if "incoming/" not in file_path:
        print("⛔ Not an incoming file — ignored")
        return ("OK", 200)

    # First part → client ID
    client_id = file_path.split("/")[0]
    print(f"👤 Client ID: {client_id}")

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_path)

    if not blob.exists():
        print("⛔ File missing — skip")
        return ("OK", 200)

    # Download PDF
    local_path = f"/tmp/{os.path.basename(file_path)}"
    blob.download_to_filename(local_path)
    print(f"📥 Downloaded: {local_path}")

    # Extract text
    text = extract_text(local_path)

    # Gemini
    parsed = call_gemini(text)

    # Insert BigQuery
    insert_bigquery(client_id, file_path, parsed)

    # Move file
    move_to_processed(bucket, file_path)

    print("🎉 DONE")
    return ("OK", 200)
