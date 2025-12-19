import os
import json
import datetime
import pdfplumber

from flask import Flask, request
from google.cloud import storage, bigquery
from vertexai import init
from vertexai.generative_models import GenerativeModel


app = Flask(__name__)

# ==========================
# CONFIG
# ==========================
BUCKET_NAME = "pdf_platform_main"
DATASET = "etl_reports"
TABLE = "documents"

# Safe project ID detection for Cloud Run
PROJECT_ID = (
    os.environ.get("GOOGLE_CLOUD_PROJECT")
    or os.environ.get("GOOGLE_CLOUD_PROJECT_NUMBER")
    or "pdf-etl-479411"
)

print(f"🔧 Using PROJECT_ID = {PROJECT_ID}")


# ==========================
# MOVE FILE TO /processed/
# ==========================
def move_to_processed(bucket, file_path):
    new_path = file_path.replace("incoming/", "processed/")
    src_blob = bucket.blob(file_path)

    if not src_blob.exists():
        print("⛔ File missing — skipping move")
        return None

    bucket.copy_blob(src_blob, bucket, new_path)
    src_blob.delete()

    print(f"📦 File moved → {new_path}")
    return new_path


# ==========================
# EXTRACT PDF TEXT
# ==========================
def extract_text(local_path):
    try:
        with pdfplumber.open(local_path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)
    except Exception as e:
        print("❌ PDF extract error:", e)
        return ""


# ==========================
# GEMINI CALL
# ==========================
def call_gemini(text):
    try:
        init(project=PROJECT_ID, location="us-central1")
        model = GenerativeModel("gemini-2.5-flash")

        prompt = f"""
Extract structured information and return ONLY JSON.

PDF TEXT:
{text}
"""
        response = model.generate_content(prompt)

    except Exception as e:
        print("❌ Gemini API error:", e)
        return {
            "summary": "",
            "key_points": [],
            "extracted_fields": {},
            "tables": [],
            "raw_text": text,
            "error": str(e),
        }

    # Try parsing JSON safely
    try:
        return json.loads(response.text)
    except Exception:
        print("⚠️ Gemini returned non-JSON")
        return {
            "summary": "",
            "key_points": [],
            "extracted_fields": {},
            "tables": [],
            "raw_text": text,
            "error": response.text,
        }


# ==========================
# WRITE TO BIGQUERY
# ==========================
def insert_bigquery(client_id, filename, parsed):
    table = f"{PROJECT_ID}.{DATASET}.{TABLE}"
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

    try:
        errors = client.insert_rows_json(table, [row])
        if errors:
            print("❌ BigQuery insert error:", errors)
        else:
            print("✅ Inserted into BigQuery")
    except Exception as e:
        print("❌ BigQuery FAILURE:", e)


# ==========================
# MAIN HANDLER
# ==========================
@app.post("/")
def event_handler():
    event = request.get_json(silent=True)

    # Always return 200 so Cloud Run does NOT retry forever
    if not event:
        print("⚠️ Empty event payload")
        return ("ok", 200)

    data = event.get("data")
    if not data:
        print("⚠️ Missing data in event")
        return ("ok", 200)

    file_path = data.get("name")
    bucket_name = data.get("bucket")

    if not file_path or not bucket_name:
        print("⚠️ Invalid GCS payload")
        return ("ok", 200)

    print("🟦 EVENT:", json.dumps({"name": file_path, "bucket": bucket_name}, indent=2))

    # Ignore folder events
    if file_path.endswith("/"):
        print("⛔ Folder — ignored")
        return ("ok", 200)

    # Only process incoming/
    if not file_path.startswith("incoming/"):
        print("⛔ Not inside incoming/ — ignored")
        return ("ok", 200)

    # Extract client_id
    parts = file_path.split("/")
    if len(parts) < 3:
        print("⚠️ Invalid file structure:", file_path)
        return ("ok", 200)

    _, client_id, _ = parts
    print(f"👤 Client ID = {client_id}")

    # Download file
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    if not blob.exists():
        print("⛔ File not found in bucket")
        return ("ok", 200)

    local_path = f"/tmp/{os.path.basename(file_path)}"
    blob.download_to_filename(local_path)
    print(f"📥 Downloaded → {local_path}")

    # Extract → Gemini → BigQuery
    text = extract_text(local_path)
    parsed = call_gemini(text)
    insert_bigquery(client_id, file_path, parsed)

    # Move file
    move_to_processed(bucket, file_path)

    print("🎉 DONE — FULL SUCCESS")
    return ("ok", 200)
