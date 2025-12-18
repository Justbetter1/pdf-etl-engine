import os
import json
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


def move_to_processed(bucket, file_path):
    new_path = file_path.replace("incoming/", "processed/")
    src_blob = bucket.blob(file_path)

    if not src_blob.exists():
        print("⛔ File missing — skipping move")
        return None

    bucket.copy_blob(src_blob, bucket, new_path)
    src_blob.delete()
    print(f"📦 File moved to {new_path}")
    return new_path


def extract_text(local_path):
    try:
        with pdfplumber.open(local_path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)
    except Exception as e:
        print("❌ PDF extraction error:", e)
        return ""


def call_gemini(text):
    init(project=os.environ.get("GOOGLE_CLOUD_PROJECT", "pdf-etl-479411"), location="us-central1")
    model = GenerativeModel("gemini-2.5-flash")

    prompt = f"""
Extract structured information and return ONLY JSON.

PDF TEXT:
{text}
"""

    # --- Gemini call with timeout ---
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "timeout": 120  # allow up to 2 minutes
            }
        )
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

    # --- Try to parse JSON ---
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
        print("❌ BigQuery error:", errors)
    else:
        print("✅ Inserted into BigQuery")


@app.post("/")
def event_handler():
    event = request.get_json(silent=True)

    if not event:
        print("⚠️ No JSON in request")
        return ("ignored", 200)

    # Detect if event is wrapped (Eventarc standard)
    if "data" in event and isinstance(event["data"], dict):
        payload = event["data"]
    else:
        # Your trigger sends RAW GCS OBJECT
        payload = event  

    print("🟦 RAW/PARSED EVENT:", json.dumps(payload, indent=2))

    file_path = payload.get("name")
    bucket_name = payload.get("bucket")

    if not file_path or not bucket_name:
        print("⚠️ Missing name/bucket → Not valid event")
        return ("ignored", 200)

    print("=======================")
    print(f"📄 Triggered: {file_path}")
    print("=======================")

    # Skip folders
    if file_path.endswith("/"):
        print("⛔ Folder event — ignored")
        return ("ok", 200)

    # Skip processed
    if file_path.startswith("processed/"):
        print("⛔ Already processed — ignored")
        return ("ok", 200)

    # Only process incoming/
    if not file_path.startswith("incoming/"):
        print("⛔ Not incoming — ignored")
        return ("ok", 200)

    # incoming/Client1/filename.pdf
    parts = file_path.split("/")
    if len(parts) < 3:
        print("⚠️ Invalid path")
        return ("ok", 200)

    _, client_id, _ = parts
    print(f"👤 Client ID: {client_id}")

    # Download → Extract → Gemini → BigQuery → Move
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    if not blob.exists():
        print("⛔ File missing on GCS")
        return ("ok", 200)

    local_path = f"/tmp/{os.path.basename(file_path)}"
    blob.download_to_filename(local_path)
    print(f"📥 Downloaded to {local_path}")

    text = extract_text(local_path)
    parsed = call_gemini(text)
    insert_bigquery(client_id, file_path, parsed)
    move_to_processed(bucket, file_path)

    print("🎉 DONE")
    return ("ok", 200)


