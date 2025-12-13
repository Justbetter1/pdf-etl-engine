import base64
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
# Helper: Move PDF to processed/
###########################################################
def move_to_processed(bucket, file_path):
    new_path = file_path.replace("incoming/", "processed/")
    src = bucket.blob(file_path)

    # If file already moved / missing → skip
    if not src.exists():
        print("⛔ File missing during move — ignoring.")
        return None

    bucket.copy_blob(src, bucket, new_path)
    src.delete()

    print(f"📦 File moved to {new_path}")
    return new_path


###########################################################
# Extract text safely
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
# Gemini
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
        print("⚠️ Gemini returned non-JSON")
        return {
            "summary": "",
            "raw_text": text,
            "error": response.text,
        }


###########################################################
# BigQuery insertion
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
        print("❌ BigQuery error:", errors)
    else:
        print("✅ Inserted into BigQuery")


###########################################################
# MAIN ENTRY — Eventarc → Cloud Run → Flask POST
###########################################################
@app.post("/")
def handler():
    envelope = request.get_json(silent=True)

    if not envelope:
        print("⚠️ No Eventarc envelope")
        return ("Ignored", 200)

    # New Eventarc payload style:
    event = envelope.get("protoPayload", {})
    resource = event.get("resourceName", "")

    if "/objects/" not in resource:
        print("⚠️ Invalid event format")
        return ("Ignored", 200)

    file_path = resource.split("/objects/")[1]

    print("========================")
    print(f"📄 Triggered: {file_path}")
    print("========================")

    if file_path.endswith("/"):
        print("⛔ Folder event — skip")
        return ("OK", 200)

    if "processed/" in file_path:
        print("⛔ Already processed — skip")
        return ("OK", 200)

    if "incoming/" not in file_path:
        print("⛔ Not an incoming file — skip")
        return ("OK", 200)

    client_id = file_path.split("/")[0]
    print(f"👤 Client ID: {client_id}")

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_path)

    if not blob.exists():
        print("⛔ File missing (already processed?)")
        return ("OK", 200)

    local_path = f"/tmp/{os.path.basename(file_path)}"
    blob.download_to_filename(local_path)
    print(f"📥 Downloaded: {local_path}")

    text = extract_text(local_path)
    parsed = call_gemini(text)

    insert_bigquery(client_id, file_path, parsed)

    move_to_processed(bucket, file_path)

    print("🎉 DONE")
    return ("OK", 200)
