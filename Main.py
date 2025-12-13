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


############################################################
# Move file to processed/
############################################################
def move_to_processed(bucket, file_path):
    new_path = file_path.replace("incoming/", "processed/")
    src = bucket.blob(file_path)

    if not src.exists():
        print("⛔ File already moved or missing — skip move.")
        return None

    print(f"📦 Moving → {new_path}")
    bucket.copy_blob(src, bucket, new_path)
    src.delete()
    return new_path


############################################################
# Extract text
############################################################
def extract_text(local_path):
    try:
        with pdfplumber.open(local_path) as pdf:
            pages = [(p.extract_text() or "") for p in pdf.pages]
        return "\n".join(pages)
    except Exception as e:
        print("❌ PDF read error:", e)
        return ""


############################################################
# Gemini
############################################################
def call_gemini(text):
    init(project=os.environ["GOOGLE_CLOUD_PROJECT"], location="us-central1")
    model = GenerativeModel("gemini-2.5-flash")

    prompt = f"""
Extract important information and return ONLY JSON.
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


############################################################
# Insert into BigQuery
############################################################
def insert_bigquery(client_id, filename, parsed):
    client = bigquery.Client()
    table = f"{os.environ['GOOGLE_CLOUD_PROJECT']}.{DATASET}.{TABLE}"

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

    err = client.insert_rows_json(table, [row])
    if err:
        print("❌ BigQuery insert error:", err)
    else:
        print("✅ Inserted into BigQuery")


############################################################
# MAIN ENTRY — Eventarc → Cloud Run → POST /
############################################################
@app.post("/")
def process_pdf():
    payload = request.get_json(silent=True)
    if not payload:
        print("⚠️ No payload received")
        return ("Ignored", 200)

    # Eventarc payload structure
    proto = payload.get("protoPayload", {})
    resource = proto.get("resourceName", "")

    if "/objects/" not in resource:
        print("⚠️ Not a valid GCS event")
        return ("Ignored", 200)

    file_path = resource.split("/objects/")[1]

    print("================================")
    print(f"📄 Triggered: {file_path}")
    print("================================")

    # Skip folder triggers
    if file_path.endswith("/"):
        print("⛔ Folder event — ignored")
        return ("OK", 200)

    # Skip non-incoming
    if "incoming/" not in file_path:
        print("⛔ Not in incoming/ — ignored")
        return ("OK", 200)

    # Skip already processed
    if "processed/" in file_path:
        print("⛔ Already processed — ignored")
        return ("OK", 200)

    # Extract client id
    client_id = file_path.split("/")[0]
    print(f"👤 Client ID: {client_id}")

    # Download file
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_path)

    if not blob.exists():
        print("⛔ File missing (likely already moved)")
        return ("OK", 200)

    # Always use a file path NOT ending in '/'
    local_filename = os.path.basename(file_path)
    local_path = f"/tmp/{local_filename}"

    print(f"📥 Downloading to: {local_path}")
    blob.download_to_filename(local_path)

    # Extract text → Gemini → BigQuery
    text = extract_text(local_path)
    parsed = call_gemini(text)
    insert_bigquery(client_id, file_path, parsed)

    # Move to processed
    move_to_processed(bucket, file_path)

    print("🎉 COMPLETED SUCCESSFULLY")
    return ("OK", 200)


############################################################
# Local dev support (Cloud Run needs this to start properly)
############################################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
