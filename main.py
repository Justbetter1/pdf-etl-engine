import os
import json
import datetime
import re
import time
import firebase_admin
from firebase_admin import auth, credentials, firestore
from flask import Flask, request, jsonify
from google.cloud import storage, bigquery
from google import genai
from google.genai import types

app = Flask(__name__)

# CONFIG
PROJECT_ID = "pdf-etl-479411"
DATASET = "etl_reports"
LOCATION = "us-central1"

# Initialize Clients
# Cloud Run automatically handles credentials if initialize_app has no arguments
firebase_admin.initialize_app()
db = firestore.client()
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# ==========================================
# 🛡️ SECURITY: AUTHENTICATION BOUNCER
# ==========================================
def get_user_id(req):
    """Verifies the Firebase Bearer Token and returns the UID"""
    auth_header = req.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        print("🔒 Auth Failed: No Bearer Token")
        return None
    
    try:
        token = auth_header.split("Bearer ")[1]
        decoded_token = auth.verify_id_token(token)
        return decoded_token["uid"]
    except Exception as e:
        print(f"🔒 Auth Error: {e}")
        return None

# ==========================================
# ✨ ONBOARDING: SETUP ACCOUNT
# ==========================================
@app.post("/setup-account")
def setup_account():
    uid = get_user_id(request)
    if not uid:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket("pdf_platform_main")
        
        # 1. Create Folder Placeholders (GCS mimics folders using file paths)
        for folder in ["incoming", "processed"]:
            blob_path = f"{folder}/{uid}/.placeholder"
            blob = bucket.blob(blob_path)
            blob.upload_from_string("Folder Initialized")
            print(f"📁 Created: {blob_path}")

        # 2. Initialize Firestore Brain
        db.collection("tenants").document(uid).collection("folders").document("default").set({
            "hint": "Please extract all relevant data points from this document.",
            "is_trained": False,
            "created_at": datetime.datetime.utcnow()
        })
        
        return jsonify({"status": "success", "message": "Environment ready"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# 🧠 THE BRAIN: EXTRACTION ENGINE
# ==========================================
def extract_with_gemini(pdf_bytes, uid):
    # Fetch personalized hint from Firestore
    user_ref = db.collection("tenants").document(uid).collection("folders").document("default").get()
    user_hint = user_ref.to_dict().get("hint", "Extract all data.") if user_ref.exists else "Extract all data."

    prompt = f"""
    Analyze this PDF. User Instructions: {user_hint}
    Return a FLAT JSON object.
    - Numbers: float
    - Dates: YYYY-MM-DD
    - Keys: lowercase_underscores
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=[types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    
    clean_json = re.sub(r',\s*([\]}])', r'\1', response.text)
    data = json.loads(clean_json)
    return data[0] if isinstance(data, list) else data

# ==========================================
# 📊 THE BUILDER: BIGQUERY SYNC
# ==========================================
def sync_bigquery_schema(uid, extracted_data):
    bq_client = bigquery.Client()
    # Table name is based on UID (sanitized)
    clean_uid = re.sub(r'[^a-zA-Z0-9_]', '_', uid).lower()
    table_id = f"{PROJECT_ID}.{DATASET}.user_{clean_uid}"
    
    try:
        table = bq_client.get_table(table_id)
    except Exception:
        schema = [
            bigquery.SchemaField("row_id", "STRING"),
            bigquery.SchemaField("file_name", "STRING"),
            bigquery.SchemaField("uploaded_at", "TIMESTAMP"),
        ]
        table = bq_client.create_table(bigquery.Table(table_id, schema=schema))
        time.sleep(3)
        table = bq_client.get_table(table_id)

    existing_columns = {field.name for field in table.schema}
    new_fields = []

    for key, value in extracted_data.items():
        col_name = f"kpi_{re.sub(r'[^a-zA-Z0-9_]', '_', key).lower()}"
        if col_name not in existing_columns:
            dtype = "FLOAT" if isinstance(value, (int, float)) else "STRING"
            new_fields.append(bigquery.SchemaField(col_name, dtype))

    if new_fields:
        table.schema += new_fields
        bq_client.update_table(table, ["schema"])
    
    return table_id

# ==========================================
# 🚀 MAIN TRIGGER
# ==========================================
@app.post("/")
def handle_event():
    # Only allow Cloud Storage triggers or Auth'd users
    # For a purely automated trigger, you might bypass UID check if internal
    # But for a user-facing API, we keep security high:
    uid = get_user_id(request)
    
    payload = request.get_json(silent=True) or {}
    data = payload.get("data", payload)
    file_path = data.get("name", "")

    # Skip if not an incoming file
    if "incoming/" not in file_path: return ("OK", 200)

    # Automatically extract UID from the path if not provided in header
    # Path: incoming/{uid}/filename.pdf
    if not uid:
        try:
            uid = file_path.split("/")[1]
        except:
            return ("Missing UID", 400)

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(data.get("bucket", "pdf_platform_main"))
        blob = bucket.blob(file_path)
        pdf_content = blob.download_as_bytes()
        
        kpis = extract_with_gemini(pdf_content, uid)
        target_table = sync_bigquery_schema(uid, kpis)

        row = {
            "row_id": str(datetime.datetime.now().timestamp()),
            "file_name": file_path,
            "uploaded_at": datetime.datetime.utcnow().isoformat()
        }
        for k, v in kpis.items():
            row[f"kpi_{re.sub(r'[^a-zA-Z0-9_]', '_', k).lower()}"] = v

        bq_client = bigquery.Client()
        bq_client.insert_rows_json(target_table, [row])
        
        # Cleanup
        new_path = file_path.replace("incoming/", "processed/")
        bucket.copy_blob(blob, bucket, new_path)
        blob.delete()
        
        return jsonify({"status": "processed", "file": file_path}), 200

    except Exception as e:
        print(f"🔥 Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
