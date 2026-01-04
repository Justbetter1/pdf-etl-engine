import os
import json
import datetime
import re
import time
import firebase_admin
from firebase_admin import auth, firestore
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from google.cloud import storage, bigquery
from google import genai
from google.genai import types

app = Flask(__name__)

# 🛡️ Enable CORS for the whole app
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# --- CONFIGURATION ---
PROJECT_ID = "pdf-etl-479411"
DATASET = "etl_reports"
LOCATION = "us-central1"

# Initialize Clients once at startup
if not firebase_admin._apps:
    firebase_admin.initialize_app()
db = firestore.client()
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# ==========================================
# 🛡️ HELPER: AUTHENTICATION
# ==========================================
def get_user_id(req):
    auth_header = req.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    try:
        token = auth_header.split("Bearer ")[1]
        decoded_token = auth.verify_id_token(token)
        return decoded_token["uid"]
    except Exception:
        return None

# ==========================================
# 📊 HELPER: BIGQUERY DYNAMIC SCHEMA
# ==========================================
def sync_bigquery_schema(uid, extracted_data):
    bq_client = bigquery.Client()
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
        time.sleep(3) # Wait for BQ propagation
        table = bq_client.get_table(table_id)

    existing_cols = {field.name for field in table.schema}
    new_fields = []
    for key, value in extracted_data.items():
        col_name = f"kpi_{re.sub(r'[^a-zA-Z0-9_]', '_', key).lower()}"
        if col_name not in existing_cols:
            dtype = "FLOAT" if isinstance(value, (int, float)) else "STRING"
            new_fields.append(bigquery.SchemaField(col_name, dtype))

    if new_fields:
        table.schema += new_fields
        bq_client.update_table(table, ["schema"])
    return table_id

# ==========================================
# ✨ ROUTE 1: ACCOUNT SETUP
# ==========================================
@app.route("/setup-account", methods=["POST", "OPTIONS"])
def setup_account():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    uid = get_user_id(request)
    if not uid: return jsonify({"error": "Unauthorized"}), 401
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket("pdf_platform_main")
        
        for folder in ["incoming", "processed"]:
            blob = bucket.blob(f"{folder}/{uid}/.placeholder")
            blob.upload_from_string("Init")

        db.collection("tenants").document(uid).collection("folders").document("default").set({
            "hint": "Extract all key data points into a flat JSON format.",
            "is_trained": False,
            "created_at": datetime.datetime.utcnow()
        })
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# 🚀 ROUTE 2: PDF ENGINE (THE BRAIN)
# ==========================================
@app.route("/", methods=["POST", "OPTIONS"])
def handle_event():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    payload = request.get_json(silent=True) or {}
    data = payload.get("data", payload)
    file_path = data.get("name", "")

    # 1. Skip system files and non-PDFs
    if ".placeholder" in file_path or not file_path.lower().endswith(".pdf"):
        print(f"⏩ Skipping: {file_path}")
        return jsonify({"status": "ignored"}), 200

    # 2. Extract UID from path if not in headers (common for GCS triggers)
    uid = get_user_id(request)
    if not uid and "/" in file_path:
        uid = file_path.split("/")[1]

    if not uid: 
        return jsonify({"error": "No User ID identified"}), 400

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket("pdf_platform_main")
        blob = bucket.blob(file_path)
        
        # 🛡️ RACE CONDITION CHECK: Ensure file hasn't been moved by a concurrent retry
        if not blob.exists():
            print(f"⚠️ File already moved or deleted: {file_path}")
            return jsonify({"status": "already_processed"}), 200

        pdf_bytes = blob.download_as_bytes()
        
        # 3. GEMINI EXTRACTION
        user_ref = db.collection("tenants").document(uid).collection("folders").document("default").get()
        hint = user_ref.to_dict().get("hint", "Extract data") if user_ref.exists else "Extract data"
        
        prompt = f"Analyze this PDF. Instructions: {hint}. Return ONLY flat JSON. No markdown."
        
        resp = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        
        # 🛡️ JSON ROBUSTNESS: Clean potential Markdown formatting
        try:
            clean_text = re.sub(r'^```json\s*|```$', '', resp.text.strip(), flags=re.MULTILINE)
            kpis = json.loads(clean_text)
            if isinstance(kpis, list): kpis = kpis[0]
        except Exception as json_err:
            print(f"❌ JSON Parse Error: {json_err}. Raw text: {resp.text[:100]}")
            raise Exception("Invalid JSON format from AI")

        # 4. BIGQUERY SAVE
        target_table = sync_bigquery_schema(uid, kpis)
        row = {
            "row_id": f"row_{int(time.time())}",
            "file_name": file_path,
            "uploaded_at": datetime.datetime.utcnow().isoformat()
        }
        for k, v in kpis.items():
            safe_key = f"kpi_{re.sub(r'[^a-zA-Z0-9_]', '_', k).lower()}"
            row[safe_key] = v

        bq_errors = bigquery.Client().insert_rows_json(target_table, [row])
        if bq_errors:
            raise Exception(f"BigQuery Error: {bq_errors}")

        # 5. MOVE FILE (Last Step)
        new_path = file_path.replace("incoming/", "processed/")
        bucket.copy_blob(blob, bucket, new_path)
        blob.delete()
        
        print(f"✅ Success: {file_path}")
        return jsonify({"status": "processed", "file": file_path}), 200

    except Exception as e:
        print(f"🔥 Engine Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "POST")
    return response, 204

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
