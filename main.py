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

# 🛡️ GLOBAL CORS
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# --- CONFIGURATION ---
PROJECT_ID = "pdf-etl-479411"
DATASET = "etl_reports"
LOCATION = "us-central1"
BUCKET_NAME = "pdf_platform_main"

if not firebase_admin._apps:
    firebase_admin.initialize_app()
db = firestore.client()
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# ==========================================
# 🛡️ AUTHENTICATION HELPER
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
# ✨ ROUTE: ACCOUNT SETUP (FIXED)
# ==========================================
@app.route("/setup-account", methods=["GET", "POST", "OPTIONS"])
def setup_account():
    if request.method == "OPTIONS": 
        return _build_cors_preflight_response()
    
    # Allow GET just to verify the endpoint is alive in browser
    if request.method == "GET":
        return jsonify({"status": "online", "message": "Use POST with Auth to setup account"}), 200

    uid = get_user_id(request)
    if not uid: 
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # Create user folder structure in Storage
        for folder in ["incoming", "processed"]:
            blob = bucket.blob(f"{folder}/{uid}/.placeholder")
            blob.upload_from_string("Init")

        # Create basic folder docs in Firestore
        for folder_type in ["invoices", "manifests"]:
            db.collection("tenants").document(uid).collection("folders").document(folder_type).set({
                "name": folder_type.capitalize(),
                "status": "setup",
                "is_trained": False,
                "selected_kpis": [],
                "created_at": datetime.datetime.utcnow()
            }, merge=True)

        return jsonify({"status": "success", "uid": uid}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# 📊 BIGQUERY SCHEMA SYNC
# ==========================================
def sync_bigquery_schema(uid, folder_id, kpi_list):
    bq_client = bigquery.Client()
    clean_uid = re.sub(r'[^a-zA-Z0-9_]', '_', uid).lower()
    clean_folder = re.sub(r'[^a-zA-Z0-9_]', '_', folder_id).lower()
    table_id = f"{PROJECT_ID}.{DATASET}.{clean_uid}_{clean_folder}"
    
    try:
        table = bq_client.get_table(table_id)
    except Exception:
        schema = [
            bigquery.SchemaField("row_id", "STRING"),
            bigquery.SchemaField("file_name", "STRING"),
            bigquery.SchemaField("uploaded_at", "TIMESTAMP"),
        ]
        table = bq_client.create_table(bigquery.Table(table_id, schema=schema))
        time.sleep(2)
        table = bq_client.get_table(table_id)

    existing_cols = {field.name for field in table.schema}
    new_fields = []
    for kpi in kpi_list:
        col_name = f"kpi_{re.sub(r'[^a-zA-Z0-9_]', '_', kpi).lower()}"
        if col_name not in existing_cols:
            new_fields.append(bigquery.SchemaField(col_name, "STRING"))

    if new_fields:
        table.schema += new_fields
        bq_client.update_table(table, ["schema"])
    return table_id

# ==========================================
# ✨ ROUTE: MASTER PDF ANALYSIS
# ==========================================
@app.route("/analyze-master", methods=["POST", "OPTIONS"])
def analyze_master():
    if request.method == "OPTIONS": return _build_cors_preflight_response()
    uid = get_user_id(request)
    if not uid: return jsonify({"error": "Unauthorized"}), 401
    
    payload = request.get_json()
    file_path = payload.get("file_path")
    folder_id = payload.get("folder_id")

    try:
        storage_client = storage.Client()
        blob = storage_client.bucket(BUCKET_NAME).blob(file_path)
        pdf_bytes = blob.download_as_bytes()

        prompt = "Analyze this PDF. List every data field/KPI found. Return ONLY a JSON object with {field_name: example_value}."
        resp = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )

        clean_text = re.sub(r'^```json\s*|```$', '', resp.text.strip(), flags=re.MULTILINE)
        detected_data = json.loads(clean_text)

        db.collection("tenants").document(uid).collection("folders").document(folder_id).update({
            "master_file_url": f"gs://{BUCKET_NAME}/{file_path}",
            "status": "training"
        })
        return jsonify({"detected_kpis": detected_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# ✅ ROUTE: CONFIRM SELECTED KPIs
# ==========================================
@app.route("/confirm-kpis", methods=["POST", "OPTIONS"])
def confirm_kpis():
    if request.method == "OPTIONS": return _build_cors_preflight_response()
    uid = get_user_id(request)
    payload = request.get_json()
    folder_id = payload.get("folder_id")
    selected_kpis = payload.get("selected_kpis")

    try:
        db.collection("tenants").document(uid).collection("folders").document(folder_id).update({
            "selected_kpis": selected_kpis,
            "status": "active",
            "is_trained": True
        })
        sync_bigquery_schema(uid, folder_id, selected_kpis)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# 🚀 ROUTE: BATCH ENGINE
# ==========================================
@app.route("/", methods=["POST", "OPTIONS"])
def process_child_pdf():
    if request.method == "OPTIONS": return _build_cors_preflight_response()
    payload = request.get_json(silent=True) or {}
    data = payload.get("data", payload)
    file_path = data.get("name", "")

    if ".placeholder" in file_path or not file_path.lower().endswith(".pdf"):
        return jsonify({"status": "ignored"}), 200

    parts = file_path.split("/")
    if len(parts) < 3: return jsonify({"error": "Invalid path"}), 400
    uid, folder_id = parts[1], parts[2]

    try:
        folder_ref = db.collection("tenants").document(uid).collection("folders").document(folder_id).get()
        if not folder_ref.exists: raise Exception("Folder not trained")
        kpi_list = folder_ref.to_dict().get("selected_kpis", [])

        storage_client = storage.Client()
        blob = storage_client.bucket(BUCKET_NAME).blob(file_path)
        pdf_bytes = blob.download_as_bytes()

        prompt = f"Extract only these fields from this PDF: {kpi_list}. Return JSON."
        resp = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        
        clean_text = re.sub(r'^```json\s*|```$', '', resp.text.strip(), flags=re.MULTILINE)
        extracted = json.loads(clean_text)

        table_id = sync_bigquery_schema(uid, folder_id, kpi_list)
        row = {
            "row_id": f"row_{int(time.time())}",
            "file_name": file_path,
            "uploaded_at": datetime.datetime.utcnow().isoformat()
        }
        for k in kpi_list:
            safe_key = f"kpi_{re.sub(r'[^a-zA-Z0-9_]', '_', k).lower()}"
            row[safe_key] = str(extracted.get(k, ""))

        bigquery.Client().insert_rows_json(table_id, [row])
        new_path = file_path.replace("incoming/", "processed/")
        storage_client.bucket(BUCKET_NAME).copy_blob(blob, storage_client.bucket(BUCKET_NAME), new_path)
        blob.delete()
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response, 204

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
