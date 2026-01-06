import os
import re
import datetime
import time
import json
import firebase_admin
from firebase_admin import auth, firestore
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from google.cloud import storage, bigquery
from google import genai
from google.genai import types

# 1. Initialize Flask & CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# 2. Configuration - DOUBLE CHECK THESE IN GCP CONSOLE
PROJECT_ID = "pdf-etl-479411"
DATASET = "etl_reports"
LOCATION = "us-central1"
# Make sure this matches your bucket name exactly
BUCKET_NAME = "pdf_platform_main" 

# 3. Initialize Google Services
try:
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    db = firestore.client()
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    print(f"✅ Service initialized. Target Bucket: {BUCKET_NAME}")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

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
    except Exception as e:
        print(f"⚠️ Auth Error: {e}")
        return None

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
# ✨ 1. ACCOUNT SETUP
# ==========================================
@app.route("/setup-account", methods=["POST", "OPTIONS"])
def setup_account():
    if request.method == "OPTIONS": return _build_cors_preflight_response()
    uid = get_user_id(request)
    if not uid: return jsonify({"error": "Unauthorized"}), 401
    
    try:
        db.collection("tenants").document(uid).set({
            "account_status": "active",
            "setup_date": datetime.datetime.utcnow()
        }, merge=True)
        return jsonify({"status": "success", "uid": uid}), 200
    except Exception as e:
        print(f"❌ Account Setup Error: {e}")
        return jsonify({"error": str(e)}), 500

# ==========================================
# 📂 2. DYNAMIC FOLDER CREATION (FIXED)
# ==========================================
@app.route("/create-folder", methods=["POST", "OPTIONS"])
def create_folder():
    if request.method == "OPTIONS": return _build_cors_preflight_response()
    uid = get_user_id(request)
    if not uid: return jsonify({"error": "Unauthorized"}), 401
    
    try:
        payload = request.get_json()
        name = payload.get("name")
        folder_id = re.sub(r'[^a-zA-Z0-9_]', '_', name).lower()

        # 1. Create Folders in Storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # Check if bucket exists first
        if not bucket.exists():
            print(f"❌ Error: Bucket {BUCKET_NAME} does not exist!")
            return jsonify({"error": f"Bucket {BUCKET_NAME} not found"}), 500

        # Upload placeholders
        m_path = f"incoming/{uid}/{folder_id}/master/.placeholder"
        b_path = f"incoming/{uid}/{folder_id}/batch/.placeholder"
        
        bucket.blob(m_path).upload_from_string("init")
        bucket.blob(b_path).upload_from_string("init")
        print(f"✅ Created Storage paths for {folder_id}")

        # 2. Update Firestore
        db.collection("tenants").document(uid).collection("folders").document(folder_id).set({
            "display_name": name,
            "folder_id": folder_id,
            "is_trained": False,
            "status": "waiting_for_training",
            "created_at": datetime.datetime.utcnow()
        })
        print(f"✅ Updated Firestore for {folder_id}")

        return jsonify({"status": "success", "folder_id": folder_id}), 200
    except Exception as e:
        print(f"❌ Create Folder Failed: {e}")
        return jsonify({"error": str(e)}), 500

# ==========================================
# 🧠 3. MASTER PDF ANALYSIS
# ==========================================
@app.route("/analyze-master", methods=["POST", "OPTIONS"])
def analyze_master():
    if request.method == "OPTIONS": return _build_cors_preflight_response()
    uid = get_user_id(request)
    if not uid: return jsonify({"error": "Unauthorized"}), 401
    
    payload = request.get_json()
    file_path = payload.get("file_path") 

    try:
        storage_client = storage.Client()
        blob = storage_client.bucket(BUCKET_NAME).blob(file_path)
        pdf_bytes = blob.download_as_bytes()

        prompt = "List data labels in this PDF. Return ONLY JSON {field: example}."
        resp = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        detected_data = json.loads(re.sub(r'^```json\s*|```$', '', resp.text.strip(), flags=re.MULTILINE))
        return jsonify({"detected_kpis": detected_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# ✅ 4. CONFIRM SELECTED KPIs
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
            "is_trained": True,
            "status": "active"
        })
        sync_bigquery_schema(uid, folder_id, selected_kpis)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# 🚜 5. BATCH ENGINE (GCS TRIGGER)
# ==========================================
@app.route("/", methods=["POST"])
def gcs_trigger_handler():
    payload = request.get_json(silent=True) or {}
    data = payload.get("data", payload)
    file_path = data.get("name", "")

    if not file_path.lower().endswith(".pdf") or ".placeholder" in file_path:
        return jsonify({"status": "ignored"}), 200

    parts = file_path.split("/")
    if len(parts) < 5 or parts[3] != "batch":
        return jsonify({"status": "ignored_path"}), 200
    
    uid, folder_id = parts[1], parts[2]

    try:
        folder_ref = db.collection("tenants").document(uid).collection("folders").document(folder_id).get()
        folder_data = folder_ref.to_dict()
        if not folder_ref.exists or not folder_data.get("is_trained"):
            return jsonify({"status": "waiting_for_training"}), 200

        kpis = folder_data.get("selected_kpis", [])
        storage_client = storage.Client()
        blob = storage_client.bucket(BUCKET_NAME).blob(file_path)
        pdf_bytes = blob.download_as_bytes()

        prompt = f"Extract fields: {kpis}. Return JSON."
        resp = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        extracted = json.loads(re.sub(r'^```json\s*|```$', '', resp.text.strip(), flags=re.MULTILINE))

        table_id = sync_bigquery_schema(uid, folder_id, kpis)
        row = {
            "row_id": f"row_{int(time.time())}",
            "file_name": file_path.split("/")[-1],
            "uploaded_at": datetime.datetime.utcnow().isoformat()
        }
        for k in kpis:
            safe_k = f"kpi_{re.sub(r'[^a-zA-Z0-9_]', '_', k).lower()}"
            row[safe_k] = str(extracted.get(k, "N/A"))

        bigquery.Client().insert_rows_json(table_id, [row])
        
        new_path = file_path.replace("incoming/", "processed/")
        storage_client.bucket(BUCKET_NAME).copy_blob(blob, storage_client.bucket(BUCKET_NAME), new_path)
        blob.delete()

        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 200

# ==========================================
# 📈 6. FETCH RESULTS API
# ==========================================
@app.route("/get-results", methods=["GET", "OPTIONS"])
def get_results():
    if request.method == "OPTIONS": return _build_cors_preflight_response()
    uid = get_user_id(request)
    if not uid: return jsonify({"error": "Unauthorized"}), 401
    
    folder_id = request.args.get("folder_id")
    if not folder_id: return jsonify({"error": "folder_id required"}), 400

    try:
        bq_client = bigquery.Client()
        clean_uid = re.sub(r'[^a-zA-Z0-9_]', '_', uid).lower()
        clean_folder = re.sub(r'[^a-zA-Z0-9_]', '_', folder_id).lower()
        table_id = f"{PROJECT_ID}.{DATASET}.{clean_uid}_{clean_folder}"
        
        query = f"SELECT * FROM `{table_id}` ORDER BY uploaded_at DESC LIMIT 100"
        query_job = bq_client.query(query)
        results = [dict(row) for row in query_job]
        
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"results": []}), 200

# --- CORS PREFLIGHT ---
def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response, 204

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
