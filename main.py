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

# 1. Initialize Flask
app = Flask(__name__)

# 2. Strong CORS Configuration
CORS(app, resources={r"/*": {
    "origins": "*",
    "allow_headers": ["Authorization", "Content-Type", "Accept"],
    "methods": ["GET", "POST", "OPTIONS"],
    "max_age": 3600
}}, supports_credentials=True)

# 3. Configuration
PROJECT_ID = "pdf-etl-479411"
DATASET = "etl_reports"
LOCATION = "us-central1"
BUCKET_NAME = "pdf_platform_main" 

# 4. Initialize Google Services
try:
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    db = firestore.client()
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    print(f"🚀 Backend fully operational. Project: {PROJECT_ID}")
except Exception as e:
    print(f"❌ Startup Error: {e}")

# ==========================================
# 🛡️ AUTHENTICATION HELPER
# ==========================================
def get_user_id(req):
    auth_header = req.headers.get("Authorization")
    if not auth_header:
        return None
    try:
        token = auth_header.split("Bearer ")[1]
        decoded_token = auth.verify_id_token(token)
        return decoded_token["uid"]
    except Exception as e:
        print(f"❌ Auth Error: {e}")
        return None

# ==========================================
# 📊 BIGQUERY SCHEMA SYNC & TABLE CREATION
# ==========================================
def sync_bigquery_schema(uid, folder_id, kpi_list):
    bq_client = bigquery.Client()
    # Clean names for BigQuery compatibility
    clean_uid = re.sub(r'[^a-zA-Z0-9_]', '_', uid).lower()
    clean_folder = re.sub(r'[^a-zA-Z0-9_]', '_', folder_id).lower()
    table_id = f"{PROJECT_ID}.{DATASET}.{clean_uid}_{clean_folder}"
    
    try:
        table = bq_client.get_table(table_id)
    except Exception:
        # Create table if it doesn't exist with base columns
        schema = [
            bigquery.SchemaField("row_id", "STRING"),
            bigquery.SchemaField("file_name", "STRING"),
            bigquery.SchemaField("uploaded_at", "TIMESTAMP"),
        ]
        table = bq_client.create_table(bigquery.Table(table_id, schema=schema))
        time.sleep(2) # Wait for propagation
        table = bq_client.get_table(table_id)

    existing_cols = {field.name for field in table.schema}
    new_fields = []
    
    for kpi in kpi_list:
        # Construct the column name consistently
        col_name = f"kpi_{re.sub(r'[^a-zA-Z0-9_]', '_', kpi).lower()}"
        if col_name not in existing_cols:
            new_fields.append(bigquery.SchemaField(col_name, "STRING"))

    if new_fields:
        table.schema += new_fields
        bq_client.update_table(table, ["schema"])
        print(f"✅ Table {table_id} updated with new columns.")
        
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
            "setup_date": datetime.datetime.utcnow(),
            "uid": uid
        }, merge=True)
        return jsonify({"status": "success", "uid": uid}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# 📂 2. DYNAMIC FOLDER CREATION
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

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # Initialize storage structure with placeholders
        bucket.blob(f"incoming/{uid}/{folder_id}/master/.placeholder").upload_from_string("init")
        bucket.blob(f"incoming/{uid}/{folder_id}/batch/.placeholder").upload_from_string("init")

        folder_data = {
            "display_name": name,
            "folder_id": folder_id,
            "is_trained": False,
            "status": "waiting_for_training",
            "created_at": datetime.datetime.utcnow().isoformat(),
            "owner": uid
        }
        db.collection("tenants").document(uid).collection("folders").document(folder_id).set(folder_data)

        return jsonify({"status": "success", "folder_id": folder_id, "folder": folder_data}), 200
    except Exception as e:
        print(f"❌ Create Folder Error: {e}")
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
    print(f"🔍 LOG: Analyzing master file: {file_path}")

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(file_path)
        
        if not blob.exists():
            return jsonify({"error": f"File {file_path} not found"}), 404

        pdf_bytes = blob.download_as_bytes()

        # Enhanced prompt for strict JSON and multi-PDF support
        prompt = "Extract all data labels and headers found in this document. Return ONLY a valid JSON object of {field_name: example_value}. Ensure keys are descriptive."
        
        resp = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        
        # Clean JSON from markdown if necessary
        raw_text = resp.text.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r'^```json\s*|```$', '', raw_text, flags=re.MULTILINE)
        
        detected_dict = json.loads(raw_text)
        if isinstance(detected_dict, list):
            detected_dict = detected_dict[0] if len(detected_dict) > 0 else {}
        
        formatted_kpis = [{"key": k, "value": str(v)} for k, v in detected_dict.items()]
        
        return jsonify({"detected_kpis": formatted_kpis}), 200
    except Exception as e:
        print(f"❌ Analyze Master Crash: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ==========================================
# ✅ 4. CONFIRM SELECTED KPIs
# ==========================================
@app.route("/confirm-kpis", methods=["POST", "OPTIONS"])
def confirm_kpis():
    if request.method == "OPTIONS": return _build_cors_preflight_response()
    uid = get_user_id(request)
    if not uid: return jsonify({"error": "Unauthorized"}), 401
    
    try:
        payload = request.get_json()
        folder_id = payload.get("folder_id")
        selected_kpis = payload.get("selected_kpis")

        db.collection("tenants").document(uid).collection("folders").document(folder_id).update({
            "selected_kpis": selected_kpis,
            "is_trained": True,
            "status": "active"
        })
        
        # Create/Update BigQuery Table
        sync_bigquery_schema(uid, folder_id, selected_kpis)
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"❌ Confirm KPIs Error: {e}")
        return jsonify({"error": str(e)}), 500

# ==========================================
# 🚜 5. BATCH ENGINE (GCS TRIGGER HANDLER)
# ==========================================
@app.route("/", methods=["POST", "OPTIONS"])
def gcs_trigger_handler():
    if request.method == "OPTIONS": return _build_cors_preflight_response()
    
    payload = request.get_json(silent=True) or {}
    data = payload.get("data", payload)
    file_path = data.get("name", "")

    # CRITICAL FIX: Ignore files already in processed or placeholders
    if "processed/" in file_path or ".placeholder" in file_path or not file_path.lower().endswith(".pdf"):
        return jsonify({"status": "ignored"}), 200

    parts = file_path.split("/")
    # Expected: incoming/{uid}/{folder_id}/batch/{filename}
    if len(parts) < 5 or parts[0] != "incoming" or parts[3] != "batch":
        return jsonify({"status": "ignored_path"}), 200
    
    uid = parts[1]
    folder_id = parts[2]

    try:
        # 1. Get training data from Firestore
        folder_ref = db.collection("tenants").document(uid).collection("folders").document(folder_id).get()
        if not folder_ref.exists:
            return jsonify({"error": "Folder not trained"}), 200
            
        folder_data = folder_ref.to_dict()
        kpis = folder_data.get("selected_kpis", [])

        # 2. Extract data using Gemini
        storage_client = storage.Client()
        source_bucket = storage_client.bucket(BUCKET_NAME)
        blob = source_bucket.blob(file_path)
        pdf_bytes = blob.download_as_bytes()

        prompt = f"Extract the values for these specific keys: {kpis}. Return ONLY a JSON object."
        resp = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        
        raw_extract = resp.text.strip()
        if raw_extract.startswith("```"):
            raw_extract = re.sub(r'^```json\s*|```$', '', raw_extract, flags=re.MULTILINE)
        
        extracted_data = json.loads(raw_extract)
        if isinstance(extracted_data, list):
            extracted_data = extracted_data[0]

        # 3. Insert into BigQuery
        table_id = sync_bigquery_schema(uid, folder_id, kpis)
        row = {
            "row_id": f"row_{int(time.time())}",
            "file_name": file_path.split("/")[-1],
            "uploaded_at": datetime.datetime.utcnow().isoformat()
        }
        
        # Map extracted data to BigQuery kpi_ columns
        for k in kpis:
            safe_col_name = f"kpi_{re.sub(r'[^a-zA-Z0-9_]', '_', k).lower()}"
            row[safe_col_name] = str(extracted_data.get(k, "N/A"))

        bq_client = bigquery.Client()
        errors = bq_client.insert_rows_json(table_id, [row])
        
        if errors:
            print(f"❌ BigQuery Insert Errors: {errors}")
            return jsonify({"error": "BigQuery Insert Failed"}), 200

        # 4. Move file to processed (This prevents re-triggering due to the check at start)
        new_path = file_path.replace("incoming/", "processed/")
        source_bucket.copy_blob(blob, source_bucket, new_path)
        blob.delete()

        print(f"✅ Successfully processed {file_path}")
        return jsonify({"status": "success"}), 200

    except Exception as e:
        print(f"❌ Batch Engine Error: {str(e)}")
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
        print(f"❌ Fetch Results Error: {e}")
        return jsonify({"results": []}), 200

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization,Accept")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    response.headers.add("Access-Control-Max-Age", "3600")
    return response, 204

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
