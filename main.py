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

def get_user_email(req):
    """Get user email from Firebase token."""
    auth_header = req.headers.get("Authorization")
    if not auth_header:
        return None
    try:
        token = auth_header.split("Bearer ")[1]
        decoded_token = auth.verify_id_token(token)
        return decoded_token.get("email", "").lower()
    except Exception as e:
        print(f"❌ Auth Error: {e}")
        return None

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization,Accept")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    response.headers.add("Access-Control-Max-Age", "3600")
    return response, 204

# ==========================================
# 📊 BIGQUERY SCHEMA SYNC & TABLE CREATION
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
        context_hint = payload.get("context_hint", "")
        folder_id = re.sub(r'[^a-zA-Z0-9_]', '_', name).lower()

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        
        bucket.blob(f"incoming/{uid}/{folder_id}/master/.placeholder").upload_from_string("init")
        bucket.blob(f"incoming/{uid}/{folder_id}/batch/.placeholder").upload_from_string("init")

        folder_data = {
            "display_name": name,
            "folder_id": folder_id,
            "context_hint": context_hint,
            "is_trained": False,
            "status": "waiting_for_training",
            "created_at": datetime.datetime.utcnow().isoformat(),
            "owner": uid,
            "shared_with": {}
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
    context_hint = payload.get("context_hint", "")
    
    print(f"🔍 LOG: Analyzing master with context: {context_hint}")

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(file_path)
        
        if not blob.exists():
            return jsonify({"error": f"File {file_path} not found"}), 404

        pdf_bytes = blob.download_as_bytes()

        prompt = f"""
        Extract all data labels and headers found in this document. 
        USER CONTEXT: {context_hint if context_hint else "Generic business document."}
        Return ONLY a valid JSON object of {{field_name: example_value}}. 
        Ensure keys are descriptive and relevant to the provided USER CONTEXT.
        """
        
        resp = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            ),
        )
        
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
        
        sync_bigquery_schema(uid, folder_id, selected_kpis)
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"❌ Confirm KPIs Error: {e}")
        return jsonify({"error": str(e)}), 500

# ==========================================
# 📋 5. GET KPIs (for shared folder status)
# ==========================================
@app.route("/get-kpis", methods=["GET", "OPTIONS"])
def get_kpis():
    if request.method == "OPTIONS": return _build_cors_preflight_response()
    uid = get_user_id(request)
    if not uid: return jsonify({"error": "Unauthorized"}), 401
    
    folder_id = request.args.get("folder_id")
    owner_id = request.args.get("owner_id")
    
    if not folder_id: return jsonify({"error": "folder_id required"}), 400

    try:
        target_uid = owner_id if owner_id else uid
        
        folder_ref = db.collection("tenants").document(target_uid).collection("folders").document(folder_id).get()
        
        if not folder_ref.exists:
            return jsonify({"error": "Folder not found"}), 404
            
        folder_data = folder_ref.to_dict()
        
        # Permission check: must be owner or have a share document
        # For shared users, we check the global shares collection
        is_owner = uid == folder_data.get("owner")
        has_share = False
        
        if not is_owner:
            # Check global shares collection for this user
            shares_query = db.collection("shares").where("folderId", "==", folder_id).where("ownerId", "==", target_uid).get()
            for share_doc in shares_query:
                share_data = share_doc.to_dict()
                # Match by email from the token (you may need to get email from token)
                # For simplicity, we also check shared_with map if it exists
                if uid in folder_data.get("shared_with", {}):
                    has_share = True
                    break
            # Also accept if shared_with map contains uid
            if uid in folder_data.get("shared_with", {}):
                has_share = True
        
        if not is_owner and not has_share:
            # Fallback: allow if there's any share doc for this folder (looser check)
            # This handles email-based sharing where we can't easily match uid
            shares_query = db.collection("shares").where("folderId", "==", folder_id).where("ownerId", "==", target_uid).get()
            if len(list(shares_query)) > 0:
                has_share = True
        
        if not is_owner and not has_share:
            return jsonify({"error": "Access denied"}), 403
        
        return jsonify({
            "is_trained": folder_data.get("is_trained", False),
            "selected_kpis": folder_data.get("selected_kpis", []),
            "context_hint": folder_data.get("context_hint", ""),
            "status": folder_data.get("status", "unknown")
        }), 200
        
    except Exception as e:
        print(f"❌ Get KPIs Error: {e}")
        return jsonify({"error": str(e)}), 500

# ==========================================
# 📤 6. UPLOAD BATCH FILE (for shared users)
# ==========================================
@app.route("/upload-batch-file", methods=["POST", "OPTIONS"])
def upload_batch_file():
    if request.method == "OPTIONS": return _build_cors_preflight_response()
    
    uid = get_user_id(request)
    user_email = get_user_email(request)
    
    if not uid or not user_email:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        # Get form data
        folder_id = request.form.get("folder_id")
        owner_id = request.form.get("owner_id")
        file = request.files.get("file")

        if not folder_id or not owner_id or not file:
            return jsonify({"error": "Missing required fields: folder_id, owner_id, or file"}), 400

        # Validate file is PDF
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        # Sanitize email for document ID lookup
        sanitized_email = re.sub(r'[@.]', '_', user_email)
        share_doc_id = f"{owner_id}_{folder_id}_{sanitized_email}"

        # Verify share permission in Firestore
        share_ref = db.collection("shares").document(share_doc_id).get()

        if not share_ref.exists:
            return jsonify({"error": "Share not found. You do not have access to this folder."}), 403

        share_data = share_ref.to_dict()
        permission = share_data.get("permission", "view")

        if permission != "edit":
            return jsonify({"error": "You have view-only access. Upload not permitted."}), 403

        # Sanitize filename
        original_filename = file.filename or "unnamed.pdf"
        sanitized_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', original_filename)

        # Upload to Firebase Storage with admin privileges
        storage_path = f"incoming/{owner_id}/{folder_id}/batch/{sanitized_filename}"
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(storage_path)
        
        # Upload file content
        blob.upload_from_file(file, content_type="application/pdf")

        print(f"✅ Shared user {user_email} uploaded {sanitized_filename} to {storage_path}")

        return jsonify({
            "success": True,
            "path": storage_path,
            "filename": sanitized_filename
        }), 200

    except Exception as e:
        print(f"❌ Upload Batch File Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ==========================================
# 🚜 7. BATCH ENGINE (GCS TRIGGER HANDLER)
# ==========================================
@app.route("/", methods=["POST", "OPTIONS"])
def gcs_trigger_handler():
    if request.method == "OPTIONS": return _build_cors_preflight_response()
    
    payload = request.get_json(silent=True) or {}
    data = payload.get("data", payload)
    file_path = data.get("name", "")

    if "processed/" in file_path or ".placeholder" in file_path or not file_path.lower().endswith(".pdf"):
        return jsonify({"status": "ignored"}), 200

    parts = file_path.split("/")
    if len(parts) < 5 or parts[0] != "incoming" or parts[3] != "batch":
        return jsonify({"status": "ignored_path"}), 200
    
    uid = parts[1]
    folder_id = parts[2]

    try:
        folder_ref = db.collection("tenants").document(uid).collection("folders").document(folder_id).get()
        if not folder_ref.exists:
            return jsonify({"error": "Folder not trained"}), 200
            
        folder_data = folder_ref.to_dict()
        kpis = folder_data.get("selected_kpis", [])
        context_hint = folder_data.get("context_hint", "")

        storage_client = storage.Client()
        source_bucket = storage_client.bucket(BUCKET_NAME)
        blob = source_bucket.blob(file_path)
        pdf_bytes = blob.download_as_bytes()

        prompt = f"""
        Extract the values for these specific keys: {kpis}. 
        CONTEXT: {context_hint if context_hint else "Generic data extraction."}
        Return ONLY a JSON object. If a value is missing, return "N/A".
        """
        
        resp = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            ),
        )
        
        raw_extract = resp.text.strip()
        if raw_extract.startswith("```"):
            raw_extract = re.sub(r'^```json\s*|```$', '', raw_extract, flags=re.MULTILINE)
        
        extracted_data = json.loads(raw_extract)
        if isinstance(extracted_data, list):
            extracted_data = extracted_data[0]

        owner_uid = folder_data.get("owner", uid)
        table_id = sync_bigquery_schema(owner_uid, folder_id, kpis)
        row = {
            "row_id": f"row_{int(time.time())}",
            "file_name": file_path.split("/")[-1],
            "uploaded_at": datetime.datetime.utcnow().isoformat()
        }
        
        for k in kpis:
            safe_col_name = f"kpi_{re.sub(r'[^a-zA-Z0-9_]', '_', k).lower()}"
            row[safe_col_name] = str(extracted_data.get(k, "N/A"))

        bq_client = bigquery.Client()
        errors = bq_client.insert_rows_json(table_id, [row])
        
        if errors:
            print(f"❌ BigQuery Insert Errors: {errors}")
            return jsonify({"error": "BigQuery Insert Failed"}), 200

        new_path = file_path.replace("incoming/", "processed/")
        source_bucket.copy_blob(blob, source_bucket, new_path)
        blob.delete()

        print(f"✅ Successfully processed {file_path}")
        return jsonify({"status": "success"}), 200

    except Exception as e:
        print(f"❌ Batch Engine Error: {str(e)}")
        return jsonify({"error": str(e)}), 200

# ==========================================
# 📈 8. FETCH RESULTS API
# ==========================================
@app.route("/get-results", methods=["GET", "OPTIONS"])
def get_results():
    if request.method == "OPTIONS": return _build_cors_preflight_response()
    uid = get_user_id(request)
    if not uid: return jsonify({"error": "Unauthorized"}), 401
    
    folder_id = request.args.get("folder_id")
    owner_id = request.args.get("owner_id")  # NEW: accept owner_id for shared folders
    
    if not folder_id: return jsonify({"error": "folder_id required"}), 400

    try:
        # Use owner_id if provided (shared folder case), otherwise use requesting user
        target_uid = owner_id if owner_id else uid
        
        folder_ref = db.collection("tenants").document(target_uid).collection("folders").document(folder_id).get()
        folder_data = None

        if folder_ref.exists:
            folder_data = folder_ref.to_dict()
        else:
            # Fallback: search for the folder across tenants (shared case)
            tenants = db.collection("tenants").stream()
            for tenant in tenants:
                f_ref = db.collection("tenants").document(tenant.id).collection("folders").document(folder_id).get()
                if f_ref.exists:
                    fd = f_ref.to_dict()
                    if uid == fd.get("owner") or uid in fd.get("shared_with", {}):
                        folder_data = fd
                        break

        if not folder_data:
            return jsonify({"error": "Folder not found or access denied"}), 404

        owner_uid = folder_data["owner"]

        # Permission check
        if not (uid == owner_uid or uid in folder_data.get("shared_with", {})):
            # Also check global shares collection
            shares_query = db.collection("shares").where("folderId", "==", folder_id).where("ownerId", "==", owner_uid).get()
            has_share = len(list(shares_query)) > 0
            if not has_share:
                return jsonify({"error": "Unauthorized"}), 403

        clean_uid = re.sub(r'[^a-zA-Z0-9_]', '_', owner_uid).lower()
        clean_folder = re.sub(r'[^a-zA-Z0-9_]', '_', folder_id).lower()
        table_id = f"{PROJECT_ID}.{DATASET}.{clean_uid}_{clean_folder}"
        
        bq_client = bigquery.Client()
        query = f"SELECT * FROM `{table_id}` ORDER BY uploaded_at DESC LIMIT 100"
        query_job = bq_client.query(query)
        results = [dict(row) for row in query_job]
        
        return jsonify({"results": results}), 200
    except Exception as e:
        print(f"❌ Fetch Results Error: {e}")
        return jsonify({"results": []}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
