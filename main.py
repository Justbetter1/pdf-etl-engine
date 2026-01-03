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
# When running on Cloud Run, initialize_app() automatically uses the service account
firebase_admin.initialize_app()
db = firestore.client()
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# ==========================================
# NEW: AUTHENTICATION BOUNCER
# ==========================================
def get_user_id(req):
    """Verifies the Firebase Token and returns the UID"""
    auth_header = req.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    try:
        token = auth_header.split("Bearer ")[1]
        decoded_token = auth.verify_id_token(token)
        return decoded_token["uid"]
    except Exception as e:
        print(f"🔒 Auth Error: {e}")
        return None

# ==========================================
# 1. THE BRAIN (Modified to use Firestore Hints)
# ==========================================
def extract_with_gemini(pdf_bytes, uid):
    # Fetch the user's specific hint from Firestore
    # Path: tenants/{uid}/folders/default
    user_ref = db.collection("tenants").document(uid).collection("folders").document("default").get()
    user_data = user_ref.to_dict() if user_ref.exists else {}
    user_hint = user_data.get("hint", "Extract all data points.")

    prompt = f"""
    Analyze this PDF. User Instruction: {user_hint}
    Return a FLAT JSON object. 
    - Numbers: float
    - Dates: YYYY-MM-DD
    - Keys: kpi_ + lowercase_underscores
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=[types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    
    clean_json = re.sub(r',\s*([\]}])', r'\1', response.text)
    return json.loads(clean_json)

# ==========================================
# 2. THE HANDLER (Updated for Auth & UIDs)
# ==========================================
@app.post("/")
def handle_event():
    # Verify User Identity first!
    uid = get_user_id(request)
    if not uid:
        return jsonify({"error": "Unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    data = payload.get("data", payload)
    file_path = data.get("name", "")

    # Ensure user is only touching their own folder: incoming/{uid}/
    if f"incoming/{uid}/" not in file_path:
        return jsonify({"error": "Invalid file path for user"}), 403

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(data.get("bucket"))
        blob = bucket.blob(file_path)
        pdf_content = blob.download_as_bytes()
        
        # Process using the UID for personalized hints
        kpis = extract_with_gemini(pdf_content, uid)

        # BigQuery table is named after the UID (cleaned)
        target_table = sync_bigquery_schema(uid, kpis)

        # ... (rest of your existing BigQuery insert logic) ...
        
        # Move to processed/{uid}/
        new_path = file_path.replace("incoming/", "processed/")
        bucket.copy_blob(blob, bucket, new_path)
        blob.delete()
        
        return jsonify({"status": "success", "uid": uid}), 200

    except Exception as e:
        print(f"🔥 System Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
