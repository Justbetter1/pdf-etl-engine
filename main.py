import os
import json
import datetime
import re
import time
import firebase_admin
from firebase_admin import auth, credentials, firestore
from flask import Flask, request, jsonify
from flask_cors import CORS  # <--- NEW IMPORT
from google.cloud import storage, bigquery
from google import genai
from google.genai import types

app = Flask(__name__)

# ==========================================
# 🛡️ CORS SETUP: Allow your frontend to talk to this backend
# ==========================================
# This allows all origins (*). For production, you can replace "*" 
# with your actual Lovable/Vercel URL.
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers=["Authorization", "Content-Type"])

# CONFIG
PROJECT_ID = "pdf-etl-479411"
DATASET = "etl_reports"
LOCATION = "us-central1"

# Initialize Clients
firebase_admin.initialize_app()
db = firestore.client()
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# ==========================================
# 🛡️ SECURITY: AUTHENTICATION BOUNCER
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
        
        # 1. Create Folder Placeholders
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
        print(f"❌ Setup Error: {e}")
        return jsonify({"error": str(e)}), 500

# [ ... Keep the rest of your extract_with_gemini and handle_event functions exactly as they are ... ]

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
