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
from dateutil import parser as date_parser

# ==========================================
# 1. Initialize Flask
# ==========================================
app = Flask(__name__)

CORS(app, resources={r"/*": {
    "origins": "*",
    "allow_headers": ["Authorization", "Content-Type", "Accept"],
    "methods": ["GET", "POST", "OPTIONS"],
    "max_age": 3600
}}, supports_credentials=True)

# ==========================================
# 2. Configuration
# ==========================================
PROJECT_ID = "pdf-etl-479411"
DATASET = "etl_reports"
LOCATION = "us-central1"
BUCKET_NAME = "pdf_platform_main"

# ==========================================
# 3. Initialize Services
# ==========================================
if not firebase_admin._apps:
    firebase_admin.initialize_app()

db = firestore.client()
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# ==========================================
# AUTH HELPERS
# ==========================================
def get_user_id(req):
    h = req.headers.get("Authorization")
    if not h:
        return None
    try:
        token = h.split("Bearer ")[1]
        return auth.verify_id_token(token)["uid"]
    except Exception:
        return None

def _build_cors_preflight_response():
    r = make_response()
    r.headers.add("Access-Control-Allow-Origin", "*")
    r.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization,Accept")
    r.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    r.headers.add("Access-Control-Max-Age", "3600")
    return r, 204

# ==========================================
# SAFE JSON PARSER (CRITICAL FIX)
# ==========================================
def safe_json(text):
    if not text:
        return {}

    text = text.strip()
    text = re.sub(r'^```json\s*|```$', '', text, flags=re.MULTILINE)

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {}

# ==========================================
# BIGQUERY HELPERS
# ==========================================
def convert_value_for_bq(value, ai_type):
    if value in (None, "", "N/A", "---"):
        return None

    val = str(value).strip()

    if ai_type == "number":
        try:
            cleaned = re.sub(r'[^\d\.-]', '', val)
            return float(cleaned)
        except Exception:
            return None

    if ai_type == "date":
        try:
            return date_parser.parse(val, fuzzy=True).strftime("%Y-%m-%d")
        except Exception:
            return None

    return val

# ==========================================
# ROUTE: ANALYZE MASTER (FIXED)
# ==========================================
@app.route("/analyze-master", methods=["POST", "OPTIONS"])
def analyze_master():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    uid = get_user_id(request)
    if not uid:
        return jsonify({"error": "Unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    file_path = payload.get("file_path")
    context_hint = payload.get("context_hint", "")

    if not file_path:
        return jsonify({"error": "Missing file_path"}), 400

    bucket = storage.Client().bucket(BUCKET_NAME)
    blob = bucket.blob(file_path)

    if not blob.exists():
        return jsonify({"error": "File not found"}), 404

    pdf_bytes = blob.download_as_bytes()

    prompt = f"""
Analyze this MASTER PDF to detect stable KPI labels.

Context:
{context_hint or "Generic business document"}

Rules:
- JSON only
- Keys = KPI names
- Ignore row values

Format:
{{ "KPI Name": "" }}
"""

    resp = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=[types.Part.from_bytes(pdf_bytes, "application/pdf"), prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0
        )
    )

    data = safe_json(resp.text)

    return jsonify({
        "detected_kpis": [{"key": k, "value": ""} for k in data.keys()]
    }), 200

# ==========================================
# ROUTE: CREATE FOLDER
# ==========================================
@app.route("/create-folder", methods=["POST", "OPTIONS"])
def create_folder():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    uid = get_user_id(request)
    if not uid:
        return jsonify({"error": "Unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    name = payload.get("name")
    context_hint = payload.get("context_hint", "")

    if not name:
        return jsonify({"error": "Missing folder name"}), 400

    folder_id = re.sub(r'[^a-zA-Z0-9_]', '_', name).lower()
    bucket = storage.Client().bucket(BUCKET_NAME)

    bucket.blob(f"incoming/{uid}/{folder_id}/master/.init").upload_from_string("")
    bucket.blob(f"incoming/{uid}/{folder_id}/batch/.init").upload_from_string("")

    db.collection("tenants").document(uid).collection("folders").document(folder_id).set({
        "display_name": name,
        "folder_id": folder_id,
        "context_hint": context_hint,
        "is_trained": False,
        "status": "waiting_for_training",
        "created_at": datetime.datetime.utcnow().isoformat(),
        "selected_kpis": [],
        "kpi_metadata": []
    })

    return jsonify({"status": "success", "folder_id": folder_id}), 200

# ==========================================
# ROUTE: BATCH PDF ENGINE
# ==========================================
@app.route("/", methods=["POST", "OPTIONS"])
def gcs_trigger_handler():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    payload = request.get_json(silent=True) or {}
    data = payload.get("data", payload)
    file_path = data.get("name", "")

    if not file_path.endswith(".pdf") or "processed/" in file_path:
        return jsonify({"status": "ignored"}), 200

    parts = file_path.split("/")
    if len(parts) < 5 or parts[3] != "batch":
        return jsonify({"status": "ignored_path"}), 200

    uid, folder_id = parts[1], parts[2]
    folder_ref = db.collection("tenants").document(uid).collection("folders").document(folder_id).get()
    if not folder_ref.exists:
        return jsonify({"status": "not_trained"}), 200

    folder = folder_ref.to_dict()
    kpis = folder.get("selected_kpis", [])
    meta = {k["name"]: k["type"] for k in folder.get("kpi_metadata", [])}

    blob = storage.Client().bucket(BUCKET_NAME).blob(file_path)
    pdf_bytes = blob.download_as_bytes()

    prompt = f"""
Extract KPI values from this PDF.

KPIs:
{kpis}

Rules:
- Tables → column headers
- Prefer totals
- Never guess
- JSON only

{{ "KPI": "value or N/A" }}
"""

    resp = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=[types.Part.from_bytes(pdf_bytes, "application/pdf"), prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0
        )
    )

    extracted = safe_json(resp.text)

    row = {
        "row_id": f"row_{int(time.time())}",
        "file_name": os.path.basename(file_path),
        "uploaded_at": datetime.datetime.utcnow().isoformat()
    }

    for kpi in kpis:
        col = f"kpi_{re.sub(r'[^a-zA-Z0-9_]', '_', kpi).lower()}"
        row[col] = convert_value_for_bq(extracted.get(kpi), meta.get(kpi, "string"))

    bigquery.Client().insert_rows_json(
        f"{PROJECT_ID}.{DATASET}.{uid}_{folder_id}", [row]
    )

    bucket = storage.Client().bucket(BUCKET_NAME)
    bucket.copy_blob(blob, bucket, file_path.replace("incoming/", "processed/"))
    blob.delete()

    return jsonify({"status": "success"}), 200

# ==========================================
# SERVER
# ==========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
