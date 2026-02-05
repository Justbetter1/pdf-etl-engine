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
try:
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    db = firestore.client()
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    print(f"🚀 Backend ready: {PROJECT_ID}")
except Exception as e:
    print(f"❌ Startup Error: {e}")

# ==========================================
# AUTH HELPERS
# ==========================================
def get_user_id(req):
    auth_header = req.headers.get("Authorization")
    if not auth_header:
        return None
    try:
        token = auth_header.split("Bearer ")[1]
        decoded = auth.verify_id_token(token)
        return decoded["uid"]
    except Exception:
        return None

def get_user_email(req):
    auth_header = req.headers.get("Authorization")
    if not auth_header:
        return None
    try:
        token = auth_header.split("Bearer ")[1]
        decoded = auth.verify_id_token(token)
        return decoded.get("email", "").lower()
    except Exception:
        return None

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization,Accept")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    response.headers.add("Access-Control-Max-Age", "3600")
    return response, 204

# ==========================================
# BIGQUERY HELPERS
# ==========================================
def get_bigquery_type(ai_type):
    return {
        "number": "FLOAT64",
        "date": "DATE",
        "categorical": "STRING",
        "string": "STRING"
    }.get(ai_type, "STRING")

def convert_value_for_bq(value, ai_type):
    if value in (None, "", "N/A", "---"):
        return None

    val = str(value).strip()

    if ai_type == "number":
        try:
            cleaned = re.sub(r'[$€£¥,\s%]', '', val)
            if cleaned.startswith("(") and cleaned.endswith(")"):
                cleaned = "-" + cleaned[1:-1]
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
# ROUTES: analyze-master (detect KPI labels)
# ==========================================
@app.route("/analyze-master", methods=["POST", "OPTIONS"])
def analyze_master():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    uid = get_user_id(request)
    if not uid:
        return jsonify({"error": "Unauthorized"}), 401

    payload = request.get_json()
    file_path = payload.get("file_path")
    context_hint = payload.get("context_hint", "")

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_path)

    pdf_bytes = blob.download_as_bytes()

    prompt = f"""
You are analyzing a MASTER PDF template to identify stable KPI labels.

Context: {context_hint or "Generic business document"}

Return ONLY a JSON object:
{{"KPI Name": "example value"}}
"""

    resp = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=[types.Part.from_bytes(pdf_bytes, "application/pdf"), prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0
        )
    )

    text = resp.text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```json\s*|```$', '', text)

    try:
        data = json.loads(text)
    except Exception:
        # fallback: return raw text for debugging
        return jsonify({"error": "Invalid AI output", "raw": text}), 500

    if isinstance(data, list):
        data = data[0] if data else {}

    return jsonify({
        "detected_kpis": [{"key": k, "value": str(v)} for k, v in data.items()]
    }), 200

# ==========================================
# ROUTE: create-folder (restored from old working code)
# ==========================================
@app.route("/create-folder", methods=["POST", "OPTIONS"])
def create_folder():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    uid = get_user_id(request)
    if not uid:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        payload = request.get_json()
        name = payload.get("name")
        context_hint = payload.get("context_hint", "")
        if not name:
            return jsonify({"error": "Missing folder name"}), 400

        folder_id = re.sub(r'[^a-zA-Z0-9_]', '_', name).lower()

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        # Create placeholder blobs so the folder exists in GCS
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
            "shared_with": {},
            "selected_kpis": [],
            "kpi_metadata": []
        }
        db.collection("tenants").document(uid).collection("folders").document(folder_id).set(folder_data)

        return jsonify({"status": "success", "folder_id": folder_id, "folder": folder_data}), 200
    except Exception as e:
        print(f"❌ Create Folder Error: {e}")
        return jsonify({"error": str(e)}), 500

# ==========================================
# ROUTE: get-kpis (fetch folder KPIs and metadata)
# ==========================================
@app.route("/get-kpis", methods=["GET", "OPTIONS"])
def get_kpis():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    uid = get_user_id(request)
    if not uid:
        return jsonify({"error": "Unauthorized"}), 401

    folder_id = request.args.get("folder_id")
    if not folder_id:
        return jsonify({"error": "Missing folder_id"}), 400

    folder_ref = db.collection("tenants").document(uid).collection("folders").document(folder_id).get()
    if not folder_ref.exists:
        return jsonify({"error": "Folder not found"}), 404

    folder = folder_ref.to_dict()
    return jsonify({
        "selected_kpis": folder.get("selected_kpis", []),
        "kpi_metadata": folder.get("kpi_metadata", []),
        "folder": {
            "display_name": folder.get("display_name"),
            "folder_id": folder.get("folder_id"),
            "context_hint": folder.get("context_hint"),
            "status": folder.get("status"),
            "is_trained": folder.get("is_trained", False)
        }
    }), 200

# ==========================================
# 🔥 BATCH ENGINE (TABLE-AWARE KPI EXTRACTION)
# ==========================================
@app.route("/", methods=["POST", "OPTIONS"])
def gcs_trigger_handler():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    payload = request.get_json(silent=True) or {}
    data = payload.get("data", payload)
    file_path = data.get("name", "")

    if not file_path.lower().endswith(".pdf") or "processed/" in file_path:
        return jsonify({"status": "ignored"}), 200

    parts = file_path.split("/")
    if len(parts) < 5 or parts[3] != "batch":
        return jsonify({"status": "ignored_path"}), 200

    uid = parts[1]
    folder_id = parts[2]

    folder_ref = db.collection("tenants").document(uid).collection("folders").document(folder_id).get()
    if not folder_ref.exists:
        return jsonify({"status": "not_trained"}), 200

    folder = folder_ref.to_dict()
    kpis = folder.get("selected_kpis", [])
    kpi_metadata = folder.get("kpi_metadata", [])
    context_hint = folder.get("context_hint", "")

    kpi_type_lookup = {k["name"]: k["type"] for k in kpi_metadata}

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_path)
    pdf_bytes = blob.download_as_bytes()

    # Table-aware prompt with explicit column-header instruction
    prompt = f"""
You are extracting KPI values from a business PDF.

Context:
{context_hint or "Generic business document"}

KPIs:
{kpis}

RULES:
1. Detect if document has tables.
2. If tables exist, ONLY column headers are KPI field names. Row values are sample data for those KPI columns. Do NOT treat individual cells as separate KPIs.
3. Prefer totals / summary rows for totals KPIs.
4. If multiple rows exist and no clear match → return "N/A".
5. If no tables exist, use semantic label/value understanding.
6. NEVER guess. NEVER return "---".
7. Return ALL KPIs.

Return ONLY valid JSON:
{{"KPI": "value or N/A"}}
"""

    resp = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=[types.Part.from_bytes(pdf_bytes, "application/pdf"), prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0
        )
    )

    raw = resp.text.strip()
    if raw.startswith("```"):
        raw = re.sub(r'^```json\s*|```$', '', raw)

    try:
        extracted_data = json.loads(raw)
    except Exception:
        print("❌ JSON Decode Error from AI:", raw)
        return jsonify({"error": "Invalid AI output", "raw": raw}), 500

    # Merge list outputs into single dict if needed
    if isinstance(extracted_data, list):
        merged = {}
        for obj in extracted_data:
            if isinstance(obj, dict):
                merged.update(obj)
        extracted_data = merged

    table_id = f"{PROJECT_ID}.{DATASET}.{uid}_{folder_id}"
    bq_client = bigquery.Client()

    row = {
        "row_id": f"row_{int(time.time())}",
        "file_name": file_path.split("/")[-1],
        "uploaded_at": datetime.datetime.utcnow().isoformat()
    }

    for kpi in kpis:
        col = f"kpi_{re.sub(r'[^a-zA-Z0-9_]', '_', kpi).lower()}"
        raw_val = extracted_data.get(kpi, "N/A")
        kpi_type = kpi_type_lookup.get(kpi, "string")
        row[col] = convert_value_for_bq(raw_val, kpi_type)

    errors = bq_client.insert_rows_json(table_id, [row])
    if errors:
        print("❌ BQ Errors:", errors)

    # Move file to processed and delete original
    try:
        bucket.copy_blob(blob, bucket, file_path.replace("incoming/", "processed/"))
        blob.delete()
    except Exception as e:
        print("⚠️ GCS move/delete error:", e)

    print(f"✅ Processed {file_path}")
    return jsonify({"status": "success"}), 200

# ==========================================
# SERVER
# ==========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
