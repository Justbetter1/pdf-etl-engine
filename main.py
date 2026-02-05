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
from dateutil import parser as date_parser  # Add to requirements.txt: python-dateutil

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
# 🧠 AI-POWERED KPI TYPE INFERENCE
# ==========================================
def infer_kpi_types_with_ai(kpi_samples: dict) -> dict:
    """
    Use Gemini AI to intelligently analyze KPI names and sample values
    to determine their data types.

    Returns: dict mapping kpi_name -> type ("number", "date", "categorical", "string")
    """
    if not kpi_samples:
        return {}

    # Build the prompt with all KPIs
    kpi_list = []
    for kpi_name, sample_value in kpi_samples.items():
        kpi_list.append(f'- "{kpi_name}": "{sample_value}"')

    kpi_text = "\n".join(kpi_list)

    prompt = f"""
Analyze these KPI field names and their sample values. For each KPI, determine the most appropriate data type.

KPIs to analyze:
{kpi_text}

Rules for type assignment:
1. "number" - For monetary values, quantities, percentages, measurements, counts, IDs that are purely numeric
2. "date" - For dates, timestamps, periods, years, months (e.g., "2024-01-15", "January 2024", "Q1 2024")
3. "categorical" - For status values, categories, types, codes, identifiers with limited possible values (e.g., "Active", "KDC-54", "Type A", "Approved")
4. "string" - For free-form text, descriptions, names, addresses, comments, long text fields

Important:
- Alphanumeric codes like "KDC-54", "INV-001", "ABC123" are "categorical" NOT "date"
- Pure numeric IDs or reference numbers are "number"
- Short identifiers and codes are "categorical"
- Rig IDs, equipment codes, reference codes are "categorical"

Return ONLY a valid JSON object with this exact format:
{{"kpi_name": "type", "another_kpi": "type"}}

Do not include any explanation, just the JSON.
"""

    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            ),
        )

        raw_text = resp.text.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r'^```json\s*|```$', '', raw_text, flags=re.MULTILINE)

        type_mapping = json.loads(raw_text)

        # Validate types - ensure only allowed values
        valid_types = {"number", "date", "categorical", "string"}
        validated_mapping = {}
        for kpi_name, kpi_type in type_mapping.items():
            if isinstance(kpi_type, str) and kpi_type.lower() in valid_types:
                validated_mapping[kpi_name] = kpi_type.lower()
            else:
                validated_mapping[kpi_name] = "string"

        print(f"✅ AI Type Inference Result: {validated_mapping}")
        return validated_mapping

    except Exception as e:
        print(f"❌ AI Type Inference Error: {e}")
        # Fallback to basic inference
        return {kpi: "string" for kpi in kpi_samples.keys()}


def infer_kpi_type_fallback(value):
    """Fallback regex-based type inference if AI fails."""
    if value is None or value == "" or value == "N/A" or value == "---":
        return "string"

    val_str = str(value).strip()

    # Check for number
    numeric_cleaned = re.sub(r'[$€£¥,\s%]', '', val_str)
    if re.match(r'^-?\d+\.?\d*$', numeric_cleaned):
        return "number"

    # Check for alphanumeric codes (letters + numbers = categorical)
    has_letters = bool(re.search(r'[A-Za-z]', val_str))
    has_numbers = bool(re.search(r'\d', val_str))

    if has_letters and has_numbers:
        # Check for month names in dates
        month_pattern = r'^(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}$'
        if re.match(month_pattern, val_str, re.IGNORECASE):
            return "date"
        return "categorical" if len(val_str) <= 30 else "string"

    # Pure date patterns
    date_patterns = [
        r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$',
        r'^\d{1,2}[-/]\d{1,2}[-/]\d{4}$',
        r'^\d{1,2}[-/]\d{1,2}[-/]\d{2}$',
    ]
    for pattern in date_patterns:
        if re.match(pattern, val_str):
            return "date"

    # Categorical indicators
    if len(val_str) <= 25 and val_str.replace(" ", "").replace("-", "").isalpha():
        return "categorical"

    return "string"


# ==========================================
# 📊 BIGQUERY DYNAMIC TYPE HELPERS
# ==========================================
def get_bigquery_type(ai_type: str) -> str:
    """Map AI-inferred types to BigQuery column types."""
    type_mapping = {
        "number": "FLOAT64",
        "date": "DATE",
        "categorical": "STRING",
        "string": "STRING"
    }
    return type_mapping.get(ai_type, "STRING")


def convert_value_for_bq(value, ai_type: str):
    """
    Convert extracted string values to the appropriate Python type
    for BigQuery insertion based on AI-inferred type.
    """
    if value is None or value == "" or value == "N/A" or value == "---":
        return None

    val_str = str(value).strip()

    if ai_type == "number":
        try:
            cleaned = re.sub(r'[$€£¥,\s%]', '', val_str)
            if cleaned.startswith('(') and cleaned.endswith(')'):
                cleaned = '-' + cleaned[1:-1]
            return float(cleaned)
        except (ValueError, TypeError):
            print(f"⚠️ Could not convert '{value}' to number, returning None")
            return None

    elif ai_type == "date":
        try:
            parsed_date = date_parser.parse(val_str, fuzzy=True)
            return parsed_date.strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            print(f"⚠️ Could not parse '{value}' as date, returning None")
            return None

    else:
        return val_str


# ==========================================
# 📊 BIGQUERY SCHEMA SYNC & TABLE CREATION (TYPED)
# ==========================================
def sync_bigquery_schema_typed(uid, folder_id, kpi_metadata):
    bq_client = bigquery.Client()
    clean_uid = re.sub(r'[^a-zA-Z0-9_]', '_', uid).lower()
    clean_folder = re.sub(r'[^a-zA-Z0-9_]', '_', folder_id).lower()
    table_id = f"{PROJECT_ID}.{DATASET}.{clean_uid}_{clean_folder}"

    kpi_type_lookup = {}
    for kpi in kpi_metadata:
        kpi_name = kpi.get("name", "")
        kpi_type = kpi.get("type", "string")
        kpi_type_lookup[kpi_name] = kpi_type

    try:
        table = bq_client.get_table(table_id)
        existing_cols = {field.name for field in table.schema}

        new_fields = []
        for kpi in kpi_metadata:
            kpi_name = kpi.get("name", "")
            kpi_type = kpi.get("type", "string")
            col_name = f"kpi_{re.sub(r'[^a-zA-Z0-9_]', '_', kpi_name).lower()}"

            if col_name not in existing_cols:
                bq_type = get_bigquery_type(kpi_type)
                new_fields.append(bigquery.SchemaField(col_name, bq_type))
                print(f"📊 Adding column: {col_name} as {bq_type}")

        if new_fields:
            table.schema += new_fields
            bq_client.update_table(table, ["schema"])
            print(f"✅ Table {table_id} updated with {len(new_fields)} new typed columns.")

    except Exception:
        print(f"📊 Creating new table with typed schema: {table_id}")

        schema = [
            bigquery.SchemaField("row_id", "STRING"),
            bigquery.SchemaField("file_name", "STRING"),
            bigquery.SchemaField("uploaded_at", "TIMESTAMP"),
        ]

        for kpi in kpi_metadata:
            kpi_name = kpi.get("name", "")
            kpi_type = kpi.get("type", "string")
            col_name = f"kpi_{re.sub(r'[^a-zA-Z0-9_]', '_', kpi_name).lower()}"
            bq_type = get_bigquery_type(kpi_type)
            schema.append(bigquery.SchemaField(col_name, bq_type))
            print(f"📊 Column: {col_name} -> {bq_type}")

        table = bigquery.Table(table_id, schema=schema)
        bq_client.create_table(table)
        time.sleep(2)
        print(f"✅ Created typed table: {table_id}")

    return table_id, kpi_type_lookup


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
# 🧠 3. MASTER PDF ANALYSIS (PROMPT ADJUSTED ONLY)
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

        # ✅ ADJUSTED PROMPT ONLY (no other code changes)
        prompt = f"""
You are analyzing a MASTER PDF template to discover KPI fields users will want to extract.

USER CONTEXT: {context_hint if context_hint else "Generic business document."}

Find and return:
- Key/value labels (Invoice No, Total Amount, Date, Vendor, Reference, Status, etc.)
- If there are tables: include meaningful column headers as fields too (Item, Qty, Unit Price, Total, etc.)
- Prefer business KPIs over generic words.

IMPORTANT:
- Do NOT output every cell value.
- Focus on stable labels/headers that repeat across similar PDFs.

Return ONLY valid JSON object:
{{"field_name": "example_value"}}

JSON only. No explanation.
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

        # ✅ FIX: if Gemini returns list of objects, merge them (instead of taking only [0])
        if isinstance(detected_dict, list):
            merged = {}
            for item in detected_dict:
                if isinstance(item, dict):
                    merged.update(item)
            detected_dict = merged

        formatted_kpis = [{"key": k, "value": str(v)} for k, v in (detected_dict or {}).items()]

        return jsonify({"detected_kpis": formatted_kpis}), 200
    except Exception as e:
        print(f"❌ Analyze Master Crash: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ==========================================
# ✅ 4. CONFIRM SELECTED KPIs (WITH AI TYPE INFERENCE + TYPED SCHEMA)
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
        kpi_samples = payload.get("kpi_samples", {})

        print(f"🧠 Calling Gemini AI to analyze {len(kpi_samples)} KPIs...")
        kpi_types = infer_kpi_types_with_ai(kpi_samples)

        kpi_metadata = []
        for kpi_name in selected_kpis:
            sample_value = kpi_samples.get(kpi_name, "")
            inferred_type = kpi_types.get(kpi_name, infer_kpi_type_fallback(sample_value))
            kpi_metadata.append({
                "name": kpi_name,
                "sample_value": sample_value,
                "type": inferred_type
            })

        db.collection("tenants").document(uid).collection("folders").document(folder_id).update({
            "selected_kpis": selected_kpis,
            "kpi_samples": kpi_samples,
            "kpi_metadata": kpi_metadata,
            "is_trained": True,
            "status": "active"
        })

        sync_bigquery_schema_typed(uid, folder_id, kpi_metadata)

        print(f"✅ KPIs confirmed with AI-inferred types: {kpi_metadata}")
        return jsonify({"status": "success", "kpi_metadata": kpi_metadata}), 200
    except Exception as e:
        print(f"❌ Confirm KPIs Error: {e}")
        return jsonify({"error": str(e)}), 500

# ==========================================
# 📋 5. GET KPIs (with pre-computed type metadata)
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

        is_owner = uid == folder_data.get("owner")
        has_share = uid in folder_data.get("shared_with", {})

        if not is_owner and not has_share:
            shares_query = db.collection("shares").where("folderId", "==", folder_id).where("ownerId", "==", target_uid).get()
            if len(list(shares_query)) > 0:
                has_share = True

        if not is_owner and not has_share:
            return jsonify({"error": "Access denied"}), 403

        kpi_metadata = folder_data.get("kpi_metadata")

        if kpi_metadata:
            return jsonify({
                "is_trained": folder_data.get("is_trained", False),
                "selected_kpis": kpi_metadata,
                "context_hint": folder_data.get("context_hint", ""),
                "status": folder_data.get("status", "unknown")
            }), 200

        selected_kpis_raw = folder_data.get("selected_kpis", [])
        kpi_samples = folder_data.get("kpi_samples", {})

        if kpi_samples:
            kpi_types = infer_kpi_types_with_ai(kpi_samples)
        else:
            kpi_types = {}

        selected_kpis_with_types = []
        for kpi_name in selected_kpis_raw:
            sample_value = kpi_samples.get(kpi_name, "")
            kpi_type = kpi_types.get(kpi_name, infer_kpi_type_fallback(sample_value))
            selected_kpis_with_types.append({
                "name": kpi_name,
                "sample_value": sample_value,
                "type": kpi_type
            })

        return jsonify({
            "is_trained": folder_data.get("is_trained", False),
            "selected_kpis": selected_kpis_with_types,
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
        folder_id = request.form.get("folder_id")
        owner_id = request.form.get("owner_id")
        file = request.files.get("file")

        if not folder_id or not owner_id or not file:
            return jsonify({"error": "Missing required fields: folder_id, owner_id, or file"}), 400

        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        sanitized_email = re.sub(r'[@.]', '_', user_email)
        share_doc_id = f"{owner_id}_{folder_id}_{sanitized_email}"

        share_ref = db.collection("shares").document(share_doc_id).get()

        if not share_ref.exists:
            return jsonify({"error": "Share not found. You do not have access to this folder."}), 403

        share_data = share_ref.to_dict()
        permission = share_data.get("permission", "view")

        if permission != "edit":
            return jsonify({"error": "You have view-only access. Upload not permitted."}), 403

        original_filename = file.filename or "unnamed.pdf"
        sanitized_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', original_filename)

        storage_path = f"incoming/{owner_id}/{folder_id}/batch/{sanitized_filename}"
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(storage_path)

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
# 🚜 7. BATCH ENGINE (GCS TRIGGER HANDLER) - KPI PROMPT + LIST-MERGE FIX
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
        kpi_metadata = folder_data.get("kpi_metadata", [])
        context_hint = folder_data.get("context_hint", "")

        kpi_type_lookup = {}
        for kpi in kpi_metadata:
            kpi_type_lookup[kpi.get("name", "")] = kpi.get("type", "string")

        storage_client = storage.Client()
        source_bucket = storage_client.bucket(BUCKET_NAME)
        blob = source_bucket.blob(file_path)
        pdf_bytes = blob.download_as_bytes()

        # ✅ PROMPT UPDATED: force returning ALL KPI keys always (no skipping)
        prompt = f"""
You are extracting KPI values from a PDF.

CONTEXT (domain hint): {context_hint if context_hint else "Generic business document."}

KPIs (use exact key names):
{kpis}

CRITICAL RULES (read carefully):
1) Output must be ONLY valid JSON object in this exact form:
   {{ "KPI_NAME_1": "value or N/A", "KPI_NAME_2": "value or N/A", ... }}
   No extra keys, no explanations.

2) You MUST include EVERY KPI key from the KPI list in the output JSON.
   - If you cannot find a KPI, set its value to "N/A".
   - Do NOT omit keys.

3) Table-aware behavior:
   - If the document contains tables, DO NOT treat each cell as an independent value.
   - Identify the table structure (column headers + rows).
   - For each KPI, select the correct value by understanding:
     • which column/header matches the KPI label,
     • which row is relevant (e.g., latest date row, totals row, summary row, matching identifier row),
     • totals/subtotals/grand totals (prefer grand total if KPI implies it).
   - If multiple candidate values exist in a table, choose the most semantically correct one and ignore the rest.
   - If the KPI refers to a row identifier (e.g., product name, account, item, category), match that row.

4) Human-like label/value linking (pattern connection):
   - Many KPIs appear as label/value pairs, sometimes split across lines.
   - Treat these as connected if they are:
     • on the same line (e.g., "Total Amount: 1,250"),
     • on adjacent lines (label on one line, value on next),
     • aligned visually in the same row/area (even without ":"),
     • near each other with strong association words (Total, Amount, Date, ID, Status, Vendor, Customer, Reference).
   - If you see a label but the value is nearby, capture it (even if formatting is messy).

5) Duplicate fields:
   - If the same KPI appears multiple times, prefer the value in the main body (not header/footer).
   - Prefer the most complete value (e.g., "KWD 1,250.000" over "1,250").

6) Value formatting:
   - Return values exactly as shown in the PDF (keep currency symbols, %, commas).
   - If a value truly does not exist, return "N/A".
   - Do not invent values.

Now return the JSON object only.
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

        # ✅ FIX: if Gemini returns list of dicts, MERGE them (instead of taking only [0])
        if isinstance(extracted_data, list):
            merged = {}
            for item in extracted_data:
                if isinstance(item, dict):
                    merged.update(item)
            extracted_data = merged

        owner_uid = folder_data.get("owner", uid)

        if kpi_metadata:
            table_id, _ = sync_bigquery_schema_typed(owner_uid, folder_id, kpi_metadata)
        else:
            table_id = sync_bigquery_schema(owner_uid, folder_id, kpis)

        row = {
            "row_id": f"row_{int(time.time())}",
            "file_name": file_path.split("/")[-1],
            "uploaded_at": datetime.datetime.utcnow().isoformat()
        }

        for k in kpis:
            safe_col_name = f"kpi_{re.sub(r'[^a-zA-Z0-9_]', '_', k).lower()}"
            raw_value = extracted_data.get(k, "N/A")
            kpi_type = kpi_type_lookup.get(k, "string")
            typed_value = convert_value_for_bq(raw_value, kpi_type)
            row[safe_col_name] = typed_value
            print(f"📊 {k}: '{raw_value}' -> {typed_value} ({kpi_type})")

        bq_client = bigquery.Client()
        errors = bq_client.insert_rows_json(table_id, [row])

        if errors:
            print(f"❌ BigQuery Insert Errors: {errors}")
            return jsonify({"error": "BigQuery Insert Failed", "details": str(errors)}), 200

        new_path = file_path.replace("incoming/", "processed/")
        source_bucket.copy_blob(blob, source_bucket, new_path)
        blob.delete()

        print(f"✅ Successfully processed {file_path} with typed values")
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
    owner_id = request.args.get("owner_id")

    if not folder_id: return jsonify({"error": "folder_id required"}), 400

    try:
        target_uid = owner_id if owner_id else uid

        folder_ref = db.collection("tenants").document(target_uid).collection("folders").document(folder_id).get()
        folder_data = None

        if folder_ref.exists:
            folder_data = folder_ref.to_dict()
        else:
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

        if not (uid == owner_uid or uid in folder_data.get("shared_with", {})):
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
