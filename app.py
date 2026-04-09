from platform import processor
import sys
from flask import Flask, render_template, request, jsonify, send_file, session, send_from_directory
from flask_session import Session
from flask import Flask, request, session, jsonify, send_file, render_template, make_response, redirect, url_for

import os
from pathlib import Path
import pickle
import pandas as pd

from matching_engine import SellerBuyerConfig, SellerBuyerMatching
from call_report_etl import CallReport5300ETL

from LTN_loan_tape_prep import LTN_loan_tape_prep
import numpy as np
import io
from io import BytesIO
import re
import json

import requests
import zipfile

from dotenv import load_dotenv

from groq import Groq
import pdfplumber

import os

try:
    import PyPDF2
    _PYPDF2_OK = True
except ImportError:
    _PYPDF2_OK = False
 
try:
    import openpyxl
    _OPENPYXL_OK = True
except ImportError:
    _OPENPYXL_OK = False

from datetime import datetime

NCUA_CACHE_DIR = Path("ncua_cache")
NCUA_CACHE_DIR.mkdir(exist_ok=True)

# ── Last-run persistence helpers defined below after BASE_DIR ──────────────

def _save_last_run():
    """Write key session values to disk so they survive Flask restarts."""
    try:
        data = {
            'summary':          dict(session.get('summary', {})),
            'tape_size':        session.get('tape_size'),
            'matching_summary': dict(session.get('matching_summary', {})),
            'last_workflow':    session.get('last_workflow'),
        }
        with open(LAST_RUN_FILE, 'w') as f:
            json.dump(data, f)
        print(f"[persistence] last_run.json saved — tape_size={data['tape_size']}, matched={data['summary'].get('matched_loans')}")
    except Exception as e:
        print(f"[persistence] save failed: {e}")

def _load_last_run():
    """Restore last-run data into session on login if no active session."""
    try:
        if LAST_RUN_FILE.exists():
            with open(LAST_RUN_FILE, 'r') as f:
                data = json.load(f)
            if data.get('summary'):
                session['summary']          = data['summary']
                session['tape_size']        = data.get('tape_size')
                session['matching_summary'] = data.get('matching_summary', {})
                session['last_workflow']    = data.get('last_workflow')
                print(f"[persistence] last_run.json restored — tape_size={data.get('tape_size')}, matched={data['summary'].get('matched_loans')}")
    except Exception as e:
        print(f"[persistence] load failed: {e}")

load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

from agentic_intent_router import StageZeroRouter
router = StageZeroRouter(groq_client)

sys.stdout.reconfigure(line_buffering=True)
print("TEST: Output working?", flush=True)
sys.stdout.flush()
print("TEST: App starting up...", flush=True)
sys.stdout.flush()

BASE_DIR = Path(__file__).resolve().parent
LAST_RUN_FILE = BASE_DIR / "last_run.json"  # persists last run across restarts

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "Templates"),
    static_folder=str(BASE_DIR / "static"),
)

app.secret_key = os.environ.get('SECRET_KEY', 'ltn-analyzer-2026') 

UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"
UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

app.config["SESSION_TYPE"] = "filesystem" 
app.config["SESSION_PERMANENT"] = False   
Session(app)

with open(BASE_DIR / "call_summary_preloaded.pkl", "rb") as f:
    call_summary_preloaded = pickle.load(f)

with open(BASE_DIR / "call_parts_preloaded.pkl", "rb") as f:
    call_parts_preloaded = pickle.load(f)

with open(BASE_DIR / "call_ratios_preloaded.pkl", "rb") as f:
    call_ratios_preloaded = pickle.load(f)

call_summary_df = pd.DataFrame(call_summary_preloaded)
call_summary_df["as_of_date"] = pd.to_datetime(
    call_summary_df["as_of_date"]
).dt.normalize()

summary_dict = {}
for _, row in call_summary_df.iterrows():
    inst_id = row["institution_id"]
    as_of = row["as_of_date"]
    summary_dict[(inst_id, as_of)] = row
call_summary_preloaded = summary_dict

if isinstance(call_parts_preloaded, dict):
    parts_df = pd.DataFrame(call_parts_preloaded)
else:
    parts_df = call_parts_preloaded

if "as_of_date" not in parts_df.columns:
    parts_df["as_of_date"] = pd.to_datetime("2022-06-30")

parts_df["as_of_date"] = pd.to_datetime(parts_df["as_of_date"]).dt.normalize()

parts_dict = {}
for (inst_id, as_of), group in parts_df.groupby(["institution_id", "as_of_date"]):
    parts_dict[(inst_id, as_of)] = group.reset_index(drop=True)
call_parts_preloaded = parts_dict

if isinstance(call_ratios_preloaded, dict):
    ratios_df = pd.DataFrame(call_ratios_preloaded)
else:
    ratios_df = call_ratios_preloaded

if "As of Date" not in ratios_df.columns:
    ratios_df["As of Date"] = pd.to_datetime("2022-06-30")

ratios_df["As of Date"] = pd.to_datetime(ratios_df["As of Date"]).dt.normalize()
ratios_df = ratios_df.rename(columns={"institutionid": "institution_id"})

ratios_dict = {}
for _, row in ratios_df.iterrows():
    inst_id = row["institution_id"]
    as_of = row["As of Date"]
    ratios_row_df = pd.DataFrame([row])
    ratios_dict[(inst_id, as_of)] = ratios_row_df
call_ratios_preloaded = ratios_dict

ltd_lookup = dict(zip(call_summary_df["institution_id"], call_summary_df["ltd"]))
size_lookup = dict(zip(call_summary_df["institution_id"], zip(call_summary_df["total_assets"], call_summary_df["size_band"])))
state_lookup = dict(zip(call_summary_df["institution_id"], call_summary_df["state"]))

geo_neighbors = {
    "TX": ["OK", "NM", "AR", "LA"],
    "OK": ["TX", "KS", "AR", "MO"],
}

state_map = {}

config = SellerBuyerConfig(
    call_report_etl_cls=CallReport5300ETL,
    preloaded_summaries=call_summary_preloaded,
    preloaded_parts=call_parts_preloaded,
    preloaded_ratios=call_ratios_preloaded,
    geo_neighbors=geo_neighbors,
    state_map=state_map,
)
sbm = SellerBuyerMatching(config=config)

PERSONAS = {
    "analyst": {
        "label":    "Credit Analyst",
        "tone":     "technical, precise and regulation-aware",
        "defaults": {"fico_min": 660, "ltv_max": 90, "dti_max": 43},
        "output":   (
            "concise prose with specific ratios, NCUA account codes and regulatory "
            "references where relevant — no markdown tables for definitions. "
            "Always frame answers in the context of credit union loan participation, "
            "NCUA regulations, and ALM strategy. Reference regulatory thresholds, "
            "account codes, and industry benchmarks. Avoid oversimplified explanations."
        ),
        "suggests": [
            "benchmark against NCUA peer group",
            "extract specific ratios from attached 5300 report",
            "compare against industry thresholds"
        ],
    },
    "officer": {
        "label":    "Loan Officer",
        "tone":     "direct, action-oriented and filter-focused",
        "defaults": {"fico_min": 680, "ltv_max": 115, "dti_max": 45},
        "output":   (
            "actionable guidance with specific filter thresholds and next steps — "
            "skip textbook definitions, get straight to what the Loan Officer needs to DO. "
            "Reference standard boxing filters (FICO, LTV, DTI, rate, term, DPD). "
            "Always suggest the next pipeline action: upload tape, apply filters, generate report."
        ),
        "suggests": [
            "upload loan tape and run box assignments",
            "apply standard filters: FICO 680+, LTV 115%, DTI 45%",
            "generate refined portfolio HTML report"
        ],
    },
    "buyer": {
        "label":    "Buyer",
        "tone":     "opportunity-focused and liquidity-deployment minded",
        "defaults": {"fico_min": 700, "ltv_max": 90, "dti_max": 40},
        "output":   (
            "counterparty and liquidity deployment context — frame every answer around "
            "how it affects the Buyer's ability to deploy excess liquidity via loan participations. "
            "Highlight LTD positioning, seller quality scores, and participation eligibility. "
            "Avoid generic definitions — always connect back to deal opportunity."
        ),
        "suggests": [
            "find top seller recommendations for this CU",
            "benchmark LTD against peer group to confirm buyer positioning",
            "identify participation-eligible seller pools"
        ],
    },
    "seller": {
        "label":    "Seller / CU CFO",
        "tone":     "strategic, balance-sheet focused and commercially minded",
        "defaults": {"fico_min": 660, "ltv_max": 110, "dti_max": 45},
        "output":   (
            "strategic implications for the CU's balance sheet and participation eligibility — "
            "not textbook definitions. Frame every answer around how it affects the Seller's "
            "ability to offload loans, manage LTD, attract buyers, and optimise capital. "
            "Connect concepts to tape marketability, pricing, and buyer appetite."
        ),
        "suggests": [
            "find potential buyers for this loan portfolio",
            "compare LTD and net worth ratio against peer group",
            "assess tape marketability before approaching buyers"
        ],
    },
    "exec": {
        "label":    "Executive",
        "tone":     "concise, strategic and jargon-free",
        "defaults": {},
        "output":   (
            "one clear strategic implication per concept — no technical detail, no formulas, "
            "no account codes. Executives need to understand what something means for the "
            "institution's competitive position, risk profile, or growth opportunity. "
            "Maximum 3-4 sentences per answer. Always end with a strategic recommendation."
        ),
        "suggests": [
            "get a market overview for our asset band",
            "summarise portfolio health and participation readiness",
            "compare institution performance against industry benchmarks"
        ],
    },
    "director": {
        "label":    "Director, Loan Transaction Network",
        "tone":     "strategic, platform-governance focused and commercially minded",
        "defaults": {"fico_min": 660, "ltv_max": 115, "dti_max": 45},
        "output":   (
            "platform-level strategic context — frame every answer around Capital Flow AI "
            "platform governance, LTN operations, and institutional deployment decisions. "
            "Highlight cross-persona implications, pipeline performance, and strategic priorities. "
            "Concise, data-led, always actionable."
        ),
        "suggests": [
            "summarise today's platform activity across all routes",
            "compare LTD positioning against peer credit unions",
            "review counterparty match quality for the current tape"
        ],
    },
}

USERS = {
    "admin":    "capitalflow2025",
    "yuan":     "capitalflow2025",
    "haihua":   "capitalflow2025",
    "kanishk":  "capitalflow2025",
}

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '').strip()
        persona  = request.form.get('persona')

        if username not in USERS or USERS[username] != password:
            return render_template('login.html', error="Invalid username or password.")

        if persona not in PERSONAS:
            return render_template('login.html', error="Please select a persona.")

        session['username']         = username
        session['persona']          = persona
        session['persona_label']    = PERSONAS[persona]['label']
        session['persona_tone']     = PERSONAS[persona]['tone']
        session['persona_defaults'] = PERSONAS[persona]['defaults']
        session['persona_output']   = PERSONAS[persona]['output']
        session['persona_suggests'] = PERSONAS[persona]['suggests']
        session['chatbot_history']  = []
        _load_last_run()  # restore last tape/matching data across Flask restarts
        return redirect(url_for('index'))

    return render_template('login.html', error=None)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route("/clear")
def clear():
    session.clear()
    return redirect(url_for('login'))


@app.route("/")
def index():
    if 'persona' not in session:
        return redirect(url_for('login'))
    return render_template("index.html")


@app.post("/upload_call_report")
def upload_call_report():
    file = request.files.get("call_report")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    upload_path = UPLOAD_DIR / file.filename
    file.save(upload_path)

    try:
        new_cu = sbm.run_call_report_etl(upload_path)
    except Exception as e:
        print("Error in run_call_report_etl:", repr(e))
        return jsonify({"error": "Failed to run call report ETL."}), 500

    inst_id = new_cu["institution_id"]
    as_of   = new_cu["as_of_date"]

    ltd_lookup[inst_id]   = new_cu["summary"]["ltd"]
    size_lookup[inst_id]  = (new_cu["summary"]["total_assets"], new_cu["summary"]["size_band"])
    state_lookup[inst_id] = new_cu["summary"]["state"]

    print(f"\n🔢 Classifying LTD + calculating 5300 ratios for {inst_id}...")
    try:
        role, ltd = sbm.classify_ltd(
            summary_row=new_cu["summary_df"][new_cu["summary_df"]["institution_id"] == inst_id].iloc[0],
            summary_df=new_cu["summary_df"],
            ratios_df=new_cu["call_ratios"],
            label=inst_id
        )
        ratios = sbm.matching_state["health"]["ratios"]
    except Exception as e:
        print(f"Error in classify_ltd: {repr(e)}")
        role, ltd = "UNKNOWN", 0.0
        ratios = {}

    try:
        print("\n[Step 3: Building Opposite-Side Pool]")
        summary_df = pd.DataFrame(list(call_summary_preloaded.values()))
        ratios_df  = pd.concat([df for df in call_ratios_preloaded.values()])

        ratios_df = ratios_df.rename(columns={
            'institutionid': 'institution_id',
            'As of Date': 'as_of_date',
            'Total Investments': 'total_investments',
            'Total Net Worth': 'total_net_worth',
            'Commercial Loans': 'commercial_loans',
            'Indirect Loans': 'indirect_loans',
            'New Vehicle Loans': 'new_vehicle_loans',
            'Used Vehicle Loans': 'used_vehicle_loans',
        })
        summary_df = summary_df.rename(columns={
            'institutionid': 'institution_id',
            'As of Date': 'as_of_date',
            'Total Loans': 'total_loans',
            'Total Deposits': 'total_deposits',
            'Total Assets': 'total_assets',
        })

        pool = sbm.build_opposite_pool(
            summary_df=summary_df,
            ratios_df=ratios_df,
            new_inst_id=inst_id,
            new_role=role,
        )
        if not pool:
            return jsonify({"error": "No opposite-side candidates found for this quarter."}), 400
        print(f"✅ Opposite-side pool built ({len(pool)} candidates).")
    except Exception as e:
        print(f"❌ Error in build_opposite_pool: {repr(e)}")
        return jsonify({"error": "Failed to build opposite-side pool."}), 500

    try:
        parts_df = pd.concat([df for df in call_parts_preloaded.values()], ignore_index=True)
        parts_df = parts_df.rename(columns={
            'institutionid': 'institution_id',
            'assetclasscode': 'asset_class_code',
            'Outstanding Balance': 'outstanding_balance',
            'Amount Purchased YTD': 'amount_purchased_ytd',
            'Retained Balance Outstanding': 'retained_balance_outstanding',
            'Amount Sold YTD': 'amount_sold_ytd',
        })
        scores_long, pair_scores = sbm.compute_activity_scores(
            new_cu=new_cu, opposite_pool=pool, new_role=role, parts_df=parts_df
        )
    except Exception as e:
        print("Error in compute_activity_scores:", repr(e))
        return jsonify({"error": "Failed to compute activity scores."}), 500

    if scores_long.empty or pair_scores.empty:
        return jsonify({"error": "No activity data available for matching."}), 400

    try:
        if role == "BUYER":
            buyer_id = inst_id
        else:
            buyer_id = None
            for cu in pool:
                if cu["side"] == "BUYER":
                    buyer_id = cu["institution_id"]
                    break
            if buyer_id is None:
                return jsonify({"error": "No buyer counterparties found in pool."}), 400

        activity_pairs = sbm.build_activity_pairs(
            buyer_id=buyer_id, scores_long=scores_long, pair_scores=pair_scores
        )
    except Exception as e:
        print("Error in build_activity_pairs:", repr(e))
        return jsonify({"error": "Failed to build activity pairs."}), 500

    if activity_pairs.empty:
        return jsonify({"error": "No valid pairs after activity matching."}), 400

    print(f"\n📊 Calculating Capacity Score for {inst_id}...")
    try:
        health_data  = sbm.matching_state["health"]
        ratios       = health_data["ratios"]
        cu_row       = new_cu["summary_df"][new_cu["summary_df"]["institution_id"] == inst_id].iloc[0]
        total_assets = float(cu_row.get("total_assets", 0))
        capacity_score = sbm.calculate_capacity_score({"ratios": ratios, "total_assets": total_assets})
        sbm.matching_state["capacity"] = {"cu_id": inst_id, **capacity_score}
    except Exception as e:
        print(f"Error in capacity score: {repr(e)}")
        sbm.matching_state["capacity"] = {"cu_id": inst_id, "capacity_score": 0}

    try:
        opposite_pool  = pool
        seller_capacities = []
        for cu in opposite_pool:
            seller_id = cu["institution_id"]
            if seller_id == inst_id:
                continue
            try:
                seller_ratios      = cu["ratios"]
                summary_row        = cu["summary"]
                seller_health_data = {"ratios": seller_ratios, "total_assets": total_assets}
                seller_capacity_data = sbm.calculate_capacity_score(seller_health_data)
                seller_capacity    = seller_capacity_data["capacity_score"]
                seller_breakdown   = seller_capacity_data["breakdown"]
                seller_capacities.append({
                    "cu_id": seller_id, "capacity_score": seller_capacity,
                    "breakdown": seller_breakdown, "side": cu["side"]
                })
            except Exception as e:
                print(f"❌ Capacity error for {seller_id}: {repr(e)}")
                continue
            sbm.matching_state["seller_capacities"] = pd.DataFrame(seller_capacities)
    except Exception as e:
        print(f"❌ Seller capacity error: {repr(e)}")
        sbm.matching_state["seller_capacities"] = pd.DataFrame()

    try:
        buyer_capacity_data = sbm.matching_state.get("capacity")
        if buyer_capacity_data is None:
            raise RuntimeError("Buyer capacity not found.")

        buyer_capacity  = buyer_capacity_data["capacity_score"]
        buyer_breakdown = buyer_capacity_data["breakdown"]
        buyer_ratios    = pd.DataFrame([{"cu_id": inst_id, "capacity_score": buyer_capacity, **buyer_breakdown}])

        seller_cap_df = sbm.matching_state.get("seller_capacities", pd.DataFrame())
        if seller_cap_df.empty:
            raise RuntimeError("No seller capacities available.")

        seller_ratios = seller_cap_df.copy()
        if "breakdown" in seller_ratios.columns:
            breakdown_expanded = seller_ratios["breakdown"].apply(pd.Series)
            seller_ratios = pd.concat([seller_ratios.drop(columns=["breakdown"]), breakdown_expanded], axis=1)

        capacity_pairs = sbm.build_capacity_pairs(
            buyer_id=inst_id, buyer_ratios=buyer_ratios, seller_ratios=seller_ratios
        )
        sbm.matching_state["capacity_pairs"] = capacity_pairs
    except Exception as e:
        print(f"❌ Capacity pairs error: {repr(e)}")
        capacity_pairs = pd.DataFrame()
        sbm.matching_state["capacity_pairs"] = capacity_pairs

    try:
        activity_ltd_pairs  = sbm.apply_ltd_band_refinement(pairs_df=activity_pairs,  ltd_lookup=ltd_lookup, buyer_label="Caprock")
        capacity_ltd_pairs  = sbm.apply_ltd_band_refinement(pairs_df=capacity_pairs,  ltd_lookup=ltd_lookup, buyer_label="Caprock")
    except Exception as e:
        print("Error in apply_ltd_band_refinement:", repr(e))
        return jsonify({"error": "Failed in LTD band refinement."}), 500

    if activity_ltd_pairs.empty and capacity_ltd_pairs.empty:
        return jsonify({"error": "No pairs remained after LTD refinement."}), 400

    try:
        activity_size_pairs = sbm.apply_size_band_refinement(pairs_df=activity_ltd_pairs, size_lookup=size_lookup, buyer_label="Caprock")
        capacity_size_pairs = sbm.apply_size_band_refinement(pairs_df=capacity_ltd_pairs, size_lookup=size_lookup, buyer_label="Caprock")
    except Exception as e:
        print("Error in apply_size_band_refinement:", repr(e))
        return jsonify({"error": "Failed in size band refinement."}), 500

    if activity_size_pairs.empty and capacity_size_pairs.empty:
        return jsonify({"error": "No pairs remained after asset size refinement."}), 400

    try:
        activity_geo_pairs  = sbm.apply_geo_refinement(pairs_df=activity_size_pairs, state_lookup=state_lookup, buyer_label="Caprock")
        capacity_geo_pairs  = sbm.apply_geo_refinement(pairs_df=capacity_size_pairs, state_lookup=state_lookup, buyer_label="Caprock")
    except Exception as e:
        print("Error in apply_geo_refinement:", repr(e))
        return jsonify({"error": "Failed in geo refinement."}), 500

    if activity_geo_pairs.empty and capacity_geo_pairs.empty:
        return jsonify({"error": "No pairs remained after geo refinement."}), 400

    try:
        final_ranked = sbm.build_ranked_matches(activity_geo_pairs, capacity_geo_pairs)
    except Exception as e:
        print("Error in build_ranked_matches:", repr(e))
        return jsonify({"error": "Failed to build ranked matches."}), 500

    if final_ranked.empty:
        return jsonify({"error": "No ranked matches available."}), 400

    try:
        pdf_name = "Caprock_SellerBuyer_Matching_Report.pdf"
        pdf_path = REPORT_DIR / pdf_name
        sbm.generate_pdf_report(ranked_matches=final_ranked, output_path=str(pdf_path), buyer_label="Caprock")
    except Exception as e:
        print("Error in generate_pdf_report:", repr(e))
        return jsonify({"error": "Failed to generate PDF report."}), 500

    return jsonify({"pdf_url": f"/reports/{pdf_name}"})


@app.get("/reports/<name>")
def get_report(name: str):
    return send_from_directory(REPORT_DIR, name)


# =============================================================================
# LTN ANALYZER PIPELINE
# =============================================================================

@app.route('/ltn')
def ltn_analyzer():
    return render_template('ltn_analyzer.html')


@app.route('/ltn-analyze', methods=['POST'])
def ltn_analyze():
    action = request.form.get('action', 'analyze')
    if action == 'refine':
        return ltn_refine()
    elif action == 'report':
        return ltn_report()

    for key in ['raw_tape', 'final_box', 'summary', 'tape_size']:
        session.pop(key, None)

    os.makedirs('static', exist_ok=True)

    if 'loan_tape' not in request.files:
        return jsonify({"error": "No loan tape file uploaded"}), 400

    file = request.files['loan_tape']
    if not file or file.filename == '':
        return jsonify({"error": "No valid loan tape file"}), 400

    try:
        if file.filename.lower().endswith(('.xls', '.xlsx')):
            df_raw = pd.read_excel(file)
        else:
            df_raw = pd.read_csv(file)
    except Exception as e:
        print("File read error:", repr(e))
        return jsonify({"error": f"Cannot read file: {str(e)}"}), 400

    def safe_float(v):
        try:
            return float(v) if v and v != '' else None
        except:
            return None

    _raw_tape_size = len(df_raw)  # capture BEFORE pipeline modifies anything
    print(f"🔥 LTN BLOCK ENTERED — raw tape: {_raw_tape_size} rows")
    try:
        loan_boxes = LTN_loan_tape_prep().get_loan_box_definitions()
        prep = LTN_loan_tape_prep(loan_box_def=loan_boxes, df_raw=df_raw)
        print("Prep initialized OK")

        prep = prep.column_standardization()
        print(f"✅ After exact matches: {len(getattr(prep, 'unmapped_columns', []))} unmapped")

        prep = prep.column_semantic_repair(similarity_threshold=0.10)
        print(f"🎯 Suggestions found: {len(getattr(prep, 'semantic_suggestions', []))}")

        prep = prep.assign_top_ranked_columns(min_confidence=0.10)
        print(f"✅ Applied mappings: {getattr(prep, 'applied_semantic_mappings', {})}")

        prep = prep.value_format_standardization()
        if prep is None: raise ValueError("value_format_standardization returned None")

        prep = prep.data_quality_check()
        if prep is None: raise ValueError("data_quality_check returned None")

        prep = prep.missing_value_impute()
        if prep is None: raise ValueError("missing_value_impute returned None")

        prep = prep.outlier_removal()
        if prep is None: raise ValueError("outlier_removal returned None")

        prep = prep.normalization()
        if prep is None: raise ValueError("normalization returned None")

        final_box = prep.loan_box_assignment()
        print("Final box shape:", final_box.shape if final_box is not None else "None!")

        try:
            numeric_cols = ['orig_fico', 'orig_ltv', 'dti_back_end', 'original_balance', 'original_term_months']
            for col in numeric_cols:
                if col in final_box.columns:
                    final_box[col] = pd.to_numeric(final_box[col], errors='coerce')

            total_loans     = len(final_box)
            matched_loans   = len(final_box[final_box['in_box_flag'] == True])
            match_pct       = (matched_loans / total_loans * 100) if total_loans > 0 else 0
            matched_df      = final_box[final_box['in_box_flag'] == True]
            wa_fico         = matched_df['orig_fico'].mean()
            wa_ltv          = matched_df['orig_ltv'].mean(skipna=True)
            wa_dti          = matched_df['dti_back_end'].mean(skipna=True)
            wa_term         = matched_df['original_term_months'].mean(skipna=True)
            matched_balance = matched_df['original_balance'].sum(skipna=True)
        except Exception as e:
            print("Summary error:", repr(e))
            total_loans = matched_loans = match_pct = wa_fico = wa_ltv = wa_dti = wa_term = matched_balance = 0

        summary = {
            'total_loans':        int(total_loans),
            'matched_loans':      int(matched_loans),
            'match_pct':          round(match_pct, 3),
            'wa_fico':            round(float(wa_fico), 1) if wa_fico and not np.isnan(wa_fico) else None,
            'wa_ltv':             round(float(wa_ltv), 1) if wa_ltv and not np.isnan(wa_ltv) else None,
            'wa_dti':             round(float(wa_dti), 1) if wa_dti and not np.isnan(wa_dti) else None,
            'wa_term':            round(float(wa_term), 1) if wa_term and not np.isnan(wa_term) else None,
            'matched_balance_mm': round(matched_balance / 1e6, 3) if matched_balance and not np.isnan(matched_balance) else None,
        }
        summary = {k: (None if v is not None and isinstance(v, float) and np.isnan(v) else v) for k, v in summary.items()}

        for col in final_box.select_dtypes(include=['object']).columns:
            final_box[col] = final_box[col].astype(str)

        dup_count = final_box['loan_id'].duplicated().sum()
        if dup_count > 0:
            final_box = final_box.drop_duplicates(subset=['loan_id'], keep='first')

        filename = f"LTN_Full_Box_Assignment_{len(final_box)}_loans.csv"
        final_box.to_csv(f'static/{filename}', index=False)

        session['raw_tape']  = df_raw.to_dict('records')
        session['final_box'] = final_box.to_dict('records')
        session['summary']   = summary
        session['tape_size'] = _raw_tape_size  # raw rows, not post-pipeline
        print("Phase 1 session summary written:", summary)
        _save_last_run()  # persist to disk — survives Flask restarts

        response = make_response(send_file(
            f'static/{filename}',
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        ))
        return response

    except Exception as e:
        print("Pipeline error:", repr(e))
        return jsonify({"error": f"Pipeline failed: {str(e)}"}), 500


@app.route('/ltn-refine', methods=['POST'])
def ltn_refine():
    print("Phase 2 session keys:", list(session.keys()))

    loan_boxes = LTN_loan_tape_prep().get_loan_box_definitions()

    final_box_data = session.get('final_box', [])
    if not final_box_data:
        return jsonify({"error": "Run Phase 1 first"}), 400

    final_box = pd.DataFrame(final_box_data)

    # ── Fix: in_box_flag is stored as string "True"/"False" in session ──
    # apply_ui_filters() does df[df['in_box_flag'] == True] which fails on strings
    if 'in_box_flag' in final_box.columns:
        final_box['in_box_flag'] = final_box['in_box_flag'].map(
            lambda x: True if str(x).strip().lower() == 'true' else False
        )

    prep      = LTN_loan_tape_prep(loan_box_def=loan_boxes, df_raw=final_box)

    filters = {
        'fico_min':           float(request.form.get('fico_min') or 0),
        'fico_max':           float(request.form.get('fico_max') or 850),
        'ltv_min':            float(request.form.get('ltv_min') or 0),
        'ltv_max':            float(request.form.get('ltv_max') or 200),
        'dti_min':            float(request.form.get('dti_min') or 0),
        'dti_max':            float(request.form.get('dti_max') or 100),
        'term_min':           int(request.form.get('term_min') or 0),
        'term_max':           int(request.form.get('term_max') or 999),
        'balance_min':        float(request.form.get('balance_min') or 0),
        'balance_max':        float(request.form.get('balance_max') or 999999),
        'interest_rate_min':  float(request.form.get('interest_rate_min') or 0),
        'interest_rate_max':  float(request.form.get('interest_rate_max') or 18),
        'days_past_due_min':  int(request.form.get('days_past_due_min') or 0),
        'days_past_due_max':  int(request.form.get('days_past_due_max') or 999),
        'product':            request.form.getlist('product'),
        'remove_zero_balance': request.form.get('remove_zero_balance') == 'on',
        'remove_zero_rate':    request.form.get('remove_zero_rate') == 'on',
        'exclude_over_30_dq':  request.form.get('exclude_over_30_dq') == 'on',
        'remove_matured':      request.form.get('remove_matured') == 'on',
        'min_remaining_term':  int(request.form.get('min_remaining_term', '').strip())
                               if request.form.get('min_remaining_term', '').strip().isdigit() else 0,
    }

    prep.apply_ui_filters(filters)
    filtered_box  = prep.filtered_loan_box
    box_for_output = filtered_box

    if len(box_for_output) == 0:
        box_for_metrics = final_box
        print("Using full box for metrics (filters eliminated all loans)")
    else:
        box_for_metrics = box_for_output

    total_loans   = len(box_for_metrics)
    matched_loans = len(box_for_metrics[box_for_metrics['in_box_flag'] == True]) if 'in_box_flag' in box_for_metrics.columns else total_loans
    match_pct     = round((matched_loans / total_loans * 100), 1) if total_loans > 0 else 0
    wa_fico       = round(box_for_metrics['orig_fico'].mean(skipna=True), 1)       if 'orig_fico'         in box_for_metrics.columns else 0
    wa_ltv        = round(box_for_metrics['orig_ltv'].mean(skipna=True), 1)        if 'orig_ltv'          in box_for_metrics.columns else 0
    wa_dti        = round(box_for_metrics['dti_back_end'].mean(skipna=True), 1)    if 'dti_back_end'      in box_for_metrics.columns else 0
    total_balance = round(box_for_metrics['original_balance'].sum(skipna=True) / 1_000_000, 1) if 'original_balance' in box_for_metrics.columns else 0

    # ── Weighted average rate ──
    wa_rate = 0.0
    if len(box_for_output) > 0 and 'original_balance' in box_for_output.columns and 'current_rate' in box_for_output.columns:
        bal_sum = pd.to_numeric(box_for_output['original_balance'], errors='coerce').sum()
        if bal_sum > 0:
            wa_rate = float(
                (pd.to_numeric(box_for_output['original_balance'], errors='coerce') *
                 pd.to_numeric(box_for_output['current_rate'], errors='coerce')).sum() / bal_sum
            )

    # ── Write Phase 2 stats into session so /ltn-summary can return them ──
    existing_summary = session.get('summary', {})
    existing_summary.update({
        'refined_loans':       len(box_for_output),
        'wa_rate':             round(wa_rate, 4),
        'refined_filename':    f"LTN_Refined_Box_Assignment_{len(box_for_output)}_loans.csv",
        'wa_fico':             wa_fico if wa_fico and not np.isnan(wa_fico) else existing_summary.get('wa_fico'),
        'refined_balance_mm':  round(total_balance, 3),
        'matched_balance_mm':  round(total_balance, 3),  # keep in sync for chat flash
    })
    session['summary'] = existing_summary
    print(f"[ltn_refine] session summary updated: refined_loans={len(box_for_output)}, wa_fico={wa_fico}, wa_rate={wa_rate:.4f}, balance={total_balance:.2f}M")
    _save_last_run()  # persist P2 results to disk

    filename = f"LTN_Refined_Box_Assignment_{len(box_for_output)}_loans.csv"
    box_for_output.to_csv(f'static/{filename}', index=False)

    return send_file(
        f'static/{filename}',
        as_attachment=True,
        download_name=filename,
        mimetype='text/csv'
    )


# ── NEW ENDPOINT: /ltn-summary ─────────────────────────────────────────────────
@app.route('/ltn-summary')
def ltn_summary():
    """
    Returns current session summary as JSON for frontend dynamic stat card updates.
    Also computes real FICO and LTV distribution from the most recent refined CSV.
    Called by index.html immediately after Phase 2 fetch completes.
    """
    import glob

    # ── CRITICAL: only return data if there is an active session ──
    # Without this guard, stale CSVs on disk from previous Flask runs
    # pollute the dashboard with wrong values on every fresh app launch.
    if not session.get('final_box') and not session.get('summary'):
        return jsonify({'distribution': {}})

    # Always return P1 stats (matched_loans, match_pct, tape_size) from session
    # These are available as soon as Phase 1 runs, regardless of Phase 2
    summary      = dict(session.get('summary', {}))
    distribution = {}

    # Add tape_size to summary so dashboard can show it
    if session.get('tape_size'):
        summary['tape_size'] = session.get('tape_size')

    try:
        # Only read refined CSV if Phase 2 has actually run this session
        # P1 stats are already in summary above — this block adds distributions only
        if not session.get('summary', {}).get('refined_filename'):
            return jsonify({**summary, 'distribution': distribution})

        # Find most recent Phase 2 refined CSV
        refined_files = sorted(
            glob.glob('static/LTN_Refined_Box_Assignment_*.csv'),
            key=os.path.getmtime,
            reverse=True
        )

        if refined_files:
            df = pd.read_csv(refined_files[0])

            # Filter to matched loans only
            if 'in_box_flag' in df.columns:
                df = df[df['in_box_flag'] == True].copy()

            if len(df) > 0:
                for col in ['orig_fico', 'orig_ltv', 'original_balance', 'current_rate']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # ── Update P2 refined count — do NOT overwrite P1 matched_loans ──
                summary['refined_loans'] = len(df)

                # ── FICO distribution (4 bands) ──
                fico_labels = ['<660', '660–699', '700–749', '750+']
                if 'orig_fico' in df.columns:
                    df['fico_band'] = pd.cut(
                        df['orig_fico'],
                        bins=[300, 660, 700, 750, 850],
                        labels=fico_labels,
                        include_lowest=True
                    )
                    fico_dist = (
                        df.groupby('fico_band', observed=True)
                        .size()
                        .reindex(fico_labels, fill_value=0)
                        .reset_index()
                    )
                    fico_dist.columns = ['band', 'count']
                    distribution['fico'] = fico_dist.to_dict('records')

                # ── LTV distribution (4 bands) ──
                ltv_labels = ['≤ 70%', '71–80%', '81–85%', '>85%']
                if 'orig_ltv' in df.columns:
                    df['ltv_band'] = pd.cut(
                        df['orig_ltv'],
                        bins=[0, 70, 80, 85, 200],
                        labels=ltv_labels,
                        include_lowest=True
                    )
                    ltv_dist = (
                        df.groupby('ltv_band', observed=True)
                        .size()
                        .reindex(ltv_labels, fill_value=0)
                        .reset_index()
                    )
                    ltv_dist.columns = ['band', 'count']
                    distribution['ltv'] = ltv_dist.to_dict('records')

                # ── Compute wa_rate if not already in summary ──
                if 'wa_rate' not in summary and 'current_rate' in df.columns and 'original_balance' in df.columns:
                    bal_sum = df['original_balance'].sum()
                    if bal_sum > 0:
                        wa_rate = float((df['original_balance'] * df['current_rate']).sum() / bal_sum)
                        summary['wa_rate'] = round(wa_rate, 4)
                        session['summary'] = {**session.get('summary', {}), 'wa_rate': summary['wa_rate']}

    except Exception as e:
        print(f"[ltn-summary] distribution error: {e}")

    return jsonify({**summary, 'distribution': distribution})


@app.route('/ltn-report', methods=['POST'])
def ltn_report():
    if 'final_box' not in session:
        return "Upload Phase 1 first!", 400

    df_raw = pd.DataFrame(session.get('raw_tape', []))
    if len(df_raw) == 0:
        df_raw = pd.DataFrame(session['final_box'])

    df = pd.DataFrame(session['final_box'])
    # Fix: in_box_flag stored as string "True"/"False" in session — coerce to bool
    if 'in_box_flag' in df.columns:
        df['in_box_flag'] = df['in_box_flag'].map(lambda x: True if str(x).strip().lower() == 'true' else False)
    df = df[df['in_box_flag'] == True].copy()
    print(f"Report input: {len(df)} matched loans")

    fico_min          = float(request.form.get('fico_min') or 0)
    fico_max          = float(request.form.get('fico_max') or 850)
    ltv_min           = float(request.form.get('ltv_min') or 0)
    ltv_max           = float(request.form.get('ltv_max') or 100)
    dti_min           = float(request.form.get('dti_min') or 0)
    dti_max           = float(request.form.get('dti_max') or 50)
    term_min          = int(request.form.get('term_min') or 0)
    term_max          = int(request.form.get('term_max') or 999)
    balance_min       = float(request.form.get('balance_min') or 0)
    balance_max       = float(request.form.get('balance_max') or 999999999)
    interest_rate_min = float(request.form.get('interest_rate_min') or 0)
    interest_rate_max = float(request.form.get('interest_rate_max') or 18.0)
    days_past_due_min = int(request.form.get('days_past_due_min') or 0)
    days_past_due_max = int(request.form.get('days_past_due_max') or 90)

    analyzer = LTN_loan_tape_prep(df_raw=df)

    filters = {
        'fico_min':           float(request.form.get('fico_min') or 0),
        'fico_max':           float(request.form.get('fico_max') or 850),
        'ltv_min':            float(request.form.get('ltv_min') or 0),
        'ltv_max':            float(request.form.get('ltv_max') or 200),
        'dti_min':            float(request.form.get('dti_min') or 0),
        'dti_max':            float(request.form.get('dti_max') or 100),
        'term_min':           int(request.form.get('term_min') or 0),
        'term_max':           int(request.form.get('term_max') or 999),
        'balance_min':        float(request.form.get('balance_min') or 0),
        'balance_max':        float(request.form.get('balance_max') or 999999),
        'interest_rate_min':  float(request.form.get('interest_rate_min') or 0),
        'interest_rate_max':  float(request.form.get('interest_rate_max') or 18),
        'days_past_due_min':  int(request.form.get('days_past_due_min') or 0),
        'days_past_due_max':  int(request.form.get('days_past_due_max') or 999),
        'product':            request.form.getlist('product'),
        'remove_zero_balance': request.form.get('remove_zero_balance') == 'on',
        'remove_zero_rate':    request.form.get('remove_zero_rate') == 'on',
        'exclude_over_30_dq':  request.form.get('exclude_over_30_dq') == 'on',
        'remove_matured':      request.form.get('remove_matured') == 'on',
        'min_remaining_term':  int(request.form.get('min_remaining_term', '').strip())
                               if request.form.get('min_remaining_term', '').strip().isdigit() else 0,
        'wa_rate_target_min':  float(request.form.get('wa_rate_target_min', 5.0) or 5.0) / 100,
        'wa_rate_target_max':  float(request.form.get('wa_rate_target_max', 15.0) or 15.0) / 100,
    }

    analyzer.apply_ui_filters(filters)
    df_filtered = analyzer.filtered_loan_box
    print(f"📊 Report filtered: {len(df_filtered)} rows")

    tape_size       = session.get('tape_size', len(df_raw))
    matched_count   = len(df)
    filtered_count  = len(df_filtered)

    tape_match_rate  = matched_count  / tape_size     * 100 if tape_size     > 0 else 0
    filter_retention = filtered_count / matched_count * 100 if matched_count > 0 else 0
    final_retention  = filtered_count / tape_size     * 100 if tape_size     > 0 else 0

    wa_fico  = df_filtered['orig_fico'].mean()        if len(df_filtered) > 0 else 0
    wa_ltv   = df_filtered['orig_ltv'].mean()         if len(df_filtered) > 0 else 0
    wa_dti   = df_filtered['dti_back_end'].mean()     if len(df_filtered) > 0 else 0
    wa_rate  = (
        (df_filtered['original_balance'] * df_filtered['current_rate']).sum() /
        df_filtered['original_balance'].sum()
        if len(df_filtered) > 0 and df_filtered['original_balance'].sum() > 0 else 0
    )
    total_balance     = df_filtered['original_balance'].sum() / 1e6 if len(df_filtered) > 0 else 0
    wa_rate_target_min = filters['wa_rate_target_min']
    wa_rate_target_max = filters['wa_rate_target_max']

    if wa_rate_target_min * 100 <= wa_rate <= wa_rate_target_max * 100:
        yield_compliance = f"✅ Portfolio meets target (WA Rate {wa_rate:.1f}% within {wa_rate_target_min*100:.0f}–{wa_rate_target_max*100:.0f}% band)."
    elif wa_rate < wa_rate_target_min * 100:
        yield_compliance = f"❌ Portfolio yield ({wa_rate:.1f}%) falls below target {wa_rate_target_min*100:.0f}–{wa_rate_target_max*100:.0f}% band."
    else:
        yield_compliance = f"⚠️ Portfolio yield ({wa_rate:.1f}%) exceeds target band {wa_rate_target_min*100:.0f}–{wa_rate_target_max*100:.0f}%, elevated risk."

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial; margin: 40px; color: #222; }}
        .header {{ border-bottom: 3px solid #1e3a8a; padding-bottom: 15px; margin-bottom: 20px; }}
        .title {{ font-size: 24px; font-weight: bold; color: #1e3a8a; margin: 0; }}
        .subtitle {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .executive-summary {{ background: #f0f4ff; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #10b981; }}
        .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .metric-box {{ background: #fff; border: 1px solid #e5e7eb; padding: 12px; border-radius: 6px; text-align: center; }}
        .metric-value {{ font-size: 20px; font-weight: bold; color: #1e3a8a; }}
        .metric-label {{ font-size: 11px; color: #666; margin-top: 5px; }}
        .portfolio-composition {{ margin: 20px 0; }}
        .portfolio-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e5e7eb; }}
        .portfolio-row strong {{ color: #1e3a8a; }}
        .recommendations {{ background: #fffbeb; padding: 15px; border-radius: 8px; border-left: 4px solid #f59e0b; margin: 20px 0; }}
        .recommendations h3 {{ margin-top: 0; color: #d97706; }}
        .recommendations li {{ margin: 8px 0; font-size: 12px; }}
        .footer {{ text-align: center; font-size: 10px; color: #999; margin-top: 30px; border-top: 1px solid #e5e7eb; padding-top: 15px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 12px; }}
        th {{ background: #1e3a8a; color: white; padding: 8px; text-align: left; }}
        td {{ padding: 8px; border-bottom: 1px solid #e5e7eb; }}
        tr:nth-child(even) {{ background: #f9fafb; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title">LTN Portfolio Summary — Capital Flow AI</div>
        <div class="subtitle">Custom Loan Box Assignment Report | Generated {pd.Timestamp.now().strftime('%B %d, %Y at %H:%M')}</div>
    </div>

    <div class="executive-summary">
        <strong>Executive Summary:</strong> Tape processed <strong>{tape_size:,}</strong> loans total.
        Pre-refinement box rules matched <strong>{matched_count}</strong> loans
        (<strong>{tape_match_rate:.0f}%</strong> capture rate). Post-refinement filters retained
        <strong>{filtered_count}</strong> loans (<strong>{filter_retention:.0f}%</strong> retention |
        <strong>{final_retention:.0f}%</strong> final yield).
    </div>

    <div class="metrics">
        <div class="metric-box"><div class="metric-value">{tape_match_rate:.1f}%</div><div class="metric-label">Tape Match Rate</div></div>
        <div class="metric-box"><div class="metric-value">{filter_retention:.1f}%</div><div class="metric-label">Filter Retention</div></div>
        <div class="metric-box"><div class="metric-value">{final_retention:.1f}%</div><div class="metric-label">Final Yield</div></div>
        <div class="metric-box"><div class="metric-value">${total_balance:.1f}M</div><div class="metric-label">Total Balance</div></div>
        <div class="metric-box"><div class="metric-value">{wa_fico:.0f}</div><div class="metric-label">Avg FICO</div></div>
        <div class="metric-box"><div class="metric-value">{filtered_count}</div><div class="metric-label">Filtered Loans</div></div>
    </div>

    <div class="portfolio-composition">
        <strong style="color: #1e3a8a;">Portfolio Risk Profile:</strong>
        <div class="portfolio-row"><span>Average LTV</span><strong>{wa_ltv:.1f}%</strong></div>
        <div class="portfolio-row"><span>Average DTI</span><strong>{wa_dti:.1f}%</strong></div>
        <div class="portfolio-row"><span>Weighted Average Interest Rate</span><strong>{wa_rate:.1f}%</strong></div>
        <div class="portfolio-row"><span>Original Tape Size</span><strong>{tape_size} loans</strong></div>
        <div class="portfolio-row"><span>Box Matched Loans</span><strong>{matched_count} ({tape_match_rate:.1f}%)</strong></div>
        <div class="portfolio-row"><span>Final Filtered Loans</span><strong>{filtered_count} ({final_retention:.1f}%)</strong></div>
    </div>

    <strong style="color: #1e3a8a;">Yield Target Compliance (WA Rate Band Assessment):</strong>
    <div style="font-size: 1.1em; margin: 10px 0;">{yield_compliance}</div>

    <div class="filter-summary" style="margin-top: 16px; padding: 10px 12px; border: 1px solid #e5e7eb; border-radius: 6px; background-color: #f9fafb;">
        <strong style="color: #1e3a8a;">Filter &amp; Criteria Summary:</strong>
        <div style="margin-top: 6px; font-size: 12px;">
            FICO [{fico_min:.0f}–{fico_max:.0f}] |
            LTV {ltv_min:.0f}–{ltv_max:.0f}% |
            DTI {dti_min:.0f}–{dti_max:.0f}% |
            Term {term_min}–{term_max} mo |
            Balance ${balance_min/1e6:.1f}M–${balance_max/1e6:.1f}M |
            Rate {interest_rate_min:.1f}–{interest_rate_max:.1f}% |
            Days PD {days_past_due_min}–{days_past_due_max}
        </div>
    </div>

    <div class="recommendations">
        <h3>📊 Strategic Recommendations</h3>
        <ul>
            <li><strong>Portfolio Strength:</strong> {tape_match_rate:.1f}% tape cracking → {filter_retention:.1f}% refined retention = {final_retention:.1f}% deployable yield.</li>
            <li><strong>Liquidity Profile:</strong> ${total_balance:.1f}M portfolio suitable for {'prime securitization' if final_retention > 20 else 'non-QM/alt-A' if final_retention > 10 else 'special servicer pools'}.</li>
            <li><strong>Next Steps:</strong> Advance {filtered_count} loans to pricing engine; model {100-final_retention:.1f}% tape remainder for re-underwriting.</li>
            <li><strong>Risk Mitigation:</strong> Apply servicer advance on {100-tape_match_rate:.1f}% OOB; blend with higher-FICO tape for ABS eligibility.</li>
        </ul>
    </div>

    <table>
        <thead>
            <tr>
                <th>Loan ID</th><th>Product</th><th>Box ID</th><th>FICO</th>
                <th>LTV</th><th>DTI</th><th>Orig Bal</th><th>Cur Bal</th>
                <th>Rate</th><th>DPD</th><th>Term</th><th>Maturity</th>
                <th>Year</th><th>State</th><th>Status</th>
            </tr>
        </thead>
        <tbody>
            {' '.join([
                f'<tr><td>{row.loan_id}</td><td>{row.product_type}</td><td>{row.box_id}</td>'
                f'<td>{row.orig_fico:.0f}</td><td>{row.orig_ltv:.2f}%</td><td>{row.dti_back_end:.2f}%</td>'
                f'<td>${row.original_balance/1e3:.0f}K</td><td>${row.current_balance/1e3:.0f}K</td>'
                f'<td>{row.current_rate:.1f}%</td><td>{row.days_past_due}</td><td>{row.original_term_months}mo</td>'
                f'<td>{row.maturity_date}</td><td>{row.year}</td><td>{row.state}</td>'
                f'<td><span style="color:green"><strong>✓ PASS</strong></span></td></tr>'
                for _, row in df_filtered.iterrows()
            ])}
        </tbody>
    </table>

    <div class="footer">
        LTN Loan Tape Analyzer | Confidential — Capital Flow AI Internal Use<br/>
        Filters: FICO [{fico_min:.0f}–{fico_max:.0f}] | LTV {ltv_min:.0f}–{ltv_max:.0f}% |
        DTI {dti_min:.0f}–{dti_max:.0f}% | Term {term_min}–{term_max} mo |
        Rate {interest_rate_min:.1f}–{interest_rate_max:.1f}% | Days PD {days_past_due_min}–{days_past_due_max}
    </div>
</body>
</html>"""

    filename  = f"LTN_Refined_Portfolio_Report_{filtered_count}_loans.html"
    full_path = os.path.join('static', filename)
    os.makedirs('static', exist_ok=True)

    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"SENDING FILE: {full_path} EXISTS? {os.path.exists(full_path)}")
    return send_file(full_path, as_attachment=True, download_name=filename, mimetype='text/html')


######---- END OF LTN ANALYZER BLOCK ----######

# =============================================================================
# PORTFOLIO CO-PILOT — CHATBOT HELPERS
# =============================================================================

def _fetch_ncua_peers(size_band=None, state=None, year=None, quarter=None):
    now = datetime.now()
    if year is None or quarter is None:
        month = now.month
        if month <= 2:      year = now.year - 1; quarter = 3
        elif month <= 5:    year = now.year - 1; quarter = 4
        elif month <= 8:    year = now.year;     quarter = 1
        elif month <= 11:   year = now.year;     quarter = 2
        else:               year = now.year;     quarter = 3

    cache_file = NCUA_CACHE_DIR / f"ncua_{year}_q{quarter}.parquet"

    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            print(f"[MARKET_DATA] Loaded from cache: {len(df):,} CUs")
        except Exception as e:
            print(f"[MARKET_DATA] Cache read failed: {e}")
            cache_file.unlink(missing_ok=True)
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if df.empty:
        url = (f"https://ncua.gov/files/analysis/call-report-data/"
               f"call-report-data-{year}-q{quarter}.zip")
        print(f"[MARKET_DATA] Downloading: {url}")
        try:
            response = requests.get(url, timeout=120, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/zip,*/*",
                "Referer": "https://ncua.gov/analysis/credit-union-corporate-call-report-data/quarterly-data",
            })
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                target_files = [f for f in z.namelist() if f.lower().startswith("fs220") and f.lower().endswith(".txt")]
                if not target_files:
                    target_files = [f for f in z.namelist() if f.lower().endswith(".txt") and "acctdesc" not in f.lower()]
                if not target_files:
                    return pd.DataFrame()

                with z.open(target_files[0]) as f:
                    df = pd.read_csv(f, low_memory=False)

            df.columns = [c.strip().upper() for c in df.columns]
            try:
                df.to_parquet(cache_file, index=False)
            except Exception as e:
                print(f"[MARKET_DATA] Cache write failed: {e}")

        except Exception as e:
            print(f"[MARKET_DATA] Download failed: {e}")
            return pd.DataFrame()

    if df.empty:
        return df

    col_map = {
        "ACCT_010": "total_assets", "ACCT_025B": "total_loans",
        "ACCT_018": "total_deposits", "ACCT_997": "total_net_worth",
        "CU_NUMBER": "cu_number", "CU_NAME": "cu_name",
        "STATE": "state", "CYCLE_DATE": "as_of_date", "PEER_GROUP": "peer_group",
    }
    existing_map = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=existing_map)

    for col in ["total_assets", "total_loans", "total_deposits", "total_net_worth"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "total_loans" in df.columns and "total_deposits" in df.columns:
        df["ltd"] = df["total_loans"] / df["total_deposits"].replace(0, pd.NA)
    if "total_net_worth" in df.columns and "total_assets" in df.columns:
        df["net_worth_ratio"] = df["total_net_worth"] / df["total_assets"].replace(0, pd.NA)

    if "total_assets" in df.columns:
        def _size_band(a):
            if pd.isna(a):          return "Unknown"
            if a < 50_000_000:      return "SMALL"
            if a < 200_000_000:     return "LOWER MID"
            if a < 1_000_000_000:   return "UPPER MID"
            return "LARGE"
        df["size_band"] = df["total_assets"].apply(_size_band)

    filtered = df.copy()
    if state and "state" in filtered.columns:
        filtered = filtered[filtered["state"].str.upper() == state.strip().upper()]
    if size_band and "size_band" in filtered.columns:
        filtered = filtered[filtered["size_band"].str.upper() == size_band.strip().upper()]

    return filtered


def _flash_market_data(cu_metrics, peers, size_band, state):
    if peers.empty:
        return "Could not retrieve NCUA peer data at this time."

    peer_count  = len(peers)
    def peer_avg(col): return peers[col].dropna().mean() if col in peers.columns else None
    peer_ltd    = peer_avg("ltd")
    peer_nwr    = peer_avg("net_worth_ratio")
    peer_assets = peer_avg("total_assets")
    peer_loans  = peer_avg("total_loans")
    cu_ltd      = cu_metrics.get("ltd", 0)
    cu_nwr      = cu_metrics.get("net_worth_ratio", 0)
    cu_assets   = cu_metrics.get("total_assets", 0)
    cu_loans    = cu_metrics.get("total_loans", 0)
    cu_label    = cu_metrics.get("cu_label", "This CU")
    cu_role     = cu_metrics.get("role", "Unknown")

    def _vs(cu, peer, higher_is_better=True):
        if peer is None or cu is None: return "<span style='color:#64748b'>N/A</span>"
        diff = cu - peer; pct = (diff / peer * 100) if peer else 0
        color = "#10b981" if (diff >= 0) == higher_is_better else "#ef4444"
        arrow = "▲" if diff >= 0 else "▼"
        return f"<span style='color:{color};font-weight:600'>{arrow} {abs(pct):.1f}% vs peers</span>"

    def _pct(v): return f"{v:.1%}" if v is not None else "N/A"
    def _m(v):   return f"${v/1_000_000:.1f}M" if v is not None else "N/A"

    peer_label = f"{size_band} CUs" + (f" in {state}" if state else "")
    flash = (
        f"<strong>Peer Benchmark: {cu_label}</strong><br>"
        f"vs <strong>{peer_count:,} {peer_label}</strong> (NCUA live data)<br><br>"
        f"<strong>LTD:</strong> {_pct(cu_ltd)} → Role: <strong>{cu_role}</strong> | Peer avg: {_pct(peer_ltd)} {_vs(cu_ltd, peer_ltd, False)}<br><br>"
        f"<strong>Net Worth Ratio:</strong> {_pct(cu_nwr)} | Peer avg: {_pct(peer_nwr)} {_vs(cu_nwr, peer_nwr, True)}<br><br>"
        f"<strong>Total Assets:</strong> {_m(cu_assets)} | Peer avg: {_m(peer_assets)}<br><br>"
        f"<strong>Total Loans:</strong> {_m(cu_loans)} | Peer avg: {_m(peer_loans)}<br><br>"
    )
    insights = []
    if cu_ltd and peer_ltd:
        if cu_ltd < peer_ltd * 0.9:
            insights.append("LTD well below peer avg — strong BUYER position, excess liquidity to deploy.")
        elif cu_ltd > peer_ltd * 1.1:
            insights.append("LTD above peer avg — SELLER profile, well-positioned for loan participation sales.")
        else:
            insights.append("LTD in line with peers — balanced liquidity position.")
    if cu_nwr and peer_nwr:
        if cu_nwr > peer_nwr * 1.1:
            insights.append("Net worth ratio above peers — strong capital buffer for participation activity.")
        elif cu_nwr < 0.07:
            insights.append("Net worth ratio below NCUA 7% Well-Capitalised threshold — caution advised.")
    if insights:
        flash += "<strong>Strategic Insights:</strong><br>" + "".join(f"• {i}<br>" for i in insights)
    return flash


def _flash_matching(ranked, cu_label, ltd, role):
    if ranked is None or (isinstance(ranked, pd.DataFrame) and ranked.empty):
        return (f"Analysed <strong>{cu_label}</strong> — LTD: <strong>{ltd:.1%}</strong> → Role: <strong>{role}</strong>.<br>"
                "No qualifying counterparty matches found.")
    labels = ["Priority Target", "Good Fit", "Acceptable"]
    rows = ranked.to_dict('records') if isinstance(ranked, pd.DataFrame) else ranked
    lines = [
        f"<strong>#{i+1} {r.get('seller_id', r.get('buyer_id','?'))}</strong> — "
        f"{labels[i] if i < 3 else 'Match'} (Score: {r.get('final_score',0):.0f})"
        for i, r in enumerate(rows[:3])
    ]
    return (
        f"Analysed <strong>{cu_label}</strong> — LTD: <strong>{ltd:.1%}</strong> → Role: <strong>{role}</strong>.<br><br>"
        f"<strong>Top {len(lines)} Counterparty Recommendations:</strong><br>"
        + "<br>".join(lines)
        + "<br><br>Full ranking report available below."
    )


def _parse_boxing_filters(msg):
    m = msg.lower(); f = {}
    if re.search(r'\b(standard|default|no filter|fine|ok|okay)\b', m): return {}

    # ── FICO ──
    hit = re.search(r'fico\s*(?:min|>=|>|≥|\+)?\s*:?\s*[-–]?\s*(\d+)', m)
    if hit: f['fico_min'] = int(hit.group(1))
    hit = re.search(r'fico\s*(?:max|<=|<|≤)\s*:?\s*[-–]?\s*(\d+)', m)
    if hit: f['fico_max'] = int(hit.group(1))
    hit = re.search(r'fico\s+(\d+)\s*[-–]\s*(\d+)', m)
    if hit: f['fico_min'] = int(hit.group(1)); f['fico_max'] = int(hit.group(2))

    # ── LTV / DTI / Term / Balance ──
    hit = re.search(r'ltv\s*(?:max|<=|<|≤)?\s*:?\s*[-–]?\s*(\d+)', m)
    if hit: f['ltv_max'] = int(hit.group(1))
    hit = re.search(r'dti\s*(?:max|<=|<|≤)?\s*:?\s*[-–]?\s*(\d+)', m)
    if hit: f['dti_max'] = int(hit.group(1))
    hit = re.search(r'rate\s+(\d+[\.\d]*)\s*[-–to]+\s*(\d+[\.\d]*)', m)
    if hit: f['interest_rate_min'] = float(hit.group(1)); f['interest_rate_max'] = float(hit.group(2))
    hit = re.search(r'term\s*(?:max|<=|<|≤)?\s*:?\s*[-–]?\s*(\d+)', m)
    if hit: f['term_max'] = int(hit.group(1))
    hit = re.search(r'balance\s*(?:max|<=|<|≤)?\s*[-–]?\s*(\d+)', m)
    if hit: f['balance_max'] = int(hit.group(1))

    # ── Product type — handles natural language product/loan type requests ──
    if re.search(r'auto\s*new|new\s*only|only\s*new|keep.*new|new.*loan|new.*vehicle|new.*auto', m):
        f['product'] = ['AUTO_NEW']
    elif re.search(r'auto\s*used|used\s*only|only\s*used|keep.*used|used.*loan|used.*vehicle|used.*auto', m):
        f['product'] = ['AUTO_USED']
    elif re.search(r'product.*=.*old|remove.*old|old.*product|exclude.*old|type.*old', m):
        f['product_exclude'] = ['OLD']
    elif re.search(r'direct\s*only|only\s*direct|keep.*direct|product.*direct', m):
        f['product'] = ['DIRECT']
    elif re.search(r'indirect\s*only|only\s*indirect|keep.*indirect|product.*indirect', m):
        f['product'] = ['INDIRECT']

    # ── Boolean exclusion flags ──
    if re.search(r'over.?30|exclude.*dq|delinquent|30\+\s*dpd|>\s*30\s*d', m):
        f['exclude_over_30_dq'] = True
    if re.search(r'remove\s+matured|matured\s+loan|exclude.*matur', m):
        f['remove_matured'] = True
    if re.search(r'zero\s+balance|remove.*zero.*bal|exclude.*zero.*bal', m):
        f['remove_zero_balance'] = True
    if re.search(r'zero\s+rate|remove.*zero.*rate|exclude.*zero.*rate', m):
        f['remove_zero_rate'] = True

    return f


def _extract_text_for_rag(file_path, filename, user_query=""):
    fname = filename.lower(); text = ""
    try:
        if fname.endswith(".pdf"):
            keywords = [w.lower() for w in user_query.split() if len(w) > 3]
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    all_pages = []
                    for page_num, page in enumerate(pdf.pages):
                        page_text = ""
                        for table in page.extract_tables():
                            for row in table:
                                row_text = " | ".join(str(c).strip() for c in row if c and str(c).strip())
                                if row_text: page_text += row_text + "\n"
                        raw = page.extract_text()
                        if raw: page_text += raw + "\n"
                        all_pages.append((page_num + 1, page_text))
                    relevant = []
                    other    = []
                    for pn, pt in all_pages:
                        if keywords and any(kw in pt.lower() for kw in keywords):
                            relevant.append(f"\n=== PAGE {pn} (RELEVANT) ===\n{pt[:2000]}")
                        else:
                            other.append(f"\n=== PAGE {pn} ===\n{pt[:300]}")
                    text = ("".join(relevant) + "".join(other))[:15000]
            except ImportError:
                import PyPDF2
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages: text += page.extract_text() or ""
                text = text[:15000]
        elif fname.endswith((".xlsx", ".xls")):
            import openpyxl
            wb = openpyxl.load_workbook(file_path, data_only=True)
            for sheet in wb.worksheets:
                text += f"\n[Sheet: {sheet.title}]\n"
                for row in sheet.iter_rows(values_only=True):
                    row_text = " | ".join(str(c) for c in row if c is not None)
                    if row_text.strip(): text += row_text + "\n"
        elif fname.endswith(".csv"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f: text = f.read()
        elif fname.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f: text = f.read()
        elif fname.endswith(".docx"):
            import docx
            doc = docx.Document(file_path)
            text = "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        text = f"[Text extraction failed: {e}]"
    return text[:15000]


def _rag_answer(user_query, doc_text, filename):
    persona_label  = session.get('persona_label', 'user')
    persona_tone   = session.get('persona_tone', 'helpful and professional')
    persona_output = session.get('persona_output', 'clear responses')
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": (
                    f'You are Portfolio Co-Pilot. The user uploaded "{filename}". '
                    "Find ALL occurrences of what the user asks about across all pages. "
                    "For each occurrence state: 'Page N — [exact label]: $VALUE'. "
                    f"Tone: {persona_tone}. Output: {persona_output}."
                )},
                {"role": "user", "content": f"--- DOCUMENT ---\n{doc_text}\n--- END ---\n\nQuestion: {user_query}"}
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"RAG error: {e}"


def _clean_markdown(text):
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'#{1,3}\s*(.+)', r'<strong>\1</strong><br>', text)
    text = re.sub(r'^\s*[\*\-\+]\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'(\d+\.?\d*%)', r'<strong>\1</strong>', text)
    text = re.sub(r'\b(BUYER|SELLER)\b', r'<strong>\1</strong>', text)
    text = re.sub(r'\b(CU_[A-Z0-9_]+)\b', r'<strong>\1</strong>', text)
    text = text.replace('\n', '<br>')
    text = re.sub(r'(<br>\s*){3,}', '<br><br>', text)
    return text.strip()


def _general_answer(user_query, history):
    persona_tone     = session.get('persona_tone', 'helpful and professional')
    persona_label    = session.get('persona_label', 'user')
    persona_output   = session.get('persona_output', 'clear chat responses')
    persona_suggests = session.get('persona_suggests', [])
    suggests_text = (f"After answering, naturally suggest one of: {', '.join(persona_suggests)}." if persona_suggests else "")

    messages = [
        {"role": "system", "content": (
            f"You are Portfolio Co-Pilot, an intelligent loan participation assistant for credit unions built by Capital Flow AI. "
            f"You specialise in NCUA 5300 call reports, loan participation matching, loan tape boxing, LTD/LTV/DTI/FICO ratios, and credit union regulatory frameworks.\n\n"
            f"IMPORTANT DOMAIN DEFINITIONS:\n"
            f"- LTD = Loan-to-Deposit ratio (total loans / total deposits). Never define as Loan-to-Tangible Equity.\n"
            f"- LTV = Loan-to-Value ratio. DTI = Debt-to-Income ratio. FICO = credit score 300-850.\n\n"
            f"FORMATTING RULES:\n"
            f"- Use <br> for line breaks. Use <strong>value</strong> for ALL key metrics.\n"
            f"- Use • for bullets. Never use markdown ** or ##. Never use markdown tables.\n"
            f"- Keep concise — max 3 lines per section.\n\n"
            f"You are speaking with a {persona_label}. Tone: {persona_tone}. Output: {persona_output}.\n"
            f"{suggests_text}"
        )},
        *[
            {"role": role, "content": content}
            for turn in history[-6:]
            for role, content in [("user", turn.get("user","")), ("assistant", turn.get("assistant",""))]
            if content
        ],
        {"role": "user", "content": user_query}
    ]
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", messages=messages, temperature=0.5, max_tokens=600
        )
        return _clean_markdown(response.choices[0].message.content.strip())
    except Exception as e:
        return f"I encountered an error: {e}"


# =============================================================================
# CHATBOT ROUTE 1 — POST /chat/upload
# =============================================================================

@app.route('/chat/upload', methods=['POST'])
def chatbot_upload():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"answer": "No file received. Please try re-attaching it."}), 400

    original_name = file.filename
    fname_lower   = original_name.lower()

    try:
        save_path = UPLOAD_DIR / original_name
        file.save(str(save_path))
    except Exception as e:
        return jsonify({"answer": f"Could not save file: {e}"}), 500

    if any(kw in fname_lower for kw in ["5300", "call_report", "callreport"]):
        ftype = "call_report"
        hint  = (
            f"5300 Call Report staged: <strong>{original_name}</strong>.<br><br>"
            "What would you like me to do?<br>"
            "• <em>\"Find me potential buyers for this CU\"</em><br>"
            "• <em>\"Run the counterparty matching engine\"</em><br>"
            "• <em>\"Identify top seller recommendations\"</em>"
        )
    elif fname_lower.endswith((".csv", ".xlsx", ".xls")) or "tape" in fname_lower:
        ftype = "loan_tape"
        hint  = (
            f"Loan Tape staged: <strong>{original_name}</strong>.<br><br>"
            "What would you like me to do?<br>"
            "• <em>\"Run loan boxing with standard Auto yield targets\"</em><br>"
            "• <em>\"Apply custom filters before boxing\"</em><br>"
            "• <em>\"Clean this portfolio and generate box assignments\"</em>"
        )
    elif fname_lower.endswith((".pdf", ".txt", ".docx")):
        ftype = "rag_doc"
        hint  = (
            f"Document staged: <strong>{original_name}</strong>.<br><br>"
            "I can answer questions about this document. What would you like to know?"
        )
    else:
        ftype = "unknown"
        hint  = (
            f"File received: <strong>{original_name}</strong>.<br><br>"
            "Could you clarify — is this a 5300 Call Report, a Loan Tape, or a reference document?"
        )

    session['chatbot_pending_file'] = {"path": str(save_path), "name": original_name, "type": ftype}
    if 'chatbot_history' not in session:
        session['chatbot_history'] = []

    return jsonify({"answer": hint})


# =============================================================================
# CHATBOT ROUTE 2 — POST /chat/message
# =============================================================================

@app.route('/chat/message', methods=['POST'])
def chatbot_message():
    data     = request.get_json(silent=True) or {}
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify({"answer": "Please type a message."}), 400

    history = session.get('chatbot_history', [])
    pending = session.get('chatbot_pending_file')

    # ── Intent override rules ──────────────────────────────────────────────────
    if re.search(r'\bCU_[A-Z0-9_]+\b', user_msg, re.I):
        intent = "MATCHING_ENGINE"
    elif session.get('matching_ran') and any(kw in user_msg.lower() for kw in [
        "generate the report", "generate report", "download report", "yes", "sure", "ok"
    ]):
        intent = "MATCHING_ENGINE"
    elif session.get('chatbot_boxing_phase'):
        intent = "LOAN_BOXING"
    elif any(kw in user_msg.lower() for kw in [
        "am i a buyer", "am i a seller", "our ltd", "ltd position",
        "what role", "buyer or seller", "participation eligible"
    ]):
        intent = "MATCHING_ENGINE"
    else:
        try:
            intent = router.classify_intent(user_msg, persona=session.get('persona'), persona_label=session.get('persona_label'))
        except Exception as e:
            print(f"[Router error] {e}")
            intent = "GENERAL_ASSISTANCE"

    print(f"[Chatbot] intent={intent} | staged={pending['type'] if pending else 'none'}")

    # =========================================================================
    # ROUTE A — MATCHING ENGINE
    # =========================================================================
    if intent == "MATCHING_ENGINE":
        msg_lower = user_msg.lower()

        # Follow-up report generation
        if session.get('matching_ran') and any(kw in msg_lower for kw in [
            "generate the report", "generate report", "download report", "yes", "sure", "ok"
        ]):
            pdf_url = sbm.matching_state.get("pdf_url") or session.get('last_pdf_url')
            if pdf_url:
                session.pop('matching_ran', None)
                answer = "Full matching report is ready — download it below."
                history.append({"user": user_msg, "assistant": answer})
                session['chatbot_history'] = history[-10:]
                return jsonify({"answer": answer, "download_url": pdf_url})
            else:
                session.pop('matching_ran', None)
                answer = "Report link expired. Please upload the 5300 Call Report again to regenerate."
                history.append({"user": user_msg, "assistant": answer})
                session['chatbot_history'] = history[-10:]
                return jsonify({"answer": answer})

        # Has existing ranking — return from state
        has_ranking  = isinstance(sbm.matching_state.get("final_ranking"), pd.DataFrame) and not sbm.matching_state["final_ranking"].empty
        has_liquidity = bool(sbm.matching_state.get("liquidity"))

        explain_general_kw = ["how does the matching", "scoring logic", "methodology", "how are scores", "how do you rank", "how does it work"]
        explain_specific_kw = ["why is", "why was", "why ranked", "explain the score for", "explain the scoring", "explain these results"]
        report_kw = ["generate report", "create report", "download report", "full report", "pdf report", "generate the report"]
        role_kw   = ["am i a buyer", "am i a seller", "what role", "our ltd", "ltd position", "buyer or seller"]
        market_kw = ["market right now", "typical ltd", "market conditions", "industry average", "market overview"]
        whatif_kw = ["what if", "hypothetically", "scenario", "what would happen", "suppose", "assume"]
        deepdive_kw = ["tell me more about", "score breakdown", "deep dive", "breakdown for"]
        rerun_kw  = ["run again", "re-run", "rerun", "new file", "start over"]

        is_explain_general  = any(kw in msg_lower for kw in explain_general_kw)
        is_explain_specific = any(kw in msg_lower for kw in explain_specific_kw)
        is_report           = any(kw in msg_lower for kw in report_kw)
        is_role             = any(kw in msg_lower for kw in role_kw)
        is_market           = any(kw in msg_lower for kw in market_kw)
        is_whatif           = any(kw in msg_lower for kw in whatif_kw)
        is_deepdive         = any(kw in msg_lower for kw in deepdive_kw)
        is_rerun            = any(kw in msg_lower for kw in rerun_kw)

        persona_label = session.get('persona_label', 'user')

        if is_rerun:
            sbm.matching_state = {}
            session.pop('matching_ran', None)
            if not pending or pending['type'] != 'call_report':
                return jsonify({"answer": "Please upload the updated 5300 Call Report using the 📎 paperclip."})

        if is_market:
            liquidity = sbm.matching_state.get("liquidity", {})
            cu_context = ""
            if has_liquidity:
                cu_context = f"Also compare: {liquidity.get('label','This CU')} has LTD {liquidity.get('ltd',0):.1%}, role {liquidity.get('role','?')}."
            answer = _general_answer(
                f"Answer this market context question for a {persona_label}: '{user_msg}'. "
                f"Use NCUA Q4 2025 data: system LTD 83.2%, 4,287 CUs, total assets $2.43T. {cu_context}", history
            )
            history.append({"user": user_msg, "assistant": answer})
            session['chatbot_history'] = history[-10:]
            return jsonify({"answer": answer})

        if is_whatif:
            liquidity = sbm.matching_state.get("liquidity", {})
            context = (f"Current: {liquidity.get('label','?')} LTD {liquidity.get('ltd',0):.1%}, role {liquidity.get('role','?')}. LTD threshold: 75%."
                       if has_liquidity else "No CU data loaded. LTD threshold: 75%.")
            answer = _general_answer(
                f"Answer this what-if for a {persona_label}: '{user_msg}'. {context} Show before/after numbers.", history
            )
            history.append({"user": user_msg, "assistant": answer})
            session['chatbot_history'] = history[-10:]
            return jsonify({"answer": answer})

        if is_role:
            liquidity = sbm.matching_state.get("liquidity", {})
            if has_liquidity:
                inst_id = liquidity.get("label", "This CU"); ltd = liquidity.get("ltd", 0); role = liquidity.get("role", "Unknown")
            elif pending and pending['type'] == 'call_report':
                try:
                    etl = sbm.run_call_report_etl(pending['path'])
                    ltd = float(etl["summary"].get("ltd", 0))
                    role = "BUYER" if ltd < 0.75 else "SELLER"; inst_id = etl.get("institution_id", "This CU")
                except Exception as e:
                    return jsonify({"answer": f"ETL error: <code>{e}</code>"})
            else:
                return jsonify({"answer": "Please upload a 5300 Call Report to check your LTD position."})
            color = "#1D9E75" if role == "BUYER" else "#D85A30"
            answer = (f"<strong>Liquidity Position: {inst_id}</strong><br><br>• LTD: <strong>{ltd:.1%}</strong><br>"
                      f"• Role: <strong style='color:{color}'>{role}</strong><br><br>"
                      f"{'LTD below 75% — excess liquidity to deploy.' if role == 'BUYER' else 'LTD above 75% — well-positioned for loan participation sales.'}")
            history.append({"user": user_msg, "assistant": answer})
            session['chatbot_history'] = history[-10:]
            return jsonify({"answer": answer})

        if has_ranking and not is_rerun:
            final_ranking = sbm.matching_state["final_ranking"]
            liquidity     = sbm.matching_state.get("liquidity", {})
            inst_id = liquidity.get("label", "This CU"); ltd = liquidity.get("ltd", 0); role = liquidity.get("role", "BUYER")
            flash = _flash_matching(final_ranking, inst_id, ltd, role)
            if is_report:
                pdf_url = session.get('last_pdf_url')
                history.append({"user": user_msg, "assistant": flash})
                session['chatbot_history'] = history[-10:]
                return jsonify({"answer": flash, "download_url": pdf_url})
            answer = flash + "<br><br><em>Say <strong>\"Generate the report\"</strong> for the full PDF.</em>"
            session['matching_ran'] = True
            history.append({"user": user_msg, "assistant": answer})
            session['chatbot_history'] = history[-10:]
            return jsonify({"answer": answer})

        # Need file to run engine
        if not pending or pending['type'] != 'call_report':
            return jsonify({"answer": "Please upload a <strong>5300 Call Report</strong> (Excel) using the 📎 paperclip."})

        upload_path   = pending['path']
        original_name = pending['name']
        session.pop('chatbot_pending_file', None)

        try:
            with open(upload_path, 'rb') as f:
                file_bytes = f.read()
            from werkzeug.datastructures import FileStorage
            fake_file = FileStorage(
                stream=io.BytesIO(file_bytes), filename=original_name,
                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                             if original_name.lower().endswith(('.xlsx', '.xls')) else 'text/csv'
            )
            with app.test_request_context('/upload_call_report', method='POST', data={'call_report': fake_file}):
                response = upload_call_report()

            response_data = json.loads(response.get_data(as_text=True))
            if 'error' in response_data:
                return jsonify({"answer": f"Matching engine error: {response_data['error']}"})

            pdf_url = response_data.get('pdf_url')
            session['last_pdf_url'] = pdf_url

            final_ranking = sbm.matching_state.get("final_ranking", pd.DataFrame())
            liquidity     = sbm.matching_state.get("liquidity", {})
            inst_id = liquidity.get("label", original_name); ltd = liquidity.get("ltd", 0); role = liquidity.get("role", "BUYER")
            flash = _flash_matching(final_ranking, inst_id, ltd, role)

            if is_report:
                history.append({"user": user_msg, "assistant": flash})
                session['chatbot_history'] = history[-10:]
                return jsonify({"answer": flash, "download_url": pdf_url})

            answer = flash + "<br><br><em>Say <strong>\"Generate the report\"</strong> for the full PDF.</em>"
            session['matching_ran'] = True
            history.append({"user": user_msg, "assistant": answer})
            session['chatbot_history'] = history[-10:]
            return jsonify({"answer": answer})

        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify({"answer": f"Matching engine error: <code>{e}</code>"}), 500

    # =========================================================================
    # ROUTE B — LOAN BOXING
    # =========================================================================
    elif intent == "LOAN_BOXING":
        msg_lower    = user_msg.lower()
        explain_kw   = ["how does", "explain", "boxing logic", "what is box assignment"]
        report_kw    = ["generate report", "create report", "executive summary", "generate the report"]
        excel_kw     = ["download", "export", "give me the excel", "download excel"]
        rerun_kw     = ["run again", "re-run", "rerun", "new file", "start over", "fresh run"]

        is_explain = any(kw in msg_lower for kw in explain_kw)
        is_report  = any(kw in msg_lower for kw in report_kw)
        is_excel   = any(kw in msg_lower for kw in excel_kw)
        is_rerun   = any(kw in msg_lower for kw in rerun_kw)

        if is_rerun:
            for key in ['chatbot_boxing_phase', 'chatbot_boxing_filters', 'chatbot_boxing_original_name', 'summary', 'raw_tape', 'final_box']:
                session.pop(key, None)

        if pending and pending['type'] == 'loan_tape':
            if session.get('chatbot_boxing_phase') and not is_rerun:
                for key in ['chatbot_boxing_phase', 'summary', 'raw_tape', 'final_box', 'chatbot_boxing_original_name']:
                    session.pop(key, None)
            upload_path   = pending['path']
            original_name = pending['name']
        else:
            if not session.get('chatbot_boxing_phase') and not is_explain:
                return jsonify({"answer": "Please upload a <strong>Loan Tape</strong> (CSV or Excel) using the 📎 paperclip first."})
            upload_path   = None
            original_name = session.get('chatbot_boxing_original_name', 'Loan Tape')

        boxing_phase = session.get('chatbot_boxing_phase')

        if is_excel and session.get('last_boxing_csv_url') and boxing_phase == 'analyzed':
            csv_url = session.get('last_boxing_csv_url')
            answer  = "Full box assignment Excel ready to download."
            history.append({"user": user_msg, "assistant": answer})
            session['chatbot_history'] = history[-10:]
            return jsonify({"answer": answer, "download_url": csv_url})

        if is_explain and not is_rerun:
            answer = _general_answer(
                f"Explain the Capital Flow AI loan boxing methodology to a {session.get('persona_label','user')}. "
                "Cover: column standardisation, preprocessing pipeline, box definitions (FICO/LTV/DTI bands), "
                "in_box_flag logic, Phase 2 filtering, Phase 3 report generation.", history
            )
            history.append({"user": user_msg, "assistant": answer})
            session['chatbot_history'] = history[-10:]
            return jsonify({"answer": answer})

        # ── PHASE 1 ──────────────────────────────────────────────────────────
        if boxing_phase is None:
            if not upload_path:
                return jsonify({"answer": "Please upload a <strong>Loan Tape</strong> using the 📎 paperclip first."})

            session.pop('chatbot_pending_file', None)
            session['chatbot_boxing_original_name'] = original_name

            try:
                from werkzeug.datastructures import FileStorage
                with open(upload_path, 'rb') as f:
                    file_bytes = f.read()

                fake_file = FileStorage(
                    stream=io.BytesIO(file_bytes), filename=original_name,
                    content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                                 if original_name.lower().endswith(('.xlsx', '.xls')) else 'text/csv'
                )

                with app.test_request_context('/ltn-analyze', method='POST', data={'loan_tape': fake_file, 'action': 'analyze'}):
                    response  = ltn_analyze()
                    summary   = session.get('summary', {})
                    raw_tape  = session.get('raw_tape')
                    final_box = session.get('final_box')

                session['summary']              = summary
                session['raw_tape']             = raw_tape
                session['final_box']            = final_box
                session['chatbot_boxing_phase'] = 'analyzed'

                content_disp = response.headers.get('Content-Disposition', '')
                csv_filename = content_disp.split('filename=')[-1].strip('"') if 'filename=' in content_disp else 'LTN_Full_Box_Assignment.csv'
                csv_url      = f"/static/{csv_filename}"
                session['last_boxing_csv_url'] = csv_url

                wa_fico       = summary.get('wa_fico')
                fico_display  = f"{float(wa_fico):.1f}" if wa_fico else 'N/A'

                flash = (
                    f"Loan Tape <strong>{original_name}</strong> processed.<br><br>"
                    f"<strong>Phase 1 — Full Box Assignment Complete:</strong><br>"
                    f"• Total loans: <strong>{int(summary.get('total_loans',0)):,}</strong><br>"
                    f"• Matched loans: <strong>{int(summary.get('matched_loans',0)):,}</strong> "
                    f"({float(summary.get('match_pct',0)):.1f}% match rate)<br>"
                    f"• Avg FICO: <strong>{fico_display}</strong><br>"
                    f"• Matched balance: <strong>${float(summary.get('matched_balance_mm',0)):.1f}M</strong>"
                )

                if is_excel:
                    answer = flash + "<br><br>Full box assignment Excel ready to download."
                    history.append({"user": user_msg, "assistant": answer})
                    session['chatbot_history'] = history[-10:]
                    return jsonify({"answer": answer, "download_url": csv_url})

                answer = (flash + "<br><br><strong>What next?</strong><br>"
                          "• Say <strong>\"Download the Excel\"</strong> to get the full CSV<br>"
                          "• Provide filter preferences to refine the portfolio<br>"
                          "• Say <strong>\"No filters, generate the executive report\"</strong> to go straight to Phase 3")
                history.append({"user": user_msg, "assistant": answer})
                session['chatbot_history'] = history[-10:]
                return jsonify({"answer": answer})

            except Exception as e:
                import traceback; traceback.print_exc()
                session.pop('chatbot_boxing_phase', None)
                return jsonify({"answer": f"Loan boxing error: <code>{e}</code>"}), 500

        # ── PHASE 2 ──────────────────────────────────────────────────────────
        elif boxing_phase == 'analyzed':
            original_name = session.get('chatbot_boxing_original_name', 'Loan Tape')
            has_filter_values = any(kw in msg_lower for kw in ["fico","ltv","dti","term","rate","dpd","680","720","750","exclude","remove","matured","delinquent"])
            skip_to_report    = bool(re.search(r'no filter|skip|generate.*report|report now', user_msg, re.I))

            if not has_filter_values and not skip_to_report:
                answer = (
                    "Please provide filter values such as <em>\"FICO 680+, LTV max 115%, DTI max 45%\"</em> "
                    "or say <strong>\"No filters, generate the report\"</strong> to go straight to Phase 3.<br><br>"
                    "<strong>Available filters:</strong><br>"
                    "• FICO range, LTV max, DTI max, Rate range, Term max<br>"
                    "• Exclude >30 DQ, Remove matured loans, Product type"
                )
                history.append({"user": user_msg, "assistant": answer})
                session['chatbot_history'] = history[-10:]
                return jsonify({"answer": answer})

            if has_filter_values and not skip_to_report:
                filters = _parse_boxing_filters(user_msg)
                ltn_raw_tape_cache  = session.get('raw_tape')
                ltn_final_box_cache = session.get('final_box')

                form_data = {
                    'action': 'refine',
                    'fico_min':           str(filters.get('fico_min', 0)),
                    'fico_max':           str(filters.get('fico_max', 850)),
                    'ltv_min':            str(filters.get('ltv_min', 0)),
                    'ltv_max':            str(filters.get('ltv_max', 200)),
                    'dti_min':            str(filters.get('dti_min', 0)),
                    'dti_max':            str(filters.get('dti_max', 100)),
                    'term_min':           str(filters.get('term_min', 0)),
                    'term_max':           str(filters.get('term_max', 999)),
                    'balance_min':        str(filters.get('balance_min', 0)),
                    'balance_max':        str(filters.get('balance_max', 999999999)),
                    'interest_rate_min':  str(filters.get('interest_rate_min', 0)),
                    'interest_rate_max':  str(filters.get('interest_rate_max', 18)),
                    'days_past_due_min':  str(filters.get('days_past_due_min', 0)),
                    'days_past_due_max':  str(filters.get('days_past_due_max', 999)),
                    'exclude_over_30_dq': 'on' if filters.get('exclude_over_30_dq') else '',
                    'remove_matured':     'on' if filters.get('remove_matured') else '',
                }

                try:
                    with app.test_request_context('/ltn-analyze', method='POST', data=form_data):
                        session['raw_tape']  = ltn_raw_tape_cache
                        session['final_box'] = ltn_final_box_cache
                        response = make_response(ltn_analyze())
                        raw_tape_r  = session.get('raw_tape')
                        final_box_r = session.get('final_box')
                        summary_r   = session.get('summary', {})

                        content_disp = response.headers.get('Content-Disposition', '')
                        _fn = content_disp.split('filename=')[-1].strip('"') if 'filename=' in content_disp else None
                        if _fn:
                            _csv_path = os.path.join(app.static_folder, _fn)
                            if os.path.exists(_csv_path):
                                refined_df = pd.read_csv(_csv_path)
                                matched    = refined_df[refined_df['in_box_flag'] == True] if 'in_box_flag' in refined_df.columns else refined_df
                                p1_total   = session.get('summary', {}).get('total_loans', len(refined_df))
                                n_match    = len(matched)
                                summary_r  = {
                                    'total_loans':        int(p1_total),
                                    'matched_loans':      n_match,
                                    'match_pct':          round(n_match / int(p1_total) * 100, 3) if p1_total > 0 else 0,
                                    'wa_fico':            matched['orig_fico'].mean() if n_match > 0 else None,
                                    'matched_balance_mm': matched['current_balance'].sum() / 1e6 if n_match > 0 else 0,
                                    'refined_loans':      n_match,
                                }

                    session['raw_tape']              = raw_tape_r
                    session['final_box']             = final_box_r
                    session['summary']               = summary_r
                    session['chatbot_boxing_phase']  = 'refined'
                    session['chatbot_boxing_filters'] = form_data
                    # Re-read to pick up wa_fico + balance written by ltn_refine()
                    summary_r = session.get('summary', summary_r)

                    content_disp = response.headers.get('Content-Disposition', '')
                    csv_filename = content_disp.split('filename=')[-1].strip('"') if 'filename=' in content_disp else 'LTN_Refined.csv'
                    csv_url      = f"/static/{csv_filename}"
                    session['last_refined_csv_url'] = csv_url

                    applied = ", ".join(f"{k}={v}" for k, v in filters.items() if v)
                    if not applied:
                        # Parser couldn't extract filters — tell user what's supported
                        history.append({"user": user_msg, "assistant": (
                            "I couldn't parse specific filter values from that request.<br><br>"
                            "Please use formats like:<br>"
                            "&#8226; <em>FICO 720+, LTV max 85%, DTI max 40%</em><br>"
                            "&#8226; <em>Exclude &gt;30 DPD loans</em><br>"
                            "&#8226; <em>Auto New only</em><br>"
                            "&#8226; <em>Remove matured loans</em><br><br>"
                            "What filter would you like to apply?"
                        )})
                        session['chatbot_history'] = history[-10:]
                        session['chatbot_boxing_phase'] = 'refined'  # keep in refined state
                        return jsonify({"answer": history[-1]["assistant"]})
                    applied = applied  # already set above
                    wa_fico_r    = summary_r.get('wa_fico')
                    fico_disp_r  = f"{float(wa_fico_r):.1f}" if wa_fico_r else 'N/A'
                    matched_r    = int(summary_r.get('matched_loans', 0))
                    total_r      = int(summary_r.get('total_loans', 0))
                    match_pct_r  = float(summary_r.get('match_pct', 0))
                    balance_r    = float(summary_r.get('matched_balance_mm', 0))

                    flash = (
                        f"<strong>Phase 2 — Refined Portfolio Complete:</strong><br><br>"
                        f"<strong>Filters applied:</strong> <em>{applied}</em><br><br>"
                        f"• Loans retained: <strong>{matched_r:,}</strong> ({match_pct_r:.1f}% retention)<br>"
                        f"• Avg FICO: <strong>{fico_disp_r}</strong><br>"
                        f"• Retained balance: <strong>${balance_r:.2f}M</strong><br><br>"
                        f"Refined Excel ready. Say <em>\"Generate the report\"</em> for Phase 3."
                    )
                    history.append({"user": user_msg, "assistant": flash})
                    session['chatbot_history'] = history[-10:]
                    return jsonify({"answer": flash, "download_url": csv_url})

                except Exception as e:
                    import traceback; traceback.print_exc()
                    return jsonify({"answer": f"Refinement error: <code>{e}</code>"}), 500

            if skip_to_report:
                session['chatbot_boxing_phase']  = 'refined'
                session['chatbot_boxing_filters'] = {
                    'action': 'refine', 'fico_min': '0', 'fico_max': '850',
                    'ltv_min': '0', 'ltv_max': '200', 'dti_min': '0', 'dti_max': '100',
                    'term_min': '0', 'term_max': '999', 'balance_min': '0', 'balance_max': '999999999',
                    'interest_rate_min': '0', 'interest_rate_max': '18',
                    'days_past_due_min': '0', 'days_past_due_max': '999',
                    'exclude_over_30_dq': '', 'remove_matured': '',
                }
                boxing_phase = 'refined'

        # ── PHASE 3 ──────────────────────────────────────────────────────────
        if boxing_phase == 'refined' and re.search(r'report|generate|download.*report|final|html', user_msg, re.I):
            original_name = session.get('chatbot_boxing_original_name', 'Loan Tape')
            saved_filters = session.get('chatbot_boxing_filters', {})
            saved_filters['action'] = 'report'
            p3_raw_tape  = session.get('raw_tape')
            p3_final_box = session.get('final_box')
            p3_summary   = session.get('summary', {})

            try:
                with app.test_request_context('/ltn-analyze', method='POST', data=saved_filters):
                    session['raw_tape']  = p3_raw_tape
                    session['final_box'] = p3_final_box
                    session['summary']   = p3_summary
                    response = make_response(ltn_analyze())
                    content_disp  = response.headers.get('Content-Disposition', '')
                    html_filename = content_disp.split('filename=')[-1].strip('"') if 'filename=' in content_disp else 'LTN_Report.html'

                report_url  = f"/static/{html_filename}"
                report_path = os.path.join(app.static_folder, html_filename)
                if not os.path.exists(report_path):
                    return jsonify({"answer": f"Report generated but file not found at expected path."})

                for key in ['chatbot_boxing_phase','chatbot_boxing_filters','chatbot_boxing_original_name','last_boxing_csv_url','last_refined_csv_url']:
                    session.pop(key, None)

                flash = (f"<strong>Phase 3 — Executive Portfolio Report Complete:</strong><br><br>"
                         f"Full refined portfolio report for <strong>{original_name}</strong> is ready to download.")
                history.append({"user": user_msg, "assistant": flash})
                session['chatbot_history'] = history[-10:]
                return jsonify({"answer": flash, "download_url": report_url})

            except Exception as e:
                import traceback; traceback.print_exc()
                return jsonify({"answer": f"Report generation error: <code>{e}</code>"}), 500

        elif boxing_phase == 'refined':
            # ── Check if user wants to apply MORE filters on top of current refined set ──
            more_filters = any(kw in msg_lower for kw in [
                "fico", "ltv", "dti", "term", "rate", "dpd", "exclude", "remove",
                "matured", "delinquent", "product", "balance", "new only", "used only",
                "tighten", "apply", "filter", "refine further", "narrow",
                # product/loan type keywords
                "auto new", "auto used", "new loan", "used loan", "keep new",
                "keep auto", "new only", "used only", "direct only", "indirect only",
                "keep only", "only new", "only used", "loan type", "vehicle type",
                # additional natural language patterns
                "also keep", "only keep", "just keep", "keep loans", "keep the",
                "zero balance", "zero rate", "delinquent", "30 dpd", "30+ dpd",
            ])

            if more_filters:
                # Parse new filters from this message
                new_filters = _parse_boxing_filters(user_msg)

                # ── Fix 2: If parser returns nothing, tell user what's supported
                #    BEFORE running any pipeline — avoids wasted Phase 2 call ──
                if not new_filters:
                    history.append({"user": user_msg, "assistant": (
                        "I couldn't parse specific filter values from that request.<br><br>"
                        "Please use formats like:<br>"
                        "&#8226; <em>FICO 720+, LTV max 85%, DTI max 40%</em><br>"
                        "&#8226; <em>Exclude &gt;30 DPD loans</em><br>"
                        "&#8226; <em>Auto New only / Auto Used only</em><br>"
                        "&#8226; <em>Remove matured loans</em><br>"
                        "&#8226; <em>LTV max: 85%, DTI max: 45%</em><br><br>"
                        "What filter would you like to apply?"
                    )})
                    session['chatbot_history'] = history[-10:]
                    session['chatbot_boxing_phase'] = 'refined'
                    return jsonify({"answer": history[-1]["assistant"]})

                # ── Fix 1 (your observation): Accumulate filters across turns ──
                # Each new filter request MERGES with previous filters, all applied
                # together against Phase 1 output (final_box). This means:
                #   Turn 1: exclude >30 DQ          → 43 loans
                #   Turn 2: remove matured           → applies BOTH → fewer loans
                #   Turn 3: FICO 720+                → applies ALL THREE → fewer still
                prev_filters = session.get('chatbot_boxing_filters', {})
                merged = {
                    'fico_min':           prev_filters.get('fico_min', '0'),
                    'fico_max':           prev_filters.get('fico_max', '850'),
                    'ltv_min':            prev_filters.get('ltv_min', '0'),
                    'ltv_max':            prev_filters.get('ltv_max', '200'),
                    'dti_min':            prev_filters.get('dti_min', '0'),
                    'dti_max':            prev_filters.get('dti_max', '100'),
                    'term_min':           prev_filters.get('term_min', '0'),
                    'term_max':           prev_filters.get('term_max', '999'),
                    'balance_min':        prev_filters.get('balance_min', '0'),
                    'balance_max':        prev_filters.get('balance_max', '999999999'),
                    'interest_rate_min':  prev_filters.get('interest_rate_min', '0'),
                    'interest_rate_max':  prev_filters.get('interest_rate_max', '18'),
                    'days_past_due_min':  prev_filters.get('days_past_due_min', '0'),
                    'days_past_due_max':  prev_filters.get('days_past_due_max', '999'),
                    # boolean flags carry forward — once excluded, always excluded
                    'exclude_over_30_dq': prev_filters.get('exclude_over_30_dq', ''),
                    'remove_matured':     prev_filters.get('remove_matured', ''),
                }
                # Overlay new parsed values on top of previous
                if new_filters.get('fico_min'):     merged['fico_min']           = str(new_filters['fico_min'])
                if new_filters.get('fico_max'):     merged['fico_max']           = str(new_filters['fico_max'])
                if new_filters.get('ltv_max'):      merged['ltv_max']            = str(new_filters['ltv_max'])
                if new_filters.get('dti_max'):      merged['dti_max']            = str(new_filters['dti_max'])
                if new_filters.get('term_max'):     merged['term_max']           = str(new_filters['term_max'])
                if new_filters.get('balance_max'):  merged['balance_max']        = str(new_filters['balance_max'])
                if new_filters.get('interest_rate_min'): merged['interest_rate_min'] = str(new_filters['interest_rate_min'])
                if new_filters.get('interest_rate_max'): merged['interest_rate_max'] = str(new_filters['interest_rate_max'])
                if new_filters.get('exclude_over_30_dq'): merged['exclude_over_30_dq'] = 'on'
                if new_filters.get('remove_matured'):      merged['remove_matured']     = 'on'

                form_data = {'action': 'refine', **merged}
                ltn_raw_tape_cache  = session.get('raw_tape')
                ltn_final_box_cache = session.get('final_box')

                try:
                    with app.test_request_context('/ltn-analyze', method='POST', data=form_data):
                        session['raw_tape']  = ltn_raw_tape_cache
                        session['final_box'] = ltn_final_box_cache
                        response = make_response(ltn_analyze())
                        raw_tape_r  = session.get('raw_tape')
                        final_box_r = session.get('final_box')
                        summary_r   = session.get('summary', {})

                    session['raw_tape']               = raw_tape_r
                    session['final_box']              = final_box_r
                    session['summary']                = summary_r
                    session['chatbot_boxing_phase']   = 'refined'
                    session['chatbot_boxing_filters'] = form_data  # save accumulated set
                    # Re-read summary_r after session write to get ltn_refine's full output
                    summary_r = session.get('summary', summary_r)

                    content_disp = response.headers.get('Content-Disposition', '')
                    csv_filename = content_disp.split('filename=')[-1].strip('"') if 'filename=' in content_disp else 'LTN_Refined.csv'
                    csv_url      = f"/static/{csv_filename}"
                    session['last_refined_csv_url'] = csv_url

                    # Show what's now active cumulatively
                    active = []
                    if merged['exclude_over_30_dq'] == 'on': active.append('Exclude &gt;30 DPD')
                    if merged['remove_matured']     == 'on': active.append('Remove matured')
                    if merged['fico_min'] != '0':   active.append(f"FICO ≥{merged['fico_min']}")
                    if merged['fico_max'] != '850':  active.append(f"FICO ≤{merged['fico_max']}")
                    if merged['ltv_max']  != '200':  active.append(f"LTV ≤{merged['ltv_max']}%")
                    if merged['dti_max']  != '100':  active.append(f"DTI ≤{merged['dti_max']}%")
                    if merged['term_max'] != '999':  active.append(f"Term ≤{merged['term_max']}mo")
                    applied = " · ".join(active) if active else "defaults"

                    matched_r  = int(summary_r.get('refined_loans', summary_r.get('matched_loans', 0)))
                    # wa_fico and balance come from the refined box metrics
                    # ltn_refine writes wa_fico into summary from box_for_metrics
                    wa_fico_r  = summary_r.get('wa_fico')
                    fico_disp  = f"{float(wa_fico_r):.1f}" if (wa_fico_r and str(wa_fico_r) not in ['nan','None','0']) else 'N/A'
                    # balance: try refined-specific keys first, fall back to matched_balance_mm
                    balance_r  = float(summary_r.get('refined_balance_mm',
                                  summary_r.get('matched_balance_mm', 0)) or 0)

                    flash = (
                        f"<strong>Phase 2 — Re-filtered Portfolio:</strong><br><br>"
                        f"<strong>New filters applied:</strong> <em>{applied}</em><br><br>"
                        f"• Loans retained: <strong>{matched_r:,}</strong><br>"
                        f"• Avg FICO: <strong>{fico_disp}</strong><br>"
                        f"• Retained balance: <strong>${balance_r:.2f}M</strong><br><br>"
                        f'Refined Excel updated. Apply more filters or say <em>"Generate the report"</em> for Phase 3.'
                    )
                    history.append({"user": user_msg, "assistant": flash})
                    session['chatbot_history'] = history[-10:]
                    return jsonify({"answer": flash, "download_url": csv_url})

                except Exception as e:
                    import traceback; traceback.print_exc()
                    return jsonify({"answer": f"Re-filter error: <code>{e}</code>"}), 500

            else:
                answer = 'Refined portfolio ready. Apply more filters or say <em>"Generate the report"</em> to produce the full HTML executive report.'
                history.append({"user": user_msg, "assistant": answer})
                session['chatbot_history'] = history[-10:]
                return jsonify({"answer": answer})

    # =========================================================================
    # ROUTE C — KNOWLEDGE BASE / RAG
    # =========================================================================
    elif intent == "KNOWLEDGE_BASE":
        if pending and pending['type'] == 'rag_doc':
            doc_text = _extract_text_for_rag(pending['path'], pending['name'])
            answer   = _rag_answer(user_msg, doc_text, pending['name'])
        elif pending:
            label  = "5300 Call Report" if pending['type'] == 'call_report' else "Loan Tape"
            answer = _general_answer(user_msg, history)
            answer += f"<br><br><em>You still have <strong>{pending['name']}</strong> staged as a {label}.</em>"
        else:
            answer = _general_answer(user_msg, history)
        history.append({"user": user_msg, "assistant": answer})
        session['chatbot_history'] = history[-10:]
        return jsonify({"answer": answer})

    # =========================================================================
    # ROUTE D — GENERAL ASSISTANCE
    # =========================================================================
    elif intent == "GENERAL_ASSISTANCE":
        answer = _general_answer(user_msg, history)
        if pending:
            label = "5300 Call Report" if pending['type'] == 'call_report' else "Loan Tape" if pending['type'] == 'loan_tape' else "document"
            answer += f"<br><br><em>Reminder: <strong>{pending['name']}</strong> is still staged as a {label}.</em>"
        history.append({"user": user_msg, "assistant": answer})
        session['chatbot_history'] = history[-10:]
        return jsonify({"answer": answer})

    # =========================================================================
    # ROUTE E — DOCUMENT QA
    # =========================================================================
    elif intent == "DOCUMENT_QA":
        if pending:
            doc_text = _extract_text_for_rag(pending['path'], pending['name'], user_msg)
            answer   = _rag_answer(user_msg, doc_text, pending['name']) if doc_text.strip() else _general_answer(user_msg, history)
        else:
            answer = _general_answer(user_msg, history)
        history.append({"user": user_msg, "assistant": answer})
        session['chatbot_history'] = history[-10:]
        return jsonify({"answer": answer})

    # =========================================================================
    # ROUTE F — MARKET DATA
    # =========================================================================
    elif intent == "MARKET_DATA":
        liquidity = sbm.matching_state.get("liquidity", {})
        health    = sbm.matching_state.get("health", {})

        if liquidity:
            assets    = health.get("ratios", {}).get("total_assets", 0) or 0
            size_band = "SMALL" if assets < 50e6 else "LOWER MID" if assets < 200e6 else "UPPER MID" if assets < 1e9 else "LARGE"
            cu_metrics = {
                "cu_label": liquidity.get("label", "This CU"), "ltd": liquidity.get("ltd"),
                "role": liquidity.get("role"), "net_worth_ratio": health.get("ratios", {}).get("NetWorthToAssets"),
                "total_assets": assets, "total_loans": None, "total_deposits": None,
            }
            state = liquidity.get("state")
        elif pending and pending['type'] == 'call_report':
            try:
                etl = sbm.run_call_report_etl(pending['path'])
                summary_row = etl["summary"]
                assets = float(summary_row.get("total_assets", 0))
                ltd_v  = float(summary_row.get("ltd", 0))
                nwr    = float(summary_row.get("total_net_worth", 0)) / assets if assets else 0
                state  = summary_row.get("state")
                size_band = summary_row.get("size_band")
                cu_metrics = {
                    "cu_label": etl["institution_id"], "ltd": ltd_v,
                    "role": "BUYER" if ltd_v < 0.75 else "SELLER",
                    "net_worth_ratio": nwr, "total_assets": assets,
                    "total_loans": float(summary_row.get("total_loans", 0)),
                    "total_deposits": float(summary_row.get("total_deposits", 0)),
                }
            except Exception as e:
                return jsonify({"answer": f"Could not extract CU metrics: <code>{e}</code>"})
        else:
            return jsonify({"answer": "Upload a 5300 Call Report or run the matching engine to enable peer benchmarking."})

        state_match = re.search(r'\b(TX|CA|NY|FL|IL|OH|PA|GA|NC|MI|VA|WA|AZ|CO|TN|MO|IN|MD|WI|MN|OR|SC|AL|KY|LA|OK|CT|UT|NV|AR|MS|KS|NM|NE|WV|ID|HI|NH|ME|RI|MT|DE|SD|ND|AK|VT|WY)\b', user_msg, re.I)
        if state_match: state = state_match.group(1).upper()

        try:
            peers = _fetch_ncua_peers(size_band=size_band, state=state)
        except Exception as e:
            return jsonify({"answer": f"Failed to fetch NCUA data: <code>{e}</code>"})

        flash = _flash_market_data(cu_metrics=cu_metrics, peers=peers, size_band=size_band or "All", state=state or "All States")
        history.append({"user": user_msg, "assistant": flash})
        session['chatbot_history'] = history[-10:]
        return jsonify({"answer": flash})

    # Fallback
    answer = _general_answer(user_msg, history)
    history.append({"user": user_msg, "assistant": answer})
    session['chatbot_history'] = history[-10:]
    return jsonify({"answer": answer})


######---- END OF CHATBOT BLOCK ----######

# =============================================================================
# START THE SERVER
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Portfolio Co-Pilot starting on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)