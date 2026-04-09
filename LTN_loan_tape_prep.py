import os
import sys
import traceback
from warnings import filters
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
import re
import io
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
_EMBED_MODEL = SentenceTransformer('all-mpnet-base-v2')

def show_full_traceback(exc_type, exc_value, exc_traceback):
    print("\n" + "="*80)
    print("DETAILED ERROR TRACEBACK:")
    print("="*80)
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("="*80)
    
    # Extract and show the specific line
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    for line in tb_lines:
        if "line" in line.lower():
            print(f">>> {line.strip()}")
    print("="*80 + "\n")

sys.excepthook = show_full_traceback


# Class for LTN loan tape data cleaning, normalization and preparation.
# The output of the class is used in seller buyer matching.

# Helper function to define a standard name mapping for loan tape columns.

STANDARD_NAME_MAP = {
            # canonical → list of aliases (including the primary label)
            "as_of_date": [
                "As of Date", "Report Date", "Data Date", "Effective Date",
                "Snapshot Date", "Calculation Date", "Cut-off Date", "Valuation Date",
                "Processing Date", "Statement Date", "Reference Date", "Date Stamp",
                "Date of Report", "Date Ref", "Extract Date", "Status Date",
                "Current Date", "Recalc Date", "Ledger Date", "System Date", "Timestamp"
            ],
            "loan_id": [
                "Loan ID", "Loan Identifier", "Account ID", "Facility ID", "Contract ID",
                "Financing ID", "Reference ID", "Asset ID", "Credit ID", "Borrower ID",
                "Unique ID", "Loan Ref No", "Loan Number", "Client Loan ID", "Deal ID",
                "Portfolio ID", "Acct Num", "Loan Key", "ID Code", "Security ID",
                "Customer Loan ID"
            ],
            "auto_direct_indirect": [
                "Auto Direct/Indirect", "Funding Channel", "Origination Source", "Dealer Channel", 
                "Acquisition Type", "Source Type", "Auto Channel", "Originator Flag", "Direct/Indirect Indicator",
                "Channel of Acquisition", "Loan Source", "Broker/Direct", "Sales Channel", "Referral Source", 
                "Funding Path", "Source of Business", "Direct/Indirect Flag", "Channel Type", "Auto Funding Type",
                "Application Source", "Dealer/Branch"]
            ,
            "auto_new_used": [
                "Auto New/Used", "Vehicle Condition", "Collateral Status", "New/Used Status", "Auto Age", 
                "Condition Type", "Vehicle Status", "Condition of Auto", "New/Used Flag", "Car Status", 
                "Asset Condition", "Auto Type (N/U)", "Vehicle Type", "New/Used Indicator", "Classification", 
                "Used/New Code", "Vehicle Class", "Product Condition", "Auto Vintage", "Pre-Owned Flag", 
                "Vehicle Designation"
            ],

            "original_balance": [ "Original Balance", "Initial Loan Amount", "Principal Balance (Orig.)", 
                                 "Starting Balance", "Origination Amount", "Initial Principal", "Face Value", 
                                 "Contract Amount", "Funded Amount", "Gross Original Balance", "Initial Loan Value", 
                                 "Original Debt", "Commencement Balance", "Max Loan Amount", "Loan Principal", 
                                 "Initial Investment", "Total Original Loan", "Starting Principal", "Beginning Balance", 
                                 "Origination Value", "Initial Amount"],

            "current_balance":  [ "Current Balance", "Outstanding Principal", "Remaining Balance", "Present Balance", 
                                 "Ending Balance", "Principal Balance", "Current Debt", "Net Balance", "Actual Balance", 
                                 "Loan Outstanding", "Principal Remaining", "Current Loan Value", "Present Value (Loan)", 
                                 "Balance Today", "Open Balance", "Running Balance", "Facility Balance", "Loan Amount Due", 
                                 "Current Principal", "Remaining Debt"],

            "current_rate":     [ "Current Rate", "Interest Rate", "Applicable Rate", "Coupon Rate", "Nominal Rate", "Current APR", 
                                 "Effective Rate", "Loan Interest", "Annual Percentage Rate", "Present Rate", 
                                 "Current Pricing", "Rate of Interest", "Interest Percentage", "Current Yield", 
                                 "Rate Applied", "Loan Rate", "Current Interest Pct", "Prevailing Rate", "Variable Rate", 
                                 "Fixed Rate", "Rate Index"],

            "origination_date": [ "Origination Date", "Inception Date", "Funding Date", "Closing Date", "Book Date", "Start Date", 
                                 "Loan Start Date", "Effective Date (Orig.)", "Contract Date", "Approval Date", 
                                 "Signing Date", "Date of Loan", "Initial Date", "Booking Date", "Issue Date", 
                                 "Commencement Date", "Trade Date", "Date Granted", "Loan Creation Date", "Facility Date",
                                 "Date Advanced"],

            "maturity_date":    [ "Maturity Date", "Final Payment Date", "End Date", "Termination Date", "Expiry Date", 
                                 "Scheduled Maturity", "Loan End Date", "Due Date (Final)", "Date of Maturity", 
                                 "Contract End Date", "Scheduled Termination", "Final Due Date", "Payoff Date",
                                 "Principal Due Date", "Max Term Date", "Date to End", "Loan Conclusion Date", 
                                 "Expiration Date", "Final Day", "Last Payment Date", "Date Fully Paid"],

            "original_term_months": [ "Original Term (months)", "Initial Term", "Contract Term", "Loan Duration (Months)",
                                      "Origination Term", "Scheduled Term", "Initial Duration", "Maximum Term", 
                                      "Term Length (Mos.)", "Original Period", "Term in Months", "Term at Origination", 
                                      "Loan Tenure (Months)", "Full Term", "Total Term (Mos)", "Term (Original)", 
                                      "Months to Maturity (Orig.)", "Life of Loan", "Original Length", "Term (Initial)", 
                                      "Term of Contract"],

            "orig_fico":        [ "OrigFICO", "FICO Score (Orig.)", "Initial FICO", "Credit Score at Origination", 
                                 "Borrower FICO (Start)", "FICO Start", "Origination Credit Score", "Customer FICO (Initial)", 
                                 "FICO Score (Start)", "Original FICO Value", "Credit Rating (Orig.)", "FICO at Inception", "Borrower Score", 
                                 "Origination Score", "Initial Credit Rating", "FICO Score", "Credit Score", "Orig FICO Bucket", 
                                 "FICO Start Score", "Applicant FICO", "Origination Risk Score"],

            "orig_ltv":         [ "Original LTV", "Initial LTV", "LTV at Origination", "Loan-to-Value (Initial)", "Starting LTV Ratio", 
                                 "LTV (Orig.)", "Original Loan Value Ratio", "Collateral LTV", "Initial Ratio", "LTV Score", 
                                 "Origination LTV Pct", "Start LTV", "LTV (Initial)", "Loan Value Pct", "Original Collateral Ratio", 
                                 "Initial Loan Ratio", "Asset LTV", "LTV Percentage (Orig.)", "Origination LTV Ratio", "Vehicle LTV (Initial)", 
                                 "Maximum LTV"],

            "dti_back_end":     [ "DTI (Back-End)", "Back-End DTI", "Debt-to-Income Ratio (Back)", "Total DTI", "Max DTI", "Back DTI", 
                                 "Borrower DTI (Total)", "Debt-Income Ratio (Back)", "Total Debt Ratio", "DTI (Gross)", 
                                 "Debt Service Ratio", "Maximum Qualifying DTI", "Income to Debt Ratio", "DTI Backend", 
                                 "Back-End Ratio", "Total Obligation Ratio", "Debt/Income Pct", "Borrower DTI", 
                                 "Expense-to-Income Ratio", "Qualifying DTI", "Total DTI (Back)"],

            "state":            [ "City", "State", "Zip", "Location (City, State, Zip)", "Property Location", "Borrower Address Info", "Geographic Location", 
                                 "City/State/ZIP Code", "Residence Info", "Address Segment", "Geo-Code", "Loan Location", 
                                 "Collateral Location", "Borrower Geo-Data", "Address (C/S/Z)", "Postal Code Info", 
                                 "Customer Location", "Asset Location", "Region Data", "Zip Code", "State/City", 
                                 "Address Details", "Locale Data" ],  

            "year":             [ "Year", "Auto Model Year", "Model Year", "Car Year", "Year of Manufacture", 
                                 "Asset Year", "Vehicle Vintage", "Production Year", "Collateral Year", 
                                 "Year Built", "Vehicle Age", "Mfg Year", "Model Date", "Year of Car", "Auto Year", 
                                 "Year Produced", "Vehicle Mfg Year", "Year Tag", "Production Date", "Car Mfg Year", 
                                 "Vehicle Production Year"],

            "days_past_due":    ["Days Past Due", "Delinquency Days", "Aging Days", "Number of Days Delinquent",
                                 "Days Late","Arrearage Days","DPD Count","Past Due Count","Days in Default",
                                 "Loan Aging","Delinquent Days Count","Days Overdue","Loan Status (Days)",
                                 "Days Since Due","Delinquency Status","Late Days","Days Delinquent",
                                 "Days Past Maturity","Past Due Indicator","Delay Days","Payment Delay"],
            # plus any additional fields from the doc you want to support
        }


def _normalize_header(name: str) -> str:
    """
    Normalize a raw column header:
    - lowercase
    - strip leading/trailing spaces
    - collapse internal whitespace
    - remove trailing punctuation like ';', ':', ',' etc.
    - remove duplicate separators
    """

    if pd.isna(name):
        return ""
    
    name = str(name)
    
    # 1. Remove ALL non-printable + weird chars (##, ****, quotes)
    name = re.sub(r'[^\w\s\-\(\)\.\&]', ' ', name)  # Keep letters, spaces, -, (), ., &
    
    # 2. Collapse multiple spaces/tabs/newlines
    name = re.sub(r'\s+', ' ', name)
    
    # 3. Strip whitespace + punctuation
    name = name.strip(" ;:,.\"'##****")

    # if name is None:
    #     return ""

    # # strip outer whitespace + lowercase
    # s = name.strip().lower()

    # # remove trailing punctuation characters like ';', ':', ',', '.'
    # s = s.rstrip(" ;:,.")  # this specifically handles "DTI (Back-End);"

    # # replace common separators with a space
    # s = re.sub(r"[/_\-]+", " ", s)

    # # collapse multiple spaces
    # s = re.sub(r"\s+", " ", s)

    return name.lower()


def _build_inverse_header_map():
    """
    Build mapping from normalized alias -> canonical standard name.
    If multiple aliases normalize to same string for different canonicals,
    last one wins; in practice your catalog should avoid that.
    """
    inverse = {}
    for canonical, aliases in STANDARD_NAME_MAP.items():
        for alias in aliases:
            norm = _normalize_header(alias)
            inverse[norm] = canonical
    return inverse


# Build once at module import (or inside __init__)
NORMALIZED_TO_STANDARD = _build_inverse_header_map()


def map_headers_to_standard(raw_columns):
    """
    Given an iterable of raw column names, return:
      - rename_map: dict raw_name -> canonical standard name (only for mapped ones)
      - unmapped: list of raw names that did not match anything in STANDARD_NAME_MAP

    Example:
      raw "DTI (Back-End);" -> normalized "dti (back end)" ->
      maps to canonical "dti_back_end".
    """
    rename_map = {}
    unmapped = []

    for raw in raw_columns:
        norm = _normalize_header(str(raw))
        canonical = NORMALIZED_TO_STANDARD.get(norm)
        if canonical:
            rename_map[raw] = canonical
        else:
            unmapped.append(raw)

    return rename_map, unmapped

# rename_map, unmapped = map_headers_to_standard(df.columns)
# unmapped_columns = list(unmapped)

CATALOG_PATH = Path("field_catalog.json")
#FIELD_CATALOG = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

# ADD THIS DIAGNOSTIC BLOCK
print("=== CATALOG DIAGNOSTIC ===")
print(f"Catalog file exists: {CATALOG_PATH.exists()}")
print(f"Catalog file path: {CATALOG_PATH.absolute()}")

CATALOG_PATH = Path("field_catalog.json")

try:
    catalog_list = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    print(f"✓ Loaded {len(catalog_list)} catalog entries")
    
    # CONVERT LIST → DICT using standard_name as key
    FIELD_CATALOG = {entry["standard_name"]: entry for entry in catalog_list}
    print(f"✓ Converted to dict: {len(FIELD_CATALOG)} keys")
    print(f"Sample keys: {list(FIELD_CATALOG.keys())[:3]}")
    
    # Verify critical field exists
    if "original_term_months" in FIELD_CATALOG:
        print("✓ 'original_term_months' READY")
    else:
        print("✗ 'original_term_months' MISSING")
        
except Exception as e:
    print(f"✗ CATALOG FAILED: {e}")
    FIELD_CATALOG = {}

print("Catalog diagnostic complete")

def build_catalog_text(entry):
    """Build simple 'col | val1, val2, val3' format from new JSON"""
    name = entry["standard_name"]
    values = entry["example_values"][:3]  # Take first 3 values exactly
    
    return f"{name} | {', '.join(values)}"

CATALOG_STANDARD_NAMES = list(FIELD_CATALOG.keys())
CATALOG_TEXTS = [build_catalog_text(FIELD_CATALOG[key]) for key in FIELD_CATALOG]

def build_raw_col_text(col_name: str, sample_values=None) -> str:
    """
    Build EXACT SAME FORMAT as catalog: "col_name | val1, val2, val3"
    """
    if sample_values:
        examples = ", ".join(map(str, sample_values[:3]))
        return f"{col_name} | {examples}"
    return col_name  # Fallback

_EMBED_MODEL = SentenceTransformer('all-mpnet-base-v2')

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Return a 2D numpy array of shape (len(texts), dim).
    YOUR EXACT FUNCTION - handles empty list gracefully
    """
    if not texts:
        dim = _EMBED_MODEL.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype="float32")
    
    emb = _EMBED_MODEL.encode(texts, normalize_embeddings=True)
    #emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return np.asarray(emb, dtype="float32")

CATALOG_EMB = embed_texts(CATALOG_TEXTS)

def get_missing_and_candidate_cols(df, unmapped_columns=None):
    present_standards = set(df.columns) & set(CATALOG_STANDARD_NAMES)
    
    candidate_raw = unmapped_columns
    # Only non-standard columns: ["org amount", "current_bal"]
    
    missing_targets = CATALOG_STANDARD_NAMES
        
    return missing_targets, candidate_raw

# missing_targets, candidate_raw = get_missing_and_candidate_cols(self.df)

# ========== CELL 13: build_raw_embeddings() ==========

def build_raw_embeddings(df, candidate_cols):
    embeddings = []
    texts = []
    
    for col in candidate_cols:
        if col in df.columns:
            samples = df[col].dropna().head(3)
            
            # SMART date/value formatting
            if pd.api.types.is_datetime64_any_dtype(samples):
                sample_text = ", ".join(samples.dt.strftime('%Y-%m-%d').astype(str))
                full_text = f"{col} | date | {sample_text}"
            else:
                sample_text = ", ".join(samples.astype(str))
                full_text = f"{col} | {sample_text}"
                
            texts.append(full_text)
            emb = embed_texts(full_text)
            #emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            embeddings.append(emb)
    
    return np.array(embeddings), texts


def apply_business_rules(raw_col, raw_dtype, sample_values, candidates):
    """Fixed: Prioritize RAW dtype over target inference"""
    validated = []
    
    # Raw sample analysis
    try:
        nums = pd.to_numeric(pd.Series(sample_values), errors='coerce').dropna()
        min_val, max_val = nums.min(), nums.max()
        mean_val = nums.mean()
        has_decimals = (nums % 1 != 0).any()
        value_count = len(nums)
    except:
        min_val, max_val, mean_val, has_decimals, value_count = 0, 0, 0, False, 0
    
    for target, score in candidates:
        validation_score = score
        
        # *** RULE 0: BLOCK IMPOSSIBLE DTYPE MATCHES BASED ON RAW_DTYPE ***
        if raw_dtype == 'date' and not any(x in target.lower() for x in ['date', 'as_of', 'maturity']):
            validation_score -= 1.2  # Numeric target for date column
        elif raw_dtype == 'numeric':
            # BLOCK dates/IDs for numeric columns
            if any(x in target.lower() for x in ['date', 'as_of', 'maturity']):
                validation_score -= 1.5  # CRITICAL: numeric vs date
            elif any(x in target.lower() for x in ['loan_id', 'box_id']) and not (mean_val > 10000 and value_count > 1):
                validation_score -= 0.8  # Too small for ID
        elif raw_dtype == 'text' and any(x in target.lower() for x in ['balance', 'amount', 'ltv']):
            validation_score -= 0.9  # Text vs numeric field
        
        # *** RULE 1: CURRENT BALANCE/AMOUNT (high values, decimals OK) ***
        if any(x in target.lower() for x in ['balance', 'amount']) and raw_dtype == 'numeric':
            if 1000 <= min_val <= 500000 and has_decimals:
                validation_score += 0.5  # STRONG BOOST
            else:
                validation_score -= 0.3
        
        # *** RULE 2: DAYS PAST DUE (0-180 small integers) ***
        elif 'past_due' in target.lower() and raw_dtype == 'numeric':
            if 0 <= max_val <= 180 and not has_decimals and mean_val < 100:
                validation_score += 0.5  # STRONG BOOST
            else:
                validation_score -= 0.3
        
        # *** RULE 3: LOAN ID (large whole numbers only) ***
        elif any(x in target.lower() for x in ['loan_id', 'box_id']) and raw_dtype == 'numeric':
            if mean_val > 10000 and not has_decimals:
                validation_score += 0.4
            else:
                validation_score -= 0.7  # PENALIZE small decimals
        
        # RULE 4: LTV (0-150, decimals)
        elif 'ltv' in target.lower() and raw_dtype == 'numeric':
            if 0 <= mean_val <= 150 and has_decimals:
                validation_score += 0.3
            else:
                validation_score -= 0.5
        
        # RULE 5: DTI (0-55, decimals)
        elif 'dti' in target.lower() and raw_dtype == 'numeric':
            if 0 <= mean_val <= 55:
                validation_score += 0.35
        
        validated.append((target, validation_score))
    
    best_target, best_score = max(validated, key=lambda x: x[1])
    print(f"    🏆 SELECTED: {best_target} ({best_score:.3f})")
    return best_target, best_score


def suggest_matches_for_missing(df, unmapped_columns=None, similarity_threshold=0.3):
    """Enhanced semantic matching with pattern matching priority"""
    
    if unmapped_columns is None or not unmapped_columns:
        print("❌ No unmapped_columns provided")
        return []

    missing_targets = CATALOG_STANDARD_NAMES
    candidate_raw = list(unmapped_columns)
    
    print(f"Missing targets: {len(missing_targets)}")
    print(f"Candidate raws: {candidate_raw}")
    
    if not missing_targets or not candidate_raw:
        return []

    # ✅ PHASE 1: Pattern-based matching (HIGH PRIORITY)
    print("\n🔍 PHASE 1: Pattern-based matching...")
    
    PATTERN_RULES = {
        'loan_id': [r'loan.*id', r'^id$', r'loan.*number', r'account.*number'],
        'original_balance': [
            r'org.*amount', r'orig.*balance', r'original.*balance', 
            r'prin.*amount', r'loan.*amount', r'original.*amount',
            r'org.*amt'  # ✅ Catches "org amount", "org amt"
        ],
        'current_balance': [
            r'curr.*bal', r'current.*balance', r'outstanding.*balance', 
            r'remaining.*balance', r'current.*amount'
        ],
        'current_rate': [
            r'current.*rate', r'interest.*rate', r'apr', r'^rate$', r'int.*rate'
        ],
        'origination_date': [
            r'orig.*date', r'fund.*date', r'start.*date', r'loan.*date'
        ],
        'maturity_date': [
            r'mat.*date', r'end.*date', r'final.*date', r'due.*date'
        ],
        'original_term_months': [
            r'orig.*term', r'original.*term', r'term.*month', r'^term$', r'duration'
        ],
        'orig_fico': [r'fico', r'credit.*score', r'orig.*fico'],
        'orig_ltv': [r'ltv', r'loan.*value', r'orig.*ltv', r'loan.*to.*value'],
        'dti_back_end': [r'dti', r'debt.*income', r'back.*end.*dti'],
        'auto_direct_indirect': [
            r'direct.*indirect', r'channel', r'origination.*type'
        ],
        'auto_new_used': [
            r'new.*used', r'vehicle.*condition', r'vehicle.*type'
        ],
        'state': [r'^state$', r'^st$', r'state.*code'],
        'days_past_due': [r'dpd', r'past.*due', r'delinq', r'days.*past'],
        'as_of_date': [r'as.*of.*date', r'report.*date', r'snapshot.*date'],
        'year': [r'^year$', r'fiscal.*year', r'reporting.*year']
    }
    
    pattern_matches = {}
    assigned_targets = set(df.columns)  # Already mapped columns
    
    for raw_col in candidate_raw:
        # Normalize column name for matching
        raw_lower = str(raw_col).lower()
        raw_normalized = raw_lower.replace('_', ' ').replace(':', '').replace('-', ' ').strip()
        
        best_match = None
        best_pattern = None
        
        for target, patterns in PATTERN_RULES.items():
            if target in assigned_targets:
                continue
                
            for pattern in patterns:
                if re.search(pattern, raw_normalized):
                    best_match = target
                    best_pattern = pattern
                    break
            
            if best_match:
                break
        
        if best_match:
            pattern_matches[raw_col] = {
                'raw_col': raw_col,
                'target_col': best_match,
                'score': 0.95,
                'method': 'pattern',
                'raw_dtype': 'inferred',
                'sample_values': df[raw_col].dropna().head(3).tolist() if raw_col in df.columns else []
            }
            assigned_targets.add(best_match)
            print(f"✅ PATTERN: {raw_col:20s} → {best_match:25s} (matched: {best_pattern})")
    
    # Remove pattern-matched columns from candidates
    remaining_candidates = [c for c in candidate_raw if c not in pattern_matches]
    
    print(f"\n📊 Pattern matching results: {len(pattern_matches)} matched, {len(remaining_candidates)} remaining")
    
    if not remaining_candidates:
        print("✅ All columns matched by patterns!")
        return list(pattern_matches.values())
    
    # ✅ PHASE 2: Embedding-based matching for remaining columns
    print(f"\n🔍 PHASE 2: Embedding-based matching for remaining columns: {remaining_candidates}")
    
    # ✅ CHANGED: Use remaining_candidates instead of candidate_raw
    raw_emb, raw_texts = build_raw_embeddings(df, remaining_candidates)
    
    if raw_emb is None or raw_emb.shape[0] == 0:
        print("❌ No valid embeddings generated")
        return list(pattern_matches.values())
    
    print(f"raw_emb shape: {raw_emb.shape}")
    print(f"Using full catalog: {CATALOG_EMB.shape}")

    # ✅ Infer dtypes for remaining candidates
    print("\n🔍 Detecting dtypes for remaining columns...")
    raw_dtypes = {}
    
    # ✅ CHANGED: Loop over remaining_candidates
    for col in remaining_candidates:
        if col not in df.columns:
            continue
            
        sample = df[col].dropna()
        if len(sample) == 0:
            raw_dtypes[col] = 'text'
            continue
        
        if pd.api.types.is_datetime64_any_dtype(sample):
            raw_dtypes[col] = 'date'
        elif pd.api.types.is_numeric_dtype(sample):
            uniqueness = sample.nunique() / len(sample)
            val_range = sample.max() - sample.min() if len(sample) > 0 else 0
            
            if uniqueness > 0.8 and val_range > len(sample) * 0.5:
                raw_dtypes[col] = 'id'
                print(f"  {col}: ID (uniqueness={uniqueness:.2f})")
            else:
                raw_dtypes[col] = 'numeric'
                print(f"  {col}: numeric")
        elif sample.astype(str).str.match(r'\d{4}-\d{2}-\d{2}').sum() > len(sample) * 0.5:
            raw_dtypes[col] = 'date'
            print(f"  {col}: date")
        else:
            raw_dtypes[col] = 'text'
            print(f"  {col}: text")

    # ✅ Compute CATALOG dtypes (only for unassigned targets)
    print("\n🔍 Building catalog dtype mapping...")
    CATALOG_DTYPES = {}
    for name in CATALOG_STANDARD_NAMES:
        if name in assigned_targets:
            continue
            
        name_lower = name.lower()
        
        if any(x in name_lower for x in ['date', 'as_of', 'maturity', 'origination']):
            CATALOG_DTYPES[name] = 'date'
        elif 'loan_id' in name_lower or name_lower == 'id':
            CATALOG_DTYPES[name] = 'id'
        elif any(x in name_lower for x in ['balance', 'amount', 'ltv', 'dti', 'rate', 'fico', 'term', 'dpd', 'days']):
            CATALOG_DTYPES[name] = 'numeric'
        else:
            CATALOG_DTYPES[name] = 'text'

    # ✅ DTYPE PRE-FILTER
    print("\n🔍 Filtering by dtype compatibility...")
    compatible_catalog_indices = []
    
    # ✅ CHANGED: Loop over remaining_candidates
    for raw_idx, raw_col in enumerate(remaining_candidates):
        raw_type = raw_dtypes.get(raw_col, 'text')
        
        compat_idx = [
            i for i, name in enumerate(CATALOG_STANDARD_NAMES) 
            if name not in assigned_targets and CATALOG_DTYPES.get(name, 'text') == raw_type
        ]
        
        compatible_catalog_indices.append(compat_idx)
        compat_names = [CATALOG_STANDARD_NAMES[i] for i in compat_idx[:3]]
        print(f"  {raw_col} ({raw_type}): {len(compat_idx)} compatible → {compat_names}")

    # ✅ Calculate similarities only for compatible pairs
    # ✅ CHANGED: Array size uses remaining_candidates length
    sims = np.zeros((len(remaining_candidates), len(CATALOG_STANDARD_NAMES)))
    for raw_idx, compat_idx in enumerate(compatible_catalog_indices):
        if compat_idx:
            sims[raw_idx, compat_idx] = cosine_similarity(
                raw_emb[raw_idx:raw_idx+1], 
                CATALOG_EMB[compat_idx]
            )[0]

    # ✅ Get TOP 3 matches per remaining column
    top_k = 3
    top_indices = []
    top_scores = []

    # ✅ CHANGED: Loop over remaining_candidates
    for i, raw_col in enumerate(remaining_candidates):
        compat_idx = compatible_catalog_indices[i]
        
        if not compat_idx:
            print(f"⚠️ {raw_col}: No dtype-compatible targets")
            top_indices.append([])
            top_scores.append([])
            continue
        
        compatible_scores = sims[i, compat_idx]
        k = min(top_k, len(compat_idx))
        top_k_local_idx = np.argpartition(compatible_scores, -k)[-k:]
        sorted_local_idx = top_k_local_idx[np.argsort(-compatible_scores[top_k_local_idx])]
        
        global_idx = [compat_idx[local_idx] for local_idx in sorted_local_idx]
        scores = [compatible_scores[local_idx] for local_idx in sorted_local_idx]
        
        top_indices.append(global_idx)
        top_scores.append(scores)
        
        matches_str = [f'{CATALOG_STANDARD_NAMES[idx]} ({score:.3f})' for idx, score in zip(global_idx, scores)]
        print(f"{raw_col}: {matches_str}")

    # ✅ Finalize embedding matches
    embedding_matches = []
    
    # ✅ CHANGED: Loop over remaining_candidates
    for i, raw_col in enumerate(remaining_candidates):
        if not top_indices[i]:
            continue
            
        raw_dtype = raw_dtypes.get(raw_col, 'text')
        sample_values = df[raw_col].dropna().head(3).tolist() if raw_col in df.columns else []
        
        best_target = CATALOG_STANDARD_NAMES[top_indices[i][0]]
        best_score = top_scores[i][0]
        
        if best_score >= similarity_threshold and best_target not in assigned_targets:
            embedding_matches.append({
                'raw_col': raw_col,
                'target_col': best_target,
                'score': float(best_score),
                'raw_dtype': raw_dtype,
                'sample_values': sample_values
            })
            assigned_targets.add(best_target)
            print(f"✅ EMBEDDING: {raw_col} ({raw_dtype}) → {best_target} ({best_score:.3f})")

    # ✅ COMBINE pattern matches + embedding matches
    all_matches = list(pattern_matches.values()) + embedding_matches
    
    print(f"\n🎉 Total matches: {len(all_matches)} (Pattern: {len(pattern_matches)}, Embedding: {len(embedding_matches)})")
    
    # ✅ CHANGED: Return combined matches
    return all_matches



# def suggest_matches_for_missing(df, similarity_threshold: float = 0.1, unmapped_columns=None):
#     print("🔥 Starting suggest_matches_for_missing()")

#     # 🔥 USE PASSED UNMAPPED COLUMNS INSTEAD OF INTERNAL LOGIC
#     if unmapped_columns is None or not unmapped_columns:
#         print("❌ No unmapped_columns provided - returning empty suggestions")
#         return []

#     # 1) Which standards are missing, and which raw cols are candidates
#     missing_targets = CATALOG_STANDARD_NAMES  # These are your "missing" columns
#     candidate_raw = list(unmapped_columns)  

#     print(f"Missing targets: {len(missing_targets)} = {missing_targets[:3]}...")
#     print(f"Candidate raws: {len(candidate_raw)} = {candidate_raw}")

#     if not missing_targets or not candidate_raw:
#         print("❌ EARLY RETURN - no targets or candidates")
#         return []

#     # 2) Build embeddings for candidate raw cols (header + sample values)
#     print("🔥 Building raw embeddings...")
#     raw_emb, raw_texts = build_raw_embeddings(df, candidate_raw)

#     # 🔥 FIX: Skip if no valid embeddings
#     if raw_emb is None or raw_emb.shape[0] == 0:
#         print("❌ No valid embeddings generated - skipping semantic matching")
#         return []

#     print(f"raw_emb shape: {raw_emb.shape}")

#     # 3) Use FULL catalog embeddings
#     print(f"Using full catalog: {CATALOG_EMB.shape}")

#     # ✅ PASS 1: Infer raw column dtypes with improved logic
#     print("\n🔍 PASS 1: Detecting dtypes for unmapped columns...")
#     raw_dtypes = {}
#     for col in candidate_raw:
#         if col not in df.columns:
#             continue
            
#         sample = df[col].dropna()
#         if len(sample) == 0:
#             raw_dtypes[col] = 'text'
#             continue
        
#         # Check if already datetime
#         if pd.api.types.is_datetime64_any_dtype(sample):
#             raw_dtypes[col] = 'date'
#         # Check if numeric (int or float)
#         elif pd.api.types.is_numeric_dtype(sample):
#             # ✅ FIX: Distinguish between ID and numeric
#             uniqueness = sample.nunique() / len(sample)
#             val_range = sample.max() - sample.min() if len(sample) > 0 else 0
            
#             # Heuristic: High uniqueness + wide range = likely ID
#             if uniqueness > 0.8 and val_range > len(sample) * 0.5:
#                 raw_dtypes[col] = 'id'
#                 print(f"  {col}: ID (uniqueness={uniqueness:.2f}, range={val_range})")
#             else:
#                 raw_dtypes[col] = 'numeric'
#                 print(f"  {col}: numeric")
#         # Check for date patterns in text
#         elif sample.astype(str).str.match(r'\d{4}-\d{2}-\d{2}').sum() > len(sample) * 0.5:
#             raw_dtypes[col] = 'date'
#             print(f"  {col}: date (pattern)")
#         else:
#             raw_dtypes[col] = 'text'
#             print(f"  {col}: text")

#     # ✅ Pre-compute CATALOG dtypes
#     print("\n🔍 Building catalog dtype mapping...")
#     CATALOG_DTYPES = {}
#     for name in CATALOG_STANDARD_NAMES:
#         name_lower = name.lower()
        
#         if any(x in name_lower for x in ['date', 'as_of', 'maturity', 'origination']):
#             CATALOG_DTYPES[name] = 'date'
#         elif 'loan_id' in name_lower or name_lower == 'id':
#             CATALOG_DTYPES[name] = 'id'
#         elif any(x in name_lower for x in ['balance', 'amount', 'ltv', 'dti', 'rate', 'fico', 'term', 'dpd', 'days']):
#             CATALOG_DTYPES[name] = 'numeric'
#         else:
#             CATALOG_DTYPES[name] = 'text'

#     # ✅ PASS 2: DTYPE PRE-FILTER - Create compatible_catalog_indices (list of lists)
#     print("\n🔍 PASS 2: Filtering by dtype compatibility...")
#     compatible_catalog_indices = []  # ← MATCHES YOUR EXISTING CODE

#     for raw_idx, raw_col in enumerate(candidate_raw):
#         raw_type = raw_dtypes.get(raw_col, 'text')
        
#         # Find catalog indices with matching dtype
#         compat_idx = [
#             i for i, name in enumerate(CATALOG_STANDARD_NAMES) 
#             if CATALOG_DTYPES.get(name, 'text') == raw_type
#         ]
        
#         compatible_catalog_indices.append(compat_idx)  # ← SAME STRUCTURE
        
#         compat_names = [CATALOG_STANDARD_NAMES[i] for i in compat_idx[:3]]
#         print(f"  {raw_col} ({raw_type}): {len(compat_idx)} compatible → {compat_names}")

#     # ✅ Now compatible_catalog_indices works with your existing code:
#     # 5) FIXED: Create proper 2D similarity matrix
#     sims = np.zeros((len(candidate_raw), len(CATALOG_STANDARD_NAMES)))
#     for raw_idx, compat_idx in enumerate(compatible_catalog_indices):
#         if compat_idx:  # Only if compatible targets exist
#             sims[raw_idx, compat_idx] = cosine_similarity(
#                 raw_emb[raw_idx:raw_idx+1], 
#                 CATALOG_EMB[compat_idx]
#             )[0]

#     print(f"Similarity matrix: {sims.shape}")
#     print(f"Max similarity: {sims.max():.3f}")


#     # 6) Get TOP 3 matches per raw column (only from dtype-compatible targets)
#     top_k = 3
#     top_indices = []
#     top_scores = []

#     for i, raw_col in enumerate(candidate_raw):
#         compat_idx = compatible_catalog_indices[i]
        
#         if not compat_idx:  # ✅ FIX: No compatible targets
#             print(f"⚠️ {raw_col}: No dtype-compatible targets found")
#             top_indices.append([])
#             top_scores.append([])
#             continue
        
#         # Get scores only for compatible indices
#         compatible_scores = sims[i, compat_idx]
        
#         # Get top k (or all if fewer than k)
#         k = min(top_k, len(compat_idx))
#         top_k_local_idx = np.argpartition(compatible_scores, -k)[-k:]
        
#         # Sort by score descending
#         sorted_local_idx = top_k_local_idx[np.argsort(-compatible_scores[top_k_local_idx])]
        
#         # Map back to global catalog indices
#         global_idx = [compat_idx[local_idx] for local_idx in sorted_local_idx]
#         scores = [compatible_scores[local_idx] for local_idx in sorted_local_idx]
        
#         top_indices.append(global_idx)
#         top_scores.append(scores)
        
#         # Display
#         matches_str = [f'{CATALOG_STANDARD_NAMES[idx]} ({score:.3f})' for idx, score in zip(global_idx, scores)]
#         print(f"{raw_col}: {matches_str}")

#     # 7) Apply guardrails to top 3 candidates and finalize matches
#     matches = []
#     for i, raw_col in enumerate(candidate_raw):
#         if not top_indices[i]:  # ✅ FIX: Skip if no compatible targets
#             continue
            
#         raw_dtype = raw_dtypes.get(raw_col, 'text')
#         sample_values = df[raw_col].dropna().head(3).tolist()
        
#         # Get top 3 candidates for this column
#         candidates = [(CATALOG_STANDARD_NAMES[idx], score) for idx, score in zip(top_indices[i], top_scores[i])]
        
#         # Re-rank with business rules
#         best_target, final_score = apply_business_rules(raw_col, raw_dtype, sample_values, candidates)
        
#         if final_score >= similarity_threshold and best_target in missing_targets:
#             matches.append({
#                 'raw_col': raw_col,
#                 'target_col': best_target,
#                 'score': float(final_score),
#                 'raw_dtype': raw_dtype,
#                 'sample_values': sample_values
#             })
#             print(f"✅ FINAL: {raw_col} ({raw_dtype}) → {best_target} ({final_score:.3f})")

#     print(f"Found {len(matches)} confident matches")
#     return matches
    

class LTN_loan_tape_prep:
    def __init__(self, loan_box_def: pd.DataFrame = None,
                 path: str = None,
                 sheet_name: str = None,
                 df_raw: pd.DataFrame = None,
                 state_map: dict = None):
        """
        Either pass df_raw directly, or give path + sheet_name for Excel ingestion.
        """
        self.loan_box_def = loan_box_def
        self.path = path
        self.sheet_name = sheet_name
        self.state_map = state_map or {}

        self.df = df_raw.copy() if df_raw is not None else None
        self.features = None          # ltn_features
        self.finalized_loan_box = None
        self.applied_semantic_mappings = {}
    
    def _init_semantic_engine(self):
        """Initialize semantic matching once (lazy load)"""
        if not hasattr(self, '_semantic_model'):
            from sentence_transformers import SentenceTransformer
            self._semantic_model = SentenceTransformer('all-mpnet-base-v2')
            
            # Build catalog embeddings once
            self.CATALOG_STANDARD_NAMES = list(FIELD_CATALOG.keys())
            self.CATALOG_TEXTS = [build_catalog_text(FIELD_CATALOG[name]) 
                                for name in self.CATALOG_STANDARD_NAMES]
            self.CATALOG_EMB = embed_texts(self.CATALOG_TEXTS)
            print(f"✅ Semantic engine ready: CATALOG_EMB {self.CATALOG_EMB.shape}")
        return self

    # 0. Loan tape ingestion
    def loan_tape_ingestion(self):
        if self.df is None:
            if self.path is None or self.sheet_name is None:
                raise ValueError("path and sheet_name must be provided for ingestion")
            self.df = pd.read_excel(self.path, sheet_name=self.sheet_name)
            print(self.df.isna())
            print(self.df.shape)
        return self

    # 1a. Column name standardization
    def column_standardization(self):
        self.unmapped_columns = []  # Initialize safely
        
        try:
            rename_map, unmapped = map_headers_to_standard(self.df.columns)
            self.df = self.df.rename(columns=rename_map or {})
            
            # 🔥 ONE-LINE FIX:
            self.unmapped_columns = list(unmapped) if unmapped is not None else list(self.df.columns)
            print(f"✅ {len(self.unmapped_columns)} unmapped columns")

            #  # --- Raw non-standard columns metrics (immediately after standardization) ---
            # raw_cols = set(self.df.columns)
            # standard_cols = set(CATALOG_STANDARD_NAMES)


            # # Columns that are NOT already exact standard names
            # self.raw_nonstandard_cols = [c for c in raw_cols if c not in standard_cols]
            # self.raw_nonstandard_count = len(self.raw_nonstandard_cols)
            # self.raw_total_cols = len(raw_cols)
            # self.raw_nonstandard_pct = (
            # self.raw_nonstandard_count / self.raw_total_cols if self.raw_total_cols else 0
            # )
            
        except Exception as e:
            print(f"⚠️ Error in column standardization: {e}")
            self.unmapped_columns = list(self.df.columns)  # Only on REAL error

            print(self.df.isna())
        
        return self

    # 1b. Semantic matching for unmapped columns
    def column_semantic_repair(self, similarity_threshold: float = 0.10):
        """Step 2: Semantic matching - RESILIENT"""
        self._init_semantic_engine()
    
        if 'suggest_matches_for_missing' not in globals():
            print("⚠️ suggest_matches_for_missing() MISSING - skipping semantic matching")
            self.semantic_suggestions = []
            return self
        
        try:
            # 🔥 PASS UNMAPPED COLUMNS EXPLICITLY
            temp_df = self.df.copy(deep=True)
            self.semantic_suggestions = suggest_matches_for_missing(
                temp_df, 
                unmapped_columns=self.unmapped_columns,  # ADD THIS
                similarity_threshold=similarity_threshold
            )
            print(f"✅ Semantic suggestions for {len(self.unmapped_columns)} unmapped cols")
        except Exception as e:
            print(f"⚠️ Semantic matching failed ({e})")
            self.semantic_suggestions = []
        
        return self
    
    # 2a. AUTO-assign high-confidence semantic mappings
    def apply_semantic_mappings(self, auto_threshold: float = 0.10, include_top3: bool = True):
        if not self.semantic_suggestions:
            print("⚠️ No semantic suggestions available")
            return self
        
        self.all_top3_suggestions = []
        assigned_targets = set(self.df.columns)  # ONLY raw tape columns initially
        rename_map = {}
        
        print(f"🔍 Initial assigned_targets: {sorted(assigned_targets)[:5]}...")
        
        for suggestion in self.semantic_suggestions:  # Sequential order!
            raw_col = suggestion['raw_col']
            target_col = suggestion['target_col']
            score = suggestion['score']
            
            # Store ALL suggestions for review
            self.all_top3_suggestions.append({
                'raw_col': raw_col, 'best_match': target_col, 
                'best_score': score, 'sample_values': suggestion['sample_values']
            })
            
            # 🔥 FIRST-COME, FIRST-SERVED: Assign if NOT already taken
            if target_col in assigned_targets:
                print(f"⚠️ SKIP {raw_col} → {target_col} (already assigned)")
            elif score >= auto_threshold:
                rename_map[raw_col] = target_col
                assigned_targets.add(target_col)  # NOW it's taken
                print(f"✅ {raw_col} → {target_col} ({score:.3f})")
            else:
                print(f"ℹ️ SKIP {raw_col} → {target_col} (score {score:.3f} < {auto_threshold})")
        
        # Safe rename
        if rename_map:
            self.df = self.df.rename(columns=rename_map)
            self.applied_semantic_mappings = rename_map
            self.df.columns = pd.Index([str(c) for c in self.df.columns])  # Clean tuples
            print(f"✅ Applied {len(rename_map)} mappings")
        
        print(self.df.isna())
        return self

    #2b. FORCE assign ALL top-ranked columns from semantic suggestions
    def assign_top_ranked_columns(self, min_confidence: float = 0.10, dry_run: bool = False):
        """
        Assign ALL top-ranked column names from semantic suggestions 
        (regardless of auto_threshold)
        
        Args:
            min_confidence: Minimum score to auto-assign (safer than 0.75)
            dry_run: Show what WOULD be assigned without changing df
        """
        if not self.semantic_suggestions:
            print("⚠️ No semantic suggestions available")
            return self
        
        # ✅ FIX: Track existing columns to prevent duplicates
        assigned_targets = set(self.df.columns)
        assignments = {}

        for suggestion in self.semantic_suggestions:
            if suggestion['score'] >= min_confidence:
                raw_col = suggestion['raw_col']
                target_col = suggestion['target_col']

                # ✅ FIX: Check if target column already exists
                if target_col in assigned_targets:
                    print(f"⚠️ SKIP: {raw_col:15s} → {target_col:20s} (already exists)")
                else:
                    assignments[raw_col] = target_col
                    assigned_targets.add(target_col)  # ✅ Mark as taken
                    print(f"✅ ASSIGNED: {raw_col:15s} → {target_col:20s} ({suggestion['score']:.3f})")


        # DRY RUN - show assignments only
        if dry_run:
            print(f"\n🧪 DRY RUN: Would assign {len(assignments)} columns")
            for raw, target in assignments.items():
                print(f"   {raw:15s} → {target}")
            return self
        
        # EXECUTE assignments
        if assignments:
            self.df = self.df.rename(columns=assignments)
            self.applied_semantic_mappings.update(assignments)
            print(f"\n🎉 EXECUTED: {len(assignments)} columns auto-assigned!")
        else:
            print(f"⚠️ No assignments above confidence {min_confidence}")
        
        print(self.df.isna())
        return self

    # 2. Value / format standardization
    def value_format_standardization(self):
        # dates
        for c in ["as_of_date", "origination_date", "maturity_date"]:
            if c in self.df.columns:
                self.df[c] = pd.to_datetime(self.df[c], errors="coerce")

        # normalize product flags
        if "auto_direct_indirect" in self.df.columns:
            self.df["auto_direct_indirect"] = (
                self.df["auto_direct_indirect"]
                .astype(str).str.upper().str.strip()
                .replace({
                    "DIRECT": "AUTO_DIRECT",
                    "INDIRECT": "AUTO_INDIRECT",
                    "AUTO DIRECT": "AUTO_DIRECT",
                    "AUTO INDIRECT": "AUTO_INDIRECT",
                })
            )

        if "auto_new_used" in self.df.columns:
            self.df["auto_new_used"] = (
                self.df["auto_new_used"]
                .astype(str).str.upper().str.strip()
                .replace({
                    "NEW": "AUTO_NEW",
                    "USED": "AUTO_USED",
                    "AUTO NEW": "AUTO_NEW",
                    "AUTO USED": "AUTO_USED",
                })
            )

        # standardize nulls
        null_tokens = {"", " ", "NA", "N/A", "NULL", "-", "NONE"}
        self.df = self.df.map(
            lambda x: np.nan
            if isinstance(x, str) and x.strip().upper() in null_tokens
            else x
        )

        # state normalization (optional map like {"TX": "TX", "TEXAS": "TX"})
        if "state" in self.df.columns and self.state_map:
            self.df["state"] = (
                self.df["state"].astype(str).str.upper().str.strip()
                .replace(self.state_map)
            )
        return self

    def data_quality_check(self):
        print("▶ data_quality_check starting")

        # 🔥 LINE 1: FORCE CLEAN COLUMNS AT ENTRY
        self.df.columns = pd.Index([str(c).strip() for c in self.df.columns])
        print(f"🔍 CLEAN columns: {list(self.df.columns[:5])}...")

        categorical_cols = {'auto_new_used', 'auto_direct_indirect', 'state', 'loan_id'}

        # Safe iteration - POSITIONAL access only
        for i, col_name in enumerate(self.df.columns):
            if col_name in categorical_cols:
                print(f"🅰️ SKIP categorical: {col_name}")
                continue

            try:
                series = self.df.iloc[:, i].copy()
                
                # 🔥 NEW: Skip numeric conversion for date columns
                if any(date_indicator in col_name.lower() for date_indicator in 
                    ['maturity_date', 'origination_date', 'as_of_date']):
                    # Preserve datetime - attempt conversion if not already
                    date_series = pd.to_datetime(series, errors='coerce')
                    self.df.iloc[:, i] = date_series
                    print(f"📅 Preserved date col {i} '{col_name}' -> datetime64")
                    continue
                
                # Original numeric standardization for non-date columns
                numeric_series = pd.to_numeric(series, errors='coerce')
                self.df.iloc[:, i] = numeric_series
                print(f"🔢 Standardized col {i} '{col_name}' -> numeric")
                
            except Exception as e:
                print(f"⚠️ Skipped col {i} '{col_name}': {e}")
                continue

        # 🔥 NEW: Final dtype enforcement pass for known LTN date columns
        date_columns = ["as_of_date", "origination_date", "maturity_date"]
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                print(f"✅ Enforced datetime on '{col}'")

        # Safe range checks - ILoc only (FIXED)
        col_map = {"orig_fico": (300, 850), "orig_ltv": (0, 200)}
        for target_col, (min_val, max_val) in col_map.items():
            if target_col in self.df.columns:
                col_idx = list(self.df.columns).index(target_col)
                col_data = self.df.iloc[:, col_idx]  # Extract column first
                mask = (col_data < min_val) | (col_data > max_val)
                self.df.iloc[mask.values, col_idx] = np.nan  # .values converts Series to numpy array
                print(f"✅ Range check {target_col}")


        
        
        print("✅ data_quality_check finished")
        print("📊 DEBUG - About to check dtypes and sample...")
        print("Dtypes:", self.df.dtypes.to_dict())
        print("Shape:", self.df.shape)
        print("Columns sample:", list(self.df.columns[:5]))

        return self


    # 3. Data quality checks
    # def data_quality_check(self):
    #     # cast numerics
    #     num_cols = [
    #         "original_balance", "current_balance", "current_rate",
    #         "original_term_months", "orig_fico", "orig_ltv",
    #         "dti_back_end", "days_past_due"
    #     ]

    #     for col in self.df.columns:
    #         try:
    #             series = self.df[col]  # 1D Series
    #             self.df[col] = pd.to_numeric(series, errors='coerce')
    #         except Exception as e:
    #             print(f"⚠️ Skipped {col}: {e}")
    #             continue


    #     # for col in self.df.select_dtypes(include=['object', 'number']).columns:
    #     #     if col in self.df.columns:  # Defensive check
    #     #         series = self.df[col]   # Extract 1D Series first
    #     #         self.df[col] = pd.to_numeric(series, errors='coerce')

        

    #     # range checks
    #     if "orig_fico" in self.df.columns:
    #         self.df.loc[(self.df["orig_fico"] < 300) | (self.df["orig_fico"] > 850),
    #                     "orig_fico"] = np.nan
    #     if "orig_ltv" in self.df.columns:
    #         self.df.loc[(self.df["orig_ltv"] < 0) | (self.df["orig_ltv"] > 200),
    #                     "orig_ltv"] = np.nan
    #     if "dti_back_end" in self.df.columns:
    #         self.df.loc[(self.df["dti_back_end"] < 0) | (self.df["dti_back_end"] > 100),
    #                     "dti_back_end"] = np.nan
    #     if "original_term_months" in self.df.columns:
    #         self.df.loc[(self.df["original_term_months"] <= 0) |
    #                     (self.df["original_term_months"] > 120),
    #                     "original_term_months"] = np.nan
    #     if "days_past_due" in self.df.columns:
    #         self.df.loc[self.df["days_past_due"] < 0, "days_past_due"] = np.nan

    #     # consistency checks
    #     if {"current_balance", "original_balance"}.issubset(self.df.columns):
    #         bad = self.df["current_balance"] > self.df["original_balance"]
    #         self.df.loc[bad, "current_balance"] = np.nan

    #     if {"origination_date", "maturity_date"}.issubset(self.df.columns):
    #         bad = self.df["maturity_date"] <= self.df["origination_date"]
    #         self.df.loc[bad, ["origination_date", "maturity_date"]] = np.nan

    #     if {"as_of_date", "origination_date", "maturity_date"}.issubset(self.df.columns):
    #         bad = ~(
    #             (self.df["as_of_date"] >= self.df["origination_date"]) &
    #             (self.df["as_of_date"] <= self.df["maturity_date"])
    #         )
    #         self.df.loc[bad, "as_of_date"] = np.nan

    #     return self

    # 4. Missing value imputation (deterministic + median/mode) - FIXED
    def missing_value_impute(self):
        # 1. Date term calculation - SAME INDEX, NO RESET
        if {"origination_date", "maturity_date"}.issubset(self.df.columns):
            # Calculate months (returns Series)
            months = ((self.df["maturity_date"] - self.df["origination_date"])
                    .dt.days / 30.0).round()
            
            # Safe column creation
            if "original_term_months" not in self.df.columns:
                self.df["original_term_months"] = np.nan
            
            # Get mask
            mask = self.df["original_term_months"].isna()
            if isinstance(mask, pd.DataFrame):
                mask = mask.iloc[:, 0]
            
            # 🔍 DEBUG: Print lengths to find the mismatch
            print(f"DEBUG - mask sum: {mask.sum()}")
            print(f"DEBUG - months[mask] length: {len(months[mask])}")
            print(f"DEBUG - mask type: {type(mask)}")
            print(f"DEBUG - months type: {type(months)}")
            
            # ✅ SAFEST FIX: Direct assignment without filtering
            if mask.sum() > 0:
                self.df.loc[mask, "original_term_months"] = months.loc[mask]
            
            # print(f"✅ Imputed {mask.sum()} terms from dates")

        # 2. SAFE imputation loop
        columns_to_impute = self.df.columns.tolist()
        for c in columns_to_impute:
            if c not in self.df.columns:
                continue
            col_series = self.df[c]

            if isinstance(col_series, pd.DataFrame):
                print(f"⚠️ Skipping '{c}' - duplicate column detected")
                continue

            if col_series.dtype.kind in "biufc":
                med = col_series.median()
                self.df[c] = col_series.fillna(med)
            else:
                mode = col_series.mode(dropna=True)
                if not mode.empty:
                    self.df[c] = col_series.fillna(mode.iloc[0])
        
        print("✅ missing_value_impute finished - all indexes aligned")
        return self

    # 5. Outlier treatment (IQR)
    def outlier_removal(self):
        num_cols = [
            "original_balance", "current_balance", "current_rate",
            "original_term_months", "orig_fico", "orig_ltv",
            "dti_back_end"   
            # , "days_past_due"
        ]
        for c in num_cols:
            if c not in self.df.columns:
                continue
            q1 = self.df[c].quantile(0.25)
            q3 = self.df[c].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.df[c] = self.df[c].clip(lower, upper)
        return self

    # 6a. Feature normalization and engineering
    def normalization(self):
        print("\n🔍 DEBUG: Starting normalization()")
        self.features = pd.DataFrame(index=self.df.index)
        print(f"✅ Created features DataFrame with index length: {len(self.features.index)}")

        # scaled numeric features
        num_cols = [
            "orig_fico", "orig_ltv", "dti_back_end",
            "original_balance", "original_term_months", "current_rate"
        ]
        scaler = StandardScaler()
        existing = [c for c in num_cols if c in self.df.columns]
        print(f"🔍 DEBUG: Found {len(existing)} numeric columns to scale: {existing}")
        
        if existing:
            print(f"🔍 DEBUG: Fitting scaler...")
            scaled = scaler.fit_transform(self.df[existing])
            print(f"✅ Scaled shape: {scaled.shape}")
            for i, c in enumerate(existing):
                self.features[f"{c}_z"] = scaled[:, i]
                print(f"  ✅ Added {c}_z")

        # business‑friendly buckets - ✅ FIX: Ensure Series input
        print("\n🔍 DEBUG: Starting pd.cut() operations...")
        
        if "orig_fico" in self.df.columns:
            print("🔍 DEBUG: Processing orig_fico...")
            fico_series = self.df["orig_fico"]
            print(f"  Type: {type(fico_series)}, Shape: {fico_series.shape if hasattr(fico_series, 'shape') else 'N/A'}")
            if isinstance(fico_series, pd.DataFrame):
                print("  ⚠️  WARNING: orig_fico is DataFrame, converting to Series")
                fico_series = fico_series.iloc[:, 0]
            print("  Calling pd.cut()...")
            self.features["fico_band"] = pd.cut(
                fico_series,
                bins=[300, 660, 700, 750, 850],
                labels=["<660", "660‑699", "700‑749", "750+"],
                include_lowest=True,
                right=False,
            )
            print("  ✅ fico_band created")
        
        if "orig_ltv" in self.df.columns:
            print("🔍 DEBUG: Processing orig_ltv...")
            ltv_series = self.df["orig_ltv"]
            print(f"  Type: {type(ltv_series)}, Shape: {ltv_series.shape if hasattr(ltv_series, 'shape') else 'N/A'}")
            if isinstance(ltv_series, pd.DataFrame):
                print("  ⚠️  WARNING: orig_ltv is DataFrame, converting to Series")
                ltv_series = ltv_series.iloc[:, 0]
            print("  Calling pd.cut()...")
            self.features["ltv_bucket"] = pd.cut(
                ltv_series,
                bins=[0, 95, 115, 130, 200],
                labels=["≤95", "95‑115", "115‑130", ">130"],
                include_lowest=True,
                right=False,
            )
            print("  ✅ ltv_bucket created")
        
        if "dti_back_end" in self.df.columns:
            print("🔍 DEBUG: Processing dti_back_end...")
            dti_series = self.df["dti_back_end"]
            print(f"  Type: {type(dti_series)}, Shape: {dti_series.shape if hasattr(dti_series, 'shape') else 'N/A'}")
            if isinstance(dti_series, pd.DataFrame):
                print("  ⚠️  WARNING: dti_back_end is DataFrame, converting to Series")
                dti_series = dti_series.iloc[:, 0]
            print("  Calling pd.cut()...")
            self.features["dti_bucket"] = pd.cut(
                dti_series,
                bins=[0, 40, 50, 60, 100],
                labels=["≤40", "40‑50", "50‑60", ">60"],
                include_lowest=True,
                right=False,
            )
            print("  ✅ dti_bucket created")

        # one‑hot encode key categoricals
        print("\n🔍 DEBUG: Starting one-hot encoding...")
        cat_cols = []
        for c in ["auto_new_used", "auto_direct_indirect", "state"]:
            if c in self.df.columns:
                cat_cols.append(c)
        print(f"🔍 DEBUG: Found {len(cat_cols)} categorical columns: {cat_cols}")
        
        if cat_cols:
            print("🔍 DEBUG: Calling pd.get_dummies()...")
            dummies = pd.get_dummies(self.df[cat_cols], prefix=cat_cols, dtype=int)
            print(f"  Dummies shape: {dummies.shape}, Empty: {dummies.empty}")
            
            if not dummies.empty:
                print(f"🔍 DEBUG: Concatenating dummies with features...")
                print(f"  self.features shape before concat: {self.features.shape}")
                print(f"  dummies shape: {dummies.shape}")
                self.features = pd.concat([self.features, dummies], axis=1)
                print(f"  ✅ self.features shape after concat: {self.features.shape}")

        print("✅ normalization() finished\n")
        return self


    # 6b. UI‑driven filtering before loan box assignment
    def apply_ui_filters(self, filters):
                        #  fico_min=None, fico_max=None,
                        # ltv_min=None, ltv_max=None,      # ← ADD ltv_min
                        # dti_min=None, dti_max=None,      # ← ADD dti_min  
                        # term_min=None, term_max=None,    # ← ADD term_min
                        # balance_min=None, balance_max=None,  # ← ADD balance_min
                        # interest_rate_min=None, interest_rate_max=None,  # ← ADD interest_rate_min
                        # days_past_due_min=None, days_past_due_max=None,  # ← ADD days_past_due_min
                        # product=None,
                        # remove_zero_balance=False,
                        # remove_zero_rate=False,
                        # exclude_over_30_dq=False,
                        # remove_matured=False,
                        # min_remaining_term=0):
        """
        Apply UI-selected filters on the cleaned loan tape.
        All args are optional; None means 'no filter' on that dimension.

        UI → column mapping:
          - Product      -> auto_new_used  ("AUTO_NEW", "AUTO_USED")
          - FICO         -> orig_fico
          - LTV          -> orig_ltv
          - DTI          -> dti_back_end
          - Term         -> original_term_months
          - Loan Amount  -> original_balance
        """
        df = self.df.copy()

        # in_box_flag may be stored as string "True"/"False" from session round-trip
        if 'in_box_flag' in df.columns:
            df['in_box_flag'] = df['in_box_flag'].map(
                lambda x: True if str(x).strip().lower() == 'true' else False
            )
        # ── END ADD ──

        print(f"🚀 Phase 2 START: {len(df)} rows from caller")
        
        # Debug info
        print("🔍 FILTER DEBUG START")
        print(f"Input rows: {len(df)}")
        print("Columns found:", df.columns.tolist())

        # Check EACH filter column:
        key_cols = ['orig_fico', 'orig_ltv', 'dti_back_end', 'original_term_months', 
                    'original_balance', 'current_rate', 'days_past_due']
        for col in key_cols:
            if col in df.columns:
                print(f"✅ {col}: {len(df)} rows, NaN: {df[col].isna().sum()}")
                print(f"   Min/Max: {df[col].min():.0f}/{df[col].max():.0f}")
            else:
                print(f"❌ {col} MISSING!")

        # 👇 ADD THIS BLOCK (line ~45, after key_cols debug):
        print(f"🔍 Matched loans: {(df['in_box_flag'] == True).sum()}/{len(df)}")
        df = df[df['in_box_flag'] == True].copy()  # Only matched loans!
        print(f"✅ After matched filter: {len(df)} rows")

        numeric_cols = ['orig_fico', 'orig_ltv', 'dti_back_end', 
                   'original_term_months', 'original_balance',
                   'current_rate', 'days_past_due']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Product filter (AUTO_NEW / AUTO_USED)
        if filters.get('product'):  # ← filters.get() instead of product param
            product = filters['product']
            if isinstance(product, str):
                product = [product]
            df = df[df["auto_new_used"].isin(product)]

        # if product is not None:
        #     if isinstance(product, str):
        #         product = [product]
        #     df = df[df["auto_new_used"].isin(product)]


        # FICO - ADD MIN/MAX
        # FICO - same pattern (extract then filter)
        fico_min = float(filters.get('fico_min')) if filters.get('fico_min') else None
        fico_max = float(filters.get('fico_max')) if filters.get('fico_max') else None
        if fico_min is not None and 'orig_fico' in df.columns:
            df = df[df["orig_fico"] >= fico_min]
        if fico_max is not None and 'orig_fico' in df.columns:
            df = df[df["orig_fico"] <= fico_max]


        # if filters.get('fico_min') is not None and 'orig_fico' in df.columns:
        #     df = df[df["orig_fico"] >= fico_min]
        # if fico_max is not None and 'orig_fico' in df.columns:
        #     df = df[df["orig_fico"] <= fico_max]

        # ltv - ADD MIN/MAX
        ltv_min = float(filters.get('ltv_min')) if filters.get('ltv_min') else None
        ltv_max = float(filters.get('ltv_max')) if filters.get('ltv_max') else None
        if ltv_min is not None and 'orig_ltv' in df.columns:
            df = df[df["orig_ltv"] >= ltv_min]  # ← ADD THIS
        if ltv_max is not None and 'orig_ltv' in df.columns:
            df = df[df["orig_ltv"] <= ltv_max]

        # if ltv_max is not None and 'orig_ltv' in df.columns:
        #     df = df[df["orig_ltv"] <= ltv_max]

        # DTI - ADD MIN/MAX
        dti_min = float(filters.get('dti_min')) if filters.get('dti_min') else None
        dti_max = float(filters.get('dti_max')) if filters.get('dti_max') else None
        if dti_min is not None and 'dti_back_end' in df.columns:
            df = df[df["dti_back_end"] >= dti_min]  # ← ADD THIS
        if dti_max is not None and 'dti_back_end' in df.columns:
            df = df[df["dti_back_end"] <= dti_max]


        # if dti_max is not None and 'dti_back_end' in df.columns:
        #     df = df[df["dti_back_end"] <= dti_max]

        # Term - ADD MIN/MAX
        term_min = float(filters.get('term_min')) if filters.get('term_min') else None
        term_max = float(filters.get('term_max')) if filters.get('term_max') else None
        if term_min is not None and 'original_term_months' in df.columns:
            df = df[df["original_term_months"] >= term_min]
        if term_max is not None and 'original_term_months' in df.columns:
            df = df[df["original_term_months"] <= term_max]

        # if term_max is not None and 'original_term_months' in df.columns:
        #     df = df[df["original_term_months"] <= term_max]

        # Balance - ADD MIN  
        balance_min = float(filters.get('balance_min')) if filters.get('balance_min') else None
        balance_max = float(filters.get('balance_max')) if filters.get('balance_max') else None
        if balance_min is not None and 'original_balance' in df.columns:  
            df = df[df["original_balance"] >= balance_min]  
        if balance_max is not None and 'original_balance' in df.columns:
            df = df[df["original_balance"] <= balance_max]

        
        
        # if orig_balance_max is not None and 'original_balance' in df.columns:
        #     df = df[df["original_balance"] <= orig_balance_max]

        # if remove_zero_balance: df = df[df["original_balance"] > 0]
        
        # Int Rate - ADD MIN
        rate_min = float(filters.get('interest_rate_min')) if filters.get('interest_rate_min') else None
        rate_max = float(filters.get('interest_rate_max')) if filters.get('interest_rate_max') else None
        if rate_min is not None and 'current_rate' in df.columns:
            df = df[df["current_rate"] >= rate_min]
        if rate_max is not None and 'current_rate' in df.columns:
            df = df[df["current_rate"] <= rate_max]
        
        
        # DPD - ADD MIN
        dpd_min = float(filters.get('days_past_due_min')) if filters.get('days_past_due_min') else None
        dpd_max = float(filters.get('days_past_due_max')) if filters.get('days_past_due_max') else None
        if dpd_min is not None and 'days_past_due' in df.columns:
            df = df[df["days_past_due"] >= dpd_min]
        if dpd_max is not None and 'days_past_due' in df.columns:
            df = df[df["days_past_due"] <= dpd_max]
        

        if filters.get('remove_zero_balance'): 
            df = df[df["original_balance"] > 0]

        if filters.get('remove_zero_rate') and "current_rate" in df.columns: 
            df = df[df["current_rate"] > 0]

        if filters.get('exclude_over_30_dq') and 'days_past_due' in df.columns: 
            df = df[df['days_past_due'] <= 30]

        if filters.get('remove_matured') and "maturity_date" in df.columns:
            df['maturity_date'] = pd.to_datetime(df['maturity_date'], errors='coerce').dt.date
            today = pd.Timestamp.now().date()
            df = df[df['maturity_date'] > today]

        if filters.get('min_remaining_term', 0) > 0 and "maturity_date" in df.columns:
            df['maturity_date'] = pd.to_datetime(df['maturity_date'], errors='coerce')
            df['days_remaining'] = (df['maturity_date'] - pd.Timestamp.now()).dt.days
            df = df[df['days_remaining'] > 90]


        self.df = df.reset_index(drop=True)
        self.filtered_loan_box = df
        return self

    # ADD THIS METHOD HERE (after apply_ui_filters, before loan_box_assignment)
    def get_loan_box_definitions(self):
        if self.loan_box_def is None or self.loan_box_def.empty:
            xls_url = "https://docs.google.com/spreadsheets/d/1_KXon_6Z-E6M5Zkjnr4fkH_oO9ljQQ1L/export?format=xlsx"
            
            # Explicit engine for Google Sheets XLSX
            self.loan_box_def = pd.read_excel(
                xls_url, 
                sheet_name="Underwriting_rules", 
                skiprows=2,
                engine='openpyxl'  # ← FIX: Forces correct XLSX reader
            )
            
            self.loan_box_def.columns = self.loan_box_def.columns.str.lower().str.replace(' ', '_').str.strip()
            # Remove any completely empty rows
            self.loan_box_def = self.loan_box_def.dropna(how='all')
            print(f"✅ Loaded {len(self.loan_box_def)} loan boxes from G-Drive")
            print("Columns:", list(self.loan_box_def.columns))
            print(self.loan_box_def)
            print(self.df)
        
        return self.loan_box_def[['box_id', 'product_type', 'fico_min', 'fico_max', 
                                'ltv_max', 'dti_max', 'term_max_months', 'max_original_amount']]
    
    # 7. Loan box assignment
    def loan_box_assignment(self):
        df = self.df.copy()
        box_def = self.loan_box_def.copy()

        # ensure numeric in box_def
        for c in ["fico_min", "fico_max", "ltv_max", "dti_max",
                  "term_max_months", "max_original_amount"]:
            if c in box_def.columns:
                box_def[c] = pd.to_numeric(box_def[c], errors="coerce")
        
        # ensure numeric in df
        data_cols = ['orig_fico', 'orig_ltv', 'dti_back_end', 
                    'original_term_months', 'original_balance', 'current_rate',      
                    'days_past_due']
        
        # ✅ FIX 1: Check if columns exist before converting
        print("\n🔍 DEBUG: Checking required columns before box assignment")
        for col in data_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"✅ {col}: EXISTS - Sample: {df[col].dropna().head(2).tolist()}")
            else:
                print(f"❌ {col}: MISSING - will cause issues!")

        assignments = []

        for idx, row in df.iterrows():
            enriched = row.to_dict()
            loan_id = row.get("loan_id")  
            product = row.get("auto_new_used")  
            fico = row.get("orig_fico")  
            ltv = row.get("orig_ltv")
            dti = row.get("dti_back_end")
            term = row.get("original_term_months")
            bal = row.get("original_balance")
    
            # map product flag to product_type
            product_raw = str(product).strip().upper()
            if "NEW" in product_raw:
                prod_type = "AUTO_NEW"
            elif "USED" in product_raw:
                prod_type = "AUTO_USED"
            else:
                prod_type = None

            enriched["product_type"] = prod_type

            matched_box = None
            exclusion_reason = None

            if prod_type is None or pd.isna(fico) or pd.isna(ltv) or pd.isna(dti):
                exclusion_reason = "MISSING_KEY_FIELDS"
            else:
                candidates = box_def[box_def["product_type"].isin(
                    [prod_type, "AUTO_ALL"]
                )]

                for _, br in candidates.iterrows():
                    fico_min = br["fico_min"]
                    fico_max = br["fico_max"]

                    print(f"DEBUG: fico={fico} (type={type(fico)}), fico_min={fico_min} (type={type(fico_min)}), fico_max={fico_max} (type={type(fico_max)})")

                    cond_fico = True
                    if not np.isnan(fico_min):
                        cond_fico &= fico >= fico_min
                    if not np.isnan(fico_max):
                        cond_fico &= fico <= fico_max

                    print(f"DEBUG: ltv={ltv} (type={type(ltv)}), ltv_max={br['ltv_max']} (type={type(br['ltv_max'])})")
                    cond_ltv = (ltv <= br["ltv_max"])

                    print(f"DEBUG: dti={dti} (type={type(dti)}), dti_max={br['dti_max']} (type={type(br['dti_max'])})")
                    cond_dti = (dti <= br["dti_max"])

                    print(f"DEBUG: term={term} (type={type(term)}), term_max={br['term_max_months']} (type={type(br['term_max_months'])})")
                    cond_term = (
                        pd.isna(term) or term <= br["term_max_months"]
                    )

                    print(f"DEBUG: bal={bal} (type={type(bal)}), bal_max={br['max_original_amount']} (type={type(br['max_original_amount'])})")
                    cond_amt = (
                        pd.isna(bal) or bal <= br["max_original_amount"]
                    )

                    if cond_fico and cond_ltv and cond_dti and cond_term and cond_amt:
                        matched_box = br["box_id"]
                        break

                if matched_box is None:
                    exclusion_reason = "NO_BOX_MATCH"

            in_box_flag = matched_box is not None and not str(matched_box).endswith("OOB")
            box_id = matched_box or "OOB_UNMATCHED"

            enriched.update({
                    "box_id": box_id,
                    "in_box_flag": in_box_flag,
                    "exclusion_reason": exclusion_reason,
                })

            assignments.append(enriched)

        print(f"🎯 Assignments created: {len(assignments)}")

        # ✅ FIX: Merge assignments with the original dataframe to preserve all columns
        box_df = pd.DataFrame(assignments)

        print(f"🎯 Box DF shape: {box_df.shape}")
        print(f"🎯 Box DF columns: {box_df.columns.tolist()}")
        print(f"🎯 Sample product_type values: {box_df['product_type'].value_counts()}")
        print(f"🎯 Sample box_id values: {box_df['box_id'].value_counts()}")


        # box_df = self.df.copy()  # Start with full dataframe

        # # Add box assignment columns from assignments
        # if assignments:
        #     assignment_df = pd.DataFrame(assignments)
            
        #     # Merge box assignments with original data
        #     # Assuming assignments has 'loan_id' or index to join on
        #     if 'loan_id' in box_df.columns and 'loan_id' in assignment_df.columns:
        #         box_df = box_df.merge(assignment_df, on='loan_id', how='left')
        #     else:
        #         # If no loan_id, use index-based merge
        #         for col in assignment_df.columns:
        #             box_df[col] = assignment_df[col].values

        # print(f"🎯 Box DF shape: {box_df.shape}")
        # print(f"🎯 Box DF columns: {box_df.columns.tolist()}")

        self.finalized_loan_box = box_df.copy()
            
        return self.finalized_loan_box
        

        # # CORRECT:
        # box_df = pd.DataFrame(assignments)  
        # print(f"🎯 Box DF shape: {box_df.shape}") 
        
        # #self.df = self.df.dropna(subset=data_cols)
        # self.finalized_loan_box = box_df.copy()
            
        # return self.finalized_loan_box  
    
    def generate_pdf_report(self):
        """Generate professional LTN PDF report (reportlab, seller-buyer style)."""
    
        if self.finalized_loan_box is None or len(self.finalized_loan_box) == 0:
            return "No data for report"
        
        df = self.finalized_loan_box
        filtered_count = len(df)
        
        filename = f"LTN_Refined_Portfolio_Report_{filtered_count}_loans.pdf"
        pdf_path = os.path.join('static', filename)
        os.makedirs('static', exist_ok=True)
        
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph(
            "LTN Loan Tape Analyzer - Refined Portfolio Summary", 
            styles['Title']
        )
        story.append(title)
        story.append(Spacer(1, 0.3*inch))
        
        # ALM Logo (optional)
        try:
            logo = Image('static/alm_first_logo.png', width=1.5*inch, height=0.6*inch)
            story.append(logo)
        except:
            pass
        
        # Key Metrics
        total_balance = df['original_balance'].sum()
        avg_fico = df['orig_fico'].mean()
        product_mix = df['product_type'].value_counts().to_dict()
        
        metrics_data = [
            ['Metric', 'Value'],
            ['Total Loans', str(filtered_count)],
            ['Total Balance', f"${total_balance:,.0f}"],
            ['Avg FICO', f"{avg_fico:.0f}"],
            ['Products', ', '.join([f"{k}:{v}" for k,v in list(product_mix.items())[:3]])]
        ]
        metrics_table = Table(metrics_data)
        
        # Style the metrics table professionally
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))

        # Prepare top loans table data (example: top 10 loans from finalized_loan_box)
        loans_data = [['Loan ID', 'Orig FICO', 'LTV', 'DTI', 'Product', 'Reason']]
        if self.finalized_loan_box is not None:
            top_loans = self.finalized_loan_box.head(10).to_dict('records')
            for loan in top_loans:
                loans_data.append([
                    str(loan.get('loan_id', 'N/A')),
                    f"{loan.get('orig_fico', 0):.0f}",
                    f"{loan.get('orig_ltv', 0):.1%}",
                    f"{loan.get('orig_dti', 0):.1%}",
                    str(loan.get('product_type', 'N/A')),
                    str(loan.get('assignment_reason', 'N/A'))
                ])

        loans_table = Table(loans_data, colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1*inch, 1.5*inch])
        loans_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))

        # Build the PDF story (flowables)
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=getSampleStyleSheet()['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=colors.darkblue
        )
        story.append(Paragraph("LTN Loan Tape Analyzer Report", title_style))
        story.append(Spacer(1, 20))
        
        # Subtitle with timestamp
        subtitle = f"<b>Generated:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | <b>Total Loans:</b> {len(self.finalized_loan_box or []):,}"
        story.append(Paragraph(subtitle, getSampleStyleSheet()['Normal']))
        story.append(Spacer(1, 30))
        
        # Metrics section
        story.append(Paragraph("Key Portfolio Metrics", getSampleStyleSheet()['Heading2']))
        story.append(metrics_table)
        story.append(Spacer(1, 30))
        
        # Top loans section
        story.append(Paragraph("Top 10 Sample Loans", getSampleStyleSheet()['Heading2']))
        story.append(loans_table)
        
        # Generate PDF file
        import tempfile
        from pathlib import Path
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
        doc.build(story)
        pdf_buffer.seek(0)
        
        # Save to temp file for Flask download (or return buffer directly)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(pdf_buffer.getvalue())
            pdf_path = tmp.name
        
        pdf_filename = f"LTN_LoanTape_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf"
        
        return pdf_path, pdf_filename