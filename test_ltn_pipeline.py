import pandas as pd
import numpy as np
from LTN_loan_tape_prep import LTN_loan_tape_prep
# 1. Load loan box definition 
loan_box_def_path = "/content/drive/MyDrive/Loan Portfolio Intelligence/Functionality Scope/Seller-Buyer MVP Prototype/Loan Box Rules.xlsx"
loan_box_def = pd.read_excel(loan_box_def_path, sheet_name="Underwriting_rules", header=2)

# 2. Path + sheet for raw LTN loan tape
loan_tape_path = "/content/drive/MyDrive/Loan Portfolio Intelligence/Functionality Scope/Seller-Buyer MVP Prototype/LTN Loan Tape - Mock Data.xlsx"
loan_tape_sheet = "AutoLoanTape"   # change to the actual sheet name

print("loan_box_def columns:", loan_box_def.columns.tolist())

# 3. Example UI filters coming from your web app
ui_filters = {
    "product": ["AUTO_NEW", "AUTO_USED"],  # or "AUTO_NEW" only
    "fico_min": 670,
    "fico_max": 749,
    "ltv_max": 115,
    "dti_max": 45,
    "term_max": 96,
    "orig_balance_max": 150_000,
}

# 4. Optional state normalization map
state_map = {
    "TEXAS": "TX",
    "TX": "TX",
    "ARKANSAS": "AR",
    "AR": "AR",
    # add others as needed
}

# 5. Run the full pipeline
prep = (
    LTN_loan_tape_prep(
        loan_box_def=loan_box_def,
        path=loan_tape_path,
        sheet_name=loan_tape_sheet,
        state_map=state_map,
    )
    .loan_tape_ingestion()
    .column_standardization()
    .value_format_standardization()
    .data_quality_check()
    .missing_value_impute()
    .outlier_removal()
)

# 6. Apply UI‑driven filters (optional; comment out to use full tape)
prep.apply_ui_filters(
    product=ui_filters.get("product"),
    fico_min=ui_filters.get("fico_min"),
    fico_max=ui_filters.get("fico_max"),
    ltv_max=ui_filters.get("ltv_max"),
    dti_max=ui_filters.get("dti_max"),
    term_max=ui_filters.get("term_max"),
    orig_balance_max=ui_filters.get("orig_balance_max"),
)

# 7. Assign loans to boxes and get the customized loan box output
final_loan_box = prep.loan_box_assignment()

# 8. Inspect or save
print(final_loan_box.head())
final_loan_box.to_excel(
    "/content/drive/MyDrive/Loan Portfolio Intelligence/Functionality Scope/Seller-Buyer MVP Prototype/Caprock_Custom_Loan_Box.xlsx",
    index=False,
)

