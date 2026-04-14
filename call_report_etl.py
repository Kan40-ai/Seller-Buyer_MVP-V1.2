# call_report_etl.py
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # if you used this in Colab


class CallReport5300ETL:
    """
    ETL for ONE 5300 mock workbook at a time.
    Works for either:
      - 5300 Call Report - Mock Data (Pre-Loaded CU)
      - 5300 Call Report - Mock Data (New CU)
    by auto-detecting sheet names.
    """

    def __init__(
        self,
        path: str,
        state_map: Optional[Dict[str, str]] = None,
        df_summary: Optional[pd.DataFrame] = None,
        df_parts: Optional[pd.DataFrame] = None,
    ):
        self.path = Path(path)
        self.state_map = state_map or {}

        # raw frames (optional overrides) - FIXED copy() warning
        self.summary_df = df_summary.copy() if df_summary is not None else None
        self.parts_df = df_parts.copy() if df_parts is not None else None
        
        # **NEW: Initialize ratios_df to fix AttributeError**
        self.ratios_df = pd.DataFrame()

        # features
        self.summary_features = None
        self.parts_features = None
        self.ratios_features = None  # For future ratios processing

    # def __init__(
    #     self,
    #     path: str,
    #     state_map: Optional[Dict[str, str]] = None,
    #     df_summary: Optional[pd.DataFrame] = None,
    #     df_parts: Optional[pd.DataFrame] = None,
    # ):
    #     self.path = Path(path)
    #     self.state_map = state_map or {}

    #     # raw frames (optional overrides)
    #     self.summary_df = df_summary.copy() if df_summary is not None else None
    #     self.parts_df = df_parts.copy() if df_parts is not None else None

    #     # features
    #     self.summary_features = None

    # 0. Ingestion for a single file
    def call_report_ingestion(self):
        if self.summary_df is not None and self.parts_df is not None:
            return self

        xl = pd.ExcelFile(self.path)
        sheet_names = [s.strip() for s in xl.sheet_names]

        # pick summary + parts sheets based on naming pattern
        summary_sheet = None
        parts_sheet = None
        ratios_sheet = None

        for s in sheet_names:
            su = s.upper()
            if "CALL_SUMMARY" in su:
                summary_sheet = s
            elif "CALL_PARTICIPATIONS" in su:
                parts_sheet = s
            elif "CALL_RATIOS" in su:
                ratios_sheet = s

        if summary_sheet is None or parts_sheet is None:
            raise ValueError(
                f"Could not find Call_Summary / Call_Participations sheets in {self.path}"
            )

        self.summary_df = pd.read_excel(xl, sheet_name=summary_sheet, header=2)
        self.parts_df = pd.read_excel(xl, sheet_name=parts_sheet, header=2)

        if ratios_sheet is not None:
            self.ratios_df = pd.read_excel(xl, sheet_name=ratios_sheet, header=2)
        else:
            self.ratios_df = pd.DataFrame()  # Empty if no ratios sheet

        return self

    # 1. Column standardization
    def column_standardization(self):

        # 1. Call Summary columns
        if self.summary_df is not None:
            col_map_summary = {
                "As of Date ": "as_of_date",
                "Total Loans": "total_loans",
                "Total Deposits": "total_deposits",
                "Total Assets ": "total_assets",
                "LTD ": "ltd",
                "Size band": "size_band",
                "State": "state",
            }
            self.summary_df.rename(columns=col_map_summary, inplace=True)
            self.summary_df.columns = (
                self.summary_df.columns.astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"[^0-9a-zA-Z_]+", "_", regex=True)
            )

        # 2. Call Participations columns
        if self.parts_df is not None:
            col_map_parts = {
                "Outstanding Balance": "outstanding_balance",
                "Amount Purchased YTD": "amount_purchased_ytd",
                "Retained Balance Outstanding": "retained_balance_outstanding",
                "Amount Sold YTD": "amount_sold_ytd",
            }
            self.parts_df.rename(columns=col_map_parts, inplace=True)
            self.parts_df.columns = (
                self.parts_df.columns.astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"[^0-9a-zA-Z_]+", "_", regex=True)
            )

        # 3. Call Ratios columns
        if not self.ratios_df.empty:
            col_map_ratios = {
                "As of Date": "as_of_date",
                "Total Investments": "total_investments",
                "Commercial Loans": "commercial_loans", 
                "Indirect Loans": "indirect_loans",
                "Total Net Worth": "total_net_worth",
                "New Vehicle Loans": "new_vehicle_loans",
                "Used Vehicle Loans": "used_vehicle_loans",
            }
            self.ratios_df.rename(columns=col_map_ratios, inplace=True)
            self.ratios_df.columns = (
                self.ratios_df.columns.astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"[^0-9a-zA-Z_]+", "_", regex=True)
            )
        return self
    
    # 2. Value / format standardization
    def value_format_standardization(self):
        """Standardize data types and null values across all 3 dataframes"""
        
        # 1. Dates standardization (all 3 tabs have as_of_date)
        for df, name in [(self.summary_df, "summary"), (self.ratios_df, "ratios")]:
            if df is not None and not df.empty and "as_of_date" in df.columns:
                df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")

        # 2. State normalization (summary only)
        if self.summary_df is not None and "state" in self.summary_df.columns:
            self.summary_df["state"] = (
                self.summary_df["state"].astype(str).str.upper().str.strip()
            )
            if self.state_map:
                self.summary_df["state"] = self.summary_df["state"].replace(self.state_map)

        # 3. Standardize null tokens in ALL 3 dataframes
        null_tokens = {"", " ", "NA", "N/A", "NULL", "-", "NONE"}
        
        dataframes = [
            (self.summary_df, "summary"),
            (self.parts_df, "parts"), 
            (self.ratios_df, "ratios")
        ]
        
        for df, name in dataframes:
            if df is not None and not df.empty:
                df[:] = df.map(
                    lambda x: np.nan
                    if isinstance(x, str) and x.strip().upper() in null_tokens
                    else x
                )

        return self



    # 2. Value / format standardization
    # def value_format_standardization(self):
    #     # summary dates
    #     if "as_of_date" in self.summary_df.columns:
    #         self.summary_df["as_of_date"] = pd.to_datetime(
    #             self.summary_df["as_of_date"], errors="coerce"
    #         )

    #     # state normalization
    #     if "state" in self.summary_df.columns:
    #         self.summary_df["state"] = (
    #             self.summary_df["state"].astype(str).str.upper().str.strip()
    #         )
    #         if self.state_map:
    #             self.summary_df["state"] = self.summary_df["state"].replace(
    #                 self.state_map
    #             )

    #     # standardize null tokens in both tables
    #     null_tokens = {"", " ", "NA", "N/A", "NULL", "-", "NONE"}
    #     self.summary_df = self.summary_df.map(
    #         lambda x: np.nan
    #         if isinstance(x, str) and x.strip().upper() in null_tokens
    #         else x
    #     )
    #     self.parts_df = self.parts_df.map(
    #         lambda x: np.nan
    #         if isinstance(x, str) and x.strip().upper() in null_tokens
    #         else x
    #     )

    #     return self

    # 3. Data Quality Check
    def data_quality_check(self):
        """Convert columns to numeric + validate/recompute key ratios across all 3 tabs"""
        
        # 1. Numeric casting for SUMMARY
        num_cols_summary = ["total_loans", "total_deposits", "total_assets", "ltd"]
        for c in num_cols_summary:
            if self.summary_df is not None and c in self.summary_df.columns:
                self.summary_df[c] = pd.to_numeric(self.summary_df[c], errors="coerce")

        # 2. Numeric casting for PARTICIPATIONS  
        num_cols_parts = [
            "outstanding_balance",
            "amount_purchased_ytd", 
            "retained_balance_outstanding",
            "amount_sold_ytd",
        ]
        for c in num_cols_parts:
            if self.parts_df is not None and c in self.parts_df.columns:
                self.parts_df[c] = pd.to_numeric(self.parts_df[c], errors="coerce")

        # 3. **NEW: Numeric casting for RATIOS**
        num_cols_ratios = [
            "total_investments",
            "commercial_loans", 
            "indirect_loans",
            "total_net_worth",
            "new_vehicle_loans",
            "used_vehicle_loans",
        ]
        for c in num_cols_ratios:
            if not self.ratios_df.empty and c in self.ratios_df.columns:
                self.ratios_df[c] = pd.to_numeric(self.ratios_df[c], errors="coerce")

        # 4. Recompute LTD if needed (summary only)
        if (self.summary_df is not None and 
            {"total_loans", "total_deposits", "ltd"}.issubset(self.summary_df.columns)):
            mask_valid = self.summary_df["total_deposits"] > 0
            recomputed_ltd = pd.Series(
                np.where(
                    mask_valid,
                    self.summary_df["total_loans"] / self.summary_df["total_deposits"],
                    np.nan,
                ),
                index=self.summary_df.index,
            )
            ltd = self.summary_df["ltd"].fillna(recomputed_ltd)
            self.summary_df["ltd"] = ltd

        return self


    # def data_quality_check(self):
    #     # numeric casting for summary
    #     num_cols_summary = ["total_loans", "total_deposits", "total_assets", "ltd"]
    #     for c in num_cols_summary:
    #         if c in self.summary_df.columns:
    #             self.summary_df[c] = pd.to_numeric(self.summary_df[c], errors="coerce")

    #     # numeric casting for participations
    #     num_cols_parts = [
    #         "outstanding_balance",
    #         "amount_purchased_ytd",
    #         "retained_balance_outstanding",
    #         "amount_sold_ytd",
    #     ]
    #     for c in num_cols_parts:
    #         if c in self.parts_df.columns:
    #             self.parts_df[c] = pd.to_numeric(self.parts_df[c], errors="coerce")

    #     # recompute LTD if needed
    #     if {"total_loans", "total_deposits", "ltd"}.issubset(self.summary_df.columns):
    #         mask_valid = self.summary_df["total_deposits"] > 0
    #         recomputed_ltd = pd.Series(
    #             np.where(
    #                 mask_valid,
    #                 self.summary_df["total_loans"]
    #                 / self.summary_df["total_deposits"],
    #                 np.nan,
    #             ),
    #             index=self.summary_df.index,
    #         )
    #         ltd = self.summary_df["ltd"]
    #         ltd = ltd.fillna(recomputed_ltd)
    #         self.summary_df["ltd"] = ltd

    #     return self

    # 4. Missing value imputation
    def missing_value_impute(self):
        """Smart imputation: median/mode for summary, 0 for financials across all 3 tabs"""
        
        # 1. Summary: median for numeric, mode for categoricals - **FIXED chained assignment**
        if self.summary_df is not None:
            for c in self.summary_df.columns:
                if self.summary_df[c].dtype.kind in "biufc":  # numeric
                    med = self.summary_df[c].median()
                    # Line 179 fix: NO inplace=True, assign back explicitly
                    self.summary_df[c] = self.summary_df[c].fillna(med)
                else:  # categorical
                    mode = self.summary_df[c].mode(dropna=True)
                    if not mode.empty:
                        # Line 183 fix: NO inplace=True
                        self.summary_df[c] = self.summary_df[c].fillna(mode.iloc[0])

        # 2. Participations: NA numeric -> 0 - **FIXED**
        num_cols_parts = [
            "outstanding_balance",
            "amount_purchased_ytd",
            "retained_balance_outstanding", 
            "amount_sold_ytd",
        ]
        if self.parts_df is not None:
            for c in num_cols_parts:
                if c in self.parts_df.columns:
                    # Line 193 fix: NO inplace=True
                    self.parts_df[c] = self.parts_df[c].fillna(0.0)

        # 3. **NEW: Ratios: NA numeric -> 0 (financial metrics)**
        num_cols_ratios = [
            "total_investments",
            "commercial_loans",
            "indirect_loans", 
            "total_net_worth",
            "new_vehicle_loans",
            "used_vehicle_loans",
        ]
        if not self.ratios_df.empty:
            for c in num_cols_ratios:
                if c in self.ratios_df.columns:
                    self.ratios_df[c] = self.ratios_df[c].fillna(0.0)

        return self




    # 4. Missing value imputation
    # def missing_value_impute(self):
    #     # summary: median for numeric, mode for categoricals
    #     for c in self.summary_df.columns:
    #         if self.summary_df[c].dtype.kind in "biufc":
    #             med = self.summary_df[c].median()
    #             self.summary_df[c].fillna(med, inplace=True)
    #         else:
    #             mode = self.summary_df[c].mode(dropna=True)
    #             if not mode.empty:
    #                 self.summary_df[c].fillna(mode.iloc[0], inplace=True)

    #     # participations: NA numeric -> 0
    #     for c in [
    #         "outstanding_balance",
    #         "amount_purchased_ytd",
    #         "retained_balance_outstanding",
    #         "amount_sold_ytd",
    #     ]:
    #         if c in self.parts_df.columns:
    #             self.parts_df[c].fillna(0.0, inplace=True)

    #     return self

    # 5. Feature creation
    def create_summary_features(self):
        df = self.summary_df.copy()
        features = pd.DataFrame(index=df.index)

        # scaled LTD
        if "ltd" in df.columns:
            scaler = StandardScaler()
            ltd_scaled = scaler.fit_transform(df[["ltd"]])
            features["ltd_z"] = ltd_scaled[:, 0]

        # log assets
        if "total_assets" in df.columns:
            features["log_assets"] = np.log1p(df["total_assets"])

        # one-hot size band and state
        cat_cols = []
        for c in ["size_band", "state"]:
            if c in df.columns:
                cat_cols.append(c)
        if cat_cols:
            dummies = pd.get_dummies(df[cat_cols], prefix=cat_cols)
            features = pd.concat([features, dummies], axis=1)

        self.summary_features = features
        return self

    # 6. Convenience runner
    def run(self):
        return (
            self.call_report_ingestion()
            .column_standardization()
            .value_format_standardization()
            .data_quality_check()
            .missing_value_impute()
            .create_summary_features()
        )
