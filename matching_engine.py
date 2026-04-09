from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Type
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY


# ------------------------------------------------------------
# 1. Config dataclass
# ------------------------------------------------------------

@dataclass
class SellerBuyerConfig:
    call_report_etl_cls: Type
    preloaded_summaries: Dict[Tuple[str, pd.Timestamp], pd.Series]
    preloaded_parts: Dict[Tuple[str, pd.Timestamp], pd.DataFrame]
    preloaded_ratios: Dict[Tuple[str, pd.Timestamp], pd.DataFrame]
    geo_neighbors: Dict[str, List[str]]
    state_map: Dict[str, str]


# ------------------------------------------------------------
# 2. SellerBuyerMatching with full pipeline + PDF report
# ------------------------------------------------------------

class SellerBuyerMatching:
    def __init__(self, config: SellerBuyerConfig):
        self.config = config
        self.matching_state = {}  # Track intermediate results for PDF
        self.ratios_df = config.preloaded_ratios  
        self.summary_df = config.preloaded_summaries  
        self.parts_df = config.preloaded_parts

    @staticmethod
    def _pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    @staticmethod
    def _normalize_score(score: float, min_score: float, max_score: float) -> float:
        """Normalize score to 0-100 scale with proper min-max logic."""
        return 100.0 * (score - min_score) / (max_score - min_score)
        

    # -------------------- 2.1 ETL for new CU -----------------
    def run_call_report_etl(self, xlsx_path: Path) -> Dict:
        etl = self.config.call_report_etl_cls(
            path=str(xlsx_path),
            state_map=self.config.state_map,
        )
        etl.run()

        summary_df = etl.summary_df
        parts_df = etl.parts_df
        ratios_df = etl.ratios_df

        if summary_df.shape[0] != 1:
            raise ValueError(
                f"Expected exactly 1 row in summary for new CU, got {summary_df.shape[0]}"
            )

        summary_row = summary_df.iloc[0]
        inst_id = summary_row["institution_id"]
        as_of = pd.to_datetime(summary_row["as_of_date"]).normalize()

        return {
            "institution_id": inst_id,
            "as_of_date": as_of,
            "summary": summary_row,
            "summary_df": summary_df,
            "participations": parts_df,
            "call_ratios": ratios_df,
        }

    
    # -------------------- 2.2 5300 Call Ratios calculation -----------------
    def _5300_call_ratios(self, ratios_df: pd.DataFrame, summary_df: pd.DataFrame, cu_id: str, role: str) -> pd.Series:
        """
        Standalone function: Calculates 5300 ratios for specified CU_ID from call_ratio + call_summary tabs.
        
        Returns pd.Series with all key ratios + health flags.
        """
        # Merge tabs on Institution ID
        cu_data = pd.merge(
        ratios_df[ratios_df['institution_id'] == cu_id], 
        summary_df[summary_df['institution_id'] == cu_id],  
        on='institution_id'
        ).iloc[0]
        
        # Extract metrics
        total_loans = float(cu_data['total_loans'])
        total_deposits = float(cu_data['total_deposits'])
        total_assets = float(cu_data['total_assets'])
        investments = float(cu_data['total_investments'])
        comm_loans = float(cu_data['commercial_loans'])
        indirect_loans = float(cu_data['indirect_loans'])
        net_worth = float(cu_data['total_net_worth'])
        new_veh_loans = float(cu_data['new_vehicle_loans'])
        used_veh_loans = float(cu_data['used_vehicle_loans'])
                              

        # Core Ratios (NCUA 5300 standards)
        ratios = {
        # 1. Cash & Equivalents/Assets  
        'CashEquivToAssets': investments / total_assets if total_assets > 0 else 0,
        
        # 2. New Veh Loans/Assets 
        'NewVehToAssets': new_veh_loans / total_assets if total_assets > 0 else 0,
        
        # 3. Used Veh Loans/Assets 
        'UsedVehToAssets': used_veh_loans / total_assets if total_assets > 0 else 0,
        
        # 4. Loan Assets/Total Loans 
        'LoanAssetsToTotalLoans': total_loans / total_deposits if total_deposits > 0 else 0,
        
        # 5. Loan to Deposit 
        'LoanToDeposit': total_loans / total_deposits if total_deposits > 0 else 0,
        
        # 6. Loan to Share ratio 
        'LoanToShare': total_loans / total_deposits if total_deposits > 0 else 0,
        
        # 7. Net Worth Ratio 
        'NetWorthToAssets': net_worth / total_assets if total_assets > 0 else 0,
        
        # 8. Loan/Assets Ratio 
        'LoanToAssets': total_loans / total_assets if total_assets > 0 else 0}


        # # 🔥 BUYER/SELLER CLASSIFICATION
        ltd = ratios['LoanToDeposit']
        # role = "BUYER" if ltd < 0.75 else "SELLER"
        
        # 🎯 ROLE-SPECIFIC FLAGS (Your exact thresholds!)
        if role == "BUYER":
            flags = {
                'HighCash_Healthy': ratios['CashEquivToAssets'] > 0.10,      # High cash
                'LowNewVeh_Good': ratios['NewVehToAssets'] < 0.05,           # < 5%
                'LowUsedVeh_Good': ratios['UsedVehToAssets'] < 0.05,         # < 5%
                'LowLoanAssets_OK': ratios['LoanAssetsToTotalLoans'] < 0.60, # < 60%
                'LowLTD_Buyer': ltd < 0.70,                                  # < 70%
                'LowLoanShare_Buyer': ratios['LoanToShare'] < 0.60,          # < 60%
                #'HighCashGrowth': True,                                      # Mock >100%
                'NetWorth_OK': ratios['NetWorthToAssets'] < 0.06,            # > 6%
                'LowLoanAssets_Buyer': ratios['LoanToAssets'] < 0.60         # < 60%
            }
        else:  # SELLER
            flags = {
                'LowCash_Normal': ratios['CashEquivToAssets'] < 0.0465,      # 4.65%
                'HighNewVeh_Seller': ratios['NewVehToAssets'] > 0.10,        # > 10%
                'HighUsedVeh_Seller': ratios['UsedVehToAssets'] > 0.10,      # > 10%
                'HighLoanAssets_Seller': ratios['LoanAssetsToTotalLoans'] > 0.60, # > 60%
                'HighLTD_Seller': ltd > 0.90,                                # > 90%
                'HighLoanShare_Seller': ratios['LoanToShare'] > 1.00,        # > 100%
                'NetWorth_OK': ratios['NetWorthToAssets'] > 0.06,            # > 6%
                'HighLoanAssets_Seller': ratios['LoanToAssets'] > 0.80       # > 80%
            }

        # 🔥 CONDITIONAL HEALTH SCORE (4/5 = 80%, etc.)
        true_flags = sum(flags.values())
        total_flags = len(flags)
        health_score = (true_flags / total_flags) * 100

    
        result = pd.Series({**ratios, 
        **flags, 
        'role': role,
        'health_score': health_score,
        'matching_flags': true_flags,
        'total_flags': total_flags,
        'cu_id': cu_id})

        return result

    # -------------------- 2.3 LTD classification -------------
    def classify_ltd(
        self,
        summary_row: pd.Series,
        summary_df: pd.DataFrame,
        ratios_df: pd.DataFrame,
        label: str = "Caprock",
    ) -> Tuple[str, float]:
        """
        Returns (role, ltd) where role is 'BUYER' or 'SELLER'.
        """
        ltd = float(summary_row["ltd"])
        if ltd < 0.75:
            role = "BUYER"
        else:
            role = "SELLER"

        cu_id = summary_row["institution_id"]

        print("\n[Step 1: Liquidity Position]")
        print(f"{label} CU: Loan-to-Deposit Ratio (LTD) = {self._pct(ltd)} → {role}")

        # ---------- invoke 5300 ratios & health flags ----------
        try:
            ratios = self._5300_call_ratios(
                ratios_df=ratios_df,
                summary_df=summary_df,
                cu_id=cu_id,
                role=role,
            )

            print(
                f"✅ {ratios['role']} Profile | Health: {ratios['health_score']:.0f}% "
                f"({ratios['matching_flags']}/{ratios['total_flags']} flags)"
            )
            print(
                f"   Cash & Equivalent-to-Assets Ratio: {ratios['CashEquivToAssets']:.1%} "
                f"{'✅ HIGH' if ratios['CashEquivToAssets'] > 0.10 else '⚠️ LOW'}"
            )
            print(
                f"   New Vehicle Loans-to-Assets Ratio: {ratios['NewVehToAssets']:.1%} "
                f"{'✅ GOOD' if ratios['NewVehToAssets'] < 0.05 else '⚠️ HIGH'}"
            )
            print(
                f"   Used Vehicle Loans-to-Assets Ratio: {ratios['UsedVehToAssets']:.1%} "
                f"{'✅ GOOD' if ratios['UsedVehToAssets'] < 0.05 else '⚠️ HIGH'}"
            )
            print(
                f"   Loan Assets-to-Total Loans Ratio: {ratios['LoanAssetsToTotalLoans']:.1%} "
                f"{'✅ GOOD' if ratios['LoanAssetsToTotalLoans'] < 0.60 else '⚠️ HIGH'}"
            )
            print(
                f"   Loan-to-Deposit Ratio: {ratios['LoanToDeposit']:.1%} "
                f"{'✅ HEALTHY' if ratios['LoanToDeposit'] < 0.70 else '⚠️ HIGH'}"
            )
            print(
                f"   Net Worth Ratio: {ratios['NetWorthToAssets']:.1%} "
                f"{'✅ BALANCED' if ratios['NetWorthToAssets'] > 0.06 else '⚠️ LOW'}"
            )
            print(
                f"   Loan-to-Assets Ratio: {ratios['LoanToAssets']:.1%} "
                f"{'✅ BALANCED' if ratios['LoanToAssets'] < 0.60 else '⚠️ HIGH'}"
            )

            # Store in matching_state for downstream use
            self.matching_state["health"] = {
                "cu_id": cu_id,
                "role": ratios["role"],
                "ltd": ltd,
                "health_score": ratios["health_score"],
                "matching_flags": ratios["matching_flags"],
                "total_flags": ratios["total_flags"],
                "ratios": ratios.to_dict()
                if hasattr(ratios, "to_dict")
                else dict(ratios),
            }

        except Exception as e:
            print(f"Error in _5300_call_ratios: {repr(e)}")

        self.matching_state["liquidity"] = {
            "label": label,
            "ltd": ltd,
            "role": role,
        }

        return role, ltd


    # -------------------- 2.4 Opposite-side pool -------------
    def build_opposite_pool(
    self,
    summary_df: pd.DataFrame,
    ratios_df: pd.DataFrame,
    new_inst_id: str,
    new_role: str,
    label: str = "Caprock",
) -> List[Dict]:
        """
        Build a pool of opposite-side counterparties for the new CU.
        Each counterparty includes its health flags and ratio diagnostics,
        mirroring the classify_ltd() flow.
        """

        pool: List[Dict] = []
        print("\n[Step 2: Opposite-Side Counterparties]")

        for idx, row in summary_df.iterrows():
            inst_id = row["institution_id"]
            if inst_id == new_inst_id:
                continue

            other_ltd = float(row["ltd"])
            other_role = "BUYER" if other_ltd < 0.75 else "SELLER"
            if new_role == other_role:
                continue

            # ---------- Invoke same _5300_call_ratios logic ----------
            try:
                ratios = self._5300_call_ratios(
                    ratios_df=ratios_df,        # ✅ Passed parameter (not self.ratios_df)
                    summary_df=summary_df,      # ✅ Passed parameter (not self.summary_df)
                    cu_id=inst_id,
                    role=other_role,
                )

                print(
                    f"\n{inst_id} ({other_role}) | Health: {ratios['health_score']:.0f}% "
                    f"({ratios['matching_flags']}/{ratios['total_flags']} flags)"
                )
                print(
                    f"   Cash & Equivalent-to-Assets: {ratios['CashEquivToAssets']:.1%} "
                    f"{'✅ HIGH' if ratios['CashEquivToAssets'] > 0.10 else '⚠️ LOW'}"
                )
                print(
                    f"   Loan-to-Deposit Ratio: {ratios['LoanToDeposit']:.1%} "
                    f"{'✅ HEALTHY' if ratios['LoanToDeposit'] < 0.70 else '⚠️ HIGH'}"
                )
                print(
                    f"   Net Worth Ratio: {ratios['NetWorthToAssets']:.1%} "
                    f"{'✅ STRONG' if ratios['NetWorthToAssets'] > 0.06 else '⚠️ LOW'}"
                )

                pool.append(
                    {
                        "institution_id": inst_id,
                        "summary": row,                    # ✅ Use row (pd.Series), not 's'
                        "side": other_role,
                        "ltd": other_ltd,
                        "ratios": ratios.to_dict() if hasattr(ratios, "to_dict") else dict(ratios),
                        "health_score": ratios["health_score"],
                        "matching_flags": ratios["matching_flags"],
                        "total_flags": ratios["total_flags"],
                    }
                )

            except Exception as e:
                print(f"Error processing {inst_id}: {repr(e)}")
                continue

        if not pool:
            print(f"No opposite-side candidates found for {label} CU.")
            return pool

        print(f"\nSummary: Found {len(pool)} suitable opposite-side candidates for {label} CU ({new_role}).")
        self.matching_state["opposite_pool"] = pool
        return pool


    # -------------------- 2.5 Activity scores ----------------
    def compute_activity_scores(
        self,
        new_cu: Dict,
        opposite_pool: List[Dict],
        new_role: str,
        parts_df,
        cu_label: str = "Caprock",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Buyer/Seller per-asset scoring.
        """
        def _agg_by_asset(df: pd.DataFrame):
            return df.groupby("asset_class_code", dropna=False).agg(
                purchased_outstanding=("outstanding_balance", "sum"),
                purchased_ytd=("amount_purchased_ytd", "sum"),
                retained_balance=("retained_balance_outstanding", "sum"),
                sold_ytd=("amount_sold_ytd", "sum"),
            )

        records = []

        new_parts = new_cu["participations"]
        new_agg = _agg_by_asset(new_parts)
        for asset, row in new_agg.iterrows():
            if new_role == "BUYER":
                score_raw = 0.6 * row["purchased_outstanding"] + 0.4 * row["purchased_ytd"]
            else:
                score_raw = 0.5 * row["retained_balance"] + 0.5 * row["sold_ytd"]
            records.append(
                {
                    "institution_id": new_cu["institution_id"],
                    "role": new_role,
                    "asset_class_code": asset,
                    "score_raw": score_raw,
                }
            )

        opp_role = "SELLER" if new_role == "BUYER" else "BUYER"
        for cu in opposite_pool:
            inst_id = cu["institution_id"]

            cu_parts = parts_df[parts_df["institution_id"] == inst_id]
            if cu_parts.empty:
                continue

            agg = _agg_by_asset(cu_parts)
            for asset, row in agg.iterrows():
                if opp_role == "BUYER":
                    score_raw = 0.6 * row["purchased_outstanding"] + 0.4 * row["purchased_ytd"]
                else:
                    score_raw = 0.5 * row["retained_balance"] + 0.5 * row["sold_ytd"]
                records.append(
                    {
                        "institution_id": inst_id,
                        "role": opp_role,
                        "asset_class_code": asset,
                        "score_raw": score_raw,
                    }
                )

        scores_long = pd.DataFrame(records)
        if scores_long.empty:
            print("\n[Step 3: Activity Profile by Asset Class]\nNo participations data available.")
            return scores_long, pd.DataFrame(columns=["buyer_id", "seller_id", "activity_score"])

        def _scale_group(g):
            min_v = g["score_raw"].min()
            max_v = g["score_raw"].max()
            if max_v == min_v:
                g["score_scaled"] = 50.0
            else:
                g["score_scaled"] = 100.0 * (g["score_raw"] - min_v) / (max_v - min_v)
            return g

        scores_long = scores_long.groupby("asset_class_code", group_keys=False).apply(_scale_group)

        def _strength(score):
            if score >= 70:
                return "STRONG"
            elif score >= 40:
                return "MEDIUM"
            else:
                return "WEAK"

        scores_long["strength"] = scores_long["score_scaled"].apply(_strength)

        print("\n[Step 3: Activity Profile by Asset Class (Normalized Weighted Scores)]")

        new_id = new_cu["institution_id"]
        new_rows = scores_long[scores_long["institution_id"] == new_id]
        print(f"{cu_label} CU ({new_role}):")
        for _, r in new_rows.iterrows():
            print(
                f"  - {r['asset_class_code']}: {r['score_scaled']:.0f} "
                f"({r['strength']}) [weighted {new_role.lower()} score]"
            )

        opp_rows = scores_long[scores_long["institution_id"] != new_id]
        for inst in opp_rows["institution_id"].unique():
            sub = opp_rows[opp_rows["institution_id"] == inst]
            print(f"\n{inst} ({opp_role}):")
            for _, r in sub.iterrows():
                print(
                    f"  - {r['asset_class_code']}: {r['score_scaled']:.0f} "
                    f"({r['strength']}) [weighted {opp_role.lower()} score]"
                )

        pair_records = []
        for inst in opp_rows["institution_id"].unique():
            new_total = new_rows["score_scaled"].sum()
            opp_total = opp_rows[opp_rows["institution_id"] == inst]["score_scaled"].sum()

            if new_role == "BUYER":
                buyer_id = new_id
                seller_id = inst
            else:
                buyer_id = inst
                seller_id = new_id

            pair_records.append(
                {
                    "buyer_id": buyer_id,
                    "seller_id": seller_id,
                    "activity_score": new_total + opp_total,
                }
            )

        pair_scores = pd.DataFrame(pair_records)

        self.matching_state["activity_scores"] = {
            "scores_long": scores_long,
            "pair_scores": pair_scores,
        }

        return scores_long, pair_scores

    # -------------------- 2.6 Activity-based pairs -----------
    def build_activity_pairs(
        self,
        buyer_id: str,
        scores_long: pd.DataFrame,
        pair_scores: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Buyer/Seller matching logic per asset.
        """
        print("\n[Step 4: Buyer–Seller Matching Logic (Per Asset Class)]")

        if scores_long.empty or pair_scores.empty:
            print("No activity data available for matching.")
            return pd.DataFrame()

        buyer_rows = scores_long[(scores_long["institution_id"] == buyer_id) &
                                 (scores_long["role"] == "BUYER")]
        seller_rows = scores_long[scores_long["role"] == "SELLER"]

        valid_pairs = []

        for seller_id in seller_rows["institution_id"].unique():
            s_rows = seller_rows[seller_rows["institution_id"] == seller_id]

            asset_scores = []
            asset_notes = []

            for asset in sorted(set(buyer_rows["asset_class_code"]) |
                                set(s_rows["asset_class_code"])):
                b = buyer_rows[buyer_rows["asset_class_code"] == asset]
                s = s_rows[s_rows["asset_class_code"] == asset]
                if b.empty or s.empty:
                    continue

                b_strength = b.iloc[0]["strength"]
                s_strength = s.iloc[0]["strength"]

                if b_strength == "WEAK" or s_strength == "WEAK":
                    continue

                if b_strength == "STRONG" and s_strength == "STRONG":
                    level = "HIGHEST PRIORITY"
                elif (b_strength == "STRONG" and s_strength == "MEDIUM") or \
                     (b_strength == "MEDIUM" and s_strength == "STRONG"):
                    level = "HIGH PRIORITY"
                else:
                    level = "OPTIONAL"

                asset_scores.append(
                    b.iloc[0]["score_scaled"] + s.iloc[0]["score_scaled"]
                )
                asset_notes.append(f"{asset}: {b_strength} ↔ {s_strength} ({level})")

            if not asset_scores:
                continue

            total_activity = sum(asset_scores)

            valid_pairs.append(
                {
                    "buyer_id": buyer_id,
                    "seller_id": seller_id,
                    "activity_score": total_activity,
                    "activity_detail": "; ".join(asset_notes),
                }
            )

            print(f"- {buyer_id} ↔ {seller_id}:")
            for note in asset_notes:
                print(f"    {note}")
            print(f"    → Combined activity score (qualifying assets): {total_activity:.1f}")

        self.matching_state["activity_pairs"] = pd.DataFrame(valid_pairs)
        return pd.DataFrame(valid_pairs)

    # -------------------- 2.7 Balance Sheet Capacity Score dimension ------------
    def calculate_capacity_score(self, cu_health_data):
        """uses classify_ltd ratios"""
        
        ratios = cu_health_data["ratios"]  # From matching_state["health"]["ratios"]
        cu_data = cu_health_data.get("raw_data", {})  # total_assets etc.
        
        # 1. Liquidity (25%) - cash/assets
        liquidity = min(ratios["CashEquivToAssets"] * 100, 100)
        
        # 2. Leverage (25%) - loan/deposit + loan/assets
        leverage_raw = (ratios["LoanToDeposit"] + ratios["LoanToAssets"]) / 2
        leverage = max(0, 100 - (leverage_raw * 100))
        
        # 3. Capital (25%) - net worth
        capital = min(ratios["NetWorthToAssets"] * 100, 100)
        
        # 4. Size/Scale (25%) - total assets
        assets_m = cu_data.get("total_assets", 0) 
        size_score = min((assets_m / 500_000_000) ** 0.5 * 100, 100) 
        
        capacity_score = (liquidity + leverage + capital + size_score) / 4

        return {
        "capacity_score": capacity_score,
        "breakdown": {
            "liquidity": liquidity,
            "leverage": leverage,
            "capital": capital,
            "size_score": size_score
        }
    }

    # -------------------- 2.8 Capacity Pair Matching ------------
    def build_capacity_pairs(self, buyer_id: str, buyer_ratios: pd.DataFrame, 
                        seller_ratios: pd.DataFrame) -> pd.DataFrame:
        """
        SIMPLE Capacity matching (CU-level) - mirrors build_activity_pairs
        Professor: Same pattern as activity but CU-level ratios vs asset classes
        """
        print("\n[Step 2: Buyer–Seller Capacity Matching (CU-Level 5300 Ratios)]")
        print(f"🔍 build_capacity_pairs() - buyer shape: {buyer_ratios.shape}")
        print(f"🔍 build_capacity_pairs() - seller shape: {seller_ratios.shape}")
        print(f"🔍 buyer capacity: {buyer_ratios['capacity_score'].iloc[0]}")
        print(f"🔍 seller capacities: {seller_ratios['capacity_score'].tolist()}")
    
        if buyer_ratios.empty or seller_ratios.empty:
            print("No capacity data available for matching.")
            return pd.DataFrame()
        
        # Get buyer data (single row)
        buyer_row = buyer_ratios[buyer_ratios["cu_id"] == buyer_id]
        if buyer_row.empty:
            return pd.DataFrame()
        
        buyer = buyer_row.iloc[0]
        buyer_capacity = buyer["capacity_score"]
        
        # 1. Label buyer band
        def classify_capacity_band(score):
            if score >= 40: return "STRONG"
            elif score >= 11: return "MODERATE"
            else: return "WEAK"

        buyer_band = classify_capacity_band(buyer_capacity)
        print(f"🔍 Buyer {buyer_id}: {buyer_capacity:.1f} = {buyer_band}")
        
        valid_pairs = []
        
        # Loop through each seller (SIMPLE like activity loop)
        for seller_id in seller_ratios["cu_id"].unique():
            seller_row = seller_ratios[seller_ratios["cu_id"] == seller_id]
            seller = seller_row.iloc[0]
            seller_capacity = seller["capacity_score"]
            buyer = buyer_row.iloc[0]

            # Label seller band
            seller_band = classify_capacity_band(seller_capacity)
            print(f"🔍 Seller {seller_id}: {seller_capacity:.1f} = {seller_band}")
            
             # 3. Filter: reject only WEAK↔WEAK
            if buyer_band == "WEAK" and seller_band == "WEAK":
                print(f"  ❌ REJECTED: {buyer_id} ↔ {seller_id} (WEAK↔WEAK)")
                continue
            
             # 4. SIMPLE score = average capacity
            capacity_score = (buyer_capacity + seller_capacity)
        
            valid_pairs.append({
            "buyer_id": buyer_id,
            "seller_id": seller_id,
            "buyer_capacity": buyer_capacity,
            "seller_capacity": seller_capacity,
            "buyer_band": buyer_band,
            "seller_band": seller_band,
            "capacity_score": capacity_score,
            "match_type": f"{buyer_band}↔{seller_band}"
            })
            
            print(f"✅ {buyer_id} ↔ {seller_id}: {capacity_score:.1f} ({buyer_band}↔{seller_band})")
        
        capacity_df = pd.DataFrame(valid_pairs)
        self.matching_state["capacity_pairs"] = capacity_df
        return capacity_df


    # -------------------- 2.8 LTD band refinement ------------
    def apply_ltd_band_refinement(
        self,
        pairs_df: pd.DataFrame,
        ltd_lookup: Dict[str, float],
        buyer_label: str = "Caprock",
    ) -> pd.DataFrame:
        """
        LTD bands and pairing preferences.
        """
        df = pairs_df.copy()
        if df.empty:
            print("\n[Step 5: LTD Band Pairing]\nNo pairs to evaluate.")
            return df

        def _band(ltd):
            if ltd < 0.70:
                return "LOW"
            elif ltd <= 0.90:
                return "MID"
            else:
                return "HIGH"

        df["buyer_ltd"] = df["buyer_id"].map(lambda k: ltd_lookup.get(k, np.nan))
        df["seller_ltd"] = df["seller_id"].map(lambda k: ltd_lookup.get(k, np.nan))

        df = df.dropna(subset=["buyer_ltd", "seller_ltd"])
        if df.empty:
            print("\n[Step 5: LTD Band Pairing]\nLTD data missing for all pairs.")
            return df

        df["buyer_band"] = df["buyer_ltd"].apply(_band)
        df["seller_band"] = df["seller_ltd"].apply(_band)

        print("\n[Step 5: LTD Band Pairing]")

        kept_rows = []
        for _, row in df.iterrows():
            b_band = row["buyer_band"]
            s_band = row["seller_band"]

            if b_band == "LOW" and s_band == "HIGH":
                quality, bonus = "IDEAL", 25
            elif b_band == "MID" and s_band == "MID":
                quality, bonus = "GOOD", 20
            elif b_band == "LOW" and s_band == "MID":
                quality, bonus = "GOOD", 20
            elif b_band == "MID" and s_band == "LOW":
                quality, bonus = "MARGINAL", 10
            elif b_band == "HIGH" and s_band == "HIGH":
                print(
                    f"  - {buyer_label} ({b_band}) vs {row['seller_id']} ({s_band}): "
                    f"BLOCKED (0 points, down-ranked)"
                )
                continue
            else:
                quality, bonus = "MARGINAL", 0

            row["ltd_pair_quality"] = quality
            row["ltd_bonus"] = bonus
            kept_rows.append(row)

            print(
                f"  - {buyer_label} (LTD {self._pct(row['buyer_ltd'])}, {b_band}) "
                f"vs {row['seller_id']} (LTD {self._pct(row['seller_ltd'])}, {s_band}): "
                f"{quality} → +{bonus} points"
            )

        if not kept_rows:
            return pd.DataFrame(columns=df.columns)

        self.matching_state["ltd_pairs"] = pd.DataFrame(kept_rows).reset_index(drop=True)
        return self.matching_state["ltd_pairs"]

    # -------------------- 2.7 CU size band refinement --------
    def apply_size_band_refinement(
        self,
        pairs_df: pd.DataFrame,
        size_lookup: Dict[str, Tuple[float, str]],
        buyer_label: str = "Caprock",
    ) -> pd.DataFrame:
        """
        Size band compatibility scoring.
        """
        df = pairs_df.copy()
        if df.empty:
            print("\n[Step 6: CU Size Band Pairing]\nNo pairs to evaluate.")
            return df

        def _norm_band(band):
            if pd.isna(band):
                return None
            band = str(band).strip().lower()
            if band in {"small", "sm"}:
                return "Small"
            if band in {"lower mid", "lower_mid", "lower-mid"}:
                return "Lower Mid"
            if band in {"upper mid", "upper_mid", "upper-mid"}:
                return "Upper Mid"
            if band in {"large", "lg"}:
                return "Large"
            return band

        df["buyer_assets"] = df["buyer_id"].map(lambda k: size_lookup.get(k, (np.nan, None))[0])
        df["buyer_band"] = df["buyer_id"].map(lambda k: _norm_band(size_lookup.get(k, (np.nan, None))[1]))
        df["seller_assets"] = df["seller_id"].map(lambda k: size_lookup.get(k, (np.nan, None))[0])
        df["seller_band"] = df["seller_id"].map(lambda k: _norm_band(size_lookup.get(k, (np.nan, None))[1]))

        band_order = {"Small": 0, "Lower Mid": 1, "Upper Mid": 2, "Large": 3}

        df["buyer_band_idx"] = df["buyer_band"].map(band_order)
        df["seller_band_idx"] = df["seller_band"].map(band_order)
        df["band_diff"] = (df["buyer_band_idx"] - df["seller_band_idx"]).abs()

        print("\n[Step 6: CU Size Band Pairing]")

        kept_rows = []
        for _, row in df.iterrows():
            diff = row["band_diff"]

            if pd.isna(diff):
                quality, bonus = "MARGINAL", 5
            elif diff == 0:
                quality, bonus = "IDEAL", 20
            elif diff == 1:
                quality, bonus = "GOOD", 15
            elif diff == 2:
                quality, bonus = "ACCEPTABLE", 10
            else:
                quality, bonus = "MARGINAL", 5

            row["size_pair_quality"] = quality
            row["size_bonus"] = bonus
            kept_rows.append(row)

            print(
                f"  - {buyer_label} (${row['buyer_assets']/1_000_000:.1f}M, {row['buyer_band']}) "
                f"vs {row['seller_id']} (${row['seller_assets']/1_000_000:.1f}M, {row['seller_band']}): "
                f"{quality} → +{bonus} points"
            )

        self.matching_state["size_pairs"] = pd.DataFrame(kept_rows).reset_index(drop=True)
        return self.matching_state["size_pairs"]

    # -------------------- 2.8 GEO refinement -----------------
    def apply_geo_refinement(
        self,
        pairs_df: pd.DataFrame,
        state_lookup: Dict[str, str],
        buyer_label: str = "Caprock",
    ) -> pd.DataFrame:
        """
        GEO match levels.
        """
        df = pairs_df.copy()
        if df.empty:
            print("\n[Step 7: Geo-Based Matching]\nNo pairs to evaluate.")
            return df

        df["buyer_state"] = df["buyer_id"].map(state_lookup)
        df["seller_state"] = df["seller_id"].map(state_lookup)

        print("\n[Step 7: Geo-Based Matching]")

        kept_rows = []
        for _, row in df.iterrows():
            b = row["buyer_state"]
            s = row["seller_state"]
            if pd.isna(b) or pd.isna(s):
                continue

            if b == s:
                level = "STRONG GEO MATCH"
                bonus = 18
            elif s in self.config.geo_neighbors.get(b, []):
                level = "MEDIUM GEO MATCH"
                bonus = 10
            else:
                level = "WEAK GEO MATCH"
                bonus = 0

            row["geo_level"] = level
            row["geo_bonus"] = bonus
            kept_rows.append(row)

            print(
                f"  - {buyer_label} ({b}) vs {row['seller_id']} ({s}): "
                f"{level} → +{bonus} points"
            )

        if not kept_rows:
            return pd.DataFrame(columns=df.columns)

        self.matching_state["geo_pairs"] = pd.DataFrame(kept_rows).reset_index(drop=True)
        return self.matching_state["geo_pairs"]

    # -------------------- 2.9 Final ranking -----------------
    def build_ranked_matches(self, activity_pairs: pd.DataFrame, capacity_pairs: pd.DataFrame) -> pd.DataFrame:
        print("\n[Step 9: FINAL Combined Ranking]")

        # 1. Get all unique sellers
        all_sellers = pd.concat([
            activity_pairs[['seller_id']],
            capacity_pairs[['seller_id']]
        ]).drop_duplicates()

        # 2. MERGE scores from both paths (VECTORIZED!)
        df = all_sellers.merge(activity_pairs[['seller_id', 'activity_score', 'ltd_bonus', 'size_bonus', 'geo_bonus']], 
                            on='seller_id', how='left', suffixes=('', '_act'))
        df = df.merge(capacity_pairs[['seller_id', 'capacity_score', 'ltd_bonus', 'size_bonus', 'geo_bonus']], 
                    on='seller_id', how='left', suffixes=('', '_cap'))

        # 3. Calculate totals + final score
        df['activity_total'] = df['activity_score'].fillna(0) + df['ltd_bonus'].fillna(0) + df['size_bonus'].fillna(0) + df['geo_bonus'].fillna(0)
        df['capacity_total'] = df['capacity_score'].fillna(0) + df[['ltd_bonus', 'size_bonus', 'geo_bonus']].fillna(0).sum(axis=1)
        df['final_score'] = df['activity_total'] + df['capacity_total']

        # 4. Sort (FIXED)
        df = df.sort_values('final_score', ascending=False).reset_index(drop=True)

        # 5. Simple demo print
        print("Rank  Seller  Activity_Total  Capacity_Total  Final")
        print("-" * 50)
        for i, row in df.head(5).iterrows():
            print(f"{i+1}   {row['seller_id']:<8}   {row['activity_total']:>6.1f}     {row['capacity_total']:>6.1f}    {row['final_score']:>5.1f}")

        self.matching_state["final_ranking"] = df
        return df

    # -------------------- 2.10 PDF Report Generation (REVISED) --------
    def generate_pdf_report(
        self,
        ranked_matches: pd.DataFrame,
        output_path: str = "SellerBuyer_Matching_Report.pdf",
        buyer_label: str = "Caprock",
    ) -> str:
        """
        Generate a professional multi-page PDF report with:
        - Executive Summary with Top Recommendations
        - Detailed matching steps (Activity, LTD, Size, Geo)
        - Methodology & Justification
        - Regulatory alignment notes
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch,
        )

        story = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#003366'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#003366'),
            spaceAfter=8,
            spaceBefore=8,
            fontName='Helvetica-Bold',
        )

        subheading_style = ParagraphStyle(
            'SubHeading',
            parent=styles['Heading3'],
            fontSize=10,
            textColor=colors.HexColor('#003366'),
            spaceAfter=6,
            spaceBefore=4,
            fontName='Helvetica-Bold',
        )

        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=9,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        )

        header_white_style = ParagraphStyle(
            'HeaderWhite',
            parent=body_style,
            textColor=colors.whitesmoke,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER,
            fontSize=7,
        )

        # ===== PAGE 1: EXECUTIVE SUMMARY =====
        story.append(Paragraph(
            "LOAN PARTICIPATION SELLER-BUYER MATCHING REPORT",
            title_style
        ))
        story.append(Spacer(1, 0.15*inch))

        # Report metadata
        liquidity_info = self.matching_state.get("liquidity", {})
        as_of = datetime.now().strftime("%B %d, %Y")

        meta_data = [
            ["Report Generation Date:", as_of],
            ["Credit Union:", liquidity_info.get("label", buyer_label)],
            ["Role:", liquidity_info.get("role", "BUYER")],
            ["LTD Position:", self._pct(liquidity_info.get("ltd", 0))],
            ["Data Period (NCUA 5300):", f"Q2 2022 (June 30, 2022)"],
            ["Analysis Type:", "Opposite-Side Counterparty Matching"],
        ]
       
        meta_table = Table(meta_data, colWidths=[2.0*inch, 3.5*inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F0F8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 0.2*inch))

        # === TOP MATCHES (RECOMMENDATIONS) ===
        story.append(Paragraph("<b>1. TOP SELLER RECOMMENDATIONS</b>", heading_style))

        if not ranked_matches.empty:
            # Work only with this buyer's matches (report is per buyer)
            buyer_id = buyer_label
            buyer_matches = ranked_matches.copy()

            # Sort by raw final_score descending for ranking
            buyer_matches = buyer_matches.sort_values("final_score", ascending=False)

            # Define banded normalized scores by position (0-based index)
            band_values = [95, 75, 55, 35]  # Priority, Good, Acceptable, Marginal

            # Compute banded scores with ties based on final_score
            unique_scores = buyer_matches["final_score"].unique()
            score_to_band = {}
            for i, score in enumerate(unique_scores):
                band_idx = min(i, len(band_values) - 1)
                norm_val = band_values[band_idx]
                rank_num = i + 1  # 1-based rank
                score_to_band[score] = (norm_val, rank_num)

            norm_vals = []
            rank_nums = []
            for score in buyer_matches["final_score"]:
                nv, rnk = score_to_band[score]
                norm_vals.append(nv)
                rank_nums.append(rnk)

            buyer_matches["final_score_normalized"] = norm_vals
            buyer_matches["rank_num"] = rank_nums

            # Now take top 3 for the PDF
            top_matches = buyer_matches.head(3)

            rec_data = [[
                Paragraph("Rank", header_white_style),
                Paragraph("Seller CU", header_white_style),
                Paragraph("State", header_white_style),
                Paragraph("Activity", header_white_style),
                Paragraph("Capacity", header_white_style),
                Paragraph("Refinement<br/>Bonus", header_white_style),
                Paragraph("Final<br/>Score", header_white_style),
                Paragraph("Final Score (0-100) + Rank", header_white_style),
                Paragraph("Recommendation", header_white_style),
                ]]

            for idx, (_, row) in enumerate(top_matches.iterrows(), 1):
                seller = row["seller_id"]
                state = row.get("seller_state", "N/A")
                activity = f"{row['activity_score']:.0f}"
                capacity = f"{row.get('capacity_score', 0):.0f}"

                bonus_val = (
                    row.get("ltd_bonus", 0)
                    + row.get("size_bonus", 0)
                    + row.get("geo_bonus", 0)
                )
                bonus = f"+{bonus_val:.0f}"

                final_raw = row["final_score"]
                final_str = f"{final_raw:.0f}"

                norm_val = row["final_score_normalized"]
                rank_num = row["rank_num"]
                norm_str = f"{norm_val:.0f} (Rank {rank_num})"

                # Recommendation from normalized score
                if norm_val >= 80:
                    rec = "Priority Target"
                elif norm_val >= 60:
                    rec = "Good Fit"
                elif norm_val >= 40:
                    rec = "Acceptable"
                else:
                    rec = "Marginal"

                rec_data.append([
                    str(idx),
                    Paragraph(seller, body_style),
                    state,
                    activity,
                    capacity,
                    bonus,
                    final_str,
                    norm_str,
                    rec,
                ])

            # Optional: debug print to verify mapping
            print(
                buyer_matches[["seller_id", "final_score", "final_score_normalized"]]
                .head(10)
                .to_string()
                )
            
            rec_table = Table(
                rec_data,
                colWidths=[
                    0.45*inch,   # Rank
                    1.15*inch,   # Seller CU
                    0.50*inch,   # State
                    0.65*inch,   # Activity
                    0.65*inch,   # Capacity
                    0.80*inch,   # Refinement Bonus
                    0.80*inch,   # FINAL SCORE
                    0.90*inch,   # FINAL SCORE - NORMALIZED
                    1.00*inch,   # Recommendation
                ],
            )

            rec_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('TOPPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                        [colors.white, colors.HexColor('#F5F5F5')]),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                    ('LEFTPADDING', (0, 0), (-1, -1), 3),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                ]))

            story.append(rec_table)
        else:
            story.append(Paragraph("No qualifying matches found.", body_style))
        
        story.append(Spacer(1, 0.15*inch))

        # === SCORE CALCULATION METHODOLOGY (TEXT ONLY CHANGES) ===
        story.append(Paragraph("<b>2. FINAL SCORE CALCULATION METHODOLOGY</b>", heading_style))
        story.append(Paragraph(
            "<b>Final Score = Activity Base + Capacity Score + Refinement Bonuses (LTD/Size/Geo)</b><br/>"
            "• <b>Activity Base:</b> Weighted complementary transaction strengths across asset classes "
            "(0–100+ range)<br/>"
            "• <b>Capacity Score:</b> Balance sheet health across Liquidity, Leverage, Capital, Size (0-100) "
            "(0–100+ range)<br/>"
            "• <b>LTD Bonus:</b> Liquidity alignment premium (+0 to +25 points)<br/>"
            "• <b>Size Bonus:</b> Credit union size compatibility (+5 to +20 points)<br/>"
            "• <b>Geo Bonus:</b> State/Regional proximity bonus (+0 to +18 points)<br/>",
            body_style
        ))

        story.append(Paragraph(
            "<b>Score Interpretation - Final Seller-Buyer Match Score Normalized</b><br/>"
            "• <b>≥ 80:</b> Priority Target – Strong strategic fit across all dimensions<br/>"
            "• <b>60–79:</b> Good Fit – Solid complementarity with minor gaps<br/>"
            "• <b>40–59:</b> Acceptable – Viable but with notable misalignments<br/>"
            "• <b>&lt; 40:</b> Marginal – Limited fit; recommend lower priority",
            body_style
        ))

        story.append(Spacer(1, 0.15*inch))

        # === EXCLUSION CRITERIA (UNCHANGED) ===
        story.append(Paragraph("<b>3. EXCLUSION CRITERIA & RISK MITIGATION</b>", heading_style))
        story.append(Paragraph(
            "The matching engine <b>automatically rejects or down-ranks</b> counterparties based on "
            "regulatory and operational best practices:<br/>"
            "• <b>Activity Weakness:</b> Pairs where either buyer OR seller shows WEAK activity (score &lt; 40) "
            "are excluded to ensure both parties can effectively manage portfolio risk.<br/>"
            "• <b>LTD Mismatch (HIGH-HIGH):</b> High-LTD buyer paired with high-LTD seller is BLOCKED "
            "to avoid concentrated liquidity risk (per NCUA guidance).<br/>"
            "• <b>Geographic Over-Concentration:</b> Out-of-region partners receive 0 geo bonus, "
            "reflecting lower operational synergy and cross-collateral risk management.<br/>"
            "• <b>Size Disparity (2+ Bands):</b> Partnerships spanning 2+ size tiers receive minimal scoring "
            "to mitigate operational burden and monitoring complexity.",
            body_style
        ))

        story.append(PageBreak())

        # ===== PAGE 2: DETAILED MATCHING ANALYSIS =====
        story.append(Paragraph("<b>4. DETAILED MATCHING ANALYSIS BY STEP</b>", heading_style))

        # Step 1: Activity Profile Matching 
        story.append(Paragraph("<b>Step 1: Activity Profile Matching</b>", subheading_style))

        activity_pairs = self.matching_state.get("activity_pairs", pd.DataFrame())
        
        if not activity_pairs.empty:
            for _, pair_row in activity_pairs.iterrows():
                buyer_id = pair_row["buyer_id"]
                seller_id = pair_row["seller_id"]
                activity_detail = pair_row.get("activity_detail", "")
                combined_score = pair_row.get("activity_score", 0)

                # Parse activity details
                activity_lines = activity_detail.split("; ")
                
                match_para = Paragraph(f"{buyer_id} ↔ {seller_id}", body_style)
                pair_data = [["Qualified Match", match_para
                ]]

                pair_data.append(["Buyer", "Seller"])
                pair_data.append([buyer_id, seller_id])
                pair_data.append(["", ""])

                pair_data.append([
                    "Asset Class",
                    "Buyer Score\n(Normalized)",
                    "Buyer Band",
                    "Seller Score\n(Normalized)",
                    "Seller Band",
                    "Cumulative Score\n(Buyer+Seller-\nQualifying Asset\nClass Specific)"
                ])

                # Extract scores from activity lines
                for line in activity_lines:
                    # Parse: "Asset Class: STRENGTH ↔ STRENGTH (LEVEL)"
                    if ":" in line and "↔" in line:
                        parts = line.split(":")
                        asset = parts[0].strip()
                        rest = ":".join(parts[1:]).strip()
                        
                        # Extract strength bands
                        strength_parts = rest.split("(")
                        bands = strength_parts[0].strip()  # e.g., "STRONG ↔ MEDIUM"
                        level = strength_parts[1].replace(")", "").strip() if len(strength_parts) > 1 else ""
                        
                        buyer_band, seller_band = [b.strip() for b in bands.split("↔")]
                        
                        # Get scores (placeholder, in production use actual scores)
                        score_sum = 100
                        
                        # Wrap asset text and left-align later via style
                        asset_para = Paragraph(asset, body_style)
                        pair_data.append([asset_para, "50", buyer_band, "50", seller_band, str(score_sum)])

                # Summary row with cleaned label and bolded via style, not <b> tags
                pair_data.append([
                    "",
                    "",
                    "",
                    "Combined Activity Score(Qualifying Assets)",
                    "",
                    f"{combined_score:.1f}"
                ])

                pair_table = Table(
                    pair_data,
                    colWidths=[1.8*inch, 1.2*inch, 0.8*inch, 1.2*inch, 0.8*inch, 1.2*inch]
                )
                pair_table.setStyle(TableStyle([
                    # Row 0: Qualified Match + CU_CAPROCK_001 ↔ CU_ACU_001
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D3E4F8')),
                    ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#E8F0F8')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),

                    # Merge and left-align match id cell; wrap via Paragraph below
                    ('SPAN', (1, 0), (1, 0)),
                    ('ALIGN', (1, 0), (1, 0), 'LEFT'),

                    # General alignment
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

                    # Base font
                    ('FONTSIZE', (0, 0), (-1, -1), 7),

                    # Bold + larger font (size 8) for key labels:
                    # "Qualified Match" (0,0) and match id (1,0)
                    ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (1, 0), 8),

                    # "Buyer" and "Seller" row (row 1)
                    ('FONTNAME', (0, 1), (1, 1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 1), (1, 1), 8),

                    # Header row: Asset Class + score/band headers (row 4)
                    ('FONTNAME', (0, 4), (5, 4), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 4), (5, 4), 8),

                    # Combined Activity Score label (col 3, last row)
                    ('FONTNAME', (3, -1), (3, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (3, -1), (3, -1), 8),

                    # Numeric total (col 5, last row) – you already bolded this
                    ('FONTNAME', (5, -1), (5, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (5, -1), (5, -1), 8),

                    # Grid + padding
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('LEFTPADDING', (0, 0), (-1, -1), 2),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 2),
                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),

                    # Left-align asset class values
                    ('ALIGN', (0, 5), (0, -2), 'LEFT'),

                    # Row striping for body rows
                    ('ROWBACKGROUNDS', (0, 2), (-1, -2),
                        [colors.white, colors.HexColor('#F9F9F9')]),

                    # Highlight combined row
                    ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#F0D8D8')),
                ]))

                story.append(pair_table)
                story.append(Spacer(1, 0.06*inch))

            story.append(Paragraph(
            "<b>Buyer/Seller Scoring Logic (Per Asset Class):</b><br/>"
            "• <b>For BUYERS (per asset class):</b><br/>"
            "&nbsp;&nbsp;• Buyer_Score (Weighted) = purchased_outstanding × 0.6 "
            "+ amount_purchased_ytd × 0.4<br/>"
            "&nbsp;&nbsp;• Normalize score to 0–100 band (min–max scaling)<br/><br/>"
            "• <b>For SELLERS (per asset class):</b><br/>"
            "&nbsp;&nbsp;• Seller_Score (Weighted) = retained_balance × 0.5 "
            "+ amount_sold_ytd × 0.5<br/>"
            "&nbsp;&nbsp;• Normalize score to 0–100 band (min–max scaling)<br/><br/>"
            "<b>Buyer/Seller Categorization (Industry Benchmarking/Thresholds):</b><br/>"
            "• Score ≥ 70 → STRONG (experienced, active)<br/>"
            "• Score 40–69 → MEDIUM (baseline capability)<br/>"
            "• Score &lt; 40 → WEAK (dropped from matching)<br/><br/>"
            "<b>MATCHING QUALIFICATION CRITERION (PER ASSET CLASS):</b><br/>"
            "• <b>VALID PAIRINGS (per asset class):</b><br/>"
            "&nbsp;&nbsp;• Strong Buyer ↔ Strong Seller → Highest Priority<br/>"
            "&nbsp;&nbsp;• Strong Buyer ↔ Medium Seller → High Priority<br/>"
            "&nbsp;&nbsp;• Medium Buyer ↔ Strong Seller → High Priority<br/>"
            "&nbsp;&nbsp;• Medium Buyer ↔ Medium Seller → Optional<br/><br/>"
            "• <b>REJECTED PAIRINGS:</b><br/>"
            "&nbsp;&nbsp;• Any ↔ Weak Seller → Dropped<br/>"
            "&nbsp;&nbsp;• Weak Buyer ↔ Any → Dropped",
            body_style
        ))

        story.append(Spacer(1, 0.08*inch))


        # Step 2: Capacity Score Matching (NEW - CU LEVEL ONLY)
        story.append(Paragraph("<b>Step 2: Capacity Score Matching</b>", subheading_style))

        capacity_pairs = self.matching_state.get("capacity_pairs", pd.DataFrame())

        if not capacity_pairs.empty:
            for _, pair_row in capacity_pairs.iterrows():
                buyer_id = pair_row["buyer_id"]
                seller_id = pair_row["seller_id"]
                buyer_capacity = pair_row.get("buyer_capacity", 0)
                seller_capacity = pair_row.get("seller_capacity", 0)
                capacity_score = pair_row.get("capacity_score", 0)
                buyer_band = pair_row.get("buyer_band", "UNKNOWN")
                seller_band = pair_row.get("seller_band", "UNKNOWN")
                match_type = pair_row.get("match_type", "")

                # CU-LEVEL table (no asset classes)
                match_para = Paragraph(f"{buyer_id} ↔ {seller_id}", body_style)
                pair_data = [["Qualified Match", match_para]]

                pair_data.append(["Buyer", "Seller"])
                pair_data.append([buyer_id, seller_id])
                pair_data.append(["", ""])

                # Capacity header (CU-level metrics)
                pair_data.append([
                    "Metric",
                    "Buyer Score", 
                    "Buyer Band",
                    "Seller Score",
                    "Seller Band",
                    "Pair Score\n(Buyer+Seller)"
                ])

                # 4 Capacity metrics (liquidity, leverage, capital, size)
                for metric in ['liquidity', 'leverage', 'capital', 'size_score']:
                    buyer_metric = pair_row.get(f'buyer_{metric}', 0)
                    seller_metric = pair_row.get(f'seller_{metric}', 0)
                    
                    pair_data.append([
                        metric.replace('_', ' ').title(),
                        f"{buyer_metric:.1f}",
                        buyer_band,
                        f"{seller_metric:.1f}",
                        seller_band,
                        f"{buyer_metric + seller_metric:.1f}"
                    ])

                # Summary row
                pair_data.append([
                    "",
                    "",
                    "",
                    "Combined Capacity Score",
                    "",
                    f"{capacity_score:.1f}"
                ])

                # IDENTICAL Table styling as Activity section
                pair_table = Table(
                    pair_data,
                    colWidths=[1.8*inch, 1.2*inch, 0.8*inch, 1.2*inch, 0.8*inch, 1.2*inch]
                )
                pair_table.setStyle(TableStyle([  # SAME STYLE AS ACTIVITY TABLE
                    # Row 0: Qualified Match + CU_CAPROCK_001 ↔ CU_ACU_001
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D3E4F8')),
                    ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#E8F0F8')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),

                    # Merge and left-align match id cell
                    ('SPAN', (1, 0), (1, 0)),
                    ('ALIGN', (1, 0), (1, 0), 'LEFT'),

                    # General alignment
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

                    # Base font
                    ('FONTSIZE', (0, 0), (-1, -1), 7),

                    # Bold headers
                    ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (1, 0), 8),
                    ('FONTNAME', (0, 1), (1, 1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 1), (1, 1), 8),
                    ('FONTNAME', (0, 4), (5, 4), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 4), (5, 4), 8),

                    # Summary row bold
                    ('FONTNAME', (3, -1), (5, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (3, -1), (5, -1), 8),

                    # Grid + padding
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('LEFTPADDING', (0, 0), (-1, -1), 2),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 2),
                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),

                    # Row striping
                    ('ROWBACKGROUNDS', (0, 2), (-1, -2), [colors.white, colors.HexColor('#F9F9F9')]),
                    ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#F0D8D8')),
                ]))

                story.append(pair_table)
                story.append(Spacer(1, 0.06*inch))

            # Capacity scoring explanation (similar format)
            story.append(Paragraph(
                "<b>Buyer/Seller Capacity Scoring Logic (CU-Level):</b><br/>"
                "• <b>4 Metrics (25% each):</b> Liquidity, Leverage, Capital Adequacy, Size/Scale<br/>"
                "• <b>Scoring:</b> Raw financial ratios × 100 (no artificial caps)<br/>"
                "• <b>Bands:</b> STRONG ≥35, MODERATE 15-34, WEAK <15<br/><br/>"
                "<b>MATCHING:</b> All pairs except WEAK↔WEAK<br/>"
                "<b>Pair Score:</b> Buyer Capacity + Seller Capacity (Simple Sum)",
                body_style
            ))

            story.append(Spacer(1, 0.08*inch))


        # ---------- Step 3: LTD Band Pairing ----------
        story.append(Paragraph("<b>Step 2: Loan-to-Deposit (LTD) Band Pairing</b>", subheading_style))

        ltd_pairs = self.matching_state.get("ltd_pairs", pd.DataFrame())
        if not ltd_pairs.empty:
            ltd_data = [["Buyer", "Buyer LTD", "Buyer Band",
                         "Seller", "Seller LTD", "Seller Band",
                         "Quality", "Bonus"]]
            for _, r in ltd_pairs.iterrows():
                ltd_data.append([
                    r["buyer_id"],
                    self._pct(r["buyer_ltd"]),
                    r["buyer_band"],
                    r["seller_id"],
                    self._pct(r["seller_ltd"]),
                    r["seller_band"],
                    r["ltd_pair_quality"],
                    f"+{r['ltd_bonus']:.0f}",
                ])

            ltd_table = Table(
                ltd_data,
                colWidths=[0.9*inch, 0.8*inch, 0.7*inch,
                           0.9*inch, 0.8*inch, 0.7*inch,
                           0.8*inch, 0.6*inch],
            )
            ltd_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                    [colors.white, colors.HexColor('#F5F5F5')]),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('LEFTPADDING', (0, 0), (-1, -1), 2),
                ('RIGHTPADDING', (0, 0), (-1, -1), 2),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(ltd_table)
            story.append(Spacer(1, 0.1*inch))

        story.append(Paragraph(
            "<b>LTD Band Pairing - Industry Benchmarking:</b><br/>"
            "• LTD Bands: LOW (LTD &lt; 70%), MID (70–90%), HIGH (LTD &gt; 90%)<br/>"
            "<b>PAIRING PREFERENCES (per match):</b><br/>"
            "Low Buyer ↔ High Seller&nbsp;&nbsp;&nbsp;&nbsp;→ IDEAL (+25 points)<br/>"
            "Mid Buyer ↔ Mid Seller&nbsp;&nbsp;&nbsp;&nbsp;→ GOOD (+20 points)<br/>"
            "Low Buyer ↔ Mid Seller&nbsp;&nbsp;&nbsp;&nbsp;→ GOOD (+20 points)<br/>"
            "Mid Buyer ↔ Low Seller&nbsp;&nbsp;&nbsp;&nbsp;→ MARGINAL (+10 points)<br/>"
            "High Buyer ↔ High Seller&nbsp;&nbsp;&nbsp;&nbsp;→ BLOCKED (0 points, down-ranked)<br/>"
            "• Ideal Pairing: Low-Liquidity BUYER (needs funds) ↔ High-Liquidity SELLER (excess funds)<br/>"
            "• Risk Mitigation: HIGH-HIGH pairs are BLOCKED to prevent systemic liquidity crises.",
            body_style
        ))

        story.append(Spacer(1, 0.1*inch))

        # ---------- Step 3: Size Band Compatibility ----------
        story.append(Paragraph("<b>Step 3: Credit Union - Asset Size Band Compatibility</b>", subheading_style))

        size_pairs = self.matching_state.get("size_pairs", pd.DataFrame())
        if not size_pairs.empty:
            size_data = [["Buyer", "Buyer Assets", "Buyer Band",
                          "Seller", "Seller Assets", "Seller Band",
                          "Quality", "Bonus"]]
            for _, r in size_pairs.iterrows():
                size_data.append([
                    r["buyer_id"],
                    f"${r['buyer_assets']/1_000_000:.1f}M",
                    r["buyer_band"],
                    r["seller_id"],
                    f"${r['seller_assets']/1_000_000:.1f}M",
                    r["seller_band"],
                    r["size_pair_quality"],
                    f"+{r['size_bonus']:.0f}",
                ])

            size_table = Table(
                size_data,
                colWidths=[0.9*inch, 0.9*inch, 0.7*inch,
                           0.9*inch, 0.9*inch, 0.7*inch,
                           0.8*inch, 0.6*inch],
            )
            size_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                    [colors.white, colors.HexColor('#F5F5F5')]),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('LEFTPADDING', (0, 0), (-1, -1), 2),
                ('RIGHTPADDING', (0, 0), (-1, -1), 2),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(size_table)
            story.append(Spacer(1, 0.1*inch))

        story.append(Paragraph(
            "<b>Asset Size Band - Industry Benchmarking:</b><br/>"
            "• Size Bands: Small (&lt; $50M) → Lower Mid ($50M–$200M) → Upper Mid ($200M–$1B) → Large (&gt; $1B)<br/>"
            "• Same Band = IDEAL (+20 pts): Similar operational complexity and risk management maturity.<br/>"
            "• Adjacent Band = GOOD (+15 pts): Manageable scale differences.<br/>"
            "• 2+ Bands Apart = MARGINAL (+5 pts): Significant operational & governance misalignment.",
            body_style
        ))

        story.append(Spacer(1, 0.1*inch))

        # ---------- Step 4: Geographic Proximity ----------
        story.append(Paragraph("<b>Step 4: Geographic (State-Based) Proximity</b>", subheading_style))

        geo_pairs = self.matching_state.get("geo_pairs", pd.DataFrame())
        if not geo_pairs.empty:
            geo_data = [["Buyer", "Buyer State", "Seller", "Seller State", "Geo Level", "Bonus"]]
            for _, r in geo_pairs.iterrows():
                geo_data.append([
                    r["buyer_id"],
                    r["buyer_state"],
                    r["seller_id"],
                    r["seller_state"],
                    r["geo_level"],
                    f"+{r['geo_bonus']:.0f}",
                ])

            geo_table = Table(
                geo_data,
                colWidths=[1.0*inch, 0.8*inch, 1.0*inch, 0.8*inch, 1.5*inch, 0.6*inch],
            )
            geo_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                    [colors.white, colors.HexColor('#F5F5F5')]),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('LEFTPADDING', (0, 0), (-1, -1), 2),
                ('RIGHTPADDING', (0, 0), (-1, -1), 2),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(geo_table)
            story.append(Spacer(1, 0.1*inch))

        story.append(Paragraph(
            "<b>Geo Based - Matching + Scoring Logic:</b><br/>"
            "• Same State (+18 pts): Strong operational synergy, shared member base understanding, local regulatory alignment.<br/>"
            "• Neighboring State (+10 pts): Moderate synergy with some cross-border complexities.<br/>"
            "• Out-of-Region (0 pts): Minimal geographic advantage but not excluded; may fit niche strategies.",
            body_style
        ))

        story.append(PageBreak())

        # ===== PAGE 3: REGULATORY CONTEXT & NEXT STEPS =====
        story.append(Paragraph("<b>5. REGULATORY & BEST PRACTICES ALIGNMENT</b>", heading_style))
        story.append(Paragraph(
            "This matching framework aligns with <b>NCUA guidance</b> (Evaluating Loan Participation Programs) "
            "and <b>ACU best practices</b> on loan participation due diligence:<br/>"
            "• <b>Risk Assessment:</b> Multi-dimensional scoring prevents single-point failures.<br/>"
            "• <b>Liquidity Stress Testing:</b> LTD band logic reflects regulatory concern for systemic risk.<br/>"
            "• <b>Operational Capability:</b> Size band matching ensures comparable credit administration and monitoring capacity.<br/>"
            "• <b>Geographic Diversification:</b> Bonus structure encourages prudent geographic spread while allowing "
            "regional clustering for operational efficiency.",
            body_style
        ))

        story.append(Spacer(1, 0.15*inch))

        story.append(Paragraph("<b>7. RECOMMENDED NEXT STEPS</b>", heading_style))
        story.append(Paragraph(
            "1. <b>Schedule Meetings:</b> Engage top-ranked sellers to assess cultural fit and operational alignment.<br/>"
            "2. <b>Legal Review:</b> Obtain copies of seller's participation agreements, audit reports, and past performance data "
            "(per NCUA guidelines).<br/>"
            "3. <b>Credit Analysis:</b> Conduct independent underwriting review of sample loans held by seller.<br/>"
            "4. <b>Board Approval:</b> Present matching results with risk/return justification to board for strategic alignment.<br/>"
            "5. <b>Ongoing Monitoring:</b> Establish quarterly performance dashboards tracking delinquency rates, "
            "collateral valuations, and cash flow metrics.",
            body_style
        ))

        doc.build(story)
        print(f"\n✓ PDF Report generated: {output_path}")
        return output_path