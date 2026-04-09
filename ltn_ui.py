import streamlit as st
import pandas as pd
from LTN_loan_tape_prep import LTN_loan_tape_prep

# Page config - Professional finance layout
st.set_page_config(
    layout="wide", 
    page_title="LTN Loan Tape Analyzer | ALM First",
    page_icon="🏦",
    initial_sidebar_state="expanded"
)

# 🔥 MOODY'S/S&P LEVEL CSS - Production finance UI
st.markdown("""
<style>
/* Professional Moody's/S&P finance dashboard */
.stApp { 
    background: linear-gradient(135deg, #0a0e17 0%, #1a1f2e 100%);
    color: #e8ecef;
}
.main { 
    padding: 2rem; 
    background: rgba(10,14,23,0.97); 
    border-radius: 16px; 
    backdrop-filter: blur(10px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.3);
}
h1 { 
    color: #00d4ff !important; 
    text-shadow: 0 0 30px rgba(0,212,255,0.4);
    font-family: 'Arial Black', sans-serif !important;
    font-size: 2.5rem;
}
.stMetric > label { color: #b8bcc5 !important; font-weight: 600; }
.stMetric > div > div > div > div { 
    color: #00d4aa !important; 
    font-size: 2.2rem; 
    font-weight: bold;
    text-shadow: 0 0 10px rgba(0,212,170,0.3);
}
.stButton > button { 
    background: linear-gradient(45deg, #0066cc, #00d4aa) !important;
    color: white !important; 
    border-radius: 12px !important; 
    border: none !important; 
    height: 3.5rem !important;
    font-weight: bold !important;
    box-shadow: 0 8px 25px rgba(0,102,204,0.3) !important;
    font-size: 1.1rem;
}
.stButton > button:hover {
    background: linear-gradient(45deg, #00d4aa, #0066cc) !important;
    box-shadow: 0 12px 35px rgba(0,102,204,0.4) !important;
}
.sidebar .sidebar-content { 
    background: linear-gradient(180deg, #1a1f2e 0%, #2a2f3e 100%) !important;
}
.stDataFrame { background-color: #1e1e1e; }
.stExpander > div > label { color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

# 🎨 PROFESSIONAL HEADER with FIXED ALM LOGO
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.empty()
with col2:
    st.markdown("# 🏦 **LTN Loan Tape Analyzer**")
    st.markdown("*Advanced Portfolio Matching Engine* | ALM First Financial Advisors")
with col3:
    st.markdown("""
    <div style='text-align: right; padding-top: 0.5rem;'>
        <img src='./alm_logo.png' width='160' height='80' alt='ALM First' 
             onerror="this.style.display='none'">
        <div style='font-size: 0.8rem; color: #b8bcc5; margin-top: 0.2rem;'>Financial Advisors</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Enterprise file upload
uploaded_file = st.file_uploader(
    "📁 **Upload LTN Loan Tape CSV**", 
    type="csv",
    help="Standard format: FICO, LTV, DTI, term, balance, product_type/auto_new_used"
)

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.success(f"✅ **Pipeline Complete** | {len(df_raw):,} loans processed")
    
    # Full LTN data pipeline
    prep = LTN_loan_tape_prep(df_raw=df_raw)
    prep.column_standardization().value_format_standardization() \
        .data_quality_check().missing_value_impute().outlier_removal()
    
    box_df = prep.get_loan_box_definitions()
    
    if box_df.empty:
        st.error("❌ **Loan Box Configuration Error**")
    else:
        st.success("✅ **Portfolio filtering enabled**")
        
        # Enterprise sidebar
        with st.sidebar:
            st.header("🔧 **Portfolio Criteria**")
            st.markdown("─" * 40)
            
            products = sorted(box_df['product_type'].dropna().unique())
            selected_products = st.multiselect(
                "🎯 **Product Selection**", options=products, 
                default=products, help="AUTO_NEW, AUTO_USED, AUTO_ALL"
            )
            
            fico_min, fico_max = st.slider(
                "📊 **FICO Spectrum**",
                min_value=int(box_df['fico_min'].min()) if 'fico_min' in box_df.columns else 300,
                max_value=int(box_df['fico_max'].max()) if 'fico_max' in box_df.columns else 850,
                value=(670, 749)
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                ltv_max = st.slider("📈 **LTV Ceiling (%)**", 0.0, 200.0, 115.0, 5.0)
                term_max = st.slider("⏱️ **Maturity Cap (mo)**", 0, 360, 96, 12)
            with col_b:
                dti_max = st.slider("⚖️ **DTI Threshold (%)**", 0.0, 100.0, 45.0, 5.0)
                balance_max = st.slider("💰 **Balance Ceiling ($)**", 0, 1000000, 150000, 25000)
            
            st.markdown("─" * 40)
            
            if st.button("🚀 **EXECUTE PORTFOLIO MATCH**", type="primary", use_container_width=True):
                with st.spinner("🔄 Loan processing → Filter application → Box assignment..."):
                    ui_filters = {
                        "product": selected_products,
                        "fico_min": fico_min, "fico_max": fico_max,
                        "ltv_max": ltv_max, "dti_max": dti_max,
                        "term_max": term_max, "orig_balance_max": balance_max
                    }
                    prep.apply_ui_filters(ui_filters)
                    st.session_state.final_df = prep.loan_box_assignment()
                    st.rerun()
        
        # Executive dashboard
        if 'final_df' not in st.session_state:
            st.session_state.final_df = None
        
        if st.session_state.final_df is not None and len(st.session_state.final_df) > 0:
            st.markdown("## 📊 **Executive Portfolio Summary**")
            
            final_df = st.session_state.final_df
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1: st.metric("**Portfolio Size**", f"{len(final_df):,}")
            with col2: st.metric("**WA FICO**", f"{final_df['orig_fico'].mean():.0f}")
            with col3: st.metric("**WA LTV**", f"{final_df['orig_ltv'].mean():.1f}%")
            with col4: st.metric("**WA DTI**", f"{final_df['dti_back_end'].mean():.1f}%")
            with col5: st.metric("**AUM ($MM)**", f"${final_df['original_balance'].sum()/1e6:.1f}")
            
            st.divider()
            st.markdown("## 📋 **Qualified Loan Assignments**")
            st.dataframe(final_df, use_container_width=True)
            
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "💾 **Export ALM First Loan Box**",
                csv, "ALM_First_LTN_Portfolio.csv", "text/csv",
                use_container_width=True
            )
        elif st.session_state.final_df is not None:
            st.warning("⚠️ **No qualifying loans** | Expand filter criteria")
        
        # Transparency expanders
        c1, c2 = st.columns(2)
        with c1:
            with st.expander("🔍 **Input Validation**"):
                st.dataframe(df_raw.head())
        with c2:
            with st.expander("📈 **Box Methodology**"):
                st.dataframe(box_df)
        
        st.markdown("---")
        st.caption("*Powered by ALM First LTN Pipeline Engine*")

else:
    st.info("👈 **Upload CSV → Filter → EXECUTE → Export portfolio**")