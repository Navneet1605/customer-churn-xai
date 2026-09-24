"""
app.py
======
Interactive Streamlit Dashboard for Explainable Customer Churn Prediction
"""

import os
import sys
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Append project root to sys.path to allow imports if needed
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# Page configuration
st.set_page_config(
    page_title="Churn Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS & Suppress Chrome
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Suppress default Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Base container padding */
.block-container {
    padding: 3rem 4rem !important;
    max-width: 1400px;
}

/* Global Typography & Colors */
html, body, [class*="css"] {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

:root {
    --bg-color: #050505;
    --surface-color: #0a0a0a;
    --border-color: #1a2e23;
    --text-primary: #ecfdf5;
    --text-muted: #a7f3d0;
    --accent: #10b981;
    --accent-hover: #34d399;
    --glow-color: rgba(52, 211, 153, 0.4);
}

/* Hide Sidebar Completely */
[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
}

/* Top Navigation Pills */
.stRadio > div[role="radiogroup"] {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 12px;
    padding-bottom: 2rem;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 2rem;
}
.stRadio > div[role="radiogroup"] > label {
    background: var(--surface-color);
    border: 1px solid var(--border-color);
    padding: 10px 24px;
    border-radius: 30px;
    transition: all 0.3s ease;
    cursor: pointer;
    margin: 0 !important;
}
.stRadio > div[role="radiogroup"] > label:hover {
    box-shadow: 0 0 15px var(--glow-color);
    border-color: var(--accent-hover);
    transform: translateY(-2px);
    color: var(--text-primary);
}
/* Hide the actual radio circle to make it look like a pure button/pill */
.stRadio > div[role="radiogroup"] label > div:first-child {
    display: none;
}

/* Metric card styling */
.metric-card {
    background: linear-gradient(145deg, #0f1712 0%, #050505 100%);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 0 20px var(--glow-color);
    border-color: var(--accent-hover);
}

.metric-title {
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
}

.metric-value {
    font-size: 2.25rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.1;
}

/* Input Widgets Override */
div[data-baseweb="select"] > div {
    background-color: var(--bg-color);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    transition: all 0.3s ease;
}
div[data-baseweb="select"]:hover > div, div[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent-hover);
    box-shadow: 0 0 15px var(--glow-color);
}
.stSlider div[data-baseweb="slider"] div[role="slider"] {
    background-color: var(--text-primary);
}
.stSlider div[data-baseweb="slider"] div[data-testid="stTickBar"] > div {
    background-color: var(--accent);
}

/* Content Blocks & Callouts */
.insight-box {
    background: rgba(16, 185, 129, 0.15);
    border-left: 3px solid var(--accent-hover);
    border-radius: 6px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    color: var(--text-primary);
    font-weight: 300;
    line-height: 1.6;
    letter-spacing: 0.02em;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    border-bottom: 1px solid var(--border-color);
}
.stTabs [data-baseweb="tab"] {
    padding-bottom: 1rem;
    color: var(--text-muted);
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--text-primary) !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-color);
    border-radius: 8px;
    overflow: hidden;
}

/* Titles and Headers */
h1 {
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    margin-bottom: 2rem !important;
    color: #ffffff;
}
h2, h3 {
    font-weight: 600 !important;
    letter-spacing: -0.01em;
    color: #e2e8f0;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper Data Loaders (Cached for high performance)
# ---------------------------------------------------------
@st.cache_data
def load_eda_insights():
    p = BASE_DIR / "reports" / "eda_and_business_insights.json"
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_model_evaluation():
    p = BASE_DIR / "reports" / "model_evaluation_metrics.json"
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_shap_insights():
    p = BASE_DIR / "reports" / "shap_explanations.json"
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return {}

@st.cache_resource
def load_best_model_and_explainer():
    """Loads champion model pipeline and serialized SHAP explainer."""
    model_meta = load_model_evaluation()
    best_file = model_meta.get("best_model_file")
    pipeline = None
    if best_file:
        # best_file paths in json are relative to customer-churn-xai, so we can use BASE_DIR
        best_path = Path(best_file)
        if not best_path.is_absolute():
             best_path = BASE_DIR / best_path
        if best_path.exists():
            with open(best_path, "rb") as f:
                pipeline = pickle.load(f)

    explainer_path = BASE_DIR / "models" / "shap_explainer.pkl"
    explainer_obj = None
    if explainer_path.exists():
        with open(explainer_path, "rb") as f:
            explainer_obj = pickle.load(f)

    return pipeline, explainer_obj

@st.cache_data
def load_processed_sample_data():
    p = BASE_DIR / "data" / "processed" / "engineered_telco.parquet"
    if p.exists():
        return pd.read_parquet(p)
    raw_p = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if raw_p.exists():
        return pd.read_csv(raw_p)
    return pd.DataFrame()


# Load datasets & artifacts
eda_data = load_eda_insights()
model_eval = load_model_evaluation()
shap_data = load_shap_insights()
champion_pipeline, live_explainer = load_best_model_and_explainer()
df_sample = load_processed_sample_data()


# ---------------------------------------------------------
# Top Navigation
# ---------------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="margin-bottom: 0 !important; color: var(--accent); font-size: 3rem;">Churn Intelligence</h1>
        <p style="color: var(--text-muted); font-size: 1.1rem; letter-spacing: 0.05em;">Distributed Pipeline and Model Explainability</p>
    </div>
    """, unsafe_allow_html=True
)

page_selection = st.radio(
    "Navigation",
    [
        "Executive Overview",
        "Data Overview & Schema",
        "Exploratory Analytics",
        "Model Performance & ROC",
        "Live Churn Inference & Explainability",
        "Batch Prediction",
        "Strategic Insights & Reports"
    ],
    horizontal=True,
    label_visibility="collapsed"
)


# =========================================================
# PAGE 1: Executive Overview
# =========================================================
if page_selection == "Executive Overview":
    st.title("Executive Overview")
    st.subheader("High-Level Churn Metrics")
    
    kpis = eda_data.get("kpis", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    def metric_card(title, value):
        return f'<div class="metric-card"><div class="metric-title">{title}</div><div class="metric-value">{value}</div></div>'
        
    with col1:
        st.markdown(metric_card("Total Customer Base", f"{kpis.get('total_customers', 7043):,}"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Baseline Churn Rate", f"{kpis.get('churn_rate_pct', 26.54):.2f}%"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("Monthly Revenue at Risk", f"${kpis.get('monthly_revenue_at_risk', 139130):,.0f}"), unsafe_allow_html=True)
    with col4:
        best_auc = model_eval.get("best_metrics", {}).get("ROC_AUC", 0.8465)
        st.markdown(metric_card("Champion Model AUC", f"{best_auc:.4f}"), unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    with st.container():
        st.subheader("Architectural Flow")
        st.markdown(
            """
            <div class="insight-box" style="font-family: monospace; font-size: 0.95rem; text-align: center; font-weight: 500;">
                Data Ingestion (HDFS/Parquet) &nbsp;➔&nbsp; PySpark Preprocessing &nbsp;➔&nbsp; Spark MLlib (LR, RF, GBT) &nbsp;➔&nbsp; SHAP Interpretation &nbsp;➔&nbsp; Decision Dashboard
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container():
        st.subheader("Project Objectives")
        st.markdown("""
        *   Identify high-risk subscribers weeks prior to contract termination for proactive retention.
        *   Eliminate black-box opacity so customer success agents understand exactly why a customer intends to leave.
        *   Preserve revenue streams by targeting retention budgets exclusively on salvageable, high-LTV accounts.
        *   Provide real-time operational scoring and automated batch CSV prediction.
        """)


# =========================================================
# PAGE 2: Data Overview & Schema
# =========================================================
elif page_selection == "Data Overview & Schema":
    st.title("Data Overview & Schema")

    if not df_sample.empty:
        st.caption(f"Ingested Dataset Shape: {df_sample.shape[0]:,} rows x {df_sample.shape[1]} columns")
        
        tab_raw, tab_schema, tab_dist = st.tabs(["Data Viewer", "Schema & Quality Audit", "Storage Architecture"])
        
        with tab_raw:
            st.dataframe(df_sample.head(50), use_container_width=True)
            
        with tab_schema:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("#### Column Data Types & Unique Values")
                dtype_df = pd.DataFrame({
                    "Column": df_sample.columns,
                    "Data Type": [str(t) for t in df_sample.dtypes],
                    "Unique Count": [df_sample[c].nunique() for c in df_sample.columns],
                    "Missing Count": [df_sample[c].isna().sum() for c in df_sample.columns]
                })
                st.dataframe(dtype_df, use_container_width=True)
                
            with col_s2:
                st.markdown("#### Numerical Summary Statistics")
                num_cols = df_sample.select_dtypes(include=[np.number]).columns
                st.dataframe(df_sample[num_cols].describe().T.round(2), use_container_width=True)
                
        with tab_dist:
            st.markdown("#### HDFS Distributed Parquet Architecture")
            st.code("""
data/hdfs_simulation/warehouse/telco/
├── Contract=Month-to-month/
│   └── part-00000.parquet
├── Contract=One year/
│   └── part-00001.parquet
└── Contract=Two year/
    └── part-00002.parquet
            """, language="text")
            st.caption("Simulates an enterprise Hadoop / Spark data lake partitioned by customer contractual commitment.")
    else:
        st.warning("Processed dataset not found.")


# =========================================================
# PAGE 3: Exploratory Data Analysis (EDA)
# =========================================================
elif page_selection == "Exploratory Analytics":
    st.title("Exploratory Analytics")

    breakdowns = eda_data.get("breakdowns", {})

    with st.container():
        r1_col1, r1_col2 = st.columns(2)
        with r1_col1:
            churn_dist = eda_data.get("kpis", {})
            fig_donut = go.Figure(data=[go.Pie(
                labels=['Retained', 'Churned'],
                values=[churn_dist.get('retained_count', 5174), churn_dist.get('churn_count', 1869)],
                hole=.55,
                marker_colors=['#10b981', '#fb7185']
            )])
            fig_donut.update_layout(title="Overall Churn Ratio", height=380, template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_donut, use_container_width=True)

        with r1_col2:
            contract_data = pd.DataFrame(breakdowns.get("by_contract", []))
            if not contract_data.empty:
                fig_contract = px.bar(
                    contract_data,
                    x="Contract",
                    y=["Retained", "Churned"],
                    title="Retention by Contract Type",
                    barmode="group",
                    color_discrete_sequence=['#10b981', '#fb7185'],
                    template="plotly_dark",
                    height=380
                )
                fig_contract.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_contract, use_container_width=True)

    with st.container():
        r2_col1, r2_col2 = st.columns(2)
        with r2_col1:
            internet_data = pd.DataFrame(breakdowns.get("by_internet_service", []))
            if not internet_data.empty:
                fig_internet = px.bar(
                    internet_data,
                    y="InternetService",
                    x="ChurnRate",
                    orientation='h',
                    title="Churn Distribution by Internet Service",
                    color_discrete_sequence=['#10b981'],
                    template="plotly_dark",
                    height=380
                )
                fig_internet.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_internet, use_container_width=True)

        with r2_col2:
            payment_data = pd.DataFrame(breakdowns.get("by_payment_method", []))
            if not payment_data.empty:
                fig_payment = px.bar(
                    payment_data,
                    y="PaymentMethod",
                    x="ChurnRate",
                    orientation='h',
                    title="Churn Distribution by Payment Method",
                    color_discrete_sequence=['#10b981'],
                    template="plotly_dark",
                    height=380
                )
                fig_payment.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_payment, use_container_width=True)


# =========================================================
# PAGE 4: Model Benchmarking & Performance
# =========================================================
elif page_selection == "Model Performance & ROC":
    st.title("Model Performance & ROC")

    metrics_list = model_eval.get("metrics_summary", [])
    if metrics_list:
        m_df = pd.DataFrame(metrics_list)
        
        with st.container():
            st.markdown("### Model Leaderboard")
            st.dataframe(
                m_df[["Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "PR_AUC"]],
                use_container_width=True
            )

        with st.container():
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("#### Benchmark ROC Curve")
                fig_path = BASE_DIR / "reports" / "figures" / "roc_curves_comparison.png"
                if fig_path.exists():
                    st.image(str(fig_path), use_container_width=True)
                
            with col_m2:
                st.markdown("#### Confusion Matrices Benchmark")
                cm_path = BASE_DIR / "reports" / "figures" / "confusion_matrices_comparison.png"
                if cm_path.exists():
                    st.image(str(cm_path), use_container_width=True)
                
        with st.container():
            best_name = model_eval.get("best_model_name", "Gradient Boosted Trees")
            st.markdown(
                f"""
                <div style="padding: 1rem; background-color: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981; border-radius: 4px; color: #f8fafc; font-weight: 500;">
                    Active Champion Model: {best_name}
                </div>
                """, unsafe_allow_html=True
            )
    else:
        st.warning("Model evaluation metrics not found.")


# =========================================================
# PAGE 5: Live XAI Churn Predictor
# =========================================================
elif page_selection == "Live Churn Inference & Explainability":
    st.title("Live Churn Inference & Explainability")

    if champion_pipeline is None or live_explainer is None:
        st.error("Champion model or SHAP explainer not loaded.")
    else:
        with st.container():
            col_c1, col_c2 = st.columns(2)

            with col_c1:
                st.markdown("#### Customer Attributes")
                f_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                f_payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                f_internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                f_tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=4)
                f_monthly = st.slider("Monthly Charges", min_value=18.0, max_value=120.0, value=89.5, step=0.5)

                # Supplementary fields to meet schema requirements for prediction
                f_gender = st.selectbox("Gender", ["Female", "Male"])
                f_senior = st.selectbox("Senior Citizen", [0, 1])
                f_partner = st.selectbox("Has Partner", ["Yes", "No"])
                f_dependents = st.selectbox("Has Dependents", ["No", "Yes"])
                f_phone = st.selectbox("Phone Service", ["Yes", "No"])
                f_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                f_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                f_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                f_device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                f_tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                f_streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                f_streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                f_paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

            with col_c2:
                st.markdown("#### Real-Time Inference Output")
                total_charges_est = round(f_monthly * f_tenure, 2)
                avg_monthly_spend_est = round(total_charges_est / (f_tenure + 1), 2)
                contract_risk_score_est = 3 if "month" in f_contract.lower() else (2 if "one" in f_contract.lower() else 1)
                
                svc_count = 0
                if f_phone == "Yes": svc_count += 1
                if f_lines == "Yes": svc_count += 1
                if f_internet in ["DSL", "Fiber optic"]: svc_count += 1
                for v in [f_security, f_backup, f_device_prot, f_tech_support, f_streaming_tv, f_streaming_movies]:
                    if v == "Yes": svc_count += 1
                    
                support_usage = 1 if (f_tech_support == "Yes" or f_security == "Yes" or f_backup == "Yes") else 0
                premium_flag = 1 if f_monthly > 80.0 else 0
                long_term_flag = 1 if f_tenure > 36 else 0
                echeck_risk = 1 if "electronic check" in f_payment.lower() else 0
                
                tenure_bucket_val = "New" if f_tenure <= 12 else ("Developing" if f_tenure <= 24 else ("Stable" if f_tenure <= 48 else "Loyal"))

                input_dict = {
                    "gender": f_gender, "SeniorCitizen": f_senior, "Partner": f_partner, "Dependents": f_dependents,
                    "PhoneService": f_phone, "MultipleLines": f_lines, "InternetService": f_internet,
                    "OnlineSecurity": f_security, "OnlineBackup": f_backup, "DeviceProtection": f_device_prot,
                    "TechSupport": f_tech_support, "StreamingTV": f_streaming_tv, "StreamingMovies": f_streaming_movies,
                    "Contract": f_contract, "PaperlessBilling": f_paperless, "PaymentMethod": f_payment,
                    "tenure_bucket": tenure_bucket_val, "tenure": f_tenure, "MonthlyCharges": f_monthly,
                    "TotalCharges": total_charges_est, "avg_monthly_spend": avg_monthly_spend_est,
                    "contract_risk_score": contract_risk_score_est, "service_count": svc_count,
                    "support_usage_indicator": support_usage, "premium_customer_flag": premium_flag,
                    "long_term_customer_flag": long_term_flag, "electronic_check_risk": echeck_risk
                }

                input_df = pd.DataFrame([input_dict])

                churn_prob = float(champion_pipeline.predict_proba(input_df)[0, 1])

                churn_pct = churn_prob * 100
                if churn_pct < 30:
                    risk_level = "LOW RISK"
                    gauge_color = "#10b981"
                elif churn_pct < 60:
                    risk_level = "MODERATE RISK"
                    gauge_color = "#f59e0b"
                else:
                    risk_level = "HIGH RISK"
                    gauge_color = "#f43f5e"

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=churn_pct,
                    number={"suffix": "%", "font": {"size": 32, "color": "#ffffff", "family": "Inter, sans-serif"}, "valueformat": ".1f"},
                    title={"text": f"<b>CHURN RISK DIAL</b><br><span style='color:{gauge_color}; font-size: 14px;'>Status: {risk_level}</span>", "font": {"size": 16, "color": "#94a3b8"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569", "tickfont": {"color": "#94a3b8"}},
                        "bar": {"color": gauge_color, "thickness": 0.3},
                        "bgcolor": "rgba(15, 23, 42, 0.6)",
                        "borderwidth": 1,
                        "bordercolor": "#334155",
                        "steps": [
                            {"range": [0, 30], "color": "rgba(16, 185, 129, 0.2)"},
                            {"range": [30, 60], "color": "rgba(245, 158, 11, 0.2)"},
                            {"range": [60, 100], "color": "rgba(244, 63, 94, 0.2)"}
                        ],
                        "threshold": {
                            "line": {"color": "#ffffff", "width": 3},
                            "thickness": 0.8,
                            "value": churn_pct
                        }
                    }
                ))

                fig_gauge.update_layout(
                    height=240,
                    margin=dict(l=25, r=25, t=55, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "white"}
                )

                st.plotly_chart(fig_gauge, use_container_width=True)

                preproc = live_explainer["preprocessor"]
                explainer = live_explainer["explainer"]
                readable_names = live_explainer["readable_feature_names"]

                X_single_trans = preproc.transform(input_df)
                raw_shap = explainer.shap_values(X_single_trans)

                if isinstance(raw_shap, list) and len(raw_shap) == 2:
                    single_shap = raw_shap[1][0]
                elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
                    single_shap = raw_shap[0, :, 1]
                elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 2:
                    single_shap = raw_shap[0]
                else:
                    single_shap = raw_shap

                paired_shap = list(zip(readable_names, single_shap))
                top_contributors = sorted(paired_shap, key=lambda x: abs(x[1]), reverse=True)[:8]
                top_feats_df = pd.DataFrame(top_contributors, columns=["Feature", "SHAP Impact"])
                top_feats_df["Impact Direction"] = top_feats_df["SHAP Impact"].apply(lambda v: "Positive Churn" if v > 0 else "Negative Churn")
                
                fig_shap_bar = px.bar(
                    top_feats_df,
                    x="SHAP Impact",
                    y="Feature",
                    orientation='h',
                    color="Impact Direction",
                    color_discrete_map={"Positive Churn": "#fb7185", "Negative Churn": "#10b981"},
                    template="plotly_dark",
                    height=350
                )
                fig_shap_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_shap_bar, use_container_width=True)


# =========================================================
# PAGE 6: Batch Prediction
# =========================================================
elif page_selection == "Batch Prediction":
    st.title("Batch Prediction")

    uploaded_file = st.file_uploader("Upload CSV File for Inference", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.caption(f"Loaded {len(batch_df):,} records.")
            
            # Incorporating imports locally
            sys.path.append(str(BASE_DIR))
            from src.feature_engineering import calculate_service_count, assign_tenure_bucket, get_contract_risk
            
            df_scored = batch_df.copy()
            if "tenure" in df_scored.columns:
                df_scored["tenure_bucket"] = df_scored["tenure"].apply(assign_tenure_bucket)
            if "TotalCharges" in df_scored.columns:
                tc_numeric = pd.to_numeric(df_scored["TotalCharges"].astype(str).str.strip(), errors="coerce")
                df_scored["TotalCharges"] = tc_numeric.fillna(tc_numeric.median())
                df_scored["avg_monthly_spend"] = np.round(df_scored["TotalCharges"] / (df_scored["tenure"] + 1.0), 2)
            if "Contract" in df_scored.columns:
                df_scored["contract_risk_score"] = df_scored["Contract"].apply(get_contract_risk)
            df_scored["service_count"] = df_scored.apply(calculate_service_count, axis=1)
            df_scored["support_usage_indicator"] = df_scored.apply(
                lambda r: 1 if (str(r.get("TechSupport", "")).lower() == "yes" or str(r.get("OnlineSecurity", "")).lower() == "yes") else 0, axis=1
            )
            if "MonthlyCharges" in df_scored.columns:
                df_scored["premium_customer_flag"] = (df_scored["MonthlyCharges"] > 80.0).astype(int)
            if "tenure" in df_scored.columns:
                df_scored["long_term_customer_flag"] = (df_scored["tenure"] > 36).astype(int)
            if "PaymentMethod" in df_scored.columns:
                df_scored["electronic_check_risk"] = df_scored["PaymentMethod"].apply(lambda x: 1 if "electronic check" in str(x).lower() else 0)

            probs = champion_pipeline.predict_proba(df_scored)[:, 1]
            batch_df["Churn_Probability"] = np.round(probs, 4)
            batch_df["Risk_Category"] = np.where(probs >= 0.65, "High", np.where(probs >= 0.35, "Moderate", "Low"))

            with st.container():
                st.markdown("### Output Summary Table")
                cols_to_show = ["customerID", "Churn_Probability", "Risk_Category", "Contract", "MonthlyCharges"]
                st.dataframe(batch_df[[c for c in cols_to_show if c in batch_df.columns]].head(100), use_container_width=True)

            csv_data = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Export Scored Dataset",
                data=csv_data,
                file_name="scored_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as batch_err:
            st.error(f"Error processing dataset: {batch_err}")


# =========================================================
# PAGE 7: Strategic Insights & Reports
# =========================================================
elif page_selection == "Strategic Insights & Reports":
    st.title("Strategic Insights & Reports")

    with st.container():
        st.markdown("### Global XAI Summary")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            beeswarm_path = BASE_DIR / "reports" / "figures" / "shap_summary_beeswarm.png"
            if beeswarm_path.exists():
                st.image(str(beeswarm_path), caption="Global SHAP Beeswarm", use_container_width=True)
                
        with col_g2:
            bar_path = BASE_DIR / "reports" / "figures" / "shap_feature_importance_bar.png"
            if bar_path.exists():
                st.image(str(bar_path), caption="Mean |SHAP| Feature Importance", use_container_width=True)

    with st.container():
        st.markdown("### Structured Retention Findings")
        for rec in eda_data.get("recommendations", []):
            st.markdown(
                f"""
                <div class="insight-box">
                    <strong style="color: var(--text-primary);">Vulnerability:</strong> <span style="color: var(--text-muted);">{rec.get('finding', '')}</span><br><br>
                    <strong style="color: var(--text-primary);">Strategy:</strong> <span style="color: var(--text-muted);">{rec.get('action', '')}</span>
                </div>
                """, unsafe_allow_html=True
            )
