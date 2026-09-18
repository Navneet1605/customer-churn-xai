"""
app.py
======
Interactive Streamlit Dashboard for Explainable Customer Churn Prediction
Features:
- Multi-page navigation (Home, Data Overview, EDA, Model Benchmark, Real-time XAI Predictor, Batch Inference, Business Insights)
- Interactive Plotly visualisations
- Live SHAP local waterfall interpretability and dynamic natural language explanations
- Single-customer real-time risk scorer & Batch CSV prediction engine
- Direct PDF / Markdown report downloaders
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

# Append project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Page configuration
st.set_page_config(
    page_title="Explainable Churn Analytics | Apache Spark & XAI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
css_file = BASE_DIR / "dashboard" / "styles.css"
if css_file.exists():
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


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
    if best_file and Path(best_file).exists():
        with open(best_file, "rb") as f:
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
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
st.sidebar.title("Churn Intelligence")
st.sidebar.caption("Apache Spark MLlib & SHAP XAI System")

page_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Executive Overview",
        "📊 Data Overview & Schema",
        "📈 Exploratory Analytics (EDA)",
        "🤖 Model Performance & ROC",
        "🔍 Live XAI Churn Predictor",
        "📁 Batch Prediction Engine",
        "💡 Business Insights & Reports"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Core Stack:**\n"
    "- Apache Spark (PySpark)\n"
    "- Spark MLlib Pipeline\n"
    "- SHAP (Game Theory XAI)\n"
    "- Streamlit & Plotly"
)


# =========================================================
# PAGE 1: Executive Overview
# =========================================================
if page_selection == "🏠 Executive Overview":
    st.title("⚡ Explainable Customer Churn Prediction")
    st.subheader("Big Data Machine Learning & Explainable AI in Telecommunications")
    
    kpis = eda_data.get("kpis", {})
    
    # KPI Cards Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Customer Base</div>
            <div class="metric-value">{kpis.get('total_customers', 7043):,}</div>
            <div class="metric-subtitle">IBM Telco Ingested Records</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Baseline Churn Rate</div>
            <div class="metric-value" style="color: #ef4444;">{kpis.get('churn_rate_pct', 26.54)}%</div>
            <div class="metric-subtitle">{kpis.get('churn_count', 1869):,} Attrited Subscribers</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Monthly Revenue at Risk</div>
            <div class="metric-value" style="color: #f59e0b;">${kpis.get('monthly_revenue_at_risk', 139130):,.0f}</div>
            <div class="metric-subtitle">{kpis.get('churn_revenue_pct', 30.5)}% of Monthly Billing</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        best_name = model_eval.get("best_model_name", "Gradient Boosted Trees")
        best_auc = model_eval.get("best_metrics", {}).get("ROC_AUC", 0.845)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Champion Model AUC</div>
            <div class="metric-value" style="color: #10b981;">{best_auc:.4f}</div>
            <div class="metric-subtitle">{best_name}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture Breakdown
    st.markdown("### 🏗️ Big Data Architecture & End-to-End Pipeline")
    
    col_arch1, col_arch2 = st.columns([1.4, 1])
    with col_arch1:
        st.markdown("""
        ```mermaid
        graph LR
            A[Raw Telco Data 7043 Rows] --> B[PySpark Schema & Null Audit]
            B --> C[Parquet Partitioning HDFS]
            C --> D[Feature Engineering Engine]
            D --> E[Spark MLlib Pipelines]
            E --> F[Model Benchmark & Cross Validation]
            F --> G[SHAP Game-Theoretic Explainability]
            G --> H[Live Interactive Dashboard]
        ```
        """, unsafe_allow_html=False)

        st.markdown("""
        The platform executes a unified distributed Big Data and Explainable AI architecture:
        - **Distributed Ingestion:** Apache Spark DataFrames handle schema enforcement, automated null detection, and partitioned Parquet storage.
        - **Data Quality Auditing:** Automatic type coercion, blank string handling for `TotalCharges`, and median imputation.
        - **Feature Engineering:** Domain-specific indicators including `tenure_bucket`, `contract_risk_score`, and `service_count`.
        - **Distributed MLlib Pipelines:** Scalable `StringIndexer` + `OneHotEncoder` + `VectorAssembler` + `StandardScaler` transformations.
        - **Explainable AI Layer:** SHAP TreeExplainer decomposing every prediction into granular additive contribution scores.
        """)

    with col_arch2:
        st.markdown("#### 🎯 Strategic Project Objectives")
        st.markdown("""
        - 📉 **Proactive Churn Mitigation:** Identify high-risk subscribers weeks prior to contract termination.
        - 💡 **Explainable Reasoning:** Eliminate black-box opacity so customer success agents understand *why* a customer intends to leave.
        - 💰 **Preserve Revenue Streams:** Target retention budgets exclusively on salvageable, high-LTV accounts.
        - ⚡ **Real-Time Operational Scoring:** Single-customer live inference + automated batch CSV prediction.
        """)

    st.markdown("---")
    st.markdown("### 🚀 Technology Stack Summary")
    b1, b2, b3, b4 = st.columns(4)
    b1.success("⚡ **PySpark & Spark SQL**\nDistributed Ingestion & Aggregation")
    b2.info("🤖 **Spark MLlib & Scikit-Learn**\nLR, Random Forest & GBT Pipelines")
    b3.warning("🔬 **SHAP Interpretability**\nGlobal Summary & Local Waterfall")
    b4.error("📊 **Streamlit & Plotly**\nModern Interactive UI & Dashboards")


# =========================================================
# PAGE 2: Data Overview & Schema
# =========================================================
elif page_selection == "📊 Data Overview & Schema":
    st.title("📊 Data Overview & Quality Audit")
    st.caption("Schema Explorer, HDFS Parquet Partitioning, and Missing Value Verification")

    if not df_sample.empty:
        st.markdown(f"**Ingested Dataset Shape:** `{df_sample.shape[0]:,}` rows × `{df_sample.shape[1]}` columns")
        
        tab_raw, tab_schema, tab_dist = st.tabs(["📋 Data Viewer", "🔍 Schema & Quality Audit", "🗂️ HDFS Storage Simulation"])
        
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
            st.markdown("#### 🗄️ HDFS Distributed Parquet Architecture")
            st.code("""
data/hdfs_simulation/warehouse/telco/
├── Contract=Month-to-month/
│   └── part-00000.parquet
├── Contract=One year/
│   └── part-00001.parquet
└── Contract=Two year/
    └── part-00002.parquet
            """, language="text")
            st.info("Simulates an enterprise Hadoop / Spark data lake partitioned by customer contractual commitment.")
    else:
        st.warning("Processed dataset not found. Please run `python run_pipeline.py` first.")


# =========================================================
# PAGE 3: Exploratory Data Analysis (EDA)
# =========================================================
elif page_selection == "📈 Exploratory Analytics (EDA)":
    st.title("📈 Exploratory Data Analysis (EDA)")
    st.caption("Interactive Visualizations powered by Spark SQL and Plotly")

    breakdowns = eda_data.get("breakdowns", {})

    # Row 1: Churn Donut + Contract Churn Bar
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        churn_dist = eda_data.get("kpis", {})
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Retained', 'Churned'],
            values=[churn_dist.get('retained_count', 5174), churn_dist.get('churn_count', 1869)],
            hole=.55,
            marker_colors=['#10b981', '#ef4444']
        )])
        fig_donut.update_layout(title="<b>Overall Churn Ratio</b>", height=380, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    with r1_col2:
        contract_data = pd.DataFrame(breakdowns.get("by_contract", []))
        if not contract_data.empty:
            fig_contract = px.bar(
                contract_data,
                x="Contract",
                y=["Retained", "Churned"],
                title="<b>Customer Retention by Contract Type</b>",
                barmode="group",
                color_discrete_map={"Retained": "#10b981", "Churned": "#ef4444"},
                height=380
            )
            st.plotly_chart(fig_contract, use_container_width=True)

    # Row 2: Internet Service & Payment Method
    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        internet_data = pd.DataFrame(breakdowns.get("by_internet_service", []))
        if not internet_data.empty:
            fig_internet = px.bar(
                internet_data,
                x="InternetService",
                y="ChurnRate",
                title="<b>Churn Rate by Internet Service Type (%)</b>",
                color="ChurnRate",
                color_continuous_scale="Reds",
                text="ChurnRate",
                height=380
            )
            fig_internet.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_internet, use_container_width=True)

    with r2_col2:
        payment_data = pd.DataFrame(breakdowns.get("by_payment_method", []))
        if not payment_data.empty:
            fig_payment = px.bar(
                payment_data,
                y="PaymentMethod",
                x="ChurnRate",
                orientation='h',
                title="<b>Churn Rate by Payment Method (%)</b>",
                color="ChurnRate",
                color_continuous_scale="Blues",
                text="ChurnRate",
                height=380
            )
            fig_payment.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_payment, use_container_width=True)

    # Row 3: Tenure & Charges Distributions
    if not df_sample.empty:
        r3_col1, r3_col2 = st.columns(2)
        with r3_col1:
            fig_hist = px.histogram(
                df_sample,
                x="tenure",
                color="Churn",
                marginal="box",
                barmode="overlay",
                title="<b>Tenure Distribution by Churn Status (Months)</b>",
                color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                height=400
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with r3_col2:
            fig_box = px.box(
                df_sample,
                x="Contract",
                y="MonthlyCharges",
                color="Churn",
                title="<b>Monthly Charges vs Contract Type</b>",
                color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                height=400
            )
            st.plotly_chart(fig_box, use_container_width=True)


# =========================================================
# PAGE 4: Model Benchmarking & Performance
# =========================================================
elif page_selection == "🤖 Model Performance & ROC":
    st.title("🤖 Model Performance & ROC Benchmark")
    st.caption("Cross-Validated Performance Benchmarks across Spark MLlib & Ensemble Models")

    metrics_list = model_eval.get("metrics_summary", [])
    if metrics_list:
        m_df = pd.DataFrame(metrics_list)
        st.markdown("### 🏆 Leaderboard Comparison")
        st.dataframe(
            m_df.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "PR_AUC"], color="#1e3a8a"),
            use_container_width=True
        )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### 📈 Benchmark Figures")
            fig_path = BASE_DIR / "reports" / "figures" / "roc_curves_comparison.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Receiver Operating Characteristic (ROC) Comparison", use_container_width=True)
            
            pr_path = BASE_DIR / "reports" / "figures" / "precision_recall_curves.png"
            if pr_path.exists():
                st.image(str(pr_path), caption="Precision-Recall Benchmark Curves", use_container_width=True)

        with col_m2:
            st.markdown("#### 🔲 Confusion Matrices Benchmark")
            cm_path = BASE_DIR / "reports" / "figures" / "confusion_matrices_comparison.png"
            if cm_path.exists():
                st.image(str(cm_path), caption="Confusion Matrix Comparison Across Models", use_container_width=True)
            
            best_info = model_eval.get("best_metrics", {})
            st.success(f"""
            **Champion Model Selected:** `{model_eval.get('best_model_name')}`
            - **ROC-AUC:** `{best_info.get('ROC_AUC', 0):.4f}`
            - **Accuracy:** `{best_info.get('Accuracy', 0)*100:.2f}%`
            - **F1 Score:** `{best_info.get('F1_Score', 0):.4f}`
            - **Recall:** `{best_info.get('Recall', 0):.4f}`
            """)
    else:
        st.warning("Model evaluation metrics not found. Please run `python run_pipeline.py` first.")


# =========================================================
# PAGE 5: Live XAI Churn Predictor
# =========================================================
elif page_selection == "🔍 Live XAI Churn Predictor":
    st.title("🔍 Live Real-Time Churn Predictor & Explainable AI")
    st.caption("Interactive single-customer risk scoring with dynamic SHAP waterfall interpretability")

    if champion_pipeline is None or live_explainer is None:
        st.error("Champion model or SHAP explainer not loaded. Please run `python run_pipeline.py`.")
    else:
        st.markdown("### 📋 Enter Customer Attributes")
        with st.form("churn_prediction_form"):
            col_c1, col_c2, col_c3 = st.columns(3)

            with col_c1:
                f_gender = st.selectbox("Gender", ["Female", "Male"])
                f_senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes (Senior)" if x == 1 else "No")
                f_partner = st.selectbox("Has Partner", ["Yes", "No"])
                f_dependents = st.selectbox("Has Dependents", ["No", "Yes"])
                f_tenure = st.slider("Tenure (Months Subscribed)", min_value=1, max_value=72, value=4)
                f_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

            with col_c2:
                f_phone = st.selectbox("Phone Service", ["Yes", "No"])
                f_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                f_internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                f_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                f_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                f_device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

            with col_c3:
                f_tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                f_streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                f_streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                f_paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                f_payment = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                f_monthly = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=89.5, step=0.5)

            submitted = st.form_submit_button("⚡ Predict Churn & Explain Decision", use_container_width=True)

        if submitted:
            # Derived features calculation
            total_charges_est = round(f_monthly * f_tenure, 2)
            avg_monthly_spend_est = round(total_charges_est / (f_tenure + 1), 2)
            contract_risk_score_est = 3 if "month" in f_contract.lower() else (2 if "one" in f_contract.lower() else 1)
            
            # Service count
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
                "gender": f_gender,
                "SeniorCitizen": f_senior,
                "Partner": f_partner,
                "Dependents": f_dependents,
                "PhoneService": f_phone,
                "MultipleLines": f_lines,
                "InternetService": f_internet,
                "OnlineSecurity": f_security,
                "OnlineBackup": f_backup,
                "DeviceProtection": f_device_prot,
                "TechSupport": f_tech_support,
                "StreamingTV": f_streaming_tv,
                "StreamingMovies": f_streaming_movies,
                "Contract": f_contract,
                "PaperlessBilling": f_paperless,
                "PaymentMethod": f_payment,
                "tenure_bucket": tenure_bucket_val,
                "tenure": f_tenure,
                "MonthlyCharges": f_monthly,
                "TotalCharges": total_charges_est,
                "avg_monthly_spend": avg_monthly_spend_est,
                "contract_risk_score": contract_risk_score_est,
                "service_count": svc_count,
                "support_usage_indicator": support_usage,
                "premium_customer_flag": premium_flag,
                "long_term_customer_flag": long_term_flag,
                "electronic_check_risk": echeck_risk
            }

            input_df = pd.DataFrame([input_dict])

            # Run inference
            churn_prob = float(champion_pipeline.predict_proba(input_df)[0, 1])
            is_churn = churn_prob >= 0.50

            st.markdown("---")
            st.markdown("### 🎯 Prediction Results & Risk Assessment")

            res_col1, res_col2 = st.columns([1, 1.5])
            with res_col1:
                risk_status = "CRITICAL CHURN RISK" if churn_prob >= 0.65 else ("ELEVATED RISK" if churn_prob >= 0.35 else "LOW RISK (LOYAL)")
                risk_color = "#ef4444" if churn_prob >= 0.65 else ("#f59e0b" if churn_prob >= 0.35 else "#10b981")

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=churn_prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"<b>{risk_status}</b><br><span style='font-size:0.8em;color:gray'>Churn Probability</span>", 'font': {'size': 18}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, 35], 'color': "rgba(16, 185, 129, 0.2)"},
                            {'range': [35, 65], 'color': "rgba(245, 158, 11, 0.2)"},
                            {'range': [65, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 3},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=320, margin=dict(t=30, b=10, l=30, r=30))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_col2:
                # Calculate live SHAP values
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
                pos_feats = sorted([p for p in paired_shap if p[1] > 0], key=lambda x: x[1], reverse=True)
                neg_feats = sorted([p for p in paired_shap if p[1] < 0], key=lambda x: x[1])

                from src.explainability import generate_natural_language_explanation
                nl_explanation = generate_natural_language_explanation(
                    input_dict,
                    churn_prob,
                    pos_feats,
                    neg_feats
                )
                
                st.markdown("#### 🧠 Natural Language Executive Explanation")
                st.markdown(f"""
                <div class="insight-box">
                    {nl_explanation}
                </div>
                """, unsafe_allow_html=True)

            # Live SHAP Contribution Chart
            st.markdown("#### 🔬 Granular Feature Attribution (SHAP Forces)")
            top_contributors = sorted(paired_shap, key=lambda x: abs(x[1]), reverse=True)[:10]
            top_feats_df = pd.DataFrame(top_contributors, columns=["Feature", "SHAP Impact"])
            top_feats_df["Impact Type"] = top_feats_df["SHAP Impact"].apply(lambda v: "Increases Churn Risk" if v > 0 else "Lowers Churn Risk")
            
            fig_shap_bar = px.bar(
                top_feats_df,
                x="SHAP Impact",
                y="Feature",
                orientation='h',
                color="Impact Type",
                color_discrete_map={"Increases Churn Risk": "#ef4444", "Lowers Churn Risk": "#10b981"},
                title="<b>Top 10 Feature Drivers for this Customer</b>",
                height=380
            )
            fig_shap_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_shap_bar, use_container_width=True)


# =========================================================
# PAGE 6: Batch Inference Engine
# =========================================================
elif page_selection == "📁 Batch Prediction Engine":
    st.title("📁 Enterprise Batch Prediction Engine")
    st.caption("Upload a batch CSV file of telecom customers for high-throughput churn scoring & risk profiling.")

    uploaded_file = st.file_uploader("Upload CSV File (e.g., Telco Customers Batch)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded `{len(batch_df):,}` customer records.")
            
            # Preprocess & Feature Engineer on the fly
            from src.feature_engineering import calculate_service_count, assign_tenure_bucket, get_contract_risk
            
            df_scored = batch_df.copy()
            if "tenure" in df_scored.columns:
                df_scored["tenure_bucket"] = df_scored["tenure"].apply(assign_tenure_bucket)
            if "TotalCharges" in df_scored.columns:
                df_scored["TotalCharges"] = pd.to_numeric(df_scored["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(df_scored["TotalCharges"].median() if hasattr(df_scored["TotalCharges"], 'median') else 0)
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

            # Predict
            probs = champion_pipeline.predict_proba(df_scored)[:, 1]
            batch_df["Churn_Probability"] = np.round(probs, 4)
            batch_df["Predicted_Churn"] = np.where(probs >= 0.5, "Yes", "No")
            batch_df["Risk_Category"] = np.where(probs >= 0.65, "High", np.where(probs >= 0.35, "Moderate", "Low"))

            # Summary KPIs
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            high_risk_count = int((batch_df["Risk_Category"] == "High").sum())
            b_col1.metric("Batch Total", f"{len(batch_df):,}")
            b_col2.metric("High Risk Customers", f"{high_risk_count:,}")
            b_col3.metric("High Risk Ratio", f"{(high_risk_count/len(batch_df))*100:.1f}%")
            if "MonthlyCharges" in batch_df.columns:
                rev_risk = float(batch_df[batch_df["Risk_Category"] == "High"]["MonthlyCharges"].sum())
                b_col4.metric("High Risk Monthly Spend", f"${rev_risk:,.0f}")

            st.markdown("### 📊 Scored Batch Results")
            st.dataframe(batch_df[["customerID", "Churn_Probability", "Predicted_Churn", "Risk_Category", "Contract", "MonthlyCharges"]].head(100), use_container_width=True)

            # Download CSV
            csv_data = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored Batch CSV",
                data=csv_data,
                file_name="telco_scored_batch_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as batch_err:
            st.error(f"Error processing batch dataset: {batch_err}")
    else:
        st.info("💡 You can upload the raw `WA_Fn-UseC_-Telco-Customer-Churn.csv` file to test batch predictions.")


# =========================================================
# PAGE 7: Business Insights & Reports
# =========================================================
elif page_selection == "💡 Business Insights & Reports":
    st.title("💡 Strategic Business Insights & Executive Reports")
    st.caption("Actionable churn drivers, financial exposure analysis, and downloadable research reports.")

    # Global SHAP Figures
    st.markdown("### 🔬 Global Explainable AI Drivers")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        beeswarm_path = BASE_DIR / "reports" / "figures" / "shap_summary_beeswarm.png"
        if beeswarm_path.exists():
            st.image(str(beeswarm_path), caption="Global SHAP Beeswarm Impact Distribution", use_container_width=True)
            
    with col_g2:
        bar_path = BASE_DIR / "reports" / "figures" / "shap_feature_importance_bar.png"
        if bar_path.exists():
            st.image(str(bar_path), caption="Global Feature Importance (Mean |SHAP Value|)", use_container_width=True)

    # Executive Recommendations
    st.markdown("---")
    st.markdown("### 🎯 Executive Strategic Action Playbook")
    for rec in eda_data.get("recommendations", []):
        with st.expander(f"📌 [{rec['priority']}] {rec['theme']}", expanded=True):
            st.markdown(f"**Observed Vulnerability:** {rec['finding']}")
            st.markdown(f"**Recommended Strategy:** {rec['action']}")

    # Report Downloads
    st.markdown("---")
    st.markdown("### 📄 Download Enterprise Reports")
    col_d1, col_d2 = st.columns(2)

    pdf_file_path = BASE_DIR / "reports" / "final_report.pdf"
    if pdf_file_path.exists():
        with open(pdf_file_path, "rb") as pf:
            col_d1.download_button(
                label="📥 Download Executive PDF Report",
                data=pf.read(),
                file_name="Telco_Customer_Churn_XAI_Executive_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    md_file_path = BASE_DIR / "reports" / "final_report.md"
    if md_file_path.exists():
        with open(md_file_path, "r", encoding="utf-8") as mf:
            col_d2.download_button(
                label="📥 Download Full Markdown Research Report",
                data=mf.read(),
                file_name="Telco_Customer_Churn_XAI_Research_Report.md",
                mime="text/markdown",
                use_container_width=True
            )
