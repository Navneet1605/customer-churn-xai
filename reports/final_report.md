# Explainable Customer Churn Prediction using Apache Spark and Explainable AI
**Research & Enterprise Technical Report**

- **Project Lead / Author:** Big Data & AI Engineering Team
- **Technology Stack:** Apache Spark (PySpark), Spark MLlib, Spark SQL, SHAP, Plotly, Streamlit
- **Target Dataset:** IBM Telco Customer Churn (7,043 Instances, 21 Attributes)
- **Status:** Production-Ready End-to-End System

---

## 1. Executive Summary

Customer attrition poses one of the most substantial threats to telecommunications profitability. This project presents a distributed, scalable Machine Learning and Explainable Artificial Intelligence (XAI) framework built on **Apache Spark** and **SHAP (SHapley Additive exPlanations)**.

### Core Highlights:
- **Financial Exposure:** Total customer base represents **$456,116.60** in monthly recurring revenue. Churning subscribers jeopardize **$139,130.85/month** (**$1,669,570.20/year**), representing **30.5%** of top-line revenue.
- **Predictive Superiority:** Benchmarked Logistic Regression, Random Forest, and Gradient Boosted Trees. The champion model (**Gradient Boosted Trees**) achieved an **ROC-AUC of 0.8465** and an **Accuracy of 80.41%**.
- **Transparent Decisioning:** Integrated Game-Theoretic SHAP interpretability to deliver individual customer waterfall breakdowns, exposing actionable retention levers.

---

## 2. Dataset Architecture & Preprocessing

The IBM Telco dataset comprises demographic, account, and subscribed service variables.

- **Total Ingested Records:** 7,043
- **Baseline Churn Rate:** 26.54% (1,869 Churned vs 5,174 Retained)
- **Cleaning & Imputation:** Coerced blank whitespace strings in `TotalCharges` into floating-point format and imputed missing values using the statistical median.
- **Engineered Distributed Features:**
  1. `tenure_bucket`: Segmented lifecycle (`New`, `Developing`, `Stable`, `Loyal`).
  2. `avg_monthly_spend`: Normalized historical burn rate.
  3. `contract_risk_score`: Quantified contractual vulnerability.
  4. `service_count`: Aggregated active bundle depth across voice, broadband, and value-added services.
  5. `support_usage_indicator`: Flag for high-dependency technical support consumers.
  6. `electronic_check_risk`: Flag for high-friction billing channels.

---

## 3. Exploratory Data Analysis & Financial Intelligence

| Segmentation Dimension | Cohort | Churn Rate (%) | Retained Count | Churned Count |
|---|---|---|---|---|
| **Contract** | Month-to-month | 42.71% | 2,220 | 1,655 |
| **Contract** | One year | 11.27% | 1,307 | 166 |
| **Contract** | Two year | 2.83% | 1,647 | 48 |
| **Internet Service** | DSL | 18.96% | 1,962 | 459 |
| **Internet Service** | Fiber optic | 41.89% | 1,799 | 1,297 |
| **Internet Service** | No | 7.4% | 1,413 | 113 |
| **Payment Method** | Bank transfer (automatic) | 16.71% | 1,286 | 258 |
| **Payment Method** | Credit card (automatic) | 15.24% | 1,290 | 232 |
| **Payment Method** | Electronic check | 45.29% | 1,294 | 1,071 |
| **Payment Method** | Mailed check | 19.11% | 1,304 | 308 |

---

## 4. Machine Learning Benchmark Results

All models were evaluated using 3-fold Stratified Cross-Validation on an 80/20 holdout split.

| Model Architecture | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 80.62% | 0.6678 | 0.5374 | 0.5956 | **0.8457** | 0.6566 |
| **Random Forest** | 80.06% | 0.6716 | 0.4866 | 0.5643 | **0.8436** | 0.6564 |
| **Gradient Boosted Trees** | 80.41% | 0.6678 | 0.5214 | 0.5856 | **0.8465** | 0.6639 |

> **Champion Model Selected:** `Gradient Boosted Trees` demonstrates optimal discrimination threshold balance with highest area under the ROC curve.

---

## 5. Explainable AI (XAI) with SHAP

### Global Feature Attribution
The SHAP TreeExplainer revealed the primary macroscopic drivers across the entire customer base:
1. **Contract Type (Month-to-Month):** Strongest positive driver of churn. Customers on monthly terms can leave without contractual friction.
2. **Tenure Duration:** Strongest negative driver of churn. Each incremental month of loyalty diminishes churn probability exponentially.
3. **Monthly Charges & Fiber Optic Service:** High recurring charges accelerate churn risk when uncoupled from technical support.
4. **Internet & Support Services:** Active subscriptions to TechSupport and Online Security provide significant retention buffering.

### Local Explainability Personas

#### Persona: High Risk Churn Archetype
- **Predicted Risk:** 86.5% (Churn)
Customer exhibits a **High Churn Risk** with a predicted churn probability of **86.5%**.
- **Primary Churn Accelerators:** Tenure (SHAP impact: +0.93), Contract Risk Score (SHAP impact: +0.87), Internetservice Fiber Optic (SHAP impact: +0.27).
- **Primary Retention Anchors:** Contract One Year (SHAP impact: -0.02), Avg Monthly Spend (SHAP impact: -0.01).
💡 **Recommended Action:** Immediate intervention required. Proactively offer a discounted annual contract lock-in, complimentary tech support, or customized loyalty incentive.

#### Persona: Moderate Vulnerability Archetype
- **Predicted Risk:** 49.9% (Retain)
Customer exhibits a **Moderate Churn Risk** with a predicted churn probability of **49.9%**.
- **Primary Churn Accelerators:** Contract Risk Score (SHAP impact: +0.87), Internetservice Fiber Optic (SHAP impact: +0.41), Tenure (SHAP impact: +0.11).
- **Primary Retention Anchors:** Electronic Check Risk (SHAP impact: -0.12), Monthlycharges (SHAP impact: -0.09).
💡 **Recommended Action:** Monitor usage velocity; propose value-add digital security or automated payment discounts.

#### Persona: High Retention & Loyal Archetype
- **Predicted Risk:** 2.2% (Retain)
Customer exhibits a **Low Churn Risk** with a predicted churn probability of **2.2%**.
- **Primary Churn Accelerators:** Onlinesecurity Yes (SHAP impact: +0.06), Totalcharges (SHAP impact: +0.03), Techsupport Yes (SHAP impact: +0.01).
- **Primary Retention Anchors:** Contract Risk Score (SHAP impact: -0.79), Tenure (SHAP impact: -0.33).
💡 **Recommended Action:** High retention health. Consider upselling premium streaming tiers or family bundles.

#### Persona: Support-Deficient Tech Consumer
- **Predicted Risk:** 64.5% (Churn)
Customer exhibits a **Moderate Churn Risk** with a predicted churn probability of **64.5%**.
- **Primary Churn Accelerators:** Contract Risk Score (SHAP impact: +0.83), Internetservice Fiber Optic (SHAP impact: +0.41), Tenure (SHAP impact: +0.25).
- **Primary Retention Anchors:** Electronic Check Risk (SHAP impact: -0.09), Streamingtv Yes (SHAP impact: -0.05).
💡 **Recommended Action:** Monitor usage velocity; propose value-add digital security or automated payment discounts.

#### Persona: Price-Sensitive Senior Subscriber
- **Predicted Risk:** 72.9% (Churn)
Customer exhibits a **High Churn Risk** with a predicted churn probability of **72.9%**.
- **Primary Churn Accelerators:** Contract Risk Score (SHAP impact: +0.77), Monthlycharges (SHAP impact: +0.46), Tenure (SHAP impact: +0.44).
- **Primary Retention Anchors:** Techsupport Yes (SHAP impact: -0.13), Electronic Check Risk (SHAP impact: -0.08).
💡 **Recommended Action:** Immediate intervention required. Proactively offer a discounted annual contract lock-in, complimentary tech support, or customized loyalty incentive.


---

## 6. Strategic Executive Recommendations

### [P1 - Critical] Month-to-Month Contract Transition
- **Observed Finding:** Month-to-month subscribers account for over 88% of all churn events with a 42.7% attrition rate.
- **Strategic Action Plan:** Incentivize 1-year commitments with a 15% discount or bundled streaming benefits; implement automated renewal nudges at month 3 and month 6.

### [P1 - Critical] Fiber Optic Value & Support Gap
- **Observed Finding:** Fiber Optic customers experience higher monthly charges ($80+) and churn at 41.9% due to lack of complementary technical assistance.
- **Strategic Action Plan:** Bundle complimentary TechSupport and Online Security into high-speed fiber tiers to protect high-ARPU subscribers.

### [P2 - High] Electronic Check Payment Friction
- **Observed Finding:** Customers using Electronic Check experience a 45.3% churn rate vs 15.2% for automated bank/credit card transfers.
- **Strategic Action Plan:** Offer a one-time $10 credit to migrate electronic check customers to automated credit card / ACH autopay.

### [P3 - Medium] New Customer Lifecycle Onboarding
- **Observed Finding:** The highest attrition velocity occurs during months 1–6 (tenure <= 6).
- **Strategic Action Plan:** Deploy a high-touch 90-day onboarding program, automated satisfaction check-ins, and priority support routing for new signups.


---

## 7. Conclusion & Production Deployment

This Explainable Machine Learning system transitions predictive intelligence from an opaque 'black-box' to transparent, trust-verified decision support. By combining the big data scalability of Apache Spark with Game-Theoretic SHAP interpretability, telecommunications retention teams can proactively safeguard vulnerable revenue streams while optimizing intervention ROI.
