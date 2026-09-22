# Data Quality & Preprocessing Audit Report

**Executive Summary:**
- **Source Dataset:** IBM Telco Customer Churn
- **Initial Records Ingested:** 7,043
- **Duplicates Removed:** 0
- **Blank `TotalCharges` Cleaned & Imputed:** 11 (imputed with median = $1397.47)
- **Final Cleaned Record Count:** 7,043
- **Feature Dimension:** 22

---

## 1. Missing Values & Schema Audit

| Column | Data Type | Null Count | Null % | Unique Values |
|---|---|---|---|---|
| `customerID` | `str` | 0 | 0.0% | 7043 |
| `gender` | `str` | 0 | 0.0% | 2 |
| `SeniorCitizen` | `int32` | 0 | 0.0% | 2 |
| `Partner` | `str` | 0 | 0.0% | 2 |
| `Dependents` | `str` | 0 | 0.0% | 2 |
| `tenure` | `int32` | 0 | 0.0% | 73 |
| `PhoneService` | `str` | 0 | 0.0% | 2 |
| `MultipleLines` | `str` | 0 | 0.0% | 3 |
| `InternetService` | `str` | 0 | 0.0% | 3 |
| `OnlineSecurity` | `str` | 0 | 0.0% | 3 |
| `OnlineBackup` | `str` | 0 | 0.0% | 3 |
| `DeviceProtection` | `str` | 0 | 0.0% | 3 |
| `TechSupport` | `str` | 0 | 0.0% | 3 |
| `StreamingTV` | `str` | 0 | 0.0% | 3 |
| `StreamingMovies` | `str` | 0 | 0.0% | 3 |
| `Contract` | `str` | 0 | 0.0% | 3 |
| `PaperlessBilling` | `str` | 0 | 0.0% | 2 |
| `PaymentMethod` | `str` | 0 | 0.0% | 4 |
| `MonthlyCharges` | `float64` | 0 | 0.0% | 1585 |
| `TotalCharges` | `float64` | 0 | 0.0% | 6531 |
| `Churn` | `str` | 0 | 0.0% | 2 |
| `Churn_Numeric` | `int64` | 0 | 0.0% | 2 |

---

## 2. Numerical Feature Descriptive Statistics

| Feature | Count | Mean | Std Dev | Min | 25% | 50% (Median) | 75% | Max |
|---|---|---|---|---|---|---|---|---|
| `tenure` | 7,043 | 32.37 | 24.56 | 0.0 | 9.0 | 29.0 | 55.0 | 72.0 |
| `MonthlyCharges` | 7,043 | 64.76 | 30.09 | 18.25 | 35.5 | 70.35 | 89.85 | 118.75 |
| `TotalCharges` | 7,043 | 2281.92 | 2265.27 | 18.8 | 402.22 | 1397.48 | 3786.6 | 8684.8 |

---

## 3. Data Cleansing Rules Applied
1. **Deduplication:** Ensured `customerID` serves as a strictly unique primary key.
2. **Whitespace Coercion:** Empty space string tokens in `TotalCharges` converted to standard IEEE floating point representation and imputed with robust median value.
3. **Target Normalization:** Created binary label column `Churn_Numeric` where `Yes` maps to `1` and `No` maps to `0`.
4. **Data Integrity:** No records dropped due to data loss; 100% data retention preserved across all 7,043 customer instances.
