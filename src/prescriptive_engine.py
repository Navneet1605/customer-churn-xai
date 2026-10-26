"""
prescriptive_engine.py
======================
Prescriptive AI & Algorithmic Counterfactual Optimization Engine

Bridges predictive classification and business actionability:
1. Algorithmic Counterfactual Search (Minimal Actionable Recourse)
2. Prescriptive Retention Action Plans (Support Stabilization, Annual Lock-In, VIP Bundle)
3. Unit Economics & ROI Modeling (Customer Lifetime Value, Net Value Saved, Retention Cost)
4. Batch Cohort Prescriptive Matrix (Strategic Retention Quadrants)
"""

from typing import Dict, Any, List, Tuple, Optional
import copy
import pandas as pd
import numpy as np


# Immutable vs. Actionable Feature Sets
IMMUTABLE_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "tenure", "tenure_bucket", "long_term_customer_flag"
]

ACTIONABLE_FEATURES = [
    "Contract", "TechSupport", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "PaymentMethod", "PaperlessBilling",
    "MonthlyCharges"
]


def compute_derived_features(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recalculates all domain-engineered features based on current attribute state.
    Maintains parity with the PySpark / feature engineering training pipeline.
    """
    row = copy.deepcopy(data_dict)

    # 1. Contract Risk Score: Month-to-month=3, One year=2, Two year=1
    contract_val = str(row.get("Contract", "")).strip().lower()
    if "month" in contract_val:
        row["contract_risk_score"] = 3
    elif "one" in contract_val:
        row["contract_risk_score"] = 2
    elif "two" in contract_val:
        row["contract_risk_score"] = 1
    else:
        row["contract_risk_score"] = 2

    # 2. Payment Method Risk Flag
    pay_val = str(row.get("PaymentMethod", "")).strip().lower()
    row["electronic_check_risk"] = 1 if "electronic check" in pay_val else 0

    # 3. Service Count
    count = 0
    if str(row.get("PhoneService", "")).strip().lower() == "yes":
        count += 1
    if str(row.get("MultipleLines", "")).strip().lower() == "yes":
        count += 1
    if str(row.get("InternetService", "")).strip().lower() in ["dsl", "fiber optic"]:
        count += 1
    for svc in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        if str(row.get(svc, "")).strip().lower() == "yes":
            count += 1
    row["service_count"] = count

    # 4. Support Usage Indicator
    ts = str(row.get("TechSupport", "")).strip().lower() == "yes"
    os_sec = str(row.get("OnlineSecurity", "")).strip().lower() == "yes"
    ob = str(row.get("OnlineBackup", "")).strip().lower() == "yes"
    row["support_usage_indicator"] = 1 if (ts or os_sec or ob) else 0

    # 5. Financial metrics
    tenure_val = float(row.get("tenure", 1))
    monthly_charges = float(row.get("MonthlyCharges", 0.0))
    
    # Estimate total charges if not present or updated
    total_charges = row.get("TotalCharges")
    if total_charges is None or pd.isna(total_charges):
        total_charges = round(monthly_charges * tenure_val, 2)
    else:
        total_charges = float(total_charges)
    row["TotalCharges"] = total_charges

    row["avg_monthly_spend"] = round(total_charges / (tenure_val + 1.0), 2)
    row["premium_customer_flag"] = 1 if monthly_charges > 80.0 else 0
    row["long_term_customer_flag"] = 1 if tenure_val > 36 else 0

    # 6. Tenure Bucket
    if tenure_val <= 12:
        row["tenure_bucket"] = "New"
    elif tenure_val <= 24:
        row["tenure_bucket"] = "Developing"
    elif tenure_val <= 48:
        row["tenure_bucket"] = "Stable"
    else:
        row["tenure_bucket"] = "Loyal"

    return row


def compute_cltv(monthly_charges: float, churn_prob: float, horizon_months: int = 24) -> float:
    """
    Computes churn-adjusted expected customer lifetime value over a given horizon.
    CLTV = sum_{t=1}^H (MonthlyCharges * (1 - churn_prob)^(t / 12))
    """
    monthly_retention = max(0.01, 1.0 - (churn_prob / 12.0))
    discount_factor = 0.995  # Standard monthly cost-of-capital discount
    cltv = 0.0
    for t in range(1, horizon_months + 1):
        cltv += monthly_charges * (monthly_retention ** t) * (discount_factor ** t)
    return round(cltv, 2)


def simulate_intervention(
    base_customer: Dict[str, Any],
    interventions: Dict[str, Any],
    pipeline: Any,
    monthly_discount_usd: float = 0.0,
    service_subsidy_cost_monthly: float = 0.0
) -> Dict[str, Any]:
    """
    Simulates a hypothetical customer intervention, predicts new churn probability,
    and returns full unit economics and return on retention spend.
    """
    # 1. Baseline prediction
    df_base = pd.DataFrame([compute_derived_features(base_customer)])
    p_base = float(pipeline.predict_proba(df_base)[0, 1])

    # 2. Intervened profile
    intervened_dict = copy.deepcopy(base_customer)
    for k, v in interventions.items():
        intervened_dict[k] = v

    # Apply monthly price concession / discount
    orig_monthly = float(base_customer.get("MonthlyCharges", 50.0))
    new_monthly = max(18.0, orig_monthly - monthly_discount_usd)
    intervened_dict["MonthlyCharges"] = new_monthly

    df_intervened = pd.DataFrame([compute_derived_features(intervened_dict)])
    p_intervened = float(pipeline.predict_proba(df_intervened)[0, 1])

    # 3. Retention Economics
    risk_drop = max(0.0, p_base - p_intervened)
    risk_drop_pct = (risk_drop / p_base * 100.0) if p_base > 0 else 0.0

    annual_revenue_base = orig_monthly * 12.0
    annual_revenue_saved = round(risk_drop * annual_revenue_base, 2)

    annual_intervention_cost = round((monthly_discount_usd + service_subsidy_cost_monthly) * 12.0, 2)
    net_value_saved = round(annual_revenue_saved - annual_intervention_cost, 2)

    roi_pct = round((net_value_saved / annual_intervention_cost * 100.0), 1) if annual_intervention_cost > 0 else 0.0

    base_cltv = compute_cltv(orig_monthly, p_base)
    intervened_cltv = compute_cltv(new_monthly, p_intervened)
    cltv_gain = round(intervened_cltv - base_cltv, 2)

    return {
        "baseline_probability": round(p_base, 4),
        "intervened_probability": round(p_intervened, 4),
        "absolute_risk_drop": round(risk_drop, 4),
        "risk_drop_pct": round(risk_drop_pct, 1),
        "annual_revenue_saved": annual_revenue_saved,
        "annual_intervention_cost": annual_intervention_cost,
        "net_value_saved": net_value_saved,
        "roi_pct": roi_pct,
        "base_cltv": base_cltv,
        "intervened_cltv": intervened_cltv,
        "cltv_gain": cltv_gain,
        "intervened_profile": compute_derived_features(intervened_dict)
    }


def generate_prescriptive_packages(
    customer_dict: Dict[str, Any],
    pipeline: Any
) -> List[Dict[str, Any]]:
    """
    Generates 3 standardized enterprise retention action packages:
    1. Operational Support Stabilization (Free Tech Support + Online Security add-on)
    2. Contract & Billing Modernization (1-Year Contract + Auto-Pay + 10% Discount)
    3. Enterprise VIP Long-Term Retention (2-Year Contract + Full Suite + 15% Discount)
    """
    orig_monthly = float(customer_dict.get("MonthlyCharges", 65.0))

    packages = []

    # Package 1: Support Stabilization
    p1_interventions = {
        "TechSupport": "Yes",
        "OnlineSecurity": "Yes"
    }
    sim1 = simulate_intervention(
        customer_dict,
        p1_interventions,
        pipeline,
        monthly_discount_usd=0.0,
        service_subsidy_cost_monthly=3.0  # $3/month internal cost of service
    )
    packages.append({
        "package_id": "pkg_support_stabilization",
        "name": "Operational Support Stabilization",
        "target_persona": "Support Deficient & Friction Prone",
        "description": "Provide complimentary 24/7 Tech Support and Online Security suite at zero customer charge to eliminate service dissatisfaction.",
        "actions": [
            "Enable 24/7 Priority Tech Support",
            "Provision Online Security suite",
            "Zero upfront contract modification required"
        ],
        "metrics": sim1
    })

    # Package 2: Contract Modernization
    discount_p2 = round(orig_monthly * 0.10, 2)  # 10% promotional concession
    p2_interventions = {
        "Contract": "One year",
        "PaymentMethod": "Bank transfer (automatic)" if "electronic check" in str(customer_dict.get("PaymentMethod", "")).lower() else customer_dict.get("PaymentMethod", "Bank transfer (automatic)")
    }
    sim2 = simulate_intervention(
        customer_dict,
        p2_interventions,
        pipeline,
        monthly_discount_usd=discount_p2,
        service_subsidy_cost_monthly=0.0
    )
    packages.append({
        "package_id": "pkg_contract_modernization",
        "name": "Annual Contract & Auto-Pay Modernization",
        "target_persona": "High Flight Risk / Month-to-Month",
        "description": "Convert volatile month-to-month subscription into a 1-year committed term with a 10% loyalty discount and automated bank payment.",
        "actions": [
            "Transition contract to 1-Year commitment",
            f"Apply 10% loyalty discount (${discount_p2:.2f}/month)",
            "Migrate payment method to automated transfer"
        ],
        "metrics": sim2
    })

    # Package 3: VIP Long-Term Retention Bundle
    discount_p3 = round(orig_monthly * 0.15, 2)  # 15% promotional concession
    p3_interventions = {
        "Contract": "Two year",
        "TechSupport": "Yes",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "PaymentMethod": "Credit card (automatic)"
    }
    sim3 = simulate_intervention(
        customer_dict,
        p3_interventions,
        pipeline,
        monthly_discount_usd=discount_p3,
        service_subsidy_cost_monthly=4.0
    )
    packages.append({
        "package_id": "pkg_vip_bundle",
        "name": "Enterprise VIP Long-Term Retention Bundle",
        "target_persona": "High-Value Account at Risk",
        "description": "Full-spectrum retention lock-in featuring a 2-year commitment, full security suite, automated credit card billing, and a 15% rate guarantee.",
        "actions": [
            "Transition contract to 2-Year guaranteed lock-in",
            "Provision full security bundle (Support, Security, Backup)",
            f"Apply 15% multi-year discount (${discount_p3:.2f}/month)",
            "Setup automated credit card billing"
        ],
        "metrics": sim3
    })

    return packages


def optimize_minimal_actionable_recourse(
    customer_dict: Dict[str, Any],
    pipeline: Any,
    target_risk: float = 0.30
) -> Dict[str, Any]:
    """
    Algorithmic Counterfactual Optimization:
    Finds the minimal-cost intervention vector that lowers predicted churn probability
    below the designated safety threshold (e.g. target_risk <= 0.30).
    """
    df_base = pd.DataFrame([compute_derived_features(customer_dict)])
    p_base = float(pipeline.predict_proba(df_base)[0, 1])

    if p_base <= target_risk:
        return {
            "status": "already_safe",
            "baseline_probability": round(p_base, 4),
            "target_risk": target_risk,
            "recommended_interventions": {},
            "annual_cost": 0.0,
            "net_saved": 0.0,
            "message": "Customer risk is already below the target threshold. No costly retention intervention required."
        }

    # Actionable search space
    contracts = ["Month-to-month", "One year", "Two year"]
    support_options = ["No", "Yes"]
    security_options = ["No", "Yes"]
    payment_options = [
        customer_dict.get("PaymentMethod", "Electronic check"),
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    discounts_pct = [0.0, 0.05, 0.10, 0.15, 0.20]

    orig_monthly = float(customer_dict.get("MonthlyCharges", 60.0))

    best_candidate = None
    min_cost = float("inf")
    fallback_candidate = None
    lowest_prob_seen = 1.0

    for contract in contracts:
        for support in support_options:
            for security in security_options:
                for pay in set(payment_options):
                    for disc_pct in discounts_pct:
                        discount_usd = round(orig_monthly * disc_pct, 2)
                        service_cost = 0.0
                        if support == "Yes" and customer_dict.get("TechSupport") != "Yes":
                            service_cost += 2.0
                        if security == "Yes" and customer_dict.get("OnlineSecurity") != "Yes":
                            service_cost += 1.0

                        annual_cost = (discount_usd + service_cost) * 12.0

                        candidate_dict = copy.deepcopy(customer_dict)
                        candidate_dict["Contract"] = contract
                        candidate_dict["TechSupport"] = support
                        candidate_dict["OnlineSecurity"] = security
                        candidate_dict["PaymentMethod"] = pay
                        candidate_dict["MonthlyCharges"] = max(18.0, orig_monthly - discount_usd)

                        df_cand = pd.DataFrame([compute_derived_features(candidate_dict)])
                        prob_cand = float(pipeline.predict_proba(df_cand)[0, 1])

                        if prob_cand < lowest_prob_seen:
                            lowest_prob_seen = prob_cand
                            fallback_candidate = {
                                "interventions": {
                                    "Contract": contract,
                                    "TechSupport": support,
                                    "OnlineSecurity": security,
                                    "PaymentMethod": pay,
                                    "monthly_discount_usd": discount_usd,
                                    "discount_pct": round(disc_pct * 100, 1)
                                },
                                "intervened_probability": round(prob_cand, 4),
                                "annual_cost": round(annual_cost, 2),
                                "risk_reduction": round(p_base - prob_cand, 4)
                            }

                        if prob_cand <= target_risk:
                            if annual_cost < min_cost:
                                min_cost = annual_cost
                                best_candidate = {
                                    "interventions": {
                                        "Contract": contract,
                                        "TechSupport": support,
                                        "OnlineSecurity": security,
                                        "PaymentMethod": pay,
                                        "monthly_discount_usd": discount_usd,
                                        "discount_pct": round(disc_pct * 100, 1)
                                    },
                                    "intervened_probability": round(prob_cand, 4),
                                    "annual_cost": round(annual_cost, 2),
                                    "risk_reduction": round(p_base - prob_cand, 4)
                                }

    selected = best_candidate if best_candidate is not None else fallback_candidate
    annual_revenue_base = orig_monthly * 12.0
    revenue_saved = round(selected["risk_reduction"] * annual_revenue_base, 2)
    net_value = round(revenue_saved - selected["annual_cost"], 2)
    roi = round((net_value / selected["annual_cost"] * 100.0), 1) if selected["annual_cost"] > 0 else 0.0

    return {
        "status": "recourse_found" if best_candidate is not None else "best_effort_found",
        "baseline_probability": round(p_base, 4),
        "target_risk": target_risk,
        "intervened_probability": selected["intervened_probability"],
        "risk_drop": selected["risk_reduction"],
        "recommended_interventions": selected["interventions"],
        "annual_cost": selected["annual_cost"],
        "annual_revenue_saved": revenue_saved,
        "net_value_saved": net_value,
        "roi_pct": roi,
        "message": (
            f"Achieved target risk ({selected['intervened_probability'] * 100:.1f}% <= {target_risk * 100:.1f}%) with minimal annual retention cost."
            if best_candidate is not None else
            f"Target {target_risk * 100:.1f}% unreachable within standard bounds; achieved best-effort risk of {selected['intervened_probability'] * 100:.1f}%."
        )
    }


def classify_retention_quadrant(churn_prob: float, monthly_charges: float) -> Tuple[str, str]:
    """
    Maps an account into the Strategic Retention Resource Allocation Matrix:
    - Quadrant 1: Immediate High-Touch Retention (High Risk, High Value)
    - Quadrant 2: Automated Self-Serve Incentives (Moderate Risk, Moderate/High Value)
    - Quadrant 3: Cost-Sensitive Monitoring (High Risk, Low Value)
    - Quadrant 4: Secure Base (Low Risk)
    """
    is_high_risk = churn_prob >= 0.60
    is_moderate_risk = 0.35 <= churn_prob < 0.60
    is_high_value = monthly_charges >= 70.0

    if is_high_risk and is_high_value:
        return (
            "Immediate High-Touch Retention",
            "Assign dedicated retention specialist. Proactively offer 1-year contract lock-in with bundled tech support and 10% rate guarantee."
        )
    elif is_moderate_risk and is_high_value:
        return (
            "Automated Self-Serve Incentives",
            "Trigger automated email campaign offering free digital security add-ons or $5/month auto-pay migration credit."
        )
    elif is_high_risk and not is_high_value:
        return (
            "Cost-Sensitive Monitoring",
            "Avoid expensive outbound specialist calls; offer digital self-serve plan optimization or lower tier bundle to prevent price churn."
        )
    elif is_moderate_risk and not is_high_value:
        return (
            "Automated Self-Serve Incentives",
            "Send digital newsletter and automated app notification highlighting unused plan benefits."
        )
    else:
        return (
            "Secure Base (Maintain)",
            "Account health is sound. Do not disrupt with promotional offers; consider eligible organic upselling."
        )


def analyze_cohort_prescriptive_matrix(
    df: pd.DataFrame,
    pipeline: Any
) -> Dict[str, Any]:
    """
    Computes cohort-level prescriptive economics across all records.
    """
    records = df.to_dict(orient="records")
    derived_records = [compute_derived_features(r) for r in records]
    df_eval = pd.DataFrame(derived_records)

    probs = pipeline.predict_proba(df_eval)[:, 1]
    df_eval["predicted_churn_risk"] = np.round(probs, 4)

    monthly_charges = df_eval["MonthlyCharges"] if "MonthlyCharges" in df_eval.columns else pd.Series(65.0, index=df_eval.index)
    df_eval["MonthlyCharges"] = monthly_charges

    quadrants = []
    actions = []
    for p, m in zip(probs, monthly_charges):
        q, act = classify_retention_quadrant(p, m)
        quadrants.append(q)
        actions.append(act)

    df_eval["retention_quadrant"] = quadrants
    df_eval["recommended_strategy"] = actions

    total_records = len(df_eval)
    high_touch_count = int((df_eval["retention_quadrant"] == "Immediate High-Touch Retention").sum())
    automated_count = int((df_eval["retention_quadrant"] == "Automated Self-Serve Incentives").sum())
    cost_sensitive_count = int((df_eval["retention_quadrant"] == "Cost-Sensitive Monitoring").sum())
    secure_count = int((df_eval["retention_quadrant"] == "Secure Base (Maintain)").sum())

    # Aggregate financial projections
    total_monthly_revenue = float(monthly_charges.sum())
    total_annual_revenue = total_monthly_revenue * 12.0

    monthly_risk_exposure = float((monthly_charges * probs).sum())
    annual_risk_exposure = round(monthly_risk_exposure * 12.0, 2)

    # Assume high-touch intervention saves 50% of targeted risk with $120/yr budget
    # Assume automated intervention saves 30% of targeted risk with $25/yr budget
    high_touch_df = df_eval[df_eval["retention_quadrant"] == "Immediate High-Touch Retention"]
    auto_df = df_eval[df_eval["retention_quadrant"] == "Automated Self-Serve Incentives"]

    revenue_saved_ht = float((high_touch_df["MonthlyCharges"] * high_touch_df["predicted_churn_risk"] * 12 * 0.50).sum())
    budget_ht = float(len(high_touch_df) * 120.0)

    revenue_saved_auto = float((auto_df["MonthlyCharges"] * auto_df["predicted_churn_risk"] * 12 * 0.30).sum())
    budget_auto = float(len(auto_df) * 25.0)

    total_recoverable = round(revenue_saved_ht + revenue_saved_auto, 2)
    total_budget_needed = round(budget_ht + budget_auto, 2)
    net_economic_benefit = round(total_recoverable - total_budget_needed, 2)
    program_roi = round((net_economic_benefit / total_budget_needed * 100.0), 1) if total_budget_needed > 0 else 0.0

    return {
        "total_records": total_records,
        "quadrant_counts": {
            "Immediate High-Touch Retention": high_touch_count,
            "Automated Self-Serve Incentives": automated_count,
            "Cost-Sensitive Monitoring": cost_sensitive_count,
            "Secure Base (Maintain)": secure_count
        },
        "financial_summary": {
            "annual_revenue_exposure": annual_risk_exposure,
            "total_recoverable_revenue": total_recoverable,
            "required_retention_budget": total_budget_needed,
            "net_economic_benefit": net_economic_benefit,
            "program_roi_pct": program_roi
        },
        "scored_dataframe": df_eval
    }
