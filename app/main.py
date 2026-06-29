# import custom classes so joblib can deserialize model.pkl
from app.custom_transformers import ScaledKNNImputer, SignupQuarterImputer
from app.Initial_validation import InitialValidation


import io
import json
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager



# ─────────────────────────────────────────
# 2. LOAD ARTIFACTS AT STARTUP
# ─────────────────────────────────────────
# loaded once when FastAPI starts — not on every request

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all artifacts once at startup."""
    global pipeline, explainer, threshold, feature_names

    pipeline      = joblib.load("artifacts/model.pkl")
    explainer     = joblib.load("artifacts/shap_explainer.pkl")

    with open("artifacts/threshold_config.json") as f:
        threshold = json.load(f)["threshold"]

    # get feature names from trained pipeline
    feature_names = pipeline.named_steps["pre_block"].get_feature_names_out()

    print(f"model.pkl loaded")
    print(f"shap_explainer.pkl loaded")
    print(f"threshold loaded: {threshold:.4f}")

    yield   # app runs here

    print("Shutting down...")


# ─────────────────────────────────────────
# 3. FASTAPI APP
# ─────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction and Retention API",
    description="Predicts churn probability for customers using XGBoost + SHAP",
    version="1.0.0",
    lifespan=lifespan
)

# allow Streamlit to call FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# 4. HELPER FUNCTIONS
# ─────────────────────────────────────────

def clean_feature_name(raw_name: str) -> str:
    """
    Converts pipeline feature names to readable display names.
    "knn_block__Email_Open_Rate"       → "Email Open Rate"
    "ohe_block__Country_France"        → "Country: France"
    "ohe_block__Signup_Quarter_Q4"     → "Signup Quarter: Q4"
    "remainder__Lifetime_Value"        → "Lifetime Value"
    """
    name = raw_name.split("__")[-1]   # strip pipeline prefix

    if name.startswith("Country_"):
        return "Country: " + name.replace("Country_", "")
    if name.startswith("Signup_Quarter_"):
        return "Signup Quarter: " + name.replace("Signup_Quarter_", "")
    if name.startswith("Gender_"):
        return "Gender: " + name.replace("Gender_", "")

    return name.replace("_", " ")


def assign_risk_tier(probability: float) -> str:
    """Maps churn probability to risk tier."""
    if probability >= 0.70:
        return "High Risk"
    elif probability >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


ACTION_MAP = {
    "Discount Usage Rate":           "🎟️ Offer a personalized discount coupon",
    "Email Open Rate":               "📧 Re-engage with email campaign",
    "Cart Abandonment Rate":         "🛒 Send cart recovery reminder",
    "Customer Service Calls":        "📞 Assign a dedicated support agent",
    "Returns Rate":                  "🔄 Investigate product quality issues",
    "Days Since Last Purchase":      "⏰ Send a win-back campaign",
    "Login Frequency":               "📱 Send app re-engagement notification",
    "Session Duration Avg":          "💡 Improve website UX / recommend content",
    "Lifetime Value":                "👑 Assign to VIP retention program",
    "Average Order Value":           "🎁 Offer bundle deals or upsell",
    "Membership Years":              "🏅 Send loyalty reward or recognition",
    "Mobile App Usage":              "📲 Send push notification via app",
    "Social Media Engagement Score": "📣 Target with social media campaign",
    "Credit Balance":                "💳 Offer credit utilization incentive",
    "Total Purchases":               "🛍️ Send product recommendation email",
    "Pages Per Session":             "🔍 Improve product discovery experience",
    "Wishlist Items":                "💝 Send wishlist reminder with discount",
    "Product Reviews Written":       "⭐ Engage with review incentive program",
    "Age":                           "👥 Target with age-appropriate campaign",
    "City":                          "📍 Run location-based promotion",
    "Country":                       "🌍 Run region-specific campaign",
    "Signup Quarter":                "📅 Send anniversary or seasonal offer",
}


def get_action(clean_name: str) -> str:
    """Maps clean feature name to business action."""
    # handle OHE features like "Country: France" → "Country"
    base = clean_name.split(":")[0].strip()
    return ACTION_MAP.get(base, "📋 Manual review recommended")


def get_top1_reason(shap_row: np.ndarray) -> tuple[str, str]:
    df_impact = pd.DataFrame({
        "feature":    feature_names,
        "shap_value": shap_row
    })

    # only features pushing TOWARD churn (positive SHAP)
    df_churn_drivers = df_impact[df_impact["shap_value"] > 0]
    df_top           = df_churn_drivers.sort_values("shap_value", ascending=False)

    if df_top.empty:
        # fallback — take absolute max if no positive values
        raw_name = df_impact.loc[df_impact["shap_value"].abs().idxmax(), "feature"]
    else:
        raw_name = df_top["feature"].iloc[0]

    clean_name = clean_feature_name(raw_name)
    action     = get_action(clean_name)
    return clean_name, action


# ─────────────────────────────────────────
# 5. PYDANTIC MODEL FOR SINGLE PREDICTION
# ─────────────────────────────────────────

class SingleCustomer(BaseModel):
    Age:                          float
    Gender:                       str
    Country:                      str
    City:                         str
    Membership_Years:             float
    Login_Frequency:              float
    Session_Duration_Avg:         float
    Pages_Per_Session:            float
    Cart_Abandonment_Rate:        float
    Wishlist_Items:               float
    Total_Purchases:              float
    Average_Order_Value:          float
    Days_Since_Last_Purchase:     float
    Discount_Usage_Rate:          float
    Returns_Rate:                 float
    Email_Open_Rate:              float
    Customer_Service_Calls:       float
    Product_Reviews_Written:      float
    Social_Media_Engagement_Score: float
    Mobile_App_Usage:             float
    Payment_Method_Diversity:     float
    Lifetime_Value:               float
    Credit_Balance:               float
    Signup_Quarter:               str


# ─────────────────────────────────────────
# 6. ENDPOINTS
# ─────────────────────────────────────────

@app.get("/health")
def health_check():
    """Check if API is alive and artifacts are loaded."""
    return {
        "status":    "ok",
        "threshold": threshold,
        "message":   "Churn prediction API is running"
    }


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Accepts a CSV file upload.
    Returns predictions for all customers with risk tier,
    top reason, recommended action, and retention priority score.
    """

    # ── Read uploaded CSV ─────────────────────────────────────────────────────
    try:
        contents = await file.read()
        df       = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {str(e)}")

    # ── Validate ──────────────────────────────────────────────────────────────
    validator  = InitialValidation()
    val_result = validator.validate(df)

    if not val_result["is_valid"]:
        return {
            "is_valid":       False,
            "message":        val_result["message"],
            "missing_cols":   val_result["missing_cols"],
            "predictions":    [],
            "summary":        {},
            "invalid_report": [],
            "missing_report": [],
        }

    clean_df       = val_result["clean_df"]
    invalid_report = val_result["invalid_data_report"]
    missing_report = val_result["missing_report"]

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        probas      = pipeline.predict_proba(clean_df)[:, 1]
        predictions = (probas >= threshold).astype(int)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # ── Risk tiers ────────────────────────────────────────────────────────────
    risk_tiers = [assign_risk_tier(p) for p in probas]

    # ── SHAP — only for High Risk customers ───────────────────────────────────
    try:
        X_transformed  = pipeline.named_steps["pre_block"].transform(clean_df)
        high_risk_mask = np.array(risk_tiers) == "High Risk"
        shap_values    = explainer.shap_values(X_transformed[high_risk_mask])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP error: {str(e)}")

    # ── Revenue calculation ───────────────────────────────────────────────────
    # fill missing Lifetime_Value with median for revenue cards
    ltv = clean_df["Lifetime_Value"].copy()
    ltv = ltv.fillna(ltv.median())

    # ── Build predictions list ────────────────────────────────────────────────
    results       = []
    shap_idx      = 0   # tracks position in shap_values (only high risk rows)

    for i in range(len(clean_df)):
        tier = risk_tiers[i]

        if tier == "High Risk":
            top_reason, action = get_top1_reason(shap_values[shap_idx])
            shap_idx += 1
        else:
            top_reason = "-"
            action     = "-"

        retention_score = round(float(probas[i]) * float(ltv.iloc[i]), 2)

        results.append({
            "index":                    i,
            "churn_probability":        round(float(probas[i]), 4),
            "risk_tier":                tier,
            "top_reason":               top_reason,
            "recommended_action":       action,
            "retention_priority_score": retention_score,
            "lifetime_value":           round(float(ltv.iloc[i]), 2),
        })

    # ── Summary cards ─────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)

    high_mask   = results_df["risk_tier"] == "High Risk"
    medium_mask = results_df["risk_tier"] == "Medium Risk"
    low_mask    = results_df["risk_tier"] == "Low Risk"

    summary = {
        "total_customers":      len(results_df),
        "high_risk":            int(high_mask.sum()),
        "medium_risk":          int(medium_mask.sum()),
        "low_risk":             int(low_mask.sum()),
        "revenue_at_risk":      round(float(ltv[high_risk_mask].sum()), 2),
        "high_risk_revenue":    round(float(results_df.loc[high_mask,   "lifetime_value"].sum()), 2),
        "medium_risk_revenue":  round(float(results_df.loc[medium_mask, "lifetime_value"].sum()), 2),
        "low_risk_revenue":     round(float(results_df.loc[low_mask,    "lifetime_value"].sum()), 2),
        "total_revenue":        round(float(ltv.sum()), 2),
    }

    return {
        "is_valid":       True,
        "message":        val_result["message"],
        "summary":        summary,
        "predictions":    results,
        "invalid_report": invalid_report.to_dict(orient="records"),
        "missing_report": missing_report.to_dict(orient="records"),
    }


@app.post("/predict/single")
def predict_single(customer: SingleCustomer):
    """
    Accepts a single customer as JSON.
    Returns churn probability, risk tier, top reason, recommended action.
    """

    # ── Build single row DataFrame ────────────────────────────────────────────
    customer_df = pd.DataFrame([customer.model_dump()])

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        proba      = float(pipeline.predict_proba(customer_df)[0, 1])
        prediction = int(proba >= threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    tier = assign_risk_tier(proba)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    try:
        X_transformed       = pipeline.named_steps["pre_block"].transform(customer_df)
        shap_row            = explainer.shap_values(X_transformed)[0]
        top_reason, action  = get_top1_reason(shap_row)
    except Exception as e:
        top_reason = "unavailable"
        action     = "📋 Manual review recommended"

    return {
        "churn_probability":  round(proba, 4),
        "churn_prediction":   prediction,
        "risk_tier":          tier,
        "top_reason":         top_reason,
        "recommended_action": action,
    }
