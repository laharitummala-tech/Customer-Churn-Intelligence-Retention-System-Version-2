import sys
sys.path.insert(0, ".")  # makes app.custom_transformers importable
import shap
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from category_encoders import CountEncoder
from xgboost import XGBClassifier

# import from app package — this is the key
from app.custom_transformers import ScaledKNNImputer, SignupQuarterImputer

# ── Load data ─────────────────────────────────────────────

df = pd.read_csv(r"C:\Users\91891\Downloads\ecommerce_customer_churn_dataset.csv")
X = df.drop(columns=["Churned"])
y = df["Churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Feature groups ────────────────────────────────────────
knn_features = [
    "Social_Media_Engagement_Score", "Session_Duration_Avg",
    "Pages_Per_Session",             "Mobile_App_Usage",
    "Login_Frequency",               "Email_Open_Rate",
    "Cart_Abandonment_Rate",         "Credit_Balance",
    "Wishlist_Items",                "Product_Reviews_Written",
]
median_features = ["Returns_Rate", "Days_Since_Last_Purchase", "Age", "Total_Purchases"]
mode_features   = ["Payment_Method_Diversity", "Customer_Service_Calls"]
mar_features    = ["Discount_Usage_Rate", "Signup_Quarter"]
ohe_features    = ["Gender", "Country", "Signup_Quarter"]
freq_features   = ["City"]

# ── Pipeline ──────────────────────────────────────────────
num_negatives = int(np.sum(y_train == 0))
num_positives = int(np.sum(y_train == 1))

tuned_params = {
    "max_depth":        3,
    "min_child_weight": 6,
    "subsample":        0.6576506716888342,
    "colsample_bytree": 0.6409731188976354,
    "reg_alpha":        0.040457936191353315,
    "reg_lambda":       1.8923567798640542,
    "learning_rate":    0.05192364617661904,
    "n_estimators":     400,
    "scale_pos_weight": num_negatives / num_positives,
    "eval_metric":      "logloss",
    "random_state":     42,
}

preprocessor_blocks = ColumnTransformer(
    transformers=[
        ("knn_block",         ScaledKNNImputer(n_neighbors=5),                                          knn_features),
        ("median_block",      SimpleImputer(strategy="median"),                                          median_features),
        ("mar_quarter_block", SignupQuarterImputer(),                                                    mar_features),
        ("mode_block",        SimpleImputer(strategy="most_frequent"),                                   mode_features),
        ("ohe_block",         OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False), ohe_features),
        ("freq_encod_block",  CountEncoder(normalize=True, handle_unknown=0),                            freq_features),
    ],
    remainder="passthrough"
)

final_pipeline = Pipeline(steps=[
    ("pre_block", preprocessor_blocks),
    ("model",     XGBClassifier(**tuned_params)),
])

# ── Train ─────────────────────────────────────────────────
final_pipeline.fit(X_train, y_train)

# ── Threshold ─────────────────────────────────────────────
y_proba = final_pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
optimal_threshold = float(thresholds[np.argmax(tpr - fpr)])

# ── Save artifacts ────────────────────────────────────────
joblib.dump(final_pipeline, "artifacts/model.pkl")
with open("artifacts/threshold_config.json", "w") as f:
    json.dump({"threshold": optimal_threshold}, f, indent=2)

print("artifacts/model.pkl saved")
print(f"artifacts/threshold_config.json saved (threshold={optimal_threshold:.4f})")

raw_booster  = final_pipeline.named_steps["model"].get_booster()
explainer    = shap.TreeExplainer(raw_booster, model_output="raw")
joblib.dump(explainer, "artifacts/shap_explainer.pkl")
print("✅ artifacts/shap_explainer.pkl saved")