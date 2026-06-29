import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")



@st.cache_data
def get_driver_counts(df):
    return (
        df[df["risk_tier"] == "High Risk"]["top_reason"]
        .value_counts()
        .head(10)
        .sort_values()
    )
    
    
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False

# ─────────────────────────────────────────
# HELPER — format currency
# ─────────────────────────────────────────
def fmt_currency(value: float) -> str:
    if value >= 1_000_000:
        return f"₹{value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"₹{value/1_000:.1f}K"
    return f"₹{value:.0f}"

# ─────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────
st.title("📊 Customer Churn Prediction & Retention System")
st.caption("Upload customer data to predict churn risk and get actionable retention recommendations.")

# API status
if check_api():
    st.success("✅ API is online and ready", icon="🟢")
else:
    st.error("❌ API is offline. Make sure FastAPI is running.", icon="🔴")
    st.stop()

st.divider()

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2 = st.tabs(["📁 Batch Prediction", "👤 Single Customer"])


with tab1:
    st.subheader("Upload Customer Dataset")
    uploaded = st.file_uploader("Choose a CSV file", type="csv", key="batch_upload")

    # ── API call — only runs when new file uploaded ──────────────────────
    if uploaded:
        # only call API if this is a NEW upload
        file_id = uploaded.name + str(uploaded.size)
        if st.session_state.get("last_file_id") != file_id:
            st.session_state["last_file_id"] = file_id

            with st.spinner("Running predictions..."):
                response = requests.post(
                    f"{API_URL}/predict/batch",
                    files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
                )

            if response.status_code != 200:
                st.error(f"API error: {response.status_code}")
                st.stop()

            data = response.json()
            st.session_state["batch_data"]    = data
            st.session_state["results_df"]    = pd.DataFrame(data["predictions"])

    # ── Display — runs on every rerun but uses cached session state ──────
    if "batch_data" in st.session_state:
        data       = st.session_state["batch_data"]
        results_df = st.session_state["results_df"]

        if not data["is_valid"]:
            st.error(data["message"])
            if data.get("missing_cols"):
                st.write(", ".join(data["missing_cols"]))
            st.stop()

        # warnings
        if data.get("invalid_report"):
            with st.expander("⚠️ Invalid values detected"):
                st.dataframe(pd.DataFrame(data["invalid_report"]), use_container_width=True)

        if data.get("missing_report"):
            with st.expander("ℹ️ Missing values detected"):
                st.dataframe(pd.DataFrame(data["missing_report"]), use_container_width=True)

        # summary cards
        st.subheader("📈 Summary")
        s = data["summary"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Customers",    f"{s['total_customers']:,}")
        c2.metric("🔴 High Risk",       f"{s['high_risk']:,}", f"{s['high_risk']/s['total_customers']*100:.1f}%")
        c3.metric("🟡 Medium Risk",     f"{s['medium_risk']:,}", f"{s['medium_risk']/s['total_customers']*100:.1f}%")
        c4.metric("🟢 Low Risk",        f"{s['low_risk']:,}", f"{s['low_risk']/s['total_customers']*100:.1f}%")
        c5.metric("💰 Revenue at Risk", fmt_currency(s["revenue_at_risk"]))

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Total Revenue",       fmt_currency(s["total_revenue"]))
        r2.metric("High Risk Revenue",   fmt_currency(s["high_risk_revenue"]))
        r3.metric("Medium Risk Revenue", fmt_currency(s["medium_risk_revenue"]))
        r4.metric("Low Risk Revenue",    fmt_currency(s["low_risk_revenue"]))

        st.divider()

        # charts
        ch1, ch2 = st.columns(2)
        with ch1:
            st.subheader("Churn Risk Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            sizes   = [s["high_risk"], s["medium_risk"], s["low_risk"]]
            colors  = ["#e05c5c", "#f0a500", "#4caf50"]
            ax.pie(sizes, labels=["High Risk", "Medium Risk", "Low Risk"],
                   colors=colors, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)
            plt.close()

        with ch2:
            st.subheader("Top Churn Drivers")
            high_df = results_df[results_df["risk_tier"] == "High Risk"]
            if not high_df.empty:
                driver_counts = high_df["top_reason"].value_counts().head(10).sort_values()
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                driver_counts.plot(kind="barh", ax=ax2, color="#e05c5c")
                ax2.set_xlabel("Number of Customers")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

        st.divider()

        # ── Filters — these are fast, just filter in memory ──────────────
        st.subheader("🎯 Prediction Results")

        all_actions = sorted(
            results_df[results_df["recommended_action"] != "-"]["recommended_action"]
            .unique().tolist()
        )
        selected_action = st.selectbox(
            "Filter by Recommended Action:",
            ["Show All"] + all_actions,
            key="action_filter"
        )
        tier_filter = st.radio(
            "Filter by Risk Tier:",
            ["All", "High Risk", "Medium Risk", "Low Risk"],
            horizontal=True,
            key="tier_filter"
        )

        # apply filters — pure in-memory, instant
        display_df = results_df.copy()
        if tier_filter != "All":
            display_df = display_df[display_df["risk_tier"] == tier_filter]
        if selected_action != "Show All":
            display_df = display_df[display_df["recommended_action"] == selected_action]

        st.caption(f"Showing {len(display_df):,} customers")
        st.dataframe(
            display_df[[
                "index", "churn_probability", "risk_tier",
                "lifetime_value", "retention_priority_score",
                "top_reason", "recommended_action"
            ]].rename(columns={
                "index":                    "Customer #",
                "churn_probability":        "Churn Probability",
                "risk_tier":                "Risk Tier",
                "lifetime_value":           "Lifetime Value (₹)",
                "retention_priority_score": "Retention Priority Score",
                "top_reason":               "Top Churn Reason",
                "recommended_action":       "Recommended Action",
            }),
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.subheader("⬇️ Download Results")
        d1, d2, d3 = st.columns(3)

        with d1:
            st.download_button("Download All", results_df.to_csv(index=False),
                               "all_predictions.csv", "text/csv")
        with d2:
            high_csv = results_df[results_df["risk_tier"] == "High Risk"].to_csv(index=False)
            st.download_button("Download High Risk Only", high_csv,
                               "high_risk_customers.csv", "text/csv")
        with d3:
            if selected_action != "Show All":
                st.download_button("Download Filtered", display_df.to_csv(index=False),
                                   "filtered_customers.csv", "text/csv")
                
                
with tab2:
    st.subheader("Predict Churn for a Single Customer")
    st.caption("Fill in the customer details and click Predict.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        age              = st.number_input("Age",                    18, 90,  35)
        gender           = st.selectbox("Gender",                    ["Male", "Female", "Other"])
        country          = st.text_input("Country",                  "India")
        city             = st.text_input("City",                     "Mumbai")
        signup_quarter   = st.selectbox("Signup Quarter",            ["Q1","Q2","Q3","Q4"])
        membership_years = st.number_input("Membership Years",       0,  20,  3)

        st.markdown("**Purchase Behaviour**")
        total_purchases      = st.number_input("Total Purchases",        0,   500,  20)
        avg_order_value      = st.number_input("Average Order Value",    0.0, 10000.0, 500.0)
        days_since_purchase  = st.number_input("Days Since Last Purchase", 0, 365,  30)

    with col2:
        st.markdown("**Engagement**")
        login_frequency  = st.number_input("Login Frequency",       0,   100,  10)
        session_duration = st.number_input("Session Duration Avg",  0.0, 300.0, 30.0)
        pages_per_session= st.number_input("Pages Per Session",     0.0, 50.0,  5.0)
        mobile_app_usage = st.number_input("Mobile App Usage",      0.0, 100.0, 20.0)
        social_score     = st.number_input("Social Media Engagement Score", 0.0, 100.0, 50.0)
        email_open_rate  = st.number_input("Email Open Rate (%)",   0.0, 100.0, 30.0)

        st.markdown("**Cart & Returns**")
        cart_abandonment = st.number_input("Cart Abandonment Rate (%)", 0.0, 100.0, 40.0)
        returns_rate     = st.number_input("Returns Rate (%)",          0.0, 100.0, 10.0)
        discount_usage   = st.number_input("Discount Usage Rate (%)",   0.0, 100.0, 20.0)

    with col3:
        st.markdown("**Support & Reviews**")
        customer_service = st.number_input("Customer Service Calls", 0,  50,   2)
        product_reviews  = st.number_input("Product Reviews Written",0,  100,  3)
        payment_diversity= st.number_input("Payment Method Diversity",0, 10,   2)
        wishlist_items   = st.number_input("Wishlist Items",         0,  100,  5)

        st.markdown("**Financials**")
        lifetime_value   = st.number_input("Lifetime Value (₹)",    0.0, 500000.0, 10000.0)
        credit_balance   = st.number_input("Credit Balance (₹)",    0.0, 100000.0, 5000.0)

    st.divider()

    if st.button("🔍 Predict Churn Risk", type="primary", use_container_width=True):
        customer_payload = {
            "Age":                           float(age),
            "Gender":                        gender,
            "Country":                       country,
            "City":                          city,
            "Signup_Quarter":                signup_quarter,
            "Membership_Years":              float(membership_years),
            "Login_Frequency":               float(login_frequency),
            "Session_Duration_Avg":          float(session_duration),
            "Pages_Per_Session":             float(pages_per_session),
            "Cart_Abandonment_Rate":         float(cart_abandonment),
            "Wishlist_Items":                float(wishlist_items),
            "Total_Purchases":               float(total_purchases),
            "Average_Order_Value":           float(avg_order_value),
            "Days_Since_Last_Purchase":      float(days_since_purchase),
            "Discount_Usage_Rate":           float(discount_usage),
            "Returns_Rate":                  float(returns_rate),
            "Email_Open_Rate":               float(email_open_rate),
            "Customer_Service_Calls":        float(customer_service),
            "Product_Reviews_Written":       float(product_reviews),
            "Social_Media_Engagement_Score": float(social_score),
            "Mobile_App_Usage":              float(mobile_app_usage),
            "Payment_Method_Diversity":      float(payment_diversity),
            "Lifetime_Value":                float(lifetime_value),
            "Credit_Balance":                float(credit_balance),
        }

        with st.spinner("Predicting..."):
            response = requests.post(
                f"{API_URL}/predict/single",
                json=customer_payload
            )

        if response.status_code != 200:
            st.error(f"API error: {response.status_code} — {response.text}")
        else:
            result = response.json()

            st.divider()
            st.subheader("Prediction Result")

            tier_colors = {
                "High Risk":   "🔴",
                "Medium Risk": "🟡",
                "Low Risk":    "🟢"
            }
            icon = tier_colors.get(result["risk_tier"], "⚪")

            r1, r2, r3 = st.columns(3)
            r1.metric("Churn Probability",
                      f"{result['churn_probability']:.1%}")
            r2.metric("Risk Tier",
                      f"{icon}  {result['risk_tier']}")
            r3.metric("Prediction",
                      "Will Churn" if result["churn_prediction"] == 1 else "Will Stay")

            st.divider()

            st.subheader("💡 Retention Recommendation")
            a1, a2 = st.columns(2)
            a1.info(f"**Top Churn Reason:**\n\n{result['top_reason']}")
            a2.success(f"**Recommended Action:**\n\n{result['recommended_action']}")
