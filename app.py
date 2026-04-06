import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}
h1 {
    text-align: center;
    color: #FF4B4B;
}
div.stButton > button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ---------------- #
svm_model = pickle.load(open("svm_model.pkl", "rb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))
meta_model = pickle.load(open("meta_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))
base_customer_dict = pickle.load(open("base_customer.pkl", "rb"))

# SHAP explainer for Random Forest
explainer = shap.TreeExplainer(rf_model)

# ---------------- TITLE ---------------- #
st.title("Customer Churn Prediction")

st.write("Enter customer details to predict churn probability and risk level.")

# ---------------- INPUT SECTION ---------------- #
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72)
    monthly_charges = st.number_input("Monthly Charges")
    referrals = st.number_input("Referrals")

with col2:
    total_charges = st.number_input("Total Charges")
    age = st.slider("Age", 18, 80, 30)

st.write("")
st.write("")

# ---------------- PREDICTION ---------------- #
if st.button("Predict Churn"):

    # Create customer from base
    df = pd.DataFrame([base_customer_dict])

    # Update user inputs
    df['TenureinMonths'] = tenure
    df['MonthlyCharge'] = monthly_charges
    df['TotalCharges'] = total_charges
    df['NumberofReferrals'] = referrals
    df['Age'] = age

    # Ensure column order
    df = df[model_columns]

    # Scale
    df_scaled = scaler.transform(df)

    # Base model predictions
    svm_prob = svm_model.predict_proba(df_scaled)[:,1]
    rf_prob = rf_model.predict_proba(df_scaled)[:,1]

    # Meta model prediction
    meta_input = np.column_stack((svm_prob, rf_prob))
    final_prob = meta_model.predict_proba(meta_input)[:,1][0]

    # Risk level
    if final_prob > 0.7:
        risk = "High Risk - Give Discount"
    elif final_prob > 0.4:
        risk = "Medium Risk - Offer Plan"
    else:
        risk = "Low Risk - No Action"

    # ---------------- RESULT UI ---------------- #
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", round(final_prob, 2))

    with col2:
        if final_prob > 0.7:
            st.error(risk)
        elif final_prob > 0.4:
            st.warning(risk)
        else:
            st.success(risk)

    # ---------------- SHAP + SMART EXPLANATION ---------------- #
    
    st.subheader("Why this customer may churn:")
    
    # SHAP values (USE SCALED DATA)
    shap_values = explainer.shap_values(df_scaled)
    
    # Correct extraction
    shap_val = shap_values[:, :, 1][0]
    
    # Feature importance
    feature_importance = sorted(
        zip(model_columns, shap_val),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]
    
    # Filter noise
    threshold = 0.02
    feature_importance = [
        (f, v) for f, v in feature_importance if abs(v) > threshold
    ]
    
    # Show table
    shap_df = pd.DataFrame(feature_importance, columns=["Feature", "Impact"])
    st.subheader("Top Factors Affecting Churn")
    st.dataframe(shap_df)
    
    
    # ---------------- SMART REASON GENERATOR ---------------- #
    
    feature_impact = {
        'TenureinMonths': 'low',
        'MonthlyCharge': 'high',
        'TotalCharges': 'low',
        'InternetType_Fiber Optic': 'high',
        'Contract_Two year': 'good',
        'Contract_One year': 'good',
        'OnlineSecurity_Yes': 'good',
        'TechSupport_Yes': 'good',
        'NumberofReferrals': 'low',
        'PremiumTechSupport_Yes': 'good',
        'ChurnCategory_Competitor': 'high'
    }
    
    reason_map = {
        'TenureinMonths': 'Customer tenure',
        'MonthlyCharge': 'Monthly charges',
        'TotalCharges': 'Total spending',
        'InternetType_Fiber Optic': 'Fiber internet plan',
        'Contract_Two year': 'Long-term contract',
        'Contract_One year': 'Yearly contract',
        'OnlineSecurity_Yes': 'Online security',
        'TechSupport_Yes': 'Tech support',
        'NumberofReferrals': 'Customer referrals',
        'PremiumTechSupport_Yes': 'Premium support',
        'ChurnCategory_Competitor': 'Competitor influence'
    }
    
    st.subheader("Explanation in Simple Terms:")
    
    readable_reasons = []
    
    for feature, value in feature_importance:
        if feature in reason_map:
            impact_type = feature_impact.get(feature, 'neutral')
            feature_name = reason_map[feature]
    
            if value > 0:
                if impact_type in ['low', 'bad']:
                    readable_reasons.append(f"🔴 Risk Increase: Low {feature_name}")
                elif impact_type == 'high':
                    readable_reasons.append(f"🔴 Risk Increase: High {feature_name}")
                else:
                    readable_reasons.append(f"🔴 Risk Increase: {feature_name}")
            else:
                if impact_type == 'good':
                    readable_reasons.append(f"🟢 Risk Decrease: Strong {feature_name}")
                elif impact_type == 'high':
                    readable_reasons.append(f"🟢 Risk Decrease: Low {feature_name}")
                elif impact_type == 'low':
                    readable_reasons.append(f"🟢 Risk Decrease: High {feature_name}")
                else:
                    readable_reasons.append(f"🟢 Risk Decrease: {feature_name}")
    
    # Filter based on prediction
    if final_prob >= 0.75:
        readable_reasons = [r for r in readable_reasons if "Increase" in r]
    elif final_prob < 0.45:
        readable_reasons = [r for r in readable_reasons if "Decrease" in r]
    
    # Top 3 only
    readable_reasons = readable_reasons[:3]
    
    # Display
    for reason in readable_reasons:
        st.write(reason)
