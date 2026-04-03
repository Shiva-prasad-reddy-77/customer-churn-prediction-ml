import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# Load models
svm_model = pickle.load(open("svm_model.pkl", "rb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))
meta_model = pickle.load(open("meta_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))
base_customer_dict = pickle.load(open("base_customer.pkl", "rb"))

# SHAP explainer for Random Forest
explainer = shap.TreeExplainer(rf_model)

st.title("Customer Churn Prediction")

tenure = st.slider("Tenure", 0, 72)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")
referrals = st.number_input("Referrals")
age = st.slider("Age", 18, 80, 30)

if st.button("Predict"):

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

    st.write("Churn Probability:", round(final_prob, 2))
    st.write("Risk Level:", risk)

    # ---------------- SHAP EXPLANATION ---------------- #

    # ---------------- SHAP EXPLANATION ---------------- #

    # ---------------- SHAP EXPLANATION ---------------- #

    st.subheader("Why this customer may churn:")

    shap_values = explainer.shap_values(df)

    # Convert SHAP values to numpy array safely
    if isinstance(shap_values, list):
        shap_array = np.array(shap_values[1])
    else:
        shap_array = np.array(shap_values)

    # Make it 1D
    shap_impact = shap_array.reshape(-1)

    # If SHAP length > features, trim it
    if len(shap_impact) > len(df.columns):
        shap_impact = shap_impact[:len(df.columns)]

    # If SHAP length < features, pad with zeros
    if len(shap_impact) < len(df.columns):
        shap_impact = np.pad(shap_impact, (0, len(df.columns) - len(shap_impact)))

    # Create dataframe
    shap_df = pd.DataFrame({
        'Feature': df.columns,
        'Impact': shap_impact
    })

    # Sort by importance
    shap_df = shap_df.sort_values(by='Impact', key=abs, ascending=False)

    # Show top 5 features
    st.write(shap_df.head(5))

    # Simple explanation
    st.subheader("Explanation in Simple Terms:")

    top_features = shap_df.head(3)

    for i, row in top_features.iterrows():
        if row['Impact'] > 0:
            st.write(f"{row['Feature']} is increasing churn risk")
        else:
            st.write(f"{row['Feature']} is decreasing churn risk")

    # SHAP chart
    st.subheader("Feature Impact on Churn")
    shap.summary_plot(shap_values, df, plot_type="bar", show=False)
    st.pyplot(plt)