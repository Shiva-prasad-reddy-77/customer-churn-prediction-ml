Customer Churn Prediction using Machine Learning & Explainable AI
📌 Overview

This project predicts customer churn using a machine learning stacking model and provides explainable insights using SHAP (Explainable AI). In addition to predicting churn probability, the system also recommends business actions and highlights the key factors influencing the prediction for each customer.

This project is designed as a Decision Support System for businesses to identify high-risk customers and take preventive actions to improve customer retention.

🚀 Key Features
Customer churn prediction using Machine Learning
Stacking Ensemble Model (SVM + Random Forest + Logistic Regression)
Explainable AI using SHAP values
Customer-level churn reason generation
Business decision system (Discount / Personalized Offer / No Action)
Real-time prediction for new customers
Streamlit web application for user interface
🧠 Technologies Used
Python
Scikit-learn
Pandas
NumPy
SHAP (Explainable AI)
Streamlit (Web App)
Matplotlib (Visualization)
📊 Project Workflow
Data Collection
Data Preprocessing
Feature Engineering
Data Scaling (StandardScaler)
Model Training
Support Vector Machine (SVM)
Random Forest (RF)
Stacking Ensemble Model
Meta Model: Logistic Regression
Model Evaluation
Explainable AI using SHAP
Business Decision Engine
Streamlit Web App Deployment
📈 Model Architecture

Base Models:

Support Vector Machine (SVM)
Random Forest (RF)

Meta Model:

Logistic Regression

Final Output:

Churn Probability
Risk Category (High / Medium / Low)
Recommended Business Action
Key Contributing Factor (Reason)
🎯 Business Decision Rules
Churn Probability	Risk Level	Recommended Action
≥ 0.75	High Risk	Give Discount
0.45 – 0.75	Medium Risk	Personalized Offer
< 0.45	Low Risk	No Action
🧾 Example Output

Customer Churn Prediction Report

Churn Probability : 0.78
Risk Level        : High Risk
Recommended Action: Give Discount
Main Reason       : Low Customer Tenure
🧠 Explainable AI (SHAP)

SHAP (SHapley Additive exPlanations) is used to explain the model predictions by identifying the most important feature contributing to churn for each customer. The system extracts the top contributing feature and presents it as the main reason for churn risk.

This makes the model transparent, interpretable, and business-friendly.

💼 Business Use Case

This system helps companies:

Identify customers likely to churn
Provide targeted offers to high-risk customers
Improve customer retention
Reduce revenue loss
Make data-driven business decisions
🖥️ Streamlit Web App

The project includes a Streamlit web application where users can:

Enter customer details
Predict churn probability
View risk level
Get recommended action
See the main reason for churn

Run the app using:

streamlit run app.py
📂 Project Structure
customer-churn-prediction/
│
├── data/
├── models/
├── app.py
├── churn_model.pkl
├── scaler.pkl
├── README.md
└── requirements.txt
📌 Future Improvements
Add more advanced models (XGBoost, LightGBM)
Deploy on cloud (AWS / Azure / Render)
Add customer segmentation
Add dashboard analytics
Integrate with CRM system
👨‍💻 Author

Shiva Prasad
Machine Learning & Data Science Enthusiast

⭐ Conclusion

This project is a complete end-to-end Machine Learning project that combines:

Machine Learning
Ensemble Learning
Explainable AI
Business Decision System
Web Application

It demonstrates how AI can be used not just for prediction, but also for business decision support and customer retention strategies.
