## 📊 Project Workflow
# 📊 Customer Churn Prediction using Machine Learning & Explainable AI

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![SHAP](https://img.shields.io/badge/Explainable-AI-green)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

---

## 📌 Project Overview
This project predicts whether a customer is likely to **churn (leave the company)** using Machine Learning.  
It not only predicts churn probability but also provides:

- Risk category (High / Medium / Low)
- Recommended business action
- Main reason for churn using Explainable AI (SHAP)

This project acts as a **Customer Retention Decision Support System** for businesses.

---

## 🚀 Key Features
✔ Customer churn prediction using ML  
✔ Stacking Ensemble Model (SVM + Random Forest + Logistic Regression)  
✔ Explainable AI using SHAP  
✔ Business Decision Engine  
✔ Customer-level churn reason generation  
✔ Real-time prediction for new customers  
✔ Streamlit Web Application  

---

## 🧠 Tech Stack

| Category | Tools |
|---------|------|
| Programming | Python |
| ML Models | SVM, Random Forest |
| Ensemble | Logistic Regression (Meta Model) |
| Explainable AI | SHAP |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Web App | Streamlit |

---

## 📊 Project Workflow
Data Collection
↓
Data Preprocessing
↓
Feature Engineering
↓
Data Scaling
↓
Model Training (SVM + Random Forest)
↓
Stacking Ensemble Model
↓
Prediction
↓
Explainable AI (SHAP)
↓
Business Decision Engine
↓
Streamlit Web App



---

## 📈 Model Architecture

| Model Type | Algorithm |
|------------|-----------|
| Base Model 1 | Support Vector Machine (SVM) |
| Base Model 2 | Random Forest |
| Meta Model | Logistic Regression |
| Explainability | SHAP |
| Output | Probability + Risk + Action + Reason |

---

## 🎯 Business Decision Rules

| Churn Probability | Risk Level | Business Action |
|-------------------|------------|----------------|
| ≥ 0.75 | 🔴 High Risk | Give Discount |
| 0.45 – 0.75 | 🟠 Medium Risk | Personalized Offer |
| < 0.45 | 🟢 Low Risk | No Action |

---

## 🧾 Example Output
Customer Churn Prediction Report

Churn Probability : 0.78
Risk Level : High Risk
Recommended Action: Give Discount
Main Reason : Low Customer Tenure


---

## 🧠 Explainable AI (SHAP)
This project uses **SHAP (SHapley Additive exPlanations)** to explain model predictions.

SHAP helps to:
- Identify the most important feature affecting churn
- Provide transparency in predictions
- Make the model interpretable for business users

Instead of showing complex numbers, the system shows the **main reason** for churn such as:
- Low Tenure
- High Monthly Charges
- Fiber Internet Plan
- Lack of Tech Support

---

## 💼 Business Use Case
This system helps companies to:

- Identify customers who are likely to leave
- Provide offers to high-risk customers
- Improve customer satisfaction
- Increase customer retention
- Reduce revenue loss
- Make data-driven decisions

---

## 🖥️ Streamlit Web Application

The Streamlit app allows users to:

- Enter customer details
- Predict churn probability
- View risk level
- Get recommended action
- See main reason for churn


## ▶️ Run the App

```bash
streamlit run app.py
```


## 📂 Project Structure

```
customer-churn-prediction/
│
├── data/                 # Dataset
├── models/               # Saved models
├── app.py                # Streamlit app
├── churn_model.pkl       # Trained model
├── scaler.pkl            # Scaler
├── notebook.ipynb        # Model training notebook
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```


## 🚀 Future Improvements

- Add XGBoost / LightGBM
- Deploy on AWS / Azure
- Add Power BI Dashboard
- Customer Segmentation
- Real-time database integration


## 👨‍💻 Author

**Shiva Prasad**  
Machine Learning | Data Science | AI Enthusiast

##⭐ Project Summary

This is a complete End-to-End Machine Learning Project that includes:

Data Preprocessing
Machine Learning Models
Ensemble Learning (Stacking)
Explainable AI (SHAP)
Business Decision System
Streamlit Web Application

This project demonstrates how Machine Learning can be used not only for prediction but also for business decision-making and customer retention strategies.

📌 How to Use This Repository
# Clone the repository
git clone https://github.com/your-username/customer-churn-prediction.git

# Go to project folder
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py


