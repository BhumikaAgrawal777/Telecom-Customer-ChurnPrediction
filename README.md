# 📊 Telecom Customer Churn Prediction

This project aims to predict customer churn in a telecom company using supervised machine learning models. Churn refers to customers who leave the service provider, and predicting it helps businesses retain valuable customers and reduce revenue loss.

## 🧠 Objective

The goal is to build and evaluate machine learning models that can accurately identify which customers are likely to leave the telecom service.

---

## 📁 Dataset

- **Source:** [Kaggle or IBM Sample Dataset]
- **Records:** ~7,000+ customer entries
- **Target Variable:** `Churn` (Yes/No)
- **Features Include:**
  - Demographics: Gender, SeniorCitizen, Partner, etc.
  - Services: InternetService, StreamingTV, PhoneService, etc.
  - Account Info: Tenure, MonthlyCharges, TotalCharges, etc.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas, NumPy** – data handling
- **Matplotlib, Seaborn** – visualization
- **Scikit-learn** – machine learning
- **XGBoost** – boosting algorithm
- **GridSearchCV** – hyperparameter tuning
- **Streamlit** *(optional)* – for deployment

---

## 🔍 Exploratory Data Analysis (EDA)

Performed in-depth EDA to:
- Handle missing values & outliers
- Analyze class imbalance
- Visualize churn trends by service and payment types

---

## 🔄 Data Preprocessing

- Label Encoding and OneHot Encoding
- Feature Scaling using StandardScaler
- Train-test split (e.g., 80-20)

---

## 🤖 Models Trained

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine (SVM)
- XGBoost

### ✅ Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve

---

## 🏆 Best Model

| Model         | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| XGBoost       | 83.2%    | 0.81      | 0.76   | 0.78     |

> XGBoost performed best based on F1-score and ROC-AUC.

---

## 🖥️ How to Run

1. **Clone the repository**  
