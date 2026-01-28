import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Smart Loan Approval System", layout="wide")

st.title("🎯 Smart Loan Approval System – Stacking Model")

st.markdown("""
This system uses a **Stacking Ensemble Machine Learning model**
to predict whether a loan will be approved by combining multiple ML models.
""")

@st.cache_data
def load_data():
    return pd.read_csv("loan_train.csv")

@st.cache_resource
def train_model():
    data = load_data()

    if "Loan_ID" in data.columns:
        data.drop("Loan_ID", axis=1, inplace=True)

    data["Loan_Status"] = data["Loan_Status"].map({"Y": 1, "N": 0})

    categorical_cols = data.select_dtypes(include="object").columns
    data = pd.get_dummies(data, columns=categorical_cols)

    X = data.drop("Loan_Status", axis=1)
    y = data["Loan_Status"]

    feature_cols = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000))
    ])

    dt = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("dt", DecisionTreeClassifier(random_state=42))
    ])

    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    stack_model = StackingClassifier(
        estimators=[("lr", lr), ("dt", dt), ("rf", rf)],
        final_estimator=LogisticRegression(),
        cv=5
    )

    stack_model.fit(X_train, y_train)

    return stack_model, feature_cols


model, model_features = train_model()

st.sidebar.header("📝 Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapp_income = st.sidebar.number_input("Co-Applicant Income", min_value=0)

loan_amount = st.sidebar.number_input(
    "Loan Amount (×1000)",
    min_value=0,
    help="Enter loan amount in thousands. Example: 100 = ₹1,00,000"
)

loan_term_years = st.sidebar.number_input(
    "Loan Term (Years)", min_value=1, max_value=40, value=20
)

credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
self_employed = st.sidebar.radio("Self Employed", ["Yes", "No"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.sidebar.markdown(
    "ℹ️ **Loan Amount is in thousands (×1000)**\n\n"
    "- 100 → ₹1,00,000\n"
    "- 250 → ₹2,50,000\n"
    "- 500 → ₹5,00,000"
)

loan_term_months = loan_term_years * 12

input_df = pd.DataFrame(0, index=[0], columns=model_features)

input_df["ApplicantIncome"] = app_income
input_df["CoapplicantIncome"] = coapp_income
input_df["LoanAmount"] = loan_amount
input_df["Loan_Amount_Term"] = loan_term_months
input_df["Credit_History"] = 1 if credit_history == "Yes" else 0

if self_employed == "Yes":
    input_df["Self_Employed_Yes"] = 1
else:
    input_df["Self_Employed_No"] = 1

if property_area == "Urban":
    input_df["Property_Area_Urban"] = 1
elif property_area == "Semiurban":
    input_df["Property_Area_Semiurban"] = 1
else:
    input_df["Property_Area_Rural"] = 1

st.subheader(" Stacking Model Architecture")

st.markdown("""
**Base Models**
- Logistic Regression  
- Decision Tree  
- Random Forest  

**Meta Model**
- Logistic Regression
""")

if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    prob = model.predict_proba(input_df)[0][1]
    prediction = 1 if prob >= 0.35 else 0

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.markdown(f"### 📈 Confidence Score: **{prob*100:.2f}%**")

    st.subheader(" Business Explanation: ")

    if prediction == 1:
        st.info("""
        Based on income, credit history,applicant's financial profile and combined predictions from multiple models,
        the applicant is likely to repay the loan.
        Therefore, the stacking model approves the loan.
        """)
    else:
        st.warning("""
        Based on the applicant's financial profile and ensemble model evaluation,
        the applicant is unlikely to repay the loan.
        Therefore, the stacking model rejects the loan.
        """)