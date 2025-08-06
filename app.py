import streamlit as st
import pandas as pd
import joblib


model = joblib.load('logistic_model.pkl')
columns = pd.read_csv('model_columns.csv', header=None)[0].tolist()

st.set_page_config(page_title="Attrition Predictor", layout="centered")
st.title("🔍 Employee Attrition Prediction App")
st.markdown("Use this tool to predict whether an employee is likely to leave the company.")

def user_input():
    st.sidebar.header("Enter Employee Details")

    Age = st.sidebar.slider('Age', 18, 60, 30)
    MonthlyIncome = st.sidebar.slider('Monthly Income', 1000, 20000, 5000)
    YearsAtCompany = st.sidebar.slider('Years at Company', 0, 40, 3)
    JobSatisfaction = st.sidebar.selectbox('Job Satisfaction (1 = Low, 4 = Very High)', [1, 2, 3, 4])
    OverTime = st.sidebar.selectbox('OverTime', ['Yes', 'No'])
    BusinessTravel = st.sidebar.selectbox('Business Travel', ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])

    
    OverTime_Yes = 1 if OverTime == 'Yes' else 0
    BT_Rarely = 1 if BusinessTravel == 'Travel_Rarely' else 0
    BT_Frequently = 1 if BusinessTravel == 'Travel_Frequently' else 0

    input_data = {
        'Age': Age,
        'MonthlyIncome': MonthlyIncome,
        'YearsAtCompany': YearsAtCompany,
        'JobSatisfaction': JobSatisfaction,
        'OverTime_Yes': OverTime_Yes,
        'BusinessTravel_Travel_Rarely': BT_Rarely,
        'BusinessTravel_Travel_Frequently': BT_Frequently
    }

    
    full_input = pd.DataFrame(columns=columns)
    full_input.loc[0] = 0
    for key in input_data:
        if key in full_input.columns:
            full_input.at[0, key] = input_data[key]

    return full_input

input_df = user_input()

st.subheader("📋 Input Sent to Model")
st.write(input_df)

prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0][1]

st.subheader("🎯 Prediction:")
if prediction == 1:
    st.error(f"**Prediction: YES** ❌ — The employee is likely to leave.")
else:
    st.success(f"**Prediction: NO** ✅ — The employee is likely to stay.")

st.subheader("📊 Attrition Probability:")
st.write(f"Chance of attrition: **{proba * 100:.2f}%**")
st.progress(proba)
