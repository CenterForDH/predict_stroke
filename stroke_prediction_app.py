import streamlit as st
import numpy as np
import pandas as pd
import pickle

# 🔹 1. Load Model and Preprocessing Pipeline

def load_model():
    with open("v3_model__.pkl", "rb") as f:
        return pickle.load(f)

def load_pipeline():
    with open("preprocessing_pipeline.pkl", "rb") as f:
        return pickle.load(f)

# 🔹 2. User Input Function
def user_input():
    SEX = st.radio("Gender", ["Male", "Female"])
    Income = st.selectbox("Income Level", ["Low", "Middle", "High"])
    Region = st.radio("Region", ["Urban", "Rural"])
    
    Age_group = st.selectbox("Age Group", [
        "20–24", "25–29", "30–34", "35–39", "40–44", 
        "45–49", "50–54", "55–59", "60–64", "65–69", 
        "70–74", "75–79", "80–84", "85+"
    ])
    
    EXERCISE = st.radio("Physical activity sessions per week", ["0-2 times", "3-4 times", "5-6 times", "Everyday"])
    SMK = st.radio("Smoking Status", ["Never smoked", "Current smoker"])
    DRNK = st.radio("Alcoholic drinks per week", ["Rarely", "Sometimes", "Everyday"])
    M_HTN = st.radio("Hypertension Diagnosis", ["No", "Yes"])
    M_DIA = st.radio("Diabetes Diagnosis", ["No", "Yes"])
    
    BP_HIGH = st.slider("Systolic Blood Pressure (mmHg)", 70, 250, 120)
    BP_LWST = st.slider("Diastolic Blood Pressure (mmHg)", 40, 140, 77)
    BLDS = st.slider("Fasting Blood Glucose (mg/dL)", 50, 400, 91)
    TOT_CHOLE = st.slider("Total Cholesterol (mg/dL)", 80, 400, 189)
    HMG = st.slider("Hemoglobin (g/dL)", 7, 20, 14)
    SGOT_AST = st.slider("SGOT (AST) (IU/L)", 5, 400, 23)
    SGPT_ALT = st.slider("SGPT (ALT) (IU/L)", 5, 400, 22)
    GAMMA_GTP = st.slider("Gamma-GTP (IU/L)", 10, 700, 28)
    BMI = st.slider("Body Mass Index (BMI)", 12, 50, 23)

    # 🔹 매핑 처리
    age_mapping = {
        "20–24": 5, "25–29": 6, "30–34": 7, "35–39": 8,
        "40–44": 9, "45–49": 10, "50–54": 11, "55–59": 12,
        "60–64": 13, "65–69": 14, "70–74": 15, "75–79": 16,
        "80–84": 17, "85+": 18
    }
    Age = age_mapping[Age_group]

    Income_mapped = {"Low": 0, "Middle": 1, "High": 2}[Income]
    EXERCISE_mapped = {"0-2 times": 0, "3-4 times": 1, "5-6 times": 2, "Everyday": 3}[EXERCISE]

    # 🔹 최종 입력 데이터
    data = {
        "SEX": SEX,
        "Income": Income_mapped,
        "Region": Region,
        "Age": Age,
        "EXERCISE": EXERCISE_mapped,
        "SMK": SMK,
        "DRNK": DRNK,
        "M_HTN": M_HTN,
        "M_DIA": M_DIA,
        "BP_HIGH": BP_HIGH,
        "BP_LWST": BP_LWST,
        "BLDS": BLDS,
        "TOT_CHOLE": TOT_CHOLE,
        "HMG": HMG,
        "SGOT_AST": SGOT_AST,
        "SGPT_ALT": SGPT_ALT,
        "GAMMA_GTP": GAMMA_GTP,
        "BMI": BMI
    }
    
    return pd.DataFrame([data])

# 🔹 3. Prediction Function
def predict(input_df):
    model = load_model()
    pipeline = load_pipeline()
    processed_input = pipeline.transform(input_df)  # pipeline 적용
    proba = model.predict_proba(processed_input)[0][1]  # stroke일 확률만 반환
    return proba

# 🔹 4. Streamlit App Main
def main():
    st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")
    st.title("🧠 Stroke Risk Prediction App")
    st.markdown("Please enter your health check-up information below to estimate your **future risk of stroke**.")

    input_df = user_input()
    st.write("### Your Input Summary", input_df)

    if st.button("Predict"):
        prob = predict(input_df)
        st.subheader("📈 Prediction Result")
        st.write(f"Predicted Risk Probability: **{prob:.2%}**")

        if prob >= 0.35:
            st.error("🔴 High risk detected. We recommend seeking medical consultation.")
        else:
            st.success("🟢 Your risk appears low. Maintain a healthy lifestyle!")

if __name__ == "__main__":
    main()
