import streamlit as st
import joblib
import numpy as np

# Load our saved model "brain" and scaler
model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')

# Set up the web page header
st.title("💰 AI Loan Approval Predictor")
st.write("Enter applicant details below to evaluate credit risk instantly.")

# Create input fields for the user
age = st.number_input("Applicant Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income ($)", min_value=0, value=55000)
credit_score = st.number_input("Credit Score (300 - 850)", min_value=300, max_value=850, value=650)
loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=20000)

# Run prediction when the button is clicked
if st.button("Evaluate Application"):
    # Convert inputs into the format the model expects
    user_features = np.array([[age, income, credit_score, loan_amount]])
    scaled_features = scaler.transform(user_features)
    
    # Predict
    prediction = model.predict(scaled_features)
    
    # Display the result
    st.markdown("---")
    if prediction[0] == 1:
        st.success("🎉 **Approved!** This applicant is a low risk.")
    else:
        st.error("❌ **Rejected.** This applicant exceeds safe risk thresholds.")