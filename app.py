# app.py

import streamlit as st
import pandas as pd
import pickle

# --- Load the saved objects ---
# We load the model, scaler, and columns list that we saved in the training script
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('columns.pkl', 'rb') as f:
        training_columns = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please run the training script first.")
    st.stop()

# --- Web App Interface ---
st.title("Customer Churn Prediction 🔮")
st.write("Enter the customer's details to predict if they will churn.")

# Create input fields for the user
# Use two columns for better layout
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input('Tenure (months)', min_value=0, max_value=72, value=12)
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input('Monthly Charges ($)', min_value=18.0, max_value=120.0, value=70.0, step=0.01)
    total_charges = st.number_input('Total Charges ($)', min_value=18.0, value=1000.0, step=0.01)

with col2:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', [0, 1])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])

# Create a "Predict" button
if st.button('Predict Churn'):
    # 1. Create a dictionary from the user inputs
    customer_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'InternetService': internet_service,
        'TechSupport': tech_support,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # 2. Convert the dictionary to a pandas DataFrame
    customer_df = pd.DataFrame([customer_data])
    
    # 3. One-hot encode the categorical variables
    customer_encoded = pd.get_dummies(customer_df)
    
    # 4. Align the columns with the training data
    # This is crucial to ensure the app's data matches the model's expected input
    customer_aligned = customer_encoded.reindex(columns=training_columns, fill_value=0)
    
    # 5. Scale the data using the loaded scaler
    customer_scaled = scaler.transform(customer_aligned)
    
    # 6. Make the prediction
    prediction = model.predict(customer_scaled)
    probability = model.predict_proba(customer_scaled)
    
    # 7. Display the result
    if prediction[0] == 1:
        churn_prob = probability[0][1] * 100
        st.error(f"Prediction: Customer WILL CHURN with a {churn_prob:.2f}% probability.", icon="🚨")
    else:
        no_churn_prob = probability[0][0] * 100
        st.success(f"Prediction: Customer WILL NOT CHURN with a {no_churn_prob:.2f}% probability.", icon="✅")