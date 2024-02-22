import pandas as pd
import streamlit as st
import joblib
import numpy as np


# Loading the trained model
model = joblib.load('XGB Model.pkl')

st.page_link('XGB-Web-App/Homet.py', label='Predict')
st.page_link('XGB-Web-App/Pages/Analysis.py', label='Analysis')

tab1, tab2 = st.tabs(['Manual', 'CSV'])
with tab1:
    # column layout
    col1, col2, col3 = st.columns(3)

    # Function to convert yes/no values to binary
    def convert_yes_no_to_binary(value):
        return 1 if value == 'Yes' else 0

    # FUnction to convert gender to binary
    def convert_gender_to_binary(sex):
        return 1 if sex == 'Female' else 0


    with col1:
        gender = st.radio('Gender', ['Male', 'Female'])
        senior_citizen = st.radio('Senior Citizen', ['Yes', 'No'])
        partner = st.radio('Partner', ['Yes', 'No'])
        dependents = st.radio('Dependents', ['Yes', 'No'])
        tenure = st.number_input('Tenure', min_value=0)
        phone_service = st.radio('PhoneService', ['Yes', 'No'])
        multiplelines = st.radio('Multipltlines', ['Yes', 'No'])
        onlinesecurity = st.radio('Onlinesecurity', ['Yes', 'No'])

    with col2:
        onlinebackup = st.radio('Onlinebackup', ['Yes', 'No'])
        deviceprotection = st.radio('Deviceprotection', ['Yes', 'No'])
        techsupport = st.radio('Tech Support', ['Yes', 'No'])
        streamingtv = st.radio('Streaming TV', ['Yes', 'No'])
        streamingmovies = st.radio('Streaming Movies', ['Yes', 'No'])
        contract = st.number_input('Contract', min_value=0, max_value=24)
        paperlessbilling = st.radio('Paperlessbilling', ['Yes', 'No'])
        monthlycharges = st.number_input('Monthly Charges', format="%.8f")

    with col3:
        totalcharges = st.number_input('Total Charges', format="%.8f")
        paymentmethod_bank_transfer_automatic = st.radio('Payment Method Bank Transfer Automatic', ['Yes', 'No'])
        paymentmethod_creditcard_automatic = st.radio('Payment method Credit Card Automatic', ['Yes', 'No'])
        paymentmethod_electronic_check = st.radio('Electronic Check', ['Yes', 'No'])
        paymentmethod_mailed_check = st.radio('Payment Method Mailed Check', ['Yes', 'No'])
        dsl = st.radio('Internet Service DSL', ['Yes', 'No'])
        fiberoptic = st.radio('Internet Service Fibre Optic', ['Yes', 'No'])
        internetservice_no = st.radio('Internetservice', ['Yes', 'No'])

    # Converting input data to DataFrame
    data = {
        'Gender': [convert_gender_to_binary(gender)],
        'SeniorCitizen': [convert_yes_no_to_binary(senior_citizen)],
        'Partner': [convert_yes_no_to_binary(partner)],
        'Dependents': [convert_yes_no_to_binary(dependents)],
        'Tenure': [tenure],
        'PhoneService': [convert_yes_no_to_binary(phone_service)],
        'MultipleLines': [convert_yes_no_to_binary(multiplelines)],
        'OnlineSecurity': [convert_yes_no_to_binary(onlinesecurity)],
        'OnlineBackup': [convert_yes_no_to_binary(onlinebackup)],
        'DeviceProtection': [convert_yes_no_to_binary(deviceprotection)],
        'TechSupport': [convert_yes_no_to_binary(techsupport)],
        'StreamingTV': [convert_yes_no_to_binary(streamingtv)],
        'StreamingMovies': [convert_yes_no_to_binary(streamingmovies)],
        'Contract': [contract],
        'PaperlessBilling': [convert_yes_no_to_binary(paperlessbilling)],
        'MonthlyCharges': [monthlycharges],
        'TotalCharges': [totalcharges],
        'PaymentMethod_Bank transfer (automatic)': [convert_yes_no_to_binary(paymentmethod_bank_transfer_automatic)],
        'PaymentMethod_Credit card (automatic)': [convert_yes_no_to_binary(paymentmethod_creditcard_automatic)],
        'PaymentMethod_Electronic check': [convert_yes_no_to_binary(paymentmethod_electronic_check)],
        'PaymentMethod_Mailed check': [convert_yes_no_to_binary(paymentmethod_mailed_check)],
        'InternetService_DSL': [convert_yes_no_to_binary(dsl)],
        'InternetService_Fiber optic': [convert_yes_no_to_binary(fiberoptic)],
        'Internet Service(Yes/No)': [convert_yes_no_to_binary(internetservice_no)]
    }

    df = pd.DataFrame(data)


    # Function for making predictions
    def prediction(path):
        pred = model.predict(path)
        return pred


    def confidence(path):
        probability = model.predict_proba(path)
        return probability


    # Button to trigger prediction
    if st.button('Predict'):
        predict = prediction(df)
        proba = confidence(df)
        if predict == 0:
            st.write('Customer will not Churn')
            st.write('Probability(%):', proba[0][0]*100)
        elif predict == 1:
            st.write('Customer will Churn')
            st.write('Probability:', proba[0][1]*100)

with tab2:
    file = st.file_uploader('Upload Processed CSV', type=['csv'])


    def prediction(path):
        pred = model.predict(path)
        return pred


    row1 = st.number_input('Pick First row (Optional)', min_value=0, max_value=70000)
    row2 = st.number_input('Pick Second row (Optional)', min_value=0, max_value=70000)

    if st.button('Predict using CSV') and file is not None:
        df = pd.read_csv(file)
        if row1 >= 0 and row2 > 0:
            prediction_df = prediction(df[row1:row2])
            num_zeros = np.sum(prediction_df == 0)
            num_ones = np.sum(prediction_df == 1)
            st.write(f'Number of Customer who will not Churn : {num_zeros}')
            st.write(f'Number of Customer who will Churn : {num_ones}')
        elif row1 == 0 and row2 == 0:
            prediction_df = prediction(df)
            num_zeros = np.sum(prediction_df == 0)
            num_ones = np.sum(prediction_df == 1)
            st.write(f'Number of Customer who will not Churn : {num_zeros}')
            st.write(f'Number of Customer who will Churn : {num_ones}')
