import streamlit as st
import json
import pandas as pd
import joblib
import numpy as np

tabA, tabB, tabC = st.tabs(['Home', 'Analysis', 'Prediction'])

with tabA:
    st.title('Churn Prediction')

    with st.expander('Why XGBoost?'):
        st.write('''
        XGBoost, which stands for Extreme Gradient Boosting, is a scalable, distributed gradient-boosted decision tree
        (GBDT) machine learning library. It provides parallel tree boosting and is the leading machine learning library
        for regression, classification, and ranking problems.  
          
        **XGBoost builds upon**: supervised machine learning, decision trees, ensemble learning, and gradient boosting.  
        **Main Features**: High Performance, Regularization, Flexibility, Tree Pruning, Handling Missing Values,  
        Feature Importance,  Scalability''')

    with st.expander('HyperParameter Tuning:'):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(['First', 'Second', 'Third', 'Fourth', 'Final'])
        with tab1:
            st.write('''
            subsample = 0.9  
            reg_lambda = 1  
            reg_alpha = 1  
            n_estimators = 1952  
            min_split_loss = 1  
            min_child_weight = 21  
            max_depth = 2  
            learning_rate = 0.01  
            colsample_bytree = 0.6  
            colsample_bynode = 0.9  
            colsample_bylevel = 0.8  
              
            Test Accuracy :0.8620522749273959  
            Validation Accuracy :0.8601694915254238''')
            col1, col2 = st.columns(2)
            with col1:
                st.image('Logloss and Classification error/First Logloss.png')
            with col2:
                st.image('Logloss and Classification error/First Classification error.png')
        with tab2:
            st.write('''
            subsample = 0.6  
            reg_lambda = 0.1  
            reg_alpha = 0  
            n_estimators = 100  
            min_split_loss = 0  
            min_child_weight = 1  
            max_depth = 5  
            learning_rate = 0.05  
            colsample_bytree = 0.8999999999999999  
            colsample_bynode = 0.8999999999999999  
            colsample_bylevel = 0.7  
    
            Test Accuracy :0.8639883833494676  
            Validation Accuracy :0.8571428571428571''')
            col3, col4 = st.columns(2)
            with col3:
                st.image('Logloss and Classification error/Second Logloss.png')
            with col4:
                st.image('Logloss and Classification error/Second Classification error.png')
        with tab3:
            st.write('''
            subsample = 0.5  
            reg_lambda = 0.1  
            reg_alpha = 1  
            n_estimators = 250  
            min_split_loss = 1  
            min_child_weight = 1  
            max_depth = 6  
            learning_rate = 0.1  
            colsample_bytree = 0.5  
            colsample_bynode = 0.6  
            colsample_bylevel = 0.5  
              
            Test Accuracy :0.861568247821878  
            Validation Accuracy :0.8589588377723971''')
            col5, col6 = st.columns(2)
            with col5:
                st.image('Logloss and Classification error/Third Logloss.png')
            with col6:
                st.image('Logloss and Classification error/Third Classification error.png')
        with tab4:
            st.write('''
            subsample = 0.7  
            reg_lambda = 0  
            reg_alpha = 0.1  
            n_estimators = 339  
            min_split_loss = 0.5  
            min_child_weight = 2  
            max_depth = 3  
            learning_rate = 0.1  
            colsample_bytree = 0.6  
            colsample_bynode = 0.5  
            colsample_bylevel = 0.7  
              
            Test Accuracy :0.8659244917715392  
            Validation Accuracy :0.8541162227602905''')
            col7, col8 = st.columns(2)
            with col7:
                st.image('Logloss and Classification error/Fourth Logloss.png')
            with col8:
                st.image('Logloss and Classification error/Fourth Classification error.png')
        with tab5:
            st.write('''
            subsample = 0.5  
            reg_lambda = 0.1  
            reg_alpha = 1  
            n_estimators = 155  
            min_split_loss = 1  
            min_child_weight = 1  
            max_depth = 6  
            learning_rate = 0.1  
            colsample_bytree = 0.5  
            colsample_bynode = 0.6  
            colsample_bylevel = 0.5  
              
            Test Accuracy :0.861568247821878  
            Validation Accuracy :0.8631961259079903''')
            col9, col10 = st.columns(2)
            with col9:
                st.image('Logloss and Classification error/Fifth Logloss.png')
            with col10:
                st.image('Logloss and Classification error/Fifth Classification error.png')

    with open('eval_results.json', 'r') as f:
        results = json.load(f)

    epochs = len(results['train']['error'])
    x_axis = list(range(0, epochs))

    # Plotting Log Loss using Streamlit
    st.subheader('XGBoost Log Loss')
    st.line_chart({'Train': results['train']['logloss'], 'Validation': results['eval']['logloss']})

    # Plotting Classification Error using Streamlit
    st.subheader('XGBoost Classification Error')
    st.line_chart({'Train': results['train']['error'], 'Validation': results['eval']['error']})

with tabB:

    # Loading the trained model
    model = joblib.load('XGB Model.pkl')

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
            'PaymentMethod_Bank transfer (automatic)': [
                convert_yes_no_to_binary(paymentmethod_bank_transfer_automatic)],
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
                st.write('Probability(%):', proba[0][0] * 100)
            elif predict == 1:
                st.write('Customer will Churn')
                st.write('Probability:', proba[0][1] * 100)

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

with tabC:
    def main(path):
        st.subheader("Power BI report:")
        st.markdown(f'<iframe width="900" height="541.25" src="{path}" frameborder="0" allowfullscreen></iframe>',
                    unsafe_allow_html=True)


    report = ("https://app.powerbi.com/reportEmbed?reportId=b99a36f5-13a5-44fc-8846-76919dbef12d&autoAuth=true&ctid="
              "9ea332bc-d5e1-4211-a36b-1e06be85ea93")

    main(path=report)
