import streamlit as st
import json

st.title('Churn Prediction')

if st.checkbox('Why XGBoost?'):
    st.write('''
    XGBoost, which stands for Extreme Gradient Boosting, is a scalable, distributed gradient-boosted decision tree
    (GBDT) machine learning library. It provides parallel tree boosting and is the leading machine learning library
    for regression, classification, and ranking problems.  
      
    **XGBoost builds upon**: supervised machine learning, decision trees, ensemble learning, and gradient boosting.  
    **Main Features**: High Performance, Regularization, Flexibility, Tree Pruning, Handling Missing Values,  
    Feature Importance,  Scalability''')

if st.checkbox('HyperParameter Tuning:'):
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
        col1, col2 = st.columns(2)
        with col1:
            st.image('Logloss and Classification error/Second Logloss.png')
        with col2:
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
        col1, col2 = st.columns(2)
        with col1:
            st.image('Logloss and Classification error/Third Logloss.png')
        with col2:
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
        col1, col2 = st.columns(2)
        with col1:
            st.image('Logloss and Classification error/Fourth Logloss.png')
        with col2:
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
        col1, col2 = st.columns(2)
        with col1:
            st.image('Logloss and Classification error/Fifth Logloss.png')
        with col2:
            st.image('Logloss and Classification error/Fifth Classification error.png')
            
st.page_link('Pages/Analysis.py', label='Analysis')
st.page_link('Pages/Predict.py', label='Predict')

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
