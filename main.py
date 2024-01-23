import streamlit as st
import joblib
import pandas as pd

def main():
    st.title('Machine Learning Model Deployment')
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader('Uploaded Data:')
        st.write(df)
        model_choice = st.selectbox("Choose a model", ["Logistic Regression", "Decision Tree", "Random Forest"])
        
        if model_choice == "Logistic Regression":
            model = joblib.load('logistic_model.joblib')
        elif model_choice == "Decision Tree":
            model = joblib.load('decision_tree_model.joblib')
        elif model_choice == "Random Forest":
            model = joblib.load('random_forest_model.joblib')
        predictions = model.predict(df)
        
        st.subheader('Predictions:')
        st.write(predictions)

if __name__ == '__main__':
    main()
