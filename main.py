import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder

def main():
    st.title('Machine Learning Model Deployment')
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        cleanData(df)
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


def cleanData(df):
    df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1})
    df['MaritalStatus'] = df['MaritalStatus'].map({'Single': 0, 'Married': 1})
    df = df[df['MonthClaimed'] != '0']
    df['Month'] = pd.to_datetime(df['Month'], format='%b')
    df['Month'] = df['Month'].dt.month
    day_mapping = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }
    df['DayOfWeek'] = df['DayOfWeek'].map(day_mapping)
    df['MonthClaimed'] = pd.to_datetime(df['MonthClaimed'], format='%b')
    df['MonthClaimed'] = df['MonthClaimed'].dt.month
    df['DayOfWeekClaimed'] = df['DayOfWeekClaimed'].map(day_mapping)
    le = LabelEncoder()
    df['Make'] = le.fit_transform(df['Make'])
    columns_to_encode = ['AccidentArea',
         'Fault',
          'PolicyType',
           'VehicleCategory',
            'VehiclePrice',
           'Days_Policy_Accident',
             'Days_Policy_Claim',
           'PastNumberOfClaims',
           'AgeOfVehicle',
          'AgeOfPolicyHolder',
          'PoliceReportFiled',
          'WitnessPresent',
         'AgentType',
          'NumberOfSuppliments',
        'AddressChange_Claim',
        'NumberOfCars',
        'BasePolicy']
    le = LabelEncoder()
    for column in columns_to_encode:
        df[column] = le.fit_transform(df[column])


if __name__ == '__main__':
    main()
