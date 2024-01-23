import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split

def main():
    st.title('Insurance Fraud Detection')
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1})
        df['MaritalStatus'] = df['MaritalStatus'].map({'Single': 0, 'Married': 1})
        df = df[df['MonthClaimed'] != '0']
        df['Month'] = pd.to_datetime(df['Month'], format='%d')
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
        df['MonthClaimed'] = pd.to_datetime(df['MonthClaimed'], format='%d')
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

        st.subheader('Uploaded Data:')
        st.write(df)
        X = df.drop('FraudFound_P', axis=1)
        y = df['FraudFound_P']
        X = X.dropna()
        y = y[X.index]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        # model_choice = st.selectbox("Choose a model", ["Logistic Regression", "Decision Tree", "Random Forest"])

        # if model_choice == "Logistic Regression":
        #     model = joblib.load('logistic_model1.joblib')
        # elif model_choice == "Decision Tree":
        #     model = joblib.load('decision_tree_model1.joblib')
        # elif model_choice == "Random Forest":
        #     model = joblib.load('random_forest_model1.joblib')
        # predictions = model.predict(df)

        st.subheader('Predictions:')
        st.write(model.predict(df))
    

if __name__ == '__main__':
    main()
