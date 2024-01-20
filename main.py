import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample data for demonstration purposes
# Replace this with your actual data

# Machine Learning Model
def train_model(df):
    X = df.drop('FraudFound_p', axis=1)  # Replace 'TargetColumn' with your target variable
    y = df['TargetColumn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()  # Replace with your preferred machine learning model
    model.fit(X_train, y_train)

    return model, X_test, y_test

# Streamlit UI
def main():
    st.title('Fraud Detection App')

    # Display input fields
    st.sidebar.header('Input Fields')
    field1 = st.text_input('Field 1', value=0)
    field2 = st.text_input('Field 2', value=0)
    field3 = st.text_input('Field 3', value=0)
    field4 = st.text_input('Field 4', value=0)
    field5 = st.text_input('Field 5', value=0)
    field6 = st.text_input('Field 6', value=0)
    field7 = st.text_input('Field 7', value=0)
    field8 = st.text_input('Field 8', value=0)
    field9 = st.text_input('Field 9', value=0)
    field10 = st.text_input('Field 10', value=0)
    field11= st.text_input('Field 11', value=0)
    field12= st.text_input('Field 12', value=0)
    field13= st.text_input('Field 13', value=0)
    field14= st.text_input('Field 14', value=0)
    field15= st.text_input('Field 15', value=0)
    field16= st.text_input('Field 16', value=0)
    field17 = st.text_input('Field 17', value=0)
    field18 = st.text_input('Field 18', value=0)
    field19= st.text_input('Field 19', value=0)
    field20= st.text_input('Field 20', value=0)
    field21= st.text_input('Field 21', value=0)
    field22= st.text_input('Field 22', value=0)
    field23= st.text_input('Field 23', value=0)
    field24= st.text_input('Field 24', value=0)
    field25= st.text_input('Field 25', value=0)
    field26= st.text_input('Field 26', value=0)
    field27= st.text_input('Field 27', value=0)
    field28= st.text_input('Field 28', value=0)
    field29= st.text_input('Field 29', value=0)
    field30= st.text_input('Field 30', value=0)
    field31= st.text_input('Field 31', value=0)
    field32= st.text_input('Field 32', value=0)
    field33= st.text_input('Field 33', value=0)

    
   

    # Create a dictionary with user input
    user_data = {
        'Field1': [field1],
        'Field2': [field2],
        'Field3':[field3],
        'Field4':[field4],
        'Field5':[field5],
        'Field6':[field6],
        'Field7':[field7],
        'Field8':[field8],
        'Field9':[field9],
        'Field10':[field10],
        'Field11':[field11],
        'Field12':[field12],
        'Field13':[field13],
        'Field14':[field14],
        'Field15':[field15],
        'Field16':[field16],
        'Field17':[field17],
        'Field18':[field18],
        'Field19':[field19],
        'Field20':[field20],
        'Field21':[field21],
        'Field22':[field22],
        'Field23':[field23],
        'Field24':[field24],
        'Field25':[field25],
        'Field26':[field26],
        'Field27':[field27],
        'Field28':[field28],
        'Field29':[field29],
        'Field30':[field30],
        'Field31':[field31],
        'Field32':[field32],
        'Field33':[field33],
        'Field3':[field3],
        
    }

    user_df = pd.DataFrame(user_data)

    # Run Prediction Button
    if st.button('Run Prediction'):
        # Train the model and make predictions
        model, X_test, y_test = train_model(df)
        predictions = model.predict(user_df)

        # Display predictions
        st.subheader('Prediction Results:')
        st.write(predictions)

        # Display accuracy
        accuracy = accuracy_score(y_test, model.predict(X_test))
        st.subheader(f'Model Accuracy: {accuracy:.2%}')

if __name__ == '__main__':
    main()
