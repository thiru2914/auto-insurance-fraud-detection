import pandas
import streamlit as st
import pickle
import joblib
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

with open('logistic_model2.pkl','rb') as lg_model_file:
   logistic_model = pickle.load(lg_model_file)

#dt_model=joblib.load('dt_model1.pkl')

#with open('rf_model1.pkl','rb') as rf_model_file:
 #  random_forest_model= pickle.load(rf_model_file)

#with open('dt_model1.pkl','rb') as dt_model_file:
 #  decision_tree_model = pickle.load(dt_model_file)
#with open('if.pkl','rb') as if_model_file:
 #  if_model = pickle.load(if_model_file)

le = LabelEncoder()
def day_to_numeric(day):
    days = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    return days.get(day.title(), -1)
def make_fun(value):
    days = {'Hyundai': 0, 'Honda': 1, 'Toyota': 2, 'BMW': 3, 'Tesla': 4}
    return days.get(value.title(), -1)

def sex_fun(value):
    days = {'Male': 0, 'Female': 1}
    return days.get(value.title(), -1)

def martial_status_fun(value):
    days = {'Single': 0, 'Married': 1}
    return days.get(value.title(), -1)

def general_fun(value):
    days = {'Yes': 0, 'No': 1}
    return days.get(value.title(), -1)

myFieldsList = []
st.title("Insurance Fraud Detection")
month=st.number_input("Month",min_value=1.0, max_value=12.0)
myFieldsList.append(float(month))
weekOfMonth = st.selectbox("WeekOfMonth",['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
myFieldsList.append(float(day_to_numeric(weekOfMonth)))
dayOfWeek = st.selectbox("DayOfWeek",['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
myFieldsList.append(float(day_to_numeric(dayOfWeek)))
make=st.selectbox("Make",['Hyndai','Honda','Toyota','BMW','Tesla'])
myFieldsList.append(float(make_fun(make)))
accidentArea = st.number_input("AccidentArea")
myFieldsList.append(float(accidentArea))
dayOfWeekClaimed = st.selectbox("DayOfWeekClaimed",['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
myFieldsList.append(float(day_to_numeric(dayOfWeekClaimed)))
monthClaimed = st.number_input("MonthClaimed",min_value=1,max_value=12)
myFieldsList.append(float(monthClaimed))
weekOfMonthClaimed = st.number_input("WeekOfMonthClaimed",min_value=1,max_value=4)
myFieldsList.append(float(weekOfMonthClaimed))
sex=st.selectbox("Sex",['Male','Female'])
myFieldsList.append(float(sex_fun(sex)))
martialStatus = st.selectbox("MaritalStatus",['Single','Married'])
myFieldsList.append(float(martial_status_fun(martialStatus)))
age = st.number_input("Age",min_value=18.0,max_value=90.0)
myFieldsList.append(float(age))
fault = st.number_input("Fault")
myFieldsList.append(float(fault))
policyType = st.number_input("PolicyType")
myFieldsList.append(float(policyType))
vehicleCategory = st.number_input("VehicleCategory")#
myFieldsList.append(float(vehicleCategory))
vehiclePrice = st.number_input("VehiclePrice",min_value=0.0)
myFieldsList.append(float(vehiclePrice))
#st.selectbox('Select 1 or 0:', ['1', '0'])
fraudFound_p=float(st.selectbox("FraudFound_P", ['1','0']))
myFieldsList.append(fraudFound_p)
policyNumber = st.number_input("PolicyNumber")
myFieldsList.append(float(policyNumber))
repNumber=st.number_input("RepNumber")
myFieldsList.append(float(repNumber))
deductible = st.number_input("Deductible",min_value=0.0)
myFieldsList.append(float(deductible))
driverrating=st.number_input("DriverRating",min_value=0.0,max_value=5.0)
myFieldsList.append(float(driverrating))
days_policy_accident=st.number_input("Days_Policy_Accident",min_value=0.0)
myFieldsList.append(float(days_policy_accident))
days_policy_claim = st.number_input("Days_Policy_Claim",min_value=0.0)
myFieldsList.append(float(days_policy_claim))
pastNumberOfClaims=st.number_input("PastNumberOfClaims",min_value=0.0)
myFieldsList.append(float(pastNumberOfClaims))
ageOfVehicle = st.number_input("AgeOfVehicle",min_value=0.0)
myFieldsList.append(float(ageOfVehicle))
ageOfPolicyHolder = st.number_input("AgeOfPolicyHolder",min_value=18.0,max_value=90.0)
myFieldsList.append(float(ageOfPolicyHolder))
policeReportFiled = st.selectbox("PoliceReportFiled",['Yes','No'])
myFieldsList.append(float(general_fun(policeReportFiled)))
witnessPresent = st.selectbox("WitnessPresent",['Yes','No'])
myFieldsList.append(general_fun(witnessPresent))
agentType = 1.0
myFieldsList.append(float(agentType))
numberOfSuppliments = st.number_input("NumberOfSuppliments")
myFieldsList.append(float(numberOfSuppliments))
addressChangeClaim=st.selectbox("AddressChange_Claim",['Yes','No'])
myFieldsList.append(float(general_fun(addressChangeClaim)))
numberOfCars=st.number_input("NumberOfCars",min_value=0.0)
myFieldsList.append(float(numberOfCars))
#year=st.number_input("year", min_value=2000.0, max_value=2020.0, value=2000.0, step=1.0)
#if year< 2000.0 or year> 2020.0:
#   st.warning('Please enter a year between 2000 and 2020.')
#myFieldsList.append(year)
year = st.number_input("Year",min_value=2000.0,max_value=2024.0)
myFieldsList.append(float(year))
basePolicy = st.selectbox("BasePolicy",['Yes','No'])
myFieldsList.append(float(general_fun(basePolicy)))




choosen_model=st.selectbox("Select Machine Learning Model",["Logistic regression Model {Best Algorithm}","Random Forest Model","Decision Tree Model","Isolation Forest Model"])
pred=''
if st.button("Get Prediction"):
  if choosen_model == "Logistic regression Model {Best Algorithm}": 
     pred=logistic_model.predict(np.array(myFieldsList).reshape(1, -1))
     
  if choosen_model == "Random Forest Model":
     pred=random_forest_model.predict(np.array(myFieldsList).reshape(1, -1))
     st.write(pred)
  if choosen_model == "Decision Tree Model":
     pred = dt_model.predict(np.array(myFieldsList).reshape(1, -1))
  if choosen_model == "Isolation Forest Model":
     pred = if_model.predict(np.array(myFieldsList).reshape(1, -1))
  if pred[0]==0:
     st.write("Claim has been predicted non fradulent for the provided input")
  else:
     st.write("Claim has been predicted fradulent for the provided input")




