import streamlit as st
import pandas as pd
import numpy as np

st.title('Insurance Fraud Detection')

Fraud_Detection = ['true', 'false']

fruad = st.text_input('fraud')

if st.button('Click Me'):
    have_it = fraud.lower() in Fraud_Detection
    if have_it==true:
        st.write("true")