from distutils.command.config import config
import streamlit as st 
import numpy as np 
import pandas as pd
import joblib
st.title('Ml Web App')
st.write("""
# Predict Used Car Prices?
""")
#inputs
km_driven=st.number_input('km driven',0,200000)
Age=st.slider('Age of Car',0,20)
st.write("choose fuel type")
fuel_diesel=st.selectbox('choose 0 or 1 for diesel',[0,1])
fuel_Electric=st.selectbox('choose 0 or 1 for Electric',[0,1])
fuel_LPG=st.selectbox('choose 0 or 1 for LPG',[0,1])
fuel_Petrol=st.selectbox('choose 0 or 1 for Petrol',[0,1])
st.write("choose seller type")
seller_type_Individual=st.selectbox('choose 0 or 1 for individual',[0,1])
seller_type_Dealer=st.selectbox('choose 0 or 1 for dealer',[0,1])
transmission_Manual=st.selectbox('choose 0 or 1 for manual',[0,1])
st.write("choose owner type")
fourth_owner=st.selectbox('choose 0 or 1 for fourth owner',[0,1])
second_owner=st.selectbox('choose 0 or 1 for second owner',[0,1])
testdrive_owner=st.selectbox('choose 0 or 1 for test drive vehicle',[0,1])
third_owner=st.selectbox('choose 0 or 1 for third owner',[0,1])


ok = st.button("Predict")
#preiction button
if ok:
    lr=joblib.load("lr.pkl")
    X = pd.DataFrame([[km_driven,Age,fuel_diesel,fuel_Electric,fuel_LPG,fuel_Petrol,seller_type_Individual,seller_type_Dealer,transmission_Manual,fourth_owner,second_owner,testdrive_owner,third_owner]],columns=['km_driven', 'Age', 'fuel_Diesel', 'fuel_Electric', 'fuel_LPG',
       'fuel_Petrol', 'seller_type_Individual', 'seller_type_Trustmark Dealer',
       'transmission_Manual', 'owner_Fourth & Above Owner',
       'owner_Second Owner', 'owner_Test Drive Car', 'owner_Third Owner'])
    X = X.astype(float)   
    prediction = lr.predict(X)
    st.subheader(f"The prediction is {prediction}")
    

