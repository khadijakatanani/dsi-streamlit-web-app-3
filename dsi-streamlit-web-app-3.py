import streamlit as st
import pandas as pd
import joblib


# load the model pipeline object
model = joblib.load("model.joblib")

# add title and instructions

divider = "rainbow"

st.title("Purchase Prediction Model")
st.subheader('Enter Customer Information Below and Submit to Find out the Likelihood of the Purchase')
# st.subheader(divider)


# age input form (numerical value, 18-120)
age = st.number_input(label = "01. Customer Age:",
                      min_value = 18,
                      max_value = 120,
                      value = 35)  # pre-populated

# gender input form (categorical value)
gender = st.radio("02. Customer Gender:",
                  ["M", "F"])


# credit score input form (numerical value, 0-1000)
credit_score = st.number_input(label = "03. Credit Score:",
                      min_value = 0,
                      max_value = 1000,
                      value = 500)  # pre-populated


# submit the form

if st.button("Submit"):
    # store the data in a dataframe for prediction
    new_data = pd.DataFrame({"age": [age],
                             "gender": [gender],
                             "credit_score": [credit_score]})
    
    # apply model pipeline to the input data and extract probability
    pred_proba = model.predict_proba(new_data)[0][1]
    
    st.subheader(f"The model predicts a probability value of {pred_proba:.0%}")











