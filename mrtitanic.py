# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:40:30 2023

@author: vibes 
"""
import streamlit as st  # For creating the web app interface
import pandas as pd  # For data manipulation and handling
from joblib import load  # For loading the pre-trained machine learning model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder  # For data preprocessing
from PIL import Image

model = load('titanic_model.joblib')
def preprocess_categorical(df):
    lb = LabelEncoder()

    # Encode 'country' and 'banking_crisis' columns to numerical values
    df['Sex'] = lb.fit_transform(df['Sex'])
    return df
def preprocess_numerical(df):
    # Scale numerical columns to a specific range
    scaler = MinMaxScaler()
    numerical_cols = ['Pclass', 'Age', 'SibSp',
                      'Parch']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df
def preprocessor(input_df):
    # Preprocess categorical and numerical columns separately
    input_df = preprocess_categorical(input_df)
    input_df = preprocess_numerical(input_df)
    return input_df
def main():
    st.title('titanic Survival app')  # Title of the web app
    st.write('this App is built to predidct survival of titanic passengers')
    img = Image.open('ship.jpg')
    st.image(img, width = 500)
    input_data = {}  # Dictionary to store user input data
    col1, col2 = st.columns(2)  # Split the interface into two columns

    with col1:
        input_data['Pclass'] = st.number_input('what vPclass is the passenger',min_value=0, max_value=3)
        input_data['Sex'] = st.number_input('what sex is the passenger',min_value=0, max_value=1)
       
        input_data['Age'] = st.number_input('how old was the passenger ?', min_value=0, max_value=100)
        
    with col2:
        # Collect user inputs for other indicators
        input_data['SibSp'] = st.number_input('whats the passenger sibsp?',min_value=0, max_value=10)
        input_data['Parch'] = st.number_input('whats the passenger  parch?',min_value=0, max_value=10)
       
    input_df = pd.DataFrame([input_data])  # Convert collected data into a DataFrame
    st.write(input_df)  # Display the collected data on the app interface

    if st.button('Predict'):  # When the 'Predict' button is clicked
        final_df = preprocessor(input_df)  # Preprocess the collected data
        prediction = model.predict(final_df) # Use the model to predict the outcome
        
        # Display the prediction result
        if prediction == 1:
            st.write('the passenger survived')
        else:
            st.write('the passenger died')

# Run the main function when the script is executed directly
if __name__ == '__main__':
    main()

