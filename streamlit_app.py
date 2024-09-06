import streamlit as st
import joblib
import pandas as pd
import sys
import os

# ruta para importar clase
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
from end_to_end.code.proyect import CombinedAttributesAdder

# pipeline y modelo 
preprocessor = joblib.load('end_to_end/code/full_pipeline.pkl')
lin_reg_model = joblib.load('end_to_end/code/lin_reg_best.pkl')

# interfaz del usuario
st.title("California Housing Price Prediction - Regression")

# entradas del usuario
longitude = st.number_input("Longitude")
latitude = st.number_input("Latitude")
housing_median_age = st.number_input("Housing Median Age")
total_rooms = st.number_input("Total Rooms")
total_bedrooms = st.number_input("Total Bedrooms")
population = st.number_input("Population")
households = st.number_input("Households")
median_income = st.number_input("Median Income")
ocean_proximity = st.selectbox("Ocean Proximity", ['<1H OCEAN', 'INLAND', 'NEAR BAY', 'NEAR OCEAN', 'ISLAND'])

# df con las entradas del usuario
input_df = pd.DataFrame({
    'longitude': [longitude],
    'latitude': [latitude],
    'housing_median_age': [housing_median_age],
    'total_rooms': [total_rooms],
    'total_bedrooms': [total_bedrooms],
    'population': [population],
    'households': [households],
    'median_income': [median_income],
    'ocean_proximity': [ocean_proximity]
})

# calcular bedrooms_per_room 
input_df['bedrooms_per_room'] = input_df['total_bedrooms'] / input_df['total_rooms']

# aplicar el pipeline a los datos de entrada
input_processed = preprocessor.transform(input_df)

# seleccionar modelo
models = {
    "Linear regression": lin_reg_model,
}

selected_model = st.selectbox("Select the model you prefer", list(models.keys()))


# prediccion de precio
if st.button("Predict"):
    try:
        input_processed = preprocessor.transform(input_df)
        model = models[selected_model]
        price = model.predict(input_processed)
        st.write(f"The estimated price of the house using {selected_model} is: ${price[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")