from datetime import datetime

import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

ai_model = load_model('best_model')

current_year = datetime.now().year

st.title("Predykcja truskawek")
## TODO zmiana uniwersum AUT na SOIL NUTRIENTS!

mark = st.selectbox("Marka", [
    "opel", "audi", "bmw", "volkswagen", "ford", "mercedes-benz", "renault", "toyota", "skoda", "alfa-romeo",
    "chevrolet", "citroen", "fiat", "honda", "hyundai", "kia", "mazda", "mini", "mitsubishi", "nissan",
    "peugeot", "seat", "volvo"
])
model = st.text_input("Model")
year = st.slider("Rok produkcji", min_value=1971, max_value=2022, value=2015)
mileage = st.number_input("Przebieg (km)", min_value=1, max_value=2800000, value=100000, step=1000)
vol_engine = st.number_input("Pojemność silnika (cm³)", min_value=850, max_value=2796, value=1248, step=1)
fuel = st.selectbox("Rodzaj paliwa", [
    "Diesel", "CNG", "Gasoline", "LPG", "Hybrid", "Electric"
])
province = st.selectbox("Województwo", [
    "Mazowieckie", "Śląskie", "Opolskie", "Dolnośląskie", "Lubelskie", "Wielkopolskie", "Warmińsko-mazurskie",
    "Małopolskie", "Podkarpackie", "Kujawsko-pomorskie", "Pomorskie", "Podlaskie", "Łódzkie", "Świętokrzyskie",
    "Zachodniopomorskie", "Lubuskie", "Berlin", "Wiedeń", "Niedersachsen", "Moravian-Silesian Region",
])
age_years = current_year - year
mileage_per_year = mileage / (age_years if age_years != 0 else 1)

st.number_input("Wiek pojazdu (lata)", value=age_years, disabled=True)
st.number_input("Średni przebieg roczny (km)", value=round(mileage_per_year, 2), disabled=True)

if st.button("Oblicz"):
    input_df = pd.DataFrame([{
        "mark": mark,
        "model": model.lower(),
        "year": year,
        "mileage": mileage,
        "vol_engine": vol_engine,
        "fuel": fuel,
        "province": province,
        "age_years": age_years,
        "mileage_per_year": mileage_per_year
    }])
    prediction = predict_model(ai_model, data=input_df)
    # st.write(f"Predication: {prediction}")
    predicted_price = prediction.loc[0, 'prediction_label']
    st.success(f"Predykcja: {predicted_price} zł")