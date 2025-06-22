import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

# TODO usunięcie częśći pól (trzeba przetrenować model), poprawa interfejsu, może tłumaczenia dla wartości

ai_model = load_model('best_model')

feature_info = {
    'Photoperiod': {'type': 'categorical', 'values': ['Day Neutral', 'Short Day Period']},
    'Temperature': {'type': 'numeric', 'min': 5., 'max': 30.},
    'Rainfall': {'type': 'numeric', 'min': 0., 'max': 915.},
    'pH': {'type': 'numeric', 'min': 4., 'max': 7.},
    'Light_Hours': {'type': 'numeric', 'min': 8., 'max': 14.},
    'Light_Intensity': {'type': 'numeric', 'min': 397., 'max': 608.},
    'Rh': {'type': 'numeric', 'min': 80., 'max': 95.},
    'Nitrogen': {'type': 'numeric', 'min': 150., 'max': 200.},
    'Phosphorus': {'type': 'numeric', 'min': 110., 'max': 129.},
    'Potassium': {'type': 'numeric', 'min': 200., 'max': 250.},
    'Season': {'type': 'categorical', 'values': ['Spring', 'Summer']},
}

translations = {
    'Photoperiod': 'Długość dnia świetlnego (Photoperiod)',
    'Temperature': 'Temperatura (°C)',
    'Rainfall': 'Opady (mm)',
    'pH': 'pH gleby',
    'Light_Hours': 'Liczba godzin światła',
    'Light_Intensity': 'Natężenie światła (lux)',
    'Rh': 'Wilgotność względna (%)',
    'Nitrogen': 'Azot (N)',
    'Phosphorus': 'Fosfor (P)',
    'Potassium': 'Potas (K)',
    'Season': 'Sezon',
}

st.title("Predykcja zbioru truskawek")

user_input = {}

for feature, info in feature_info.items():
    label = translations.get(feature, feature)
    if info['type'] == 'categorical':
        default = info['values'][0] if info['values'] else None
        user_input[feature] = st.selectbox(label, options=info['values'], index=0)
    elif info['type'] == 'numeric':
        min_val = info['min']
        max_val = info['max']
        default = (min_val + max_val) / 2
        if min_val == max_val:
            user_input[feature] = st.number_input(label, value=min_val, step=0.0, format="%.4f")
        else:
            user_input[feature] = st.slider(label, min_value=min_val, max_value=max_val, value=default, format="%.4f")

if st.button("Sprawdź predykcję"):
    input_df = pd.DataFrame([user_input])
    # st.subheader("Dane wejściowe:")
    # st.dataframe(input_df)

    prediction = predict_model(ai_model, data=input_df)
    predicted_price = prediction.loc[0, 'prediction_label']
    st.success(f"Predykcja: {predicted_price}")
