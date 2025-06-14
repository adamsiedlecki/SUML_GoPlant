import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

# TODO usunięcie częśći pól (trzeba przetrenować model), poprawa interfejsu, może tłumaczenia dla wartości

ai_model = load_model('best_model')

feature_info = {
    'Fertility': {'type': 'categorical', 'values': ['Moderate']},
    'Photoperiod': {'type': 'categorical', 'values': ['Day Neutral', 'Short Day Period']},
    'Temperature': {'type': 'numeric', 'min': 11.4895951867279, 'max': 24.7851260541969},
    'Rainfall': {'type': 'numeric', 'min': 603.262466985949, 'max': 912.357821862667},
    'pH': {'type': 'numeric', 'min': 6.01954182231121, 'max': 6.83444527129809},
    'Light_Hours': {'type': 'numeric', 'min': 11.9530915745843, 'max': 14.0224375398152},
    'Light_Intensity': {'type': 'numeric', 'min': 397.678455866516, 'max': 608.36827590609},
    'Rh': {'type': 'numeric', 'min': 89.5019906720046, 'max': 95.1292528088176},
    'Nitrogen': {'type': 'numeric', 'min': 151.499465257155, 'max': 200.356849479585},
    'Phosphorus': {'type': 'numeric', 'min': 110.976211751343, 'max': 129.848646332759},
    'Potassium': {'type': 'numeric', 'min': 230.478897122857, 'max': 250.798391983322},
    'Category_pH': {'type': 'categorical', 'values': ['low_acidic']},
    'Soil_Type': {'type': 'categorical', 'values': ['Loam']},
    'Season': {'type': 'categorical', 'values': ['Spring', 'Summer']},
    'N_Ratio': {'type': 'numeric', 'min': 10.0, 'max': 10.0},
    'P_Ratio': {'type': 'numeric', 'min': 10.0, 'max': 10.0},
    'K_Ratio': {'type': 'numeric', 'min': 10.0, 'max': 10.0}
}

translations = {
    'Fertility': 'Żyzność',
    'Photoperiod': 'Fotoperiod',
    'Temperature': 'Temperatura (°C)',
    'Rainfall': 'Opady (mm)',
    'pH': 'pH gleby',
    'Light_Hours': 'Liczba godzin światła',
    'Light_Intensity': 'Natężenie światła (lux)',
    'Rh': 'Wilgotność względna (%)',
    'Nitrogen': 'Azot (N)',
    'Phosphorus': 'Fosfor (P)',
    'Potassium': 'Potas (K)',
    'Category_pH': 'Kategoria pH',
    'Soil_Type': 'Typ gleby',
    'Season': 'Sezon',
    'N_Ratio': 'Stosunek N (%)',
    'P_Ratio': 'Stosunek P (%)',
    'K_Ratio': 'Stosunek K (%)'
}

st.title("Predykcja truskawek")

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
    st.subheader("Dane wejściowe:")
    st.dataframe(input_df)

    prediction = predict_model(ai_model, data=input_df)
    predicted_price = prediction.loc[0, 'prediction_label']
    st.success(f"Predykcja: {predicted_price}")
