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
    'Season': 'Pora roku',
}

feature_explanations = {
    'Photoperiod': 'Wpływ długości dnia na kwitnienie roślin. "Day Neutral" - kwitnie niezależnie od długości dnia, "Short Day Period" - wymaga krótkich dni do kwitnienia.',
    'Temperature': 'Średnia temperatura powietrza w okresie wegetacji. Optimum dla truskawek: 15-25°C. Zbyt niska hamuje wzrost, zbyt wysoka powoduje stres.',
    'Rainfall': 'Suma opadów w sezonie wegetacyjnym. Nadmiar powoduje gnicie owoców, niedobór - zahamowanie wzrostu. Optimum: 500-700 mm/rok.',
    'pH': 'Odczyn gleby decydujący o dostępności składników pokarmowych. Truskawki preferują gleby lekko kwaśne (pH 5.5-6.5).',
    'Light_Hours': 'Długość dnia świetlnego w godzinach. Wpływa na fotosyntezę i dojrzewanie owoców. Latem: 14-16h, wiosną/jesienią: 8-12h.',
    'Light_Intensity': 'Natężenie światła decydujące o intensywności fotosyntezy. Optimum: 400-600 lux. Niższe wartości spowalniają dojrzewanie.',
    'Rh': 'Wilgotność względna powietrza. Wysoka wilgotność (>90%) sprzyja chorobom grzybowym, niska (<70%) - transpiracji i stresowi wodnemu.',
    'Nitrogen': 'Poziom azotu w glebie (kg/ha). Nadmiar powoduje bujny wzrost liści kosztem owoców, niedobór - żółknięcie liści i słaby wzrost.',
    'Phosphorus': 'Poziom fosforu w glebie (kg/ha). Kluczowy dla rozwoju korzeni i kwitnienia. Niedobór objawia się fioletowymi przebarwieniami liści.',
    'Potassium': 'Poziom potasu w glebie (kg/ha). Wpływa na jakość owoców i odporność na choroby. Niedobór powoduje brązowienie brzegów liści.',
    'Season': 'Okres zbiorów. Wiosna (kwiecień-czerwiec) i lato (lipiec-sierpień) mają różne profile pogodowe wpływające na plonowanie.',
}

st.title("Predykcja zbioru truskawek")
st.markdown("Najedź na ikonę informacji przy każdym polu, aby zobaczyć opis parametru")

user_input = {}

for feature, info in feature_info.items():
    label = translations.get(feature, feature)
    help_text = feature_explanations.get(feature, "Brak opisu")

    if info['type'] == 'categorical':
        default = info['values'][0] if info['values'] else None
        user_input[feature] = st.selectbox(label, options=info['values'], index=0, help=help_text)
    elif info['type'] == 'numeric':
        min_val = info['min']
        max_val = info['max']
        default = (min_val + max_val) / 2
        if min_val == max_val:
            user_input[feature] = st.number_input(label, value=min_val, step=0.0, format="%.4f", help=help_text)
        else:
            user_input[feature] = st.slider(label, min_value=min_val, max_value=max_val, value=default, format="%.4f", help=help_text)

if st.button("Sprawdź predykcję"):
    input_df = pd.DataFrame([user_input])
    # st.subheader("Dane wejściowe:")
    # st.dataframe(input_df)

    prediction = predict_model(ai_model, data=input_df)
    predicted_price = prediction.loc[0, 'prediction_label']
    st.success(f"Przewidywany plon: {predicted_price} kg/ha")
    st.info("Wynik przedstawia przewidywaną wydajność uprawy w kilogramach na hektar")
