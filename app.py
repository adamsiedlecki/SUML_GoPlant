import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model
import matplotlib.pyplot as plt

# TODO usunięcie częśći pól (trzeba przetrenować model), poprawa interfejsu, może tłumaczenia dla wartości

with st.expander("ℹ️ Porady dla plantatorów"):
    st.markdown("""
    **Optymalne warunki dla truskawek:**
    - Temperatura: 18-22°C w dzień, 10-13°C w nocy
    - Wilgotność: 80-85% podczas kwitnienia
    - pH gleby: 5.5-6.5
    - Nawożenie: N-P-K w proporcjach 2:1:3
    
    **Częste problemy:**
    - Zbyt wysoka temperatura (>25°C) - redukcja owocowania
    - Niskie pH (<5.0) - niedobór wapnia
    - Nadmiar azotu - bujny wzrost liści kosztem owoców
    """)

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
    st.session_state.current_prediction = predicted_price

    st.success(f"Przewidywany plon: {predicted_price} kg/ha")
    st.info("Wynik przedstawia przewidywaną wydajność uprawy w kilogramach na hektar")

    #wykresik
    fig, ax = plt.subplots(figsize=(10, 3))
    
    yield_ranges = {
        "Niski": (18.0, 19.0, "#ff4b4b"),
        "Średni": (19.0, 20.5, "#ffdc4b"),
        "Dobry": (20.5, 21.5, "#a1ff4b"),
        "Doskonały": (21.5, 22.5, "#4bff6a")
    }
    
    current_category = "Niski"
    for category, (low, high, color) in yield_ranges.items():
        if low <= predicted_price < high:
            current_category = category
            current_color = color
            break
    
    for category, (low, high, color) in yield_ranges.items():
        ax.barh(category, high - low, left=low, color=color, alpha=0.4)
    
    ax.barh(current_category, predicted_price, color=current_color, height=0.5)
    
    ax.text(predicted_price, list(yield_ranges.keys()).index(current_category), 
            f' {predicted_price:.2f} kg/ha', 
            ha='left', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(17.8, 22.5) 
    ax.set_xlabel('Plon (kg/ha)')
    ax.set_title('Poziom plonów w porównaniu do skali efektywności', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    for value in [19.0, 20.5, 21.5]:
        ax.axvline(value, color='gray', linestyle='--', alpha=0.5)
    
    st.pyplot(fig)

    if predicted_price >= 21.5:
        st.success("Bardzo dobre warunki sprzyjające bogatym zbiorom")
    elif predicted_price >= 20.5:
        st.success("Dobre warunki sprzyjające przyzwoitym zbiorom")
    elif predicted_price >= 19.0:
        st.info("Średnie warunki sprzyjające akceptowalnym zbiorom")
    else:
        st.warning("Warunki poniżej optymalnych - rozważ zmiany w uprawie")

st.divider()
with st.expander("Eksport wyników"):
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    if st.button("Zapisz obecny wynik"):
        try:
            if 'current_prediction' in st.session_state:
                new_entry = {
                    **user_input,
                    'przewidywany_plon': st.session_state.current_prediction,
                    'timestamp': pd.Timestamp.now()
                }
                st.session_state.history.append(new_entry)
                st.success("Wynik zapisany!")
            else:
                st.warning("Najpierw wykonaj predykcję przed zapisem")
        except Exception as e:
            st.error(f"Błąd podczas zapisu: {e}")
    
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        base_columns = [col for col in feature_info.keys()]
        column_order = base_columns + ['przewidywany_plon', 'timestamp']
        history_df = history_df.reindex(columns=column_order)
        
        st.dataframe(history_df)
        
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Pobierz historię",
            csv,
            "historia_predykcji_truskawek.csv",
            "text/csv")
    else:
        st.info("Brak zapisanych wyników")