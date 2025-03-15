import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv("plant_health_dataset.csv")

print("Brakujące wartości:\n", df.isnull().sum())

# Podział na cechy i zmienną docelową
X = df.drop('PlantHealth', axis=1)
y = df['PlantHealth']

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=31
)

# Inicjalizacja i trenowanie modelu Random Forest
rf_model = RandomForestClassifier(
    n_estimators=31,
    random_state=31,
    max_depth=8,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Predykcja na danych testowych
y_pred = rf_model.predict(X_test)
y_train_pred = rf_model.predict(X_train)

# Ocena modelu
print(f"\nAccuracy test: {accuracy_score(y_test, y_pred):.2f}")
print(f"Accuracy training: {accuracy_score(y_train, y_train_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Analiza ważności cech
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:\n", feature_importances)

# zapis modelu do pliku
joblib.dump(rf_model, 'plant_health_model.pkl')