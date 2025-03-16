import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("plant_health_dataset.csv")

# Konwersja tylko kolumn float na int
df[df.select_dtypes('float').columns] = df.select_dtypes('float').astype(int)

# df = df[df['PlantAge'] > 36]  # interesują nas starsze rośliny
#
# df['wigor'] = df['GreenColorIntensity'] * df['NumLeaves'] * df['PlantHeight']
# df['srodowisko'] = df['SpotArea'] * df['SunExposureLevel']
# df['czy_slonecznie'] = df['SunExposureLevel'] > 50

# df = df.drop('DustPresence', axis=1)
# df = df.drop('LeafEdgeType', axis=1)

print("Brakujące wartości:\n", df.isnull().sum())

# Obliczamy korelację
corr = df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Macierz korelacji")
plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.3)
plt.show()


# Podział na cechy i zmienną docelową
X = df.drop('PlantHealth', axis=1)
y = df['PlantHealth']

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Optymalizacja hiperparametrów
param_grid = {
    'n_estimators': [100, 120],
    'max_depth': [None, 10, 15],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}

# Model z GridSearch
rf_model = GridSearchCV(
    RandomForestClassifier(n_jobs=-1, random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)
rf_model.fit(X_train, y_train)

print("\nNajlepsze parametry:", rf_model.best_params_)

# Predykcje
y_pred = rf_model.predict(X_test)
y_train_pred = rf_model.predict(X_train)

# Ewaluacja
print(f"\nAccuracy test: {accuracy_score(y_test, y_pred):.2f}")
print(f"Accuracy training: {accuracy_score(y_train, y_train_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Analiza ważności cech
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.best_estimator_.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:\n", feature_importances)

# Zapis modelu
joblib.dump(rf_model.best_estimator_, 'plant_health_model.pkl')