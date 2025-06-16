import pandas as pd
from pycaret.regression import setup, compare_models, pull, save_model

df = pd.read_csv("Soil Nutrients.csv")

# preprocessing
df = df[df['Name'] == 'Strawberry'] # chcemy uczyć tylko o truskawce
df = df.drop(columns=['Name'])

# poniższe kolumny są mało różnorodne dla truskawki
df = df.drop(columns=['Fertility'])
df = df.drop(columns=['Category_pH'])
df = df.drop(columns=['N_Ratio'])
df = df.drop(columns=['P_Ratio'])
df = df.drop(columns=['K_Ratio'])
df = df.drop(columns=['Soil_Type'])

def train_model(train_data):
    # Initialize the PyCaret environment
    setup(data=train_data, target='Yield')

    # Train and evaluate multiple models
    best_model = compare_models(include=[
        'lr',        # Linear Regression (baseline)
        'ridge',     # Regularized linear model (L2)
        'rf',  # random forest
        'mlp',       # Perceptron wielowarstwowy
    ])

    # Optionally, pull and log comparison results
    comparison_results = pull()
    print(comparison_results)  # or use logging
    save_model(best_model, 'best_model')

    return best_model

train_model(df)