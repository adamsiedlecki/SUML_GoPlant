from pycaret.regression import *
import pandas as pd

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

setup(data=df, target='Yield')

model = load_model('best_model')
plot_model(model, plot='feature')