import pandas as pd
import numpy as np

def summarize_dataframe(df):
    summary = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_min = df[col].min()
            col_max = df[col].max()
            summary[col] = {
                'type': 'numeric',
                'min': col_min,
                'max': col_max
            }
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            unique_values = sorted(df[col].dropna().unique().tolist())
            summary[col] = {
                'type': 'categorical',
                'values': unique_values
            }
        else:
            summary[col] = {
                'type': 'unsupported',
                'note': f"Unsupported dtype: {df[col].dtype}"
            }
    return summary

# Przykład użycia:
if __name__ == "__main__":
    df = pd.read_csv("Soil Nutrients.csv")
    # df = df[df['Name'] == 'Strawberry']  # chcemy uczyć tylko o truskawce
    df = df.drop(columns=['Name'])

    liczba_wierszy = len(df)
    print(f"Liczba wierszy: {liczba_wierszy}")

    summary = summarize_dataframe(df)
    print(summary)
