import pandas as pd

file_path = "data/Bitext_Sample_Customer_Service_Training_Dataset.csv"

df = pd.read_csv(file_path)

print("Dataset successfully loaded!")
print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)
print("\nUnique categories:")
print(df["category"].unique())

print("\nNumber of unique intents:")
print(df["intent"].nunique())

print("\nIntent distribution:")
print(df["intent"].value_counts())