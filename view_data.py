import pandas as pd

file_path = "data/Bitext_Sample_Customer_Service_Training_Dataset.csv"

df = pd.read_csv(file_path)

sample = df[["utterance", "category", "intent"]].sample(20, random_state=42)

print(sample.to_string(index=False))