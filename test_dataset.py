import pandas as pd

# Load the legal cases dataset
data = pd.read_csv("dataset/legal_cases.csv")

# Display basic information
print("===== LAWASSIST DATASET =====")
print("Total cases:", len(data))
print()

print("Available columns:")
print(data.columns.tolist())
print()

print("===== CASE TITLES =====")
for title in data["case_title"]:
    print("-", title)

print()
print("===== FIRST CASE =====")
print(data.iloc[0])