import pandas as pd

data = pd.read_csv("question_A/diabetes.csv")

print("First 5 rows:")
print(data.head())

print("\nShape of dataset:")
print(data.shape)

print("\nColumn names:")
print(data.columns)

print("\nMissing values:")
print(data.isnull().sum())

print("\nBasic information:")
print(data.info())

print("\nZero values:")
print((data == 0).sum())

print("\nDuplicate rows:")
print(data.duplicated().sum())