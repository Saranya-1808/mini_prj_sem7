
import pandas as pd

FILE_PATH = "electricity_consumption_optimization_dataset.csv"

df = pd.read_csv(FILE_PATH)

print("DATASET INFORMATION")
print("-------------------")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nFIRST 5 ROWS")
print(df.head().to_string())

print("\nMISSING VALUES")
print(df.isnull().sum().to_string())

print("\nDATA TYPES")
print(df.dtypes.to_string())