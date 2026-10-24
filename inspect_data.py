
import pandas as pd

df = pd.read_csv("cleaned_energy_data.csv")

print("DATASET SHAPE:", df.shape)

# Check columns that should not contain negative values
columns_to_check = [
    "Energy Consumption (kWh)",
    "Occupancy Rate (%)",
    "Building Size (m²)",
    "Building Age (years)",
    "Equipment Age (years)",
    "Lighting Consumption (kWh)",
    "HVAC Consumption (kWh)",
    "Water Usage (liters)"
]

print("\nNEGATIVE VALUE CHECK")
print("--------------------")

for col in columns_to_check:
    if col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce")
        print(f"{col}: {int((values < 0).sum())} negative values")

print("\nMISSING VALUES")
print("--------------")
print(df.isnull().sum().sort_values(ascending=False).head(10))

print("\nENERGY CONSUMPTION BY BUILDING TYPE")
print("-----------------------------------")
print(
    df.groupby("Building Type")["Energy Consumption (kWh)"]
    .agg(["count", "mean", "min", "max"])
)

print("\nENERGY CONSUMPTION BY OCCUPANCY SCHEDULE")
print("----------------------------------------")
print(
    df.groupby("Occupancy Schedule")["Energy Consumption (kWh)"]
    .agg(["count", "mean", "min", "max"])
)