
import pandas as pd

INPUT_FILE = "cleaned_energy_data.csv"
OUTPUT_FILE = "energy_data_ready.csv"

df = pd.read_csv(INPUT_FILE)

# Remove duplicate rows
df = df.drop_duplicates()

# Convert important columns to numeric
numeric_columns = [
    "Energy Consumption (kWh)",
    "Occupancy Rate (%)",
    "Building Size (m²)",
    "Building Age (years)",
    "Equipment Age (years)"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing target values
target = "Energy Consumption (kWh)"
df = df.dropna(subset=[target])

# Keep non-negative energy consumption records
df = df[df[target] >= 0]

# Keep physically meaningful ranges for these fields
if "Occupancy Rate (%)" in df.columns:
    df = df[df["Occupancy Rate (%)"].between(0, 100)]

if "Building Size (m²)" in df.columns:
    df = df[df["Building Size (m²)"] > 0]

for col in ["Building Age (years)", "Equipment Age (years)"]:
    if col in df.columns:
        df = df[df[col] >= 0]

df.to_csv(OUTPUT_FILE, index=False)

print("Data cleaning completed!")
print("Rows before cleaning:", len(pd.read_csv(INPUT_FILE)))
print("Rows after cleaning:", len(df))
print("Saved file:", OUTPUT_FILE)