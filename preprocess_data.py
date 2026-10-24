
import pandas as pd

INPUT_FILE = "electricity_consumption_optimization_dataset.csv"
OUTPUT_FILE = "cleaned_energy_data.csv"

df = pd.read_csv(INPUT_FILE)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Remove duplicate rows
df = df.drop_duplicates()

# Convert timestamp if available
if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], errors="coerce"
    )

# Convert energy consumption to numeric values
energy_col = "Energy Consumption (kWh)"

if energy_col not in df.columns:
    raise ValueError(
        f"Column '{energy_col}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

df[energy_col] = pd.to_numeric(
    df[energy_col], errors="coerce"
)

# Remove rows without a valid energy target
df = df.dropna(subset=[energy_col])

# Save cleaned data
df.to_csv(OUTPUT_FILE, index=False)

print("Data preprocessing completed!")
print("Cleaned dataset shape:", df.shape)
print("Saved file:", OUTPUT_FILE)