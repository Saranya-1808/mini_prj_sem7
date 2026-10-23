
import pandas as pd
from sklearn.ensemble import IsolationForest

print("Loading energy dataset...")

df = pd.read_csv("energy_data_ready.csv")

target = "Energy Consumption (kWh)"

# Use energy consumption for the initial anomaly detector
energy = pd.to_numeric(df[target], errors="coerce")

df = df.loc[energy.notna()].copy()
df[target] = energy.loc[energy.notna()]

X = df[[target]]

print("Detecting unusual energy consumption...")

# Train the anomaly detection model
detector = IsolationForest(
    contamination=0.05,
    random_state=42
)

df["Anomaly Status"] = detector.fit_predict(X)

# Isolation Forest returns -1 for anomalies and 1 for normal records
df["Anomaly Status"] = df["Anomaly Status"].map({
    1: "Normal",
    -1: "Anomaly"
})

# Save the results
df.to_csv("anomaly_results.csv", index=False)

# Display summary
anomaly_count = (df["Anomaly Status"] == "Anomaly").sum()
normal_count = (df["Anomaly Status"] == "Normal").sum()

print("\nANOMALY DETECTION RESULTS")
print("-------------------------")
print("Total records:", len(df))
print("Normal records:", normal_count)
print("Anomalous records:", anomaly_count)
print("Anomaly percentage:", round(anomaly_count / len(df) * 100, 2), "%")

print("\nSample detected anomalies:")
print(
    df.loc[
        df["Anomaly Status"] == "Anomaly",
        ["Timestamp", "Building Type", target, "Anomaly Status"]
    ].head(10).to_string(index=False)
)

print("\nSUCCESS!")
print("Results saved to anomaly_results.csv")