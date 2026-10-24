
import pandas as pd

print("Loading energy dataset...")

df = pd.read_csv("energy_data_ready.csv")

energy_col = "Energy Consumption (kWh)"
recommendations = []

# Check whether required columns exist
if energy_col not in df.columns:
    raise ValueError("Energy consumption column not found.")

# Calculate overall average energy consumption
average_energy = df[energy_col].mean()

print("\nENERGY-SAVING RECOMMENDATIONS")
print("-----------------------------")
print(f"Average energy consumption: {average_energy:.2f} kWh")

# Rule 1: High energy consumption
high_energy = df[df[energy_col] > average_energy * 1.20]

if len(high_energy) > 0:
    recommendations.append(
        "Review records with energy consumption more than "
        "20% above the dataset average."
    )

# Rule 2: HVAC consumption
hvac_col = "HVAC Consumption (kWh)"

if hvac_col in df.columns:
    hvac_average = df[hvac_col].mean()

    if hvac_average > 0:
        recommendations.append(
            "Review HVAC schedules and temperature settings. "
            f"Average HVAC consumption is {hvac_average:.2f} kWh."
        )

# Rule 3: Lighting consumption
lighting_col = "Lighting Consumption (kWh)"

if lighting_col in df.columns:
    lighting_average = df[lighting_col].mean()

    if lighting_average > 0:
        recommendations.append(
            "Review lighting schedules and consider efficient "
            f"lighting. Average lighting consumption is "
            f"{lighting_average:.2f} kWh."
        )

# Rule 4: Occupancy schedule
schedule_col = "Occupancy Schedule"

if schedule_col in df.columns and energy_col in df.columns:
    schedule_energy = df.groupby(schedule_col)[energy_col].mean()

    if "Vacant" in schedule_energy.index:
        vacant_energy = schedule_energy["Vacant"]

        recommendations.append(
            "Review energy use during vacant periods. "
            f"Average consumption for records marked Vacant: "
            f"{vacant_energy:.2f} kWh."
        )

# Display recommendations
if recommendations:
    for i, recommendation in enumerate(recommendations, start=1):
        print(f"{i}. {recommendation}")
else:
    print("No recommendations generated.")

# Save recommendations
recommendation_df = pd.DataFrame({
    "Recommendation": recommendations
})

recommendation_df.to_csv(
    "energy_recommendations.csv",
    index=False
)

print("\nSUCCESS!")
print("Recommendations saved to energy_recommendations.csv")