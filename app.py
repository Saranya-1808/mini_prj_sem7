
import os
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AI Energy Management System",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ AI-Powered Energy Management System")
st.write(
    "Monitor energy consumption, view machine-learning predictions, "
    "identify unusual records, and explore energy-saving suggestions."
)

DATA_FILE = "energy_data_ready.csv"
ANOMALY_FILE = "anomaly_results.csv"
RECOMMENDATION_FILE = "energy_recommendations.csv"
MODEL_FILE = "models/energy_prediction_model.pkl"

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

if not os.path.exists(DATA_FILE):
    st.error("energy_data_ready.csv was not found. Run clean_data.py first.")
    st.stop()

df = load_data()

target = "Energy Consumption (kWh)"

if target not in df.columns:
    st.error(f"Missing required column: {target}")
    st.stop()

df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target])

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Dashboard Filters")

if "Building Type" in df.columns:
    building_types = sorted(df["Building Type"].dropna().unique().tolist())
    selected_buildings = st.sidebar.multiselect(
        "Select building type",
        building_types,
        default=building_types
    )
    filtered_df = df[df["Building Type"].isin(selected_buildings)].copy()
else:
    filtered_df = df.copy()

# ---------------- KEY METRICS ----------------
st.subheader("Energy Overview")

if filtered_df.empty:
    st.warning("No records match your selected filters.")
    st.stop()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Energy (kWh)",
    f"{filtered_df[target].sum():,.2f}"
)

col2.metric(
    "Average Energy (kWh)",
    f"{filtered_df[target].mean():,.2f}"
)

col3.metric(
    "Records Analyzed",
    f"{len(filtered_df):,}"
)

# ---------------- ENERGY CHART ----------------
st.subheader("Energy Consumption Analysis")

chart_option = st.selectbox(
    "Choose chart",
    ["Energy by Building Type", "Energy Distribution"]
)

if chart_option == "Energy by Building Type":
    if "Building Type" in filtered_df.columns:
        chart_data = (
            filtered_df.groupby("Building Type")[target]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(chart_data)
    else:
        st.info("Building Type is not available in the dataset.")

else:
    st.bar_chart(filtered_df[target].reset_index(drop=True))

# ---------------- PREDICTION MODEL ----------------
st.subheader("Energy Prediction")

if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)

        st.write("Choose a record to estimate its energy consumption.")
        record_index = st.number_input(
            "Record number",
            min_value=0,
            max_value=len(filtered_df) - 1,
            value=0,
            step=1
        )

        if st.button("Predict Energy Consumption"):
            row = filtered_df.iloc[[int(record_index)]]
            features = row.drop(
                columns=[target, "Timestamp"],
                errors="ignore"
            )

            prediction = model.predict(features)[0]

            st.metric(
                "Estimated Energy Consumption",
                f"{prediction:.2f} kWh"
            )
            st.caption(
                "This is a model estimate, not a guaranteed future forecast."
            )

    except Exception as e:
        st.error(f"Prediction could not be completed: {e}")
else:
    st.warning("Prediction model not found. Run train_clean_model.py first.")

# ---------------- ANOMALY RESULTS ----------------
st.subheader("Anomaly Detection")

if os.path.exists(ANOMALY_FILE):
    anomaly_df = pd.read_csv(ANOMALY_FILE)

    if "Anomaly Status" in anomaly_df.columns:
        anomaly_count = (
            anomaly_df["Anomaly Status"] == "Anomaly"
        ).sum()

        st.metric("Flagged Records", f"{anomaly_count:,}")

        st.dataframe(
            anomaly_df[
                anomaly_df["Anomaly Status"] == "Anomaly"
            ].head(100),
            use_container_width=True
        )
    else:
        st.warning("Anomaly Status column was not found.")
else:
    st.info("Run anomaly_detection.py to generate anomaly results.")

# ---------------- RECOMMENDATIONS ----------------
st.subheader("Energy-Saving Recommendations")

if os.path.exists(RECOMMENDATION_FILE):
    recommendation_df = pd.read_csv(RECOMMENDATION_FILE)

    if "Recommendation" in recommendation_df.columns:
        for recommendation in recommendation_df["Recommendation"].dropna():
            st.info(recommendation)
    else:
        st.warning("Recommendation column was not found.")
else:
    st.info("Run recommendations.py to generate recommendations.")

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Academic project demonstration. Results depend on dataset quality "
    "and model performance."
)