
import os
import pandas as pd
import streamlit as st
import joblib

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AI Energy Management",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. CUSTOM DESIGN
# =========================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1220 0%, #101b2d 100%);
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263449;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #9caec5;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #e2e8f0;
        margin-top: 18px;
        margin-bottom: 14px;
    }

    .metric-card {
        background: linear-gradient(145deg, #17263b, #142033);
        border: 1px solid #2c405a;
        border-radius: 14px;
        padding: 20px;
        min-height: 125px;
    }

    .metric-label {
        color: #a9bbd0;
        font-size: 14px;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #67e8f9;
        font-size: 27px;
        font-weight: 750;
        overflow-wrap: anywhere;
    }

    .metric-note {
        color: #93a8bf;
        font-size: 12px;
        margin-top: 7px;
    }

    .recommendation-card {
        background-color: #14263a;
        border-left: 4px solid #22d3ee;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        color: #e2e8f0;
    }

    .alert-card {
        background-color: #33251a;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
        color: #ffedd5;
    }

    div[data-testid="stButton"] button {
        background: linear-gradient(90deg, #0891b2, #2563eb);
        color: white;
        border: none;
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
    }

    div[data-testid="stButton"] button:hover {
        border: 1px solid #67e8f9;
        color: white;
    }

    div[data-testid="stMetric"] {
        background-color: #17263b;
        border: 1px solid #2c405a;
        padding: 15px;
        border-radius: 12px;
    }

    h1, h2, h3 {
        color: #f8fafc;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #2c405a;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. FILE PATHS
# =========================================================
DATA_FILE = "energy_data_ready.csv"
ANOMALY_FILE = "anomaly_results.csv"
RECOMMENDATION_FILE = "energy_recommendations.csv"
MODEL_FILE = "models/energy_prediction_model.pkl"

TARGET = "Energy Consumption (kWh)"

# =========================================================
# 4. LOAD DATA
# =========================================================
@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)

st.markdown(
    '<div class="main-title">⚡ AI Energy Management System</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">'
    'Intelligent energy monitoring, consumption analysis, '
    'prediction and anomaly detection'
    '</div>',
    unsafe_allow_html=True
)

if not os.path.exists(DATA_FILE):
    st.error(
        f"Dataset not found: {DATA_FILE}. "
        "Run clean_data.py first."
    )
    st.stop()

df = load_csv(DATA_FILE).copy()

if TARGET not in df.columns:
    st.error(f"Required column not found: {TARGET}")
    st.stop()

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).copy()

if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], errors="coerce"
    )

# =========================================================
# 5. SIDEBAR FILTERS
# =========================================================
st.sidebar.markdown("## ⚙️ Dashboard Controls")
st.sidebar.caption("Choose which energy records to view.")

filtered_df = df.copy()

if "Building Type" in df.columns:
    building_types = sorted(
        df["Building Type"].dropna().unique().tolist()
    )

    selected_buildings = st.sidebar.multiselect(
        "Building type",
        options=building_types,
        default=building_types
    )

    filtered_df = filtered_df[
        filtered_df["Building Type"].isin(selected_buildings)
    ]

if (
    "Timestamp" in filtered_df.columns
    and filtered_df["Timestamp"].notna().any()
):
    valid_dates = filtered_df["Timestamp"].dropna()

    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range

        filtered_df = filtered_df[
            filtered_df["Timestamp"].dt.date.between(
                start_date, end_date
            )
        ]

st.sidebar.divider()
st.sidebar.caption("Project demonstration version")
st.sidebar.caption("Dataset-based analysis; not live IoT monitoring.")

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# =========================================================
# 6. KPI CARDS
# =========================================================
st.markdown(
    '<div class="section-title">📊 Energy Overview</div>',
    unsafe_allow_html=True
)

total_energy = filtered_df[TARGET].sum()
average_energy = filtered_df[TARGET].mean()
record_count = len(filtered_df)

k1, k2, k3 = st.columns(3)

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">⚡ Total Energy Consumption</div>
        <div class="metric-value">{total_energy:,.2f}</div>
        <div class="metric-note">kWh across filtered records</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📈 Average Consumption</div>
        <div class="metric-value">{average_energy:,.2f}</div>
        <div class="metric-note">kWh per record</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🏢 Records Analyzed</div>
        <div class="metric-value">{record_count:,}</div>
        <div class="metric-note">Records after applying filters</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# 7. ENERGY CHARTS
# =========================================================
st.markdown(
    '<div class="section-title">📉 Consumption Analytics</div>',
    unsafe_allow_html=True
)

chart_left, chart_right = st.columns(2)

with chart_left:
    st.markdown("#### Average Energy by Building")

    if "Building Type" in filtered_df.columns:
        building_energy = (
            filtered_df.groupby("Building Type")[TARGET]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(building_energy)
    else:
        st.info("Building Type is unavailable.")

with chart_right:
    st.markdown("#### Energy Consumption Distribution")

    # Group consumption values into ranges for a frequency chart.
    histogram_data = filtered_df[TARGET].dropna()

    if len(histogram_data) > 0:
        counts, edges = __import__("numpy").histogram(
            histogram_data, bins=15
        )

        labels = [
            f"{edges[i]:.0f}–{edges[i + 1]:.0f}"
            for i in range(len(edges) - 1)
        ]

        distribution = pd.Series(counts, index=labels)
        st.bar_chart(distribution)
        st.caption("Number of records in each energy range.")

if (
    "Timestamp" in filtered_df.columns
    and filtered_df["Timestamp"].notna().any()
):
    st.markdown("#### Energy Consumption Over Time")

    time_data = filtered_df.dropna(subset=["Timestamp"]).copy()
    time_data["Date"] = time_data["Timestamp"].dt.floor("D")

    daily_energy = (
        time_data.groupby("Date")[TARGET]
        .mean()
        .sort_index()
    )

    st.line_chart(daily_energy)
    st.caption("Shows average energy consumption per day, not a future forecast.")

# =========================================================
# 8. ENERGY PREDICTION
# =========================================================
st.markdown(
    '<div class="section-title">🤖 Energy Prediction</div>',
    unsafe_allow_html=True
)

st.write(
    "Select a record to estimate its energy consumption "
    "using the trained machine-learning model."
)

if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)

        if len(filtered_df) > 0:
            record_number = st.number_input(
                "Select record number",
                min_value=0,
                max_value=len(filtered_df) - 1,
                value=0,
                step=1
            )

            if st.button("⚡ Predict Energy Consumption"):
                selected_row = filtered_df.iloc[[int(record_number)]]

                features = selected_row.drop(
                    columns=[TARGET, "Timestamp"],
                    errors="ignore"
                )

                prediction = float(model.predict(features)[0])

                p1, p2 = st.columns(2)

                p1.metric(
                    "Actual Consumption",
                    f"{selected_row[TARGET].iloc[0]:.2f} kWh"
                )

                p2.metric(
                    "Estimated Consumption",
                    f"{prediction:.2f} kWh"
                )

                st.info(
                    "This is an estimate for a selected historical record, "
                    "not a verified forecast of future consumption."
                )

                st.caption(
                    "The model's current test R² score was -0.0052. "
                    "Its prediction accuracy needs improvement before "
                    "reliable operational use."
                )

    except Exception as error:
        st.error(f"Could not load or use the prediction model: {error}")
else:
    st.warning(
        "Prediction model not found. Run train_clean_model.py first."
    )

# =========================================================
# 9. ANOMALY DETECTION
# =========================================================
st.markdown(
    '<div class="section-title">🚨 Anomaly Detection</div>',
    unsafe_allow_html=True
)

if os.path.exists(ANOMALY_FILE):
    anomaly_df = load_csv(ANOMALY_FILE).copy()

    if "Timestamp" in anomaly_df.columns:
        anomaly_df["Timestamp"] = pd.to_datetime(
            anomaly_df["Timestamp"], errors="coerce"
        )

    if "Anomaly Status" in anomaly_df.columns:
        anomaly_view = anomaly_df.copy()

        if (
            "Building Type" in anomaly_view.columns
            and "Building Type" in filtered_df.columns
        ):
            anomaly_view = anomaly_view[
                anomaly_view["Building Type"].isin(selected_buildings)
            ]

        if (
            "Timestamp" in anomaly_view.columns
            and "Timestamp" in filtered_df.columns
            and filtered_df["Timestamp"].notna().any()
        ):
            dates = filtered_df["Timestamp"].dropna()

            if len(dates) > 0:
                anomaly_view = anomaly_view[
                    anomaly_view["Timestamp"].dt.date.between(
                        dates.min().date(), dates.max().date()
                    )
                ]

        flagged = anomaly_view[
            anomaly_view["Anomaly Status"] == "Anomaly"
        ]

        normal_count = (
            anomaly_view["Anomaly Status"] == "Normal"
        ).sum()

        a1, a2, a3 = st.columns(3)

        a1.metric("Flagged Records", f"{len(flagged):,}")
        a2.metric("Normal Records", f"{normal_count:,}")
        a3.metric(
            "Flagged Percentage",
            f"{len(flagged) / len(anomaly_view) * 100:.2f}%"
            if len(anomaly_view) else "0%"
        )

        if len(flagged) > 0:
            st.markdown(
                '<div class="alert-card">'
                '⚠️ These records were flagged as statistical outliers. '
                'They require review and are not confirmed equipment faults.'
                '</div>',
                unsafe_allow_html=True
            )

            display_columns = [
                col for col in [
                    "Timestamp",
                    "Building Type",
                    TARGET,
                    "Anomaly Status"
                ]
                if col in flagged.columns
            ]

            st.dataframe(
                flagged[display_columns].head(100),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Anomaly Records",
                data=flagged.to_csv(index=False).encode("utf-8"),
                file_name="energy_anomalies.csv",
                mime="text/csv"
            )
        else:
            st.success("No anomalies were flagged for these filters.")
    else:
        st.warning("Anomaly Status column was not found.")
else:
    st.info("Run anomaly_detection.py to generate anomaly results.")

# =========================================================
# 10. ENERGY-SAVING RECOMMENDATIONS
# =========================================================
st.markdown(
    '<div class="section-title">💡 Energy-Saving Recommendations</div>',
    unsafe_allow_html=True
)

if os.path.exists(RECOMMENDATION_FILE):
    recommendation_df = load_csv(RECOMMENDATION_FILE)

    if "Recommendation" in recommendation_df.columns:
        recommendations = (
            recommendation_df["Recommendation"]
            .dropna()
            .tolist()
        )

        if recommendations:
            for number, recommendation in enumerate(
                recommendations, start=1
            ):
                st.markdown(
                    f'<div class="recommendation-card">'
                    f'<b>{number}.</b> {recommendation}'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("No recommendations are currently available.")
else:
    st.info("Run recommendations.py to generate recommendations.")

# =========================================================
# 11. DOWNLOAD DATA REPORT
# =========================================================
st.markdown(
    '<div class="section-title">📁 Export Energy Data</div>',
    unsafe_allow_html=True
)

st.write(
    "Download the records currently selected by your dashboard filters."
)

st.download_button(
    label="⬇️ Download Filtered Energy Data (CSV)",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_energy_report.csv",
    mime="text/csv"
)

# =========================================================
# 12. FOOTER
# =========================================================
st.divider()

st.caption(
    "AI-Powered Energy Management System | Academic Project Prototype"
)

st.caption(
    "Results depend on data quality and model performance. "
    "Recommendations should be validated before operational decisions."
)