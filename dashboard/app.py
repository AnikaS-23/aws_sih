import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AWS Anomaly Detection",
    page_icon="🌦️",
    layout="wide"
)

st.title("🌦️ Intelligent Weather Station Anomaly Detector")
st.markdown(
    "**Ministry of Earth Sciences — SIH26073**  \n"
    "Automated Sensor Fault & Outlier Detection System"
)
st.divider()


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_json("../notebooks/weather.json")

    df = df[
        ["Date/Time", "Temp_C", "Rel Hum_%", "Wind Speed_km/h"]
    ].copy()

    df.columns = [
        "timestamp",
        "temperature",
        "humidity",
        "wind_speed"
    ]

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="mixed"
    )

    df = df.dropna()

    return df


df = load_data()


# ---------------- FAULT DETECTION ----------------
def detect_frozen_sensor(df, column, window=15, threshold=1e-6):
    rolling_std = df[column].rolling(window=window).std()
    return rolling_std[rolling_std < threshold].index.tolist()


def detect_spike_fault(df, column, z_score_threshold=3.5):
    mean = df[column].mean()
    std = df[column].std()

    z_scores = np.abs(
        (df[column] - mean) / (std + 1e-6)
    )

    return df[z_scores > z_score_threshold].index.tolist()


def detect_drift_fault(df, column, window=30, drift_threshold=0.05):
    rolling_mean_recent = (
        df[column]
        .rolling(window=window, min_periods=1)
        .mean()
    )

    rolling_mean_older = rolling_mean_recent.shift(window)

    mean_diff = np.abs(
        rolling_mean_recent - rolling_mean_older
    )

    relative_drift = mean_diff / (df[column].std() + 1e-6)

    return df[
        relative_drift > drift_threshold
    ].index.tolist()


def classify_fault(df, column, index):

    value = df.loc[index, column]

    fault_type = "No Fault"
    severity = "Low"

    # Frozen sensor
    window = min(15, index)

    if window > 0:
        recent_values = df.loc[
            max(0, index-window):index,
            column
        ].values

        if len(recent_values) > 1 and np.std(recent_values) < 1e-6:
            return {
                "timestamp": df.loc[index, "timestamp"],
                "sensor": column,
                "value": value,
                "fault_type": "Frozen Sensor",
                "severity": "High"
            }

    # Spike
    mean = df[column].mean()
    std = df[column].std()

    z_score = abs(
        (value - mean) / (std + 1e-6)
    )

    if z_score > 3.5:
        fault_type = "Wild Spike"
        severity = "High"

    # Drift
    window = min(30, index)

    if fault_type == "No Fault" and window > 5:

        older_values = df.loc[
            max(0, index-2*window):
            max(0, index-window),
            column
        ]

        recent_values = df.loc[
            max(0, index-window):index,
            column
        ]

        if len(older_values) > 0 and len(recent_values) > 0:

            drift_amount = abs(
                recent_values.mean() -
                older_values.mean()
            )

            if drift_amount > 0.1 * std:
                fault_type = "Sensor Drift"
                severity = "Medium"

    return {
        "timestamp": df.loc[index, "timestamp"],
        "sensor": column,
        "value": value,
        "fault_type": fault_type,
        "severity": severity
    }


def classify_all_faults(df):

    faults = []

    for column in [
        "temperature",
        "humidity",
        "wind_speed"
    ]:

        frozen = detect_frozen_sensor(df, column)
        spikes = detect_spike_fault(df, column)
        drift = []

        suspect_indices = set(
            frozen + spikes + drift
        )

        for index in suspect_indices:

            result = classify_fault(
                df,
                column,
                index
            )

            if result["fault_type"] != "No Fault":
                faults.append(result)

    return pd.DataFrame(faults)


faults_df = classify_all_faults(df)


# ---------------- DASHBOARD METRICS ----------------

total_readings = len(df)
total_faults = len(faults_df)

high_faults = len(
    faults_df[
        faults_df["severity"] == "High"
    ]
)

medium_faults = len(
    faults_df[
        faults_df["severity"] == "Medium"
    ]
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "📡 Readings Processed",
    f"{total_readings:,}"
)

col2.metric(
    "🚨 Faults Detected",
    f"{total_faults:,}"
)

col3.metric(
    "🔴 High Severity",
    f"{high_faults:,}"
)

col4.metric(
    "🟡 Medium Severity",
    f"{medium_faults:,}"
)

st.divider()


# ---------------- SENSOR TRENDS ----------------

st.subheader("📈 Sensor Monitoring")

sensor = st.selectbox(
    "Select sensor",
    ["temperature", "humidity", "wind_speed"]
)

fig = px.line(
    df,
    x="timestamp",
    y=sensor,
    markers=False,
    title=f"{sensor.replace('_', ' ').title()} Sensor Stream"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ---------------- FAULT DISTRIBUTION ----------------

st.subheader("🚨 Fault Distribution")

if not faults_df.empty:

    fault_counts = (
        faults_df["fault_type"]
        .value_counts()
        .reset_index()
    )

    fault_counts.columns = [
        "fault_type",
        "count"
    ]

    fig2 = px.bar(
        fault_counts,
        x="fault_type",
        y="count",
        title="Detected Sensor Fault Types"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

else:
    st.success("✅ No sensor faults detected.")


# ---------------- ALERT QUEUE ----------------

st.subheader("🛠️ Field Maintenance Alert Queue")

if not faults_df.empty:

    st.warning(
        "⚠️ Sensor anomalies detected. "
        "Review the following readings for maintenance."
    )

    display_df = faults_df[
        [
            "timestamp",
            "sensor",
            "value",
            "fault_type",
            "severity"
        ]
    ].copy()

    st.dataframe(
        display_df.head(100),
        use_container_width=True
    )

else:

    st.success(
        "✅ All monitored sensors are operating normally."
    )


# ---------------- RECENT DATA ----------------

with st.expander("📋 View Recent Weather Data"):

    st.dataframe(
        df.tail(20),
        use_container_width=True
    )

st.caption(
    "SIH26073 • AWS Intelligent Anomaly Detection System"
)