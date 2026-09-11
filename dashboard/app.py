import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
        color: #f9fafb;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 16px;
        margin-top: 5px;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(
            135deg,
            #111827,
            #172033
        );
        border: 1px solid #263247;
        border-radius: 14px;
        padding: 20px;
        height: 125px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .metric-label {
        color: #9ca3af;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #f9fafb;
        font-size: 30px;
        font-weight: 700;
    }

    .metric-sub {
        color: #6ee7b7;
        font-size: 12px;
        margin-top: 5px;
    }

    /* Section headings */
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #f3f4f6;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    /* Status */
    .status-good {
        background: #052e1b;
        border: 1px solid #166534;
        padding: 12px 16px;
        border-radius: 10px;
        color: #86efac;
        font-weight: 600;
    }

    .status-warning {
        background: #3b2505;
        border: 1px solid #92400e;
        padding: 12px 16px;
        border-radius: 10px;
        color: #fbbf24;
        font-weight: 600;
    }

    /* Alert cards */
    .alert-card {
        background: #111827;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }

    .alert-card.severity-critical {
        border-left-color: #ef4444;
    }

    .alert-card.severity-high {
        border-left-color: #f97316;
    }

    .alert-card.severity-medium {
        border-left-color: #eab308;
    }

    .alert-card.severity-low {
        border-left-color: #6b7280;
    }

    .alert-title {
        color: #f9fafb;
        font-weight: 600;
    }

    .alert-text {
        color: #9ca3af;
        font-size: 13px;
    }

    /* Hide Streamlit menu/footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = BASE_DIR / "data" / "raw" / "weather.json"

ANOMALY_DATA = BASE_DIR / "src" / "anomaly_results_v2.csv"


# =========================================================
# FAULT CLASSIFICATION LOGIC
# (ported from AWS_Fault_Classification.ipynb)
# =========================================================

def detect_frozen_sensor(df, column, window=15, threshold=1e-6):
    rolling_std = df[column].rolling(window=window).std()
    frozen_indices = rolling_std[rolling_std < threshold].index.tolist()
    return frozen_indices


def detect_spike_fault(df, column, z_score_threshold=3.5):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / (std + 1e-6))
    spike_indices = df[z_scores > z_score_threshold].index.tolist()
    return spike_indices


def detect_drift_fault(df, column, window=30, drift_threshold=0.05):
    rolling_mean_recent = df[column].rolling(window=window, min_periods=1).mean()
    rolling_mean_older = rolling_mean_recent.shift(window)

    mean_diff = np.abs(rolling_mean_recent - rolling_mean_older)
    relative_drift = mean_diff / (df[column].std() + 1e-6)

    drift_indices = df[relative_drift > drift_threshold].index.tolist()
    return drift_indices


def detect_missing_data(df, column):
    missing_indices = df[df[column].isna()].index.tolist()
    return missing_indices


def classify_fault(df, column, index, timestamp_col="Date/Time"):

    fault_info = {
        "index": index,
        "timestamp": df.loc[index, timestamp_col],
        "value": df.loc[index, column],
        "fault_type": "Unknown",
        "severity": "Low",
    }

    if pd.isna(df.loc[index, column]):
        fault_info["fault_type"] = "Missing Data"
        fault_info["severity"] = "Medium"
        return fault_info

    window = min(15, index)
    if window > 0:
        recent_values = df.loc[max(0, index - window):index, column].values
        if len(recent_values) > 1 and np.std(recent_values) < 1e-6:
            fault_info["fault_type"] = "Frozen Sensor"
            fault_info["severity"] = "High"
            return fault_info

    mean = df[column].mean()
    std = df[column].std()
    z_score = abs((df.loc[index, column] - mean) / (std + 1e-6))

    if z_score > 3.5:
        fault_info["fault_type"] = "Wild Spike"
        fault_info["severity"] = "High"
        return fault_info

    window = min(30, index)
    if window > 5:
        older_values = df.loc[max(0, index - 2 * window):max(0, index - window), column]
        recent_values = df.loc[max(0, index - window):index, column]

        if len(older_values) > 0 and len(recent_values) > 0:
            older_mean = older_values.mean()
            recent_mean = recent_values.mean()
            drift_amount = abs(recent_mean - older_mean)

            if drift_amount > 0.1 * std:
                fault_info["fault_type"] = "Sensor Drift"
                fault_info["severity"] = "Medium"
                return fault_info

    fault_info["fault_type"] = "No Fault"
    fault_info["severity"] = "Low"
    return fault_info


def classify_all_faults(df, columns, timestamp_col="Date/Time"):
    faults_detected = []

    for column in columns:
        frozen_indices = detect_frozen_sensor(df, column)
        spike_indices = detect_spike_fault(df, column)
        drift_indices = detect_drift_fault(df, column)
        missing_indices = detect_missing_data(df, column)

        all_suspect_indices = set(
            frozen_indices + spike_indices + drift_indices + missing_indices
        )

        for idx in all_suspect_indices:
            fault_classification = classify_fault(df, column, idx, timestamp_col)
            fault_classification["sensor"] = column
            faults_detected.append(fault_classification)

    faults_df = pd.DataFrame(faults_detected)

    if len(faults_df) > 0:
        severity_order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        faults_df["severity_rank"] = faults_df["severity"].map(severity_order)
        faults_df = faults_df.sort_values(
            ["severity_rank", "timestamp"], ascending=[False, False]
        )
        faults_df = faults_df.drop("severity_rank", axis=1)
        faults_df = faults_df[faults_df["fault_type"] != "No Fault"]

    return faults_df


def generate_alerts_summary(faults_df):

    if len(faults_df) == 0:
        return {
            "total_faults": 0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
        }

    summary = {
        "total_faults": len(faults_df),
        "critical_count": len(faults_df[faults_df["severity"] == "Critical"]),
        "high_count": len(faults_df[faults_df["severity"] == "High"]),
        "medium_count": len(faults_df[faults_df["severity"] == "Medium"]),
        "low_count": len(faults_df[faults_df["severity"] == "Low"]),
    }

    return summary


SEVERITY_ICON = {
    "Critical": "🔴",
    "High": "🟠",
    "Medium": "🟡",
    "Low": "⚪",
}

FAULT_TYPE_EXPLANATION = {
    "Frozen Sensor": "Sensor readings have stopped changing — likely a stuck or malfunctioning sensor.",
    "Wild Spike": "A sudden, extreme value far outside the normal range — possible sensor glitch or transient event.",
    "Sensor Drift": "Readings are gradually shifting away from their baseline — possible sensor calibration drift.",
    "Missing Data": "Expected reading is missing — possible data transmission or logging failure.",
    "Unknown": "An unclassified irregularity was detected in the sensor stream.",
}


@st.cache_data
def compute_fault_classification(df, columns, timestamp_col="Date/Time"):
    faults_df = classify_all_faults(df, columns, timestamp_col)
    summary = generate_alerts_summary(faults_df)
    return faults_df, summary


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_weather_data():

    df = pd.read_json(RAW_DATA)

    df["Date/Time"] = pd.to_datetime(
        df["Date/Time"],
        format="mixed"
    )

    df = df.sort_values("Date/Time").reset_index(drop=True)

    return df


@st.cache_data
def load_anomaly_data():

    df = pd.read_csv(ANOMALY_DATA)

    df["Date/Time"] = pd.to_datetime(
        df["Date/Time"],
        format="mixed"
    )

    return df


try:

    weather_df = load_weather_data()
    anomaly_df = load_anomaly_data()

except FileNotFoundError as e:

    st.error(
        f"Required data file not found:\n\n{e}"
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🌦️ SkyGuard AI")

    st.caption(
        "Intelligent AWS Anomaly Detection"
    )

    st.divider()

    st.markdown("### 🎛️ Dashboard Controls")

    sensor_options = {
        "Temperature": "Temp_C",
        "Humidity": "Rel Hum_%",
        "Pressure": "Press_kPa"
    }

    selected_sensor_name = st.selectbox(
        "Sensor",
        list(sensor_options.keys())
    )

    selected_sensor = sensor_options[
        selected_sensor_name
    ]

    st.divider()

    st.markdown("### 🤖 ML Model")

    st.success("Isolation Forest")

    st.caption(
        "200 estimators • 1% contamination"
    )

    st.divider()

    st.markdown("### 📡 System")

    st.write("🟢 AWS Data Stream")
    st.write("🟢 ML Detection")
    st.write("🟢 Anomaly Scoring")
    st.write("🟢 Monitoring Dashboard")


# =========================================================
# RUN FAULT CLASSIFICATION
# =========================================================

fault_columns = list(sensor_options.values())

faults_df, fault_summary = compute_fault_classification(
    anomaly_df, fault_columns, "Date/Time"
)


# =========================================================
# NOTIFICATION MECHANISM
# =========================================================
# Fires a toast for the single most urgent, most recent fault
# (tracked via session_state so we only toast once per new fault).

if len(faults_df) > 0:

    latest_fault = faults_df.iloc[0]
    latest_fault_id = f"{latest_fault['timestamp']}_{latest_fault['sensor']}_{latest_fault['fault_type']}"

    if st.session_state.get("last_notified_fault") != latest_fault_id:

        st.session_state["last_notified_fault"] = latest_fault_id

        st.toast(
            f"{SEVERITY_ICON.get(latest_fault['severity'], '⚪')} "
            f"{latest_fault['fault_type']} detected on {latest_fault['sensor']} "
            f"({latest_fault['severity']} severity)",
            icon="🚨"
        )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🌦️ SkyGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Anomaly Detection for Automatic Weather Stations'
    '</div>',
    unsafe_allow_html=True
)

st.caption(
    "Ministry of Earth Sciences • SIH26073 • Disaster Management"
)

st.divider()


# =========================================================
# CALCULATE METRICS
# =========================================================

total_readings = len(anomaly_df)

total_anomalies = int(
    anomaly_df["ML_Anomaly"].sum()
)

normal_readings = (
    total_readings - total_anomalies
)

anomaly_rate = (
    total_anomalies / total_readings * 100
    if total_readings > 0
    else 0
)

# Anomaly score
mean_score = anomaly_df[
    "Anomaly_Score"
].mean()


# =========================================================
# KPI CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">📡 READINGS PROCESSED</div>
            <div class="metric-value">{total_readings:,}</div>
            <div class="metric-sub">AWS observations</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">🚨 ANOMALIES DETECTED</div>
            <div class="metric-value">{total_anomalies:,}</div>
            <div class="metric-sub">{anomaly_rate:.2f}% of readings</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">🟢 NORMAL READINGS</div>
            <div class="metric-value">{normal_readings:,}</div>
            <div class="metric-sub">Within expected pattern</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">🧠 AVG ANOMALY SCORE</div>
            <div class="metric-value">{mean_score:.3f}</div>
            <div class="metric-sub">Isolation Forest</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# SYSTEM STATUS
# =========================================================

if fault_summary["total_faults"] == 0:

    st.markdown(
        '<div class="status-good">'
        '🟢 SYSTEM STATUS — All monitored readings appear normal'
        '</div>',
        unsafe_allow_html=True
    )

else:

    st.markdown(
        f'<div class="status-warning">'
        f'🟠 SYSTEM STATUS — {fault_summary["total_faults"]} classified fault(s) require attention '
        f'({fault_summary["critical_count"]} critical, {fault_summary["high_count"]} high, '
        f'{fault_summary["medium_count"]} medium)'
        f'</div>',
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# LIVE FAULT NOTIFICATIONS PANEL
# =========================================================

st.markdown(
    '<div class="section-title">🔔 Live Fault Notifications</div>',
    unsafe_allow_html=True
)

if len(faults_df) == 0:

    st.markdown(
        '<div class="status-good">'
        '🟢 No faults classified — all sensors reporting normally.'
        '</div>',
        unsafe_allow_html=True
    )

else:

    notify_count = st.slider(
        "Number of recent notifications to show",
        min_value=3,
        max_value=min(20, len(faults_df)),
        value=min(8, len(faults_df))
    )

    for _, fault in faults_df.head(notify_count).iterrows():

        icon = SEVERITY_ICON.get(fault["severity"], "⚪")
        severity_class = f"severity-{fault['severity'].lower()}"
        explanation = FAULT_TYPE_EXPLANATION.get(
            fault["fault_type"], "An irregularity was detected."
        )

        st.markdown(
            f"""
            <div class="alert-card {severity_class}">
                <div class="alert-title">{icon} {fault['fault_type']} — {fault['sensor']} ({fault['severity']})</div>
                <div class="alert-text">{fault['timestamp']} • value: {fault['value']}</div>
                <div class="alert-text">{explanation}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# SENSOR MONITORING
# =========================================================

st.markdown(
    '<div class="section-title">📈 Live Sensor Monitoring</div>',
    unsafe_allow_html=True
)

# Last 1000 readings for cleaner visualization
plot_df = anomaly_df.tail(1000).copy()

plot_df["Status"] = np.where(
    plot_df["ML_Anomaly"] == 1,
    "Anomaly",
    "Normal"
)

fig = go.Figure()


# Normal readings
normal = plot_df[
    plot_df["ML_Anomaly"] == 0
]

fig.add_trace(
    go.Scatter(
        x=normal["Date/Time"],
        y=normal[selected_sensor],
        mode="lines",
        name="Normal",
        line=dict(width=1.5)
    )
)


# Anomalies
anomalies = plot_df[
    plot_df["ML_Anomaly"] == 1
]

fig.add_trace(
    go.Scatter(
        x=anomalies["Date/Time"],
        y=anomalies[selected_sensor],
        mode="markers",
        name="Anomaly",
        marker=dict(
            size=9,
            symbol="circle"
        )
    )
)


fig.update_layout(
    title=f"{selected_sensor_name} Sensor Stream",
    xaxis_title="Time",
    yaxis_title=selected_sensor_name,
    template="plotly_dark",
    height=430,
    hovermode="x unified",
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(
    fig,
    width="stretch"
)


# =========================================================
# SECOND ROW
# =========================================================

left, right = st.columns(2)


# ---------------------------------------------------------
# ANOMALY DISTRIBUTION
# ---------------------------------------------------------

with left:

    st.markdown(
        '<div class="section-title">🚨 Anomaly Overview</div>',
        unsafe_allow_html=True
    )

    status_counts = pd.DataFrame({
        "Status": ["Normal", "Anomaly"],
        "Count": [
            normal_readings,
            total_anomalies
        ]
    })

    fig_status = px.pie(
        status_counts,
        names="Status",
        values="Count",
        hole=0.65,
        template="plotly_dark"
    )

    fig_status.update_layout(
        height=350,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),
        showlegend=True
    )

    st.plotly_chart(
        fig_status,
        width="stretch"
    )


# ---------------------------------------------------------
# FAULT TYPE BREAKDOWN (replaces plain score histogram)
# ---------------------------------------------------------

with right:

    st.markdown(
        '<div class="section-title">🧠 Classified Fault Types</div>',
        unsafe_allow_html=True
    )

    if len(faults_df) == 0:

        st.info("No classified faults to display yet.")

    else:

        fault_type_counts = (
            faults_df["fault_type"]
            .value_counts()
            .reset_index()
        )
        fault_type_counts.columns = ["Fault Type", "Count"]

        fig_fault_types = px.bar(
            fault_type_counts,
            x="Fault Type",
            y="Count",
            template="plotly_dark",
            color="Fault Type"
        )

        fig_fault_types.update_layout(
            height=350,
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            ),
            showlegend=False
        )

        st.plotly_chart(
            fig_fault_types,
            width="stretch"
        )


# =========================================================
# SENSOR SUMMARY
# =========================================================

st.markdown(
    '<div class="section-title">📡 Sensor Health Summary</div>',
    unsafe_allow_html=True
)

sensor_data = []

for sensor_name, column in sensor_options.items():

    sensor_anomalies = int(
        anomaly_df[
            anomaly_df["ML_Anomaly"] == 1
        ][column].count()
    )

    sensor_fault_count = (
        int((faults_df["sensor"] == column).sum())
        if len(faults_df) > 0
        else 0
    )

    sensor_data.append({
        "Sensor": sensor_name,
        "Current Value": round(
            anomaly_df[column].iloc[-1],
            2
        ),
        "Average": round(
            anomaly_df[column].mean(),
            2
        ),
        "Minimum": round(
            anomaly_df[column].min(),
            2
        ),
        "Maximum": round(
            anomaly_df[column].max(),
            2
        ),
        "Classified Faults": sensor_fault_count,
        "Status": (
            "⚠️ Attention"
            if sensor_anomalies > 0 or sensor_fault_count > 0
            else "🟢 Healthy"
        )
    })


sensor_summary = pd.DataFrame(
    sensor_data
)

st.dataframe(
    sensor_summary,
    hide_index=True,
    width="stretch"
)


# =========================================================
# ANOMALY ALERT QUEUE (now includes fault classification)
# =========================================================

st.markdown(
    '<div class="section-title">🛠️ Anomaly Alert Queue</div>',
    unsafe_allow_html=True
)

alerts = anomaly_df[
    anomaly_df["ML_Anomaly"] == 1
].copy()

if alerts.empty:

    st.markdown(
        '<div class="status-good">'
        '🟢 No anomalous readings detected.'
        '</div>',
        unsafe_allow_html=True
    )

else:

    st.warning(
        f"⚠️ {len(alerts)} anomalous observations detected by the Isolation Forest model."
    )

    # Attach fault type/severity where available, matched on timestamp
    if len(faults_df) > 0:

        fault_lookup = (
            faults_df
            .drop_duplicates(subset=["timestamp", "sensor"])
            .set_index(["timestamp", "sensor"])
        )

        def lookup_fault_type(row):
            hits = []
            for sensor_name, column in sensor_options.items():
                key = (row["Date/Time"], column)
                if key in fault_lookup.index:
                    hits.append(fault_lookup.loc[key, "fault_type"])
            return ", ".join(hits) if hits else "Unclassified"

        def lookup_severity(row):
            hits = []
            for sensor_name, column in sensor_options.items():
                key = (row["Date/Time"], column)
                if key in fault_lookup.index:
                    hits.append(fault_lookup.loc[key, "severity"])
            return ", ".join(hits) if hits else "—"

        alerts["Fault Type"] = alerts.apply(lookup_fault_type, axis=1)
        alerts["Severity"] = alerts.apply(lookup_severity, axis=1)

    else:
        alerts["Fault Type"] = "Unclassified"
        alerts["Severity"] = "—"

    alert_columns = [
        "Date/Time",
        "Temp_C",
        "Rel Hum_%",
        "Press_kPa",
        "Anomaly_Score",
        "Fault Type",
        "Severity",
    ]

    alert_table = alerts[
        alert_columns
    ].sort_values(
        "Anomaly_Score"
    )

    st.dataframe(
        alert_table.head(100),
        hide_index=True,
        width="stretch"
    )


# =========================================================
# RECENT DATA
# =========================================================

with st.expander("📋 View Recent AWS Data"):

    st.dataframe(
        anomaly_df.tail(25),
        hide_index=True,
        width="stretch"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "SKYGUARD AI • SIH26073 • "
    "AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations"
)