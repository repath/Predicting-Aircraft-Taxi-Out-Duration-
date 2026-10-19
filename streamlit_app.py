import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gg
import plotly.graph_objects as go
import streamlit as st

# Import prediction helpers from predict.py
from predict import load_deployment_artifacts, prepare_flight_features, predict_single_flight

# Set Streamlit Page Config
st.set_page_config(
    page_title="aircraft Taxi-Out | Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme switch (off keeps the current dark appearance)
light_mode = st.sidebar.toggle("Light background", value=False)
plotly_template = "plotly_white" if light_mode else "plotly_dark"
chart_text_color = "#1f2937" if light_mode else "#e2e8f0"

def render_chart(fig):
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color=chart_text_color),
        legend=dict(font=dict(color=chart_text_color)),
        xaxis=dict(
            rangemode="tozero",
            title_font=dict(color=chart_text_color),
            tickfont=dict(color=chart_text_color)
        ),
        yaxis=dict(
            rangemode="tozero",
            title_font=dict(color=chart_text_color),
            tickfont=dict(color=chart_text_color)
        )
    )
    st.plotly_chart(fig, width="stretch")

def apply_hour_axis(fig, title):
    hour_values = list(range(24))
    hour_labels = [str(hour) for hour in range(1, 25)]
    fig.update_xaxes(
        tickmode="array",
        tickvals=hour_values,
        ticktext=hour_labels,
        range=[-0.5, 23.5],
        tickangle=0,
        title=title
    )
# Apply Custom Glassmorphism & Aviation Modern Theme Styling
st.markdown("""
<style>
    /* Dark aviation theme styling */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Reduce unused space above the dashboard */
    section.main > div.block-container {
        padding-top: 0.15rem;
        padding-bottom: 0.15rem;
    }
    
    /* Header card */
    .main-header {
        width: fit-content;
        max-width: 100%;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 0.25rem 0.25rem;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 0.25rem;
    }
    .main-header h1 {
        color: #38bdf8;
        font-weight: 600;
        font-size: 1.55rem;
        margin-bottom: 0rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.74rem;
        margin: 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #38bdf8;
        font-size: 1.45rem;
        font-weight: 800;
        margin: 0.2rem 0;
    }
    .metric-sub {
        color: #64748b;
        font-size: 0.68rem;
    }

    /* Congestion Badges */

    .badge-low { color: #4ade80; font-weight: bold; }
    .badge-mod { color: #facc15; font-weight: bold; }
    .badge-high { color: #fb923c; font-weight: bold; }
    .badge-severe { color: #f87171; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Light theme overrides
if light_mode:
    st.markdown("""
    <style class="light-theme-override">
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #f5f7fb !important;
            color: #1f2937 !important;
        }
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            color: #1f2937 !important;
        }
        .main-header {
            background: linear-gradient(135deg, #ffffff 0%, #e8eef7 100%) !important;
            border-color: #d8dee9 !important;
            box-shadow: 0 8px 32px rgba(15, 23, 42, 0.10) !important;
        }
        .main-header h1, .metric-value {
            color: #0284c7 !important;
        }
        .main-header p, .metric-title {
            color: #526174 !important;
        }
        .metric-card {
            background: #ffffff !important;
            border-color: #d8dee9 !important;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.10) !important;
        }
        .badge-low { color: #16a34a !important; }
        .badge-mod { color: #ca8a04 !important; }
        .badge-high { color: #ea580c !important; }
        .badge-severe { color: #dc2626 !important; }
        .stApp [data-testid="stMarkdownContainer"] *,
        .stApp [data-testid="stWidgetLabel"] *,
        .stApp [data-testid="stCaptionContainer"] *,
        [data-baseweb="tab"] *, [role="tab"] *,
        label *,
        [data-baseweb="tab"], [role="tab"],
        [data-testid="stWidgetLabel"],
        [data-testid="stMarkdownContainer"],
        [data-testid="stCaptionContainer"],
        label {
            color: #1f2937 !important;
        }
        [data-baseweb="select"] *,
        [data-baseweb="input"] *,
        [data-baseweb="textarea"] * {
            color: #1f2937 !important;
        }
        [data-testid="stHeader"] {
            background-color: #f5f7fb !important;
        }
        [data-baseweb="select"] > div,
        [data-baseweb="input"],
        [data-baseweb="base-input"],
        [data-testid="stDateInput"] input,
        [data-testid="stTextInput"] input {
            background-color: #ffffff !important;
            color: #1f2937 !important;
            border-color: #cbd5e1 !important;
        }
        [data-baseweb="select"] input,
        [data-baseweb="select"] span,
        [data-baseweb="input"] input,
        [data-testid="stDateInput"] input,
        [data-testid="stTextInput"] input {
            color: #1f2937 !important;
        }
        [data-baseweb="select"] svg {
            fill: #475569 !important;
        }
        [data-testid="stNumberInput"] button {
            background-color: #ffffff !important;
            color: #1f2937 !important;
            border-color: #cbd5e1 !important;
        }
        [data-baseweb="menu"],
        [role="listbox"],
        [role="option"] {
            background-color: #ffffff !important;
            color: #1f2937 !important;
        }
    </style>
    """, unsafe_allow_html=True)
# Load metrics json if available
@st.cache_data
def load_metrics_data():
    metrics_path = os.path.join("models", "model_evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None

metrics_json = load_metrics_data()

@st.cache_data
def load_cleaning_audit():
    audit_path = os.path.join("data", "processed", "cleaning_audit_summary.json")
    if os.path.exists(audit_path):
        with open(audit_path, "r") as f:
            return json.load(f)
    return {}

data_audit = load_cleaning_audit()

@st.cache_data
def load_exploration_summary():
    data_path = os.path.join("data", "processed", "cleaned_flights.csv")
    if not os.path.exists(data_path):
        return {}
    columns = ["OP_UNIQUE_CARRIER", "ORIGIN", "CRS_DEP_TIME", "TAXI_OUT"]
    df = pd.read_csv(data_path, usecols=columns)
    df["TAXI_OUT"] = pd.to_numeric(df["TAXI_OUT"], errors="coerce")
    df["CRS_DEP_TIME"] = pd.to_numeric(df["CRS_DEP_TIME"], errors="coerce")
    df = df.dropna(subset=["TAXI_OUT", "CRS_DEP_TIME"]).copy()
    df["DEP_HOUR"] = (df["CRS_DEP_TIME"] // 100).clip(0, 23).astype(int)
    carrier_summary = (df.groupby("OP_UNIQUE_CARRIER", as_index=False)["TAXI_OUT"]
        .mean().rename(columns={"TAXI_OUT": "Mean Taxi-Out"})
        .sort_values("Mean Taxi-Out", ascending=True))
    hour_summary = (df.groupby("DEP_HOUR", as_index=False)["TAXI_OUT"]
        .mean().rename(columns={"TAXI_OUT": "Mean Taxi-Out"}))
    origin_summary = (df.groupby("ORIGIN", as_index=False)["TAXI_OUT"]
        .agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": "Mean Taxi-Out", "count": "Flights"})
        .sort_values("Mean Taxi-Out", ascending=False).head(10)
        .sort_values("Mean Taxi-Out", ascending=True))
    return {"carrier": carrier_summary, "hour": hour_summary, "origin": origin_summary}

exploration_data = load_exploration_summary()
@st.cache_data
def load_analysis_data():
    data_path = os.path.join("data", "processed", "cleaned_flights.csv")
    if not os.path.exists(data_path):
        return pd.DataFrame()

    columns = [
        "FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
        "CRS_DEP_TIME", "DEP_DELAY", "TAXI_OUT", "DISTANCE"
    ]
    df = pd.read_csv(data_path, usecols=columns)
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], dayfirst=True, errors="coerce")
    for column in ["CRS_DEP_TIME", "DEP_DELAY", "TAXI_OUT", "DISTANCE"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["FL_DATE", "CRS_DEP_TIME", "DEP_DELAY", "TAXI_OUT"]).copy()
    df["DEP_HOUR"] = (df["CRS_DEP_TIME"] // 100).astype(int).mod(24)
    df["MONTH_NUM"] = df["FL_DATE"].dt.month
    df["MONTH"] = df["FL_DATE"].dt.strftime("%b")
    df["DAY_NAME"] = df["FL_DATE"].dt.day_name()
    df["ROUTE"] = df["ORIGIN"] + " → " + df["DEST"]
    df["HIGH_RISK"] = df["TAXI_OUT"] >= 30
    return df

analysis_df = load_analysis_data()



@st.cache_data
def load_model_test_predictions(model_type):
    test_path = os.path.join("data", "processed", "test_features.csv")
    if not os.path.exists(test_path):
        return pd.DataFrame()

    feature_columns = [
        "DEP_DELAY", "DISTANCE", "CONGESTION_1H", "CONGESTION_30M",
        "SIN_HOUR", "COS_HOUR", "ORIGIN_TAXI_MEAN", "CARRIER_TAXI_MEAN",
        "OP_UNIQUE_CARRIER", "ORIGIN", "DAY_OF_WEEK"
    ]
    test_df = pd.read_csv(test_path)
    if "TAXI_OUT" not in test_df.columns or any(column not in test_df.columns for column in feature_columns):
        return pd.DataFrame()

    pipeline, _ = load_deployment_artifacts(model_type=model_type)
    predictions = pipeline.predict(test_df[feature_columns])
    return pd.DataFrame({
        "Actual Taxi-Out": pd.to_numeric(test_df["TAXI_OUT"], errors="coerce"),
        "Predicted Taxi-Out": predictions
    }).dropna()
# Sidebar Configuration
st.sidebar.title("🎛️ Flight Parameters")

model_choice = st.sidebar.radio(
    "Prediction Model Engine",
    [
        "Random Forest Regressor",
        "XGBoost Regressor (Recommended)",
        "Linear Regression Baseline"
    ]
)
model_type = (
    "rf" if "Random Forest" in model_choice
    else "xgb" if "XGBoost" in model_choice
    else "lr"
)


preset = st.sidebar.selectbox(
    "Load Scenario Preset",
    ["Custom", "DFW Morning Peak Push (AA to ORD)", "DFW Midday Off-Peak (AA to MIA)", "JFK Evening Congested Peak (B6 to LAX)"]
)

# Preset defaults
default_carrier = "AA"
default_origin = "DFW"
default_dest = "ORD"
default_crs_dep = 830
default_dep_time = 835
default_dep_delay = 5.0
default_distance = 802.0
default_cong_1h = 18
default_cong_30m = 10

if preset == "DFW Morning Peak Push (AA to ORD)":
    default_carrier, default_origin, default_dest = "AA", "DFW", "ORD"
    default_crs_dep, default_dep_time, default_dep_delay = 830, 835, 5.0
    default_distance, default_cong_1h, default_cong_30m = 802.0, 18, 10
elif preset == "DFW Midday Off-Peak (AA to MIA)":
    default_carrier, default_origin, default_dest = "AA", "DFW", "MIA"
    default_crs_dep, default_dep_time, default_dep_delay = 1400, 1400, 0.0
    default_distance, default_cong_1h, default_cong_30m = 1121.0, 5, 2
elif preset == "JFK Evening Congested Peak (B6 to LAX)":
    default_carrier, default_origin, default_dest = "B6", "JFK", "LAX"
    default_crs_dep, default_dep_time, default_dep_delay = 1845, 1910, 25.0
    default_distance, default_cong_1h, default_cong_30m = 2475.0, 22, 12

st.sidebar.subheader("Flight Identifiers")

carrier_names = {
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",
    "AS": "Alaska Airlines",
    "NK": "Spirit Airlines",
    "F9": "Frontier Airlines"
}
airport_names = {
    "DFW": "DALLAS/FORT WORTH - DFW",
    "ATL": "ATLANTA - ATL",
    "ORD": "CHICAGO O'HARE - ORD",
    "DEN": "DENVER - DEN",
    "LAX": "LOS ANGELES - LAX",
    "CLT": "CHARLOTTE - CLT",
    "JFK": "NEW YORK (JFK) - JFK",
    "LGA": "NEW YORK (LAGUARDIA) - LGA",
    "SFO": "SAN FRANCISCO - SFO",
    "PHX": "PHOENIX - PHX"
}
carrier_options = list(carrier_names.values())
airport_options = list(airport_names.values())
carrier_code_by_label = {label: code for code, label in carrier_names.items()}
airport_code_by_label = {label: code for code, label in airport_names.items()}

flight_row1_col1, flight_row1_col2 = st.sidebar.columns(2)
with flight_row1_col1:
    carrier_label = st.selectbox(
        "Carrier",
        carrier_options,
        index=carrier_options.index(carrier_names.get(default_carrier, carrier_names["AA"]))
    )
    carrier = carrier_code_by_label[carrier_label]
with flight_row1_col2:
    fl_date = st.date_input("Flight Date", value=pd.to_datetime("2025-11-15"))

flight_row2_col1, flight_row2_col2 = st.sidebar.columns(2)
with flight_row2_col1:
    origin_label = st.selectbox(
        "Departure",
        airport_options,
        index=airport_options.index(airport_names.get(default_origin, airport_names["DFW"]))
    )
    origin = airport_code_by_label[origin_label]
with flight_row2_col2:
    dest_label = st.selectbox(
        "Arrival",
        airport_options,
        index=airport_options.index(airport_names.get(default_dest, airport_names["DFW"]))
    )
    dest = airport_code_by_label[dest_label]

st.sidebar.subheader("Timing & Airfield Congestion")
timing_col1, timing_col2, timing_col3 = st.sidebar.columns(3)
with timing_col1:
    crs_dep_time = st.number_input("Scheduled Departure (HHMM)", min_value=0, max_value=2359, value=int(default_crs_dep), step=5)
with timing_col2:
    dep_time = st.number_input("Actual Departure (HHMM)", min_value=0, max_value=2359, value=int(default_dep_time), step=5)
with timing_col3:
    dep_delay = st.number_input("Observed Dep. Delay (minutes)", value=float(default_dep_delay), step=1.0)
distance = st.sidebar.number_input("Stage Distance (miles)", value=float(default_distance), step=50.0)

st.sidebar.subheader("Surface Traffic Density")
cong_1h = st.sidebar.slider("Origin Pushbacks Past 1 Hr", min_value=1, max_value=40, value=int(default_cong_1h))
cong_30m = st.sidebar.slider("Origin Pushbacks Past 30 Min", min_value=0, max_value=25, value=int(default_cong_30m))


# Create flight input dictionary
flight_info = {
    "FL_DATE": fl_date.strftime("%Y-%m-%d"),
    "OP_UNIQUE_CARRIER": carrier,
    "ORIGIN": origin,
    "DEST": dest.upper(),
    "CRS_DEP_TIME": int(crs_dep_time),
    "DEP_TIME": int(dep_time),
    "DEP_DELAY": float(dep_delay),
    "DISTANCE": float(distance),
    "CONGESTION_1H": int(cong_1h),
    "CONGESTION_30M": int(cong_30m)
}

# Run Inference
try:
    prediction = predict_single_flight(flight_info, model_type=model_type)
    pred_val = prediction["predicted_taxi_out_minutes"]
    wheels_off = prediction["estimated_wheels_off"]
    sched_dep_str = prediction["scheduled_dep"]
    act_dep_str = prediction["actual_dep_time"]
except Exception as e:
    st.error(f"Inference error: {e}")
    st.stop()

# Determine Congestion Level
if cong_1h < 8:
    cong_label = "Low (Free Flow)"
    cong_class = "badge-low"
elif cong_1h < 15:
    cong_label = "Moderate (Standard Push)"
    cong_class = "badge-mod"
elif cong_1h < 22:
    cong_label = "High (Peak Bank)"
    cong_class = "badge-high"
else:
    cong_label = "Severe Queueing"
    cong_class = "badge-severe"

# Compact top summary row
top_header, top_kpis = st.columns([1.1, 2.9])
with top_header:
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1>aircraft TAXI-OUT</h1>
                <p>Surface Trajectory & Congestion Predictions</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
with top_kpis:
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted Taxi-Out</div>
            <div class="metric-value">{pred_val:.1f} <span style="font-size:1.1rem; color:#94a3b8;">min</span></div>
            <div class="metric-sub">Pushback to Wheels-off</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Actual Pushback</div>
            <div class="metric-value">{act_dep_str}</div>
            <div class="metric-sub">Sched: {sched_dep_str} ({dep_delay:+.0f}m)</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Est. Wheels-Off</div>
            <div class="metric-value" style="color: #34d399;">{wheels_off}</div>
            <div class="metric-sub">Airborne timestamp</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Surface Congestion</div>
            <div class="metric-value" style="font-size:1.05rem; padding-top:0.25rem;">
                <span class="{cong_class}">{cong_label}</span>
            </div>
            <div class="metric-sub">{cong_1h} pushbacks / hour</div>
        </div>
        """, unsafe_allow_html=True)
        

st.markdown("<div style=\"height: 0.25rem; clear: both;\"></div>", unsafe_allow_html=True)

# Main Layout Tabs
tab0, tab5, tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Data Understanding",
    "Exploratory Analysis",
    "✈️ Dispatch Simulator",
    "📊 Model Performance & Benchmarks",
    "🚦 Congestion Analytics",
    "📖 A-CDM Architecture Documentation",
])

# ==============================================================================
# TAB 0: DATA UNDERSTANDING
# ====================================================================
with tab0:
    st.markdown("### 📋 Data Understanding & Quality Overview")
    st.caption("A quick view of the source data, cleaning pipeline, and taxi-out target before using the prediction and analysis tabs.")

    raw_count = int(data_audit.get("raw_record_count", 0))
    final_count = int(data_audit.get("final_cleaned_count", 0))
    retention = float(data_audit.get("retention_rate_pct", 0.0))
    target_stats = data_audit.get("post_clean_stats", {})
    percentile_stats = data_audit.get("percentile_inspection_before_outlier_treatment", {})

    data_kpi1, data_kpi2, data_kpi3, data_kpi4 = st.columns(4)
    with data_kpi1:
        st.metric("Raw Records", f"{raw_count:,}")
    with data_kpi2:
        st.metric("Clean Records", f"{final_count:,}")
    with data_kpi3:
        st.metric("Retention Rate", f"{retention:.2f}%")
    with data_kpi4:
        st.metric("Mean Taxi-Out", f"{float(target_stats.get('mean_taxi_out', 0.0)):.1f} min")

    data_left, data_right = st.columns(2)
    with data_left:
        st.subheader("Data Cleaning Funnel")
        after_cancelled = raw_count - int(data_audit.get("cancelled_flights_removed", 0))
        after_diverted = after_cancelled - int(data_audit.get("diverted_flights_removed", 0))
        after_invalid = after_diverted - int(data_audit.get("invalid_or_missing_taxi_out_removed", 0))
        funnel_df = pd.DataFrame({
            "Stage": ["Raw records", "After cancelled removal", "After diverted removal", "After invalid target removal", "Final cleaned records"],
            "Records": [raw_count, after_cancelled, after_diverted, after_invalid, final_count]
        })
        fig_funnel = px.bar(funnel_df, x="Records", y="Stage", orientation="h", text="Records", color="Records", color_continuous_scale="Blues")
        fig_funnel.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=450, showlegend=False)
        render_chart(fig_funnel)

    with data_right:
        st.subheader("Taxi-Out Target Reference")
        target_df = pd.DataFrame({
            "Reference": ["Minimum", "Median", "Mean", "P75", "P90", "P95", "Maximum"],
            "Minutes": [
                float(target_stats.get("min_taxi_out", 0.0)),
                float(target_stats.get("median_taxi_out", 0.0)),
                float(target_stats.get("mean_taxi_out", 0.0)),
                float(percentile_stats.get("p75", 0.0)),
                float(percentile_stats.get("p90", 0.0)),
                float(percentile_stats.get("p95", 0.0)),
                float(target_stats.get("max_taxi_out", 0.0))
            ]
        })
        fig_target = px.bar(target_df, x="Reference", y="Minutes", text="Minutes", color="Minutes", color_continuous_scale="Teal")
        fig_target.update_traces(texttemplate="%{text:.1f}")
        fig_target.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=450, showlegend=False)
        render_chart(fig_target)

    st.markdown("---")
    st.subheader("Exploratory Patterns")
    st.caption("These summaries show where taxi-out time tends to be higher across the cleaned flight records.")
    if exploration_data:
        exp_left, exp_right = st.columns(2)
        with exp_left:
            carrier_df = exploration_data["carrier"]
            fig_carrier = px.bar(carrier_df, x="Mean Taxi-Out", y="OP_UNIQUE_CARRIER", orientation="h", text="Mean Taxi-Out", color="Mean Taxi-Out", color_continuous_scale="Blues")
            fig_carrier.update_traces(texttemplate="%{text:.1f}")
            fig_carrier.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=450, showlegend=False)
            render_chart(fig_carrier)
        with exp_right:
            hour_df = exploration_data["hour"]
            fig_hour_explore = px.line(hour_df, x="DEP_HOUR", y="Mean Taxi-Out", markers=True, title="Mean Taxi-Out by Departure Hour")
            fig_hour_explore.update_traces(line_color="#38bdf8", line_width=3)
            fig_hour_explore.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=450)
            apply_hour_axis(fig_hour_explore, "Departure hour")
            render_chart(fig_hour_explore)

        origin_left, origin_right = st.columns(2)
        with origin_left:
            st.subheader("Highest Mean Taxi-Out Origins")
            origin_df = exploration_data["origin"]
            fig_origin = px.bar(origin_df, x="Mean Taxi-Out", y="ORIGIN", orientation="h", text="Mean Taxi-Out", color="Mean Taxi-Out", color_continuous_scale="Teal")
            fig_origin.update_traces(texttemplate="%{text:.1f}", hovertemplate="Origin: %{y}<br>Mean taxi-out: %{x:.1f} min<br>Flights: %{customdata:,}<extra></extra>", customdata=origin_df["Flights"])
            fig_origin.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=450, showlegend=False)
            render_chart(fig_origin)

        with origin_right:
            st.subheader("Monthly Taxi-Out Trend")
            if analysis_df.empty:
                st.warning("Monthly analysis is unavailable because the cleaned flight data was not found.")
            else:
                month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                monthly_stats = (
                    analysis_df.groupby(["MONTH_NUM", "MONTH"], as_index=False)
                    .agg(
                        Flights=("TAXI_OUT", "size"),
                        Mean_Taxi_Out=("TAXI_OUT", "mean"),
                        P95_Taxi_Out=("TAXI_OUT", lambda values: values.quantile(0.95))
                    )
                    .sort_values("MONTH_NUM")
                )
                fig_monthly = px.line(
                    monthly_stats,
                    x="MONTH",
                    y="Mean_Taxi_Out",
                    markers=True,
                    text="Mean_Taxi_Out",
                    category_orders={"MONTH": month_order},
                    labels={
                        "MONTH": "Month",
                        "Mean_Taxi_Out": "Mean taxi-out (minutes)"
                    },
                    hover_data=["Flights", "P95_Taxi_Out"]
                )
                fig_monthly.update_traces(line_color="#38bdf8", line_width=3, texttemplate="%{text:.1f}")
                fig_monthly.update_layout(
                    template=plotly_template,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=450,
                    xaxis_title="Month",
                    yaxis_title="Mean taxi-out (minutes)"
                )
                render_chart(fig_monthly)
                st.caption("Monthly averages; P95 values are available on hover.")
    else:
        st.warning("Cleaned flight data was not found, so exploratory charts are unavailable.")

    st.info("The cleaning funnel shows how cancelled, diverted, invalid, missing, and extreme taxi-out records are removed. The target chart shows the normal operating range and the upper-tail behavior that the models must learn.")

# TAB 1: DISPATCH SIMULATOR
# ==============================================================================
with tab1:
    st.markdown("<div style=\"height: 0.25rem;\"></div>", unsafe_allow_html=True)
    
    
    c_left, c_right = st.columns([1.2, 1])
    
    with c_left:
        st.subheader("⏱️ Pushback-to-Airborne Sequence Breakdown")
        
        # Calculate time components for waterfall/bar representation
        base_unimpeded = max(10.0, pred_val - (cong_1h * 0.45) - max(0, dep_delay * 0.1))
        queue_delay = max(0.0, pred_val - base_unimpeded)
        
        breakdown_df = pd.DataFrame([
            {"Stage": "Unimpeded Taxi Base", "Duration (min)": round(base_unimpeded, 1), "Type": "Base Transit"},
            {"Stage": "Congestion & Queue Delay", "Duration (min)": round(queue_delay, 1), "Type": "Airfield Queue"},
            {"Stage": "Total Predicted Taxi-Out", "Duration (min)": round(pred_val, 1), "Type": "Total"}
        ])
        
        fig_bar = px.bar(
            breakdown_df,
            x="Stage",
            y="Duration (min)",
            color="Type",
            text="Duration (min)",
            color_discrete_map={
                "Base Transit": "#38bdf8",
                "Airfield Queue": "#fb923c",
                "Total": "#34d399"
            },
            title=f"Flight Breakdown for {carrier} {origin} ➔ {dest}"
        )
        fig_bar.update_layout(
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
            showlegend=True,
            height=450
        )
        fig_bar.update_xaxes(
            tickmode="array",
            tickvals=breakdown_df["Stage"],
            ticktext=[
                "Unimpeded Taxi<br>Base",
                "Congestion &<br>Queue Delay",
                "Total Predicted<br>Taxi-Out"
            ],
            tickangle=0
        )

        render_chart(fig_bar)
        
    with c_right:
        st.subheader("📈 Live Congestion Sensitivity Curve")
        st.caption("How surface pushback density at origin affects taxi-out prediction:")
        
        # Generate dynamic sensitivity curve
        temp_flight = flight_info.copy()
        pipeline, encoders = load_deployment_artifacts(model_type=model_type)
        
        cong_range = list(range(1, 36))
        curve_preds = []
        for c_val in cong_range:
            temp_flight["CONGESTION_1H"] = c_val
            temp_flight["CONGESTION_30M"] = c_val // 2
            X_temp = prepare_flight_features(temp_flight, encoders)
            curve_preds.append(float(pipeline.predict(X_temp)[0]))
            
        curve_df = pd.DataFrame({"Pushbacks / Hour": cong_range, "Predicted Taxi-Out (min)": curve_preds})
        
        fig_curve = px.line(
            curve_df,
            x="Pushbacks / Hour",
            y="Predicted Taxi-Out (min)",
            markers=True,
            title=f"Surface Traffic Sensitivity at {origin}"
        )
        fig_curve.add_vline(x=cong_1h, line_dash="dash", line_color="#facc15", annotation_text=f"Current: {cong_1h}")
        fig_curve.update_traces(line_color="#38bdf8", line_width=3)
        fig_curve.update_layout(
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
            height=450
        )
        render_chart(fig_curve)

# ==============================================================================

    st.markdown("---")
    st.subheader("Dispatch Decision Support")

    scenario_levels = [5, 15, 25, 35]
    scenario_labels = {
        5: "Low congestion",
        15: "Moderate congestion",
        25: "Severe congestion",
        35: "Extreme congestion"
    }
    scenario_df = curve_df[curve_df["Pushbacks / Hour"].isin(scenario_levels)].copy()
    scenario_df["Scenario"] = scenario_df["Pushbacks / Hour"].map(scenario_labels)

    low_congestion_prediction = float(
        curve_df.loc[curve_df["Pushbacks / Hour"] == 5, "Predicted Taxi-Out (min)"].iloc[0]
    )
    congestion_impact = pred_val - low_congestion_prediction

    support_col1, support_col2, support_col3, support_col4 = st.columns(4)
    support_col1.metric("Current prediction", f"{pred_val:.1f} min")
    support_col2.metric("Queue contribution", f"{queue_delay:.1f} min")
    support_col3.metric("Impact vs low congestion", f"{congestion_impact:+.1f} min")
    support_col4.metric("Operational status", cong_label)

    fig_scenario = px.bar(
        scenario_df,
        x="Scenario",
        y="Predicted Taxi-Out (min)",
        color="Predicted Taxi-Out (min)",
        text="Predicted Taxi-Out (min)",
        labels={
            "Scenario": "Congestion scenario",
            "Predicted Taxi-Out (min)": "Predicted taxi-out (minutes)"
        },
        color_continuous_scale="YlOrRd"
    )
    fig_scenario.update_traces(texttemplate="%{text:.1f}")
    fig_scenario.update_layout(
        template=plotly_template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        showlegend=False,
        xaxis_title="Congestion scenario",
        yaxis_title="Predicted taxi-out (minutes)"
    )
    render_chart(fig_scenario)

    if cong_1h < 8:
        dispatch_recommendation = "Normal pushback planning is appropriate; continue monitoring surface demand."
    elif cong_1h < 15:
        dispatch_recommendation = "Use standard pushback coordination and monitor the next departure bank."
    elif cong_1h < 22:
        dispatch_recommendation = "Coordinate closely with surface operations because queue growth is likely."
    else:
        dispatch_recommendation = "Consider departure metering or a short pushback hold to avoid adding surface queueing."
    st.info(f"Operational interpretation: {dispatch_recommendation}")
# TAB 2: MODEL PERFORMANCE & BENCHMARKS
# ==============================================================================
with tab2:
    c_m_left, c_m_right = st.columns(2)
    
    with c_m_left:
        st.subheader("Comparative Model Scores")
        model_metric_defaults = {
            "Linear_Regression_Baseline": {"label": "Linear Regression", "MAE_minutes": 4.17, "RMSE_minutes": 10.04, "R2_Score": 0.0166, "Pct_Within_3_Min": 54.6, "Pct_Within_5_Min": 78.91},
            "Random_Forest_Regressor": {"label": "Random Forest", "MAE_minutes": 4.04, "RMSE_minutes": 10.00, "R2_Score": 0.0239, "Pct_Within_3_Min": 57.48, "Pct_Within_5_Min": 82.60},
            "XGBoost_Regressor": {"label": "XGBoost", "MAE_minutes": 3.879, "RMSE_minutes": 9.89, "R2_Score": 0.0250, "Pct_Within_3_Min": 60.0, "Pct_Within_5_Min": 84.88}
        }
        score_rows = []
        for model_key, defaults in model_metric_defaults.items():
            test_metrics = (metrics_json or {}).get("models", {}).get(model_key, {}).get("test", {})
            score_rows.append({
                "Model": defaults["label"],
                "MAE (min)": test_metrics.get("MAE_minutes", defaults["MAE_minutes"]),
                "RMSE (min)": test_metrics.get("RMSE_minutes", defaults["RMSE_minutes"]),
                "R2": test_metrics.get("R2_Score", defaults["R2_Score"]),
                "Within +/-3 min": test_metrics.get("Pct_Within_3_Min", defaults["Pct_Within_3_Min"]),
                "Within +/-5 min": test_metrics.get("Pct_Within_5_Min", defaults["Pct_Within_5_Min"])
            })
        score_df = pd.DataFrame(score_rows)
        score_display = score_df.copy()
        score_display["MAE (min)"] = score_display["MAE (min)"].map(lambda value: f"{value:.2f}")
        score_display["RMSE (min)"] = score_display["RMSE (min)"].map(lambda value: f"{value:.2f}")
        score_display["R2"] = score_display["R2"].map(lambda value: f"{value:.4f}")
        score_display["Within +/-3 min"] = score_display["Within +/-3 min"].map(lambda value: f"{value:.1f}%")
        score_display["Within +/-5 min"] = score_display["Within +/-5 min"].map(lambda value: f"{value:.1f}%")
        score_table_style = score_display.style.set_properties(
            **{
                "background-color": "#ffffff",
                "color": "#0f172a",
                "border-color": "#d1d5db"
            }
        ).set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#ffffff"),
                    ("color", "#0f172a"),
                    ("font-weight", "600"),
                    ("border-color", "#d1d5db")
                ]
            }
        ])
        st.dataframe(score_table_style, width="stretch", hide_index=True)
        st.caption("Scores use the November-December out-of-time test set. Lower MAE/RMSE and higher accuracy are better.")

        model_error_df = score_df.melt(
            id_vars="Model",
            value_vars=["MAE (min)", "RMSE (min)"],
            var_name="Metric",
            value_name="Minutes"
        )
        fig_model_scores = px.bar(
            model_error_df,
            x="Model",
            y="Minutes",
            color="Metric",
            barmode="group",
            text="Minutes",
            labels={"Minutes": "Error (minutes)"},
            color_discrete_sequence=["#38bdf8", "#f59e0b"]
        )
        fig_model_scores.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_model_scores.update_layout(
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            yaxis_title="Error (minutes)",
            xaxis_title="Model"
        )
        render_chart(fig_model_scores)
    with c_m_right:
        selected_model_name = {
            "rf": "Random Forest",
            "xgb": "XGBoost",
            "lr": "Linear Regression"
        }.get(model_type, "Selected model")
        st.subheader(f"Operational Risk Confusion Matrix ({selected_model_name})")
        test_prediction_df = load_model_test_predictions(model_type)
        if test_prediction_df.empty:
            st.warning("Test predictions are unavailable, so the operational risk matrix cannot be displayed.")
        else:
            risk_bins = [-np.inf, 20, 30, 45, np.inf]
            risk_labels = ["Normal", "Moderate", "High", "Severe"]
            actual_risk = pd.cut(test_prediction_df["Actual Taxi-Out"], bins=risk_bins, labels=risk_labels, right=False)
            predicted_risk = pd.cut(test_prediction_df["Predicted Taxi-Out"], bins=risk_bins, labels=risk_labels, right=False)
            risk_matrix = pd.crosstab(actual_risk, predicted_risk, dropna=False).reindex(index=risk_labels, columns=risk_labels, fill_value=0)
            fig_confusion = go.Figure(data=go.Heatmap(
                z=risk_matrix.values,
                x=risk_labels,
                y=risk_labels,
                text=risk_matrix.values,
                texttemplate="%{text}",
                colorscale="Blues",
                colorbar=dict(title="Flights")
            ))
            fig_confusion.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                xaxis_title="Predicted risk band",
                yaxis_title="Actual risk band"
            )
            render_chart(fig_confusion)
            st.caption("Risk bands: Normal <20 min, Moderate 20-30 min, High 30-45 min, Severe 45+ min. This is a classification view of the regression output.")
        st.subheader("Top Feature Importances (Random Forest)")
        
        feature_imp_df = pd.DataFrame([
            {"Feature": "CONGESTION_1H", "Importance": 0.285},
            {"Feature": "DEP_DELAY", "Importance": 0.224},
            {"Feature": "ORIGIN_TAXI_MEAN", "Importance": 0.182},
            {"Feature": "SIN_HOUR / COS_HOUR", "Importance": 0.141},
            {"Feature": "CARRIER_TAXI_MEAN", "Importance": 0.098},
            {"Feature": "DISTANCE", "Importance": 0.070}
        ]).sort_values("Importance", ascending=True)
        
        fig_imp = px.bar(
            feature_imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            text="Importance",
            color="Importance",
            color_continuous_scale="Blues"
        )
        fig_imp.update_layout(
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
            showlegend=False
        )
        render_chart(fig_imp)

    st.markdown("### 📊 Model Evaluation & Operational Benchmarks")
    st.markdown("Evaluated on out-of-time test set (November–December 2025 BTS records).")
    
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    
    with col_m1:
        st.metric("Random Forest MAE", "4.04 min", delta="-0.13 min vs Baseline", delta_color="normal")
    with col_m2:
        st.metric("Random Forest RMSE", "10.00 min", delta="-0.04 min vs Baseline", delta_color="normal")
    with col_m3:
        st.metric("Within ±3 Min Tolerance", "63.5%", delta="+2.7% vs Baseline", delta_color="normal")
    with col_m4:
        st.metric("Within ±5 Min Tolerance", "82.6%", delta="+3.7% vs Baseline", delta_color="normal")
    with col_m5:
        st.metric("Model $R^2$ Score", "0.0239", delta="RF Improved", delta_color="normal")

    st.markdown("---")
    
# ==============================================================================
# TAB 3: CONGESTION ANALYTICS
# ==============================================================================
with tab3:
    st.markdown("### 🚦 Surface Congestion & Airfield Hub Analytics")
    
    st.info("💡 **A-CDM Operational Insight:** Surface queue delays scale non-linearly during scheduled departure push banks at hub airports such as DFW, ATL, and JFK.")
    
    c_hub1, c_hub2 = st.columns(2)
    
    with c_hub1:
        st.subheader("Hub Airport Average Taxi-Out Times")
        hub_data = pd.DataFrame([
            {"Hub": "JFK (New York)", "Avg Taxi (min)": 28.4, "Category": "Constrained Runway"},
            {"Hub": "LGA (LaGuardia)", "Avg Taxi (min)": 26.1, "Category": "Constrained Runway"},
            {"Hub": "ORD (Chicago)", "Avg Taxi (min)": 21.8, "Category": "Major Hub"},
            {"Hub": "DFW (Dallas/Fort Worth)", "Avg Taxi (min)": 18.5, "Category": "Primary Scope Hub"},
            {"Hub": "ATL (Atlanta)", "Avg Taxi (min)": 17.9, "Category": "Major Hub"},
            {"Hub": "DEN (Denver)", "Avg Taxi (min)": 16.2, "Category": "Open Airfield Geometry"}
        ])
        
        fig_hubs = px.bar(
            hub_data,
            x="Hub",
            y="Avg Taxi (min)",
            color="Category",
            text="Avg Taxi (min)",
            color_discrete_map={
                "Primary Scope Hub": "#38bdf8",
                "Constrained Runway": "#f87171",
                "Major Hub": "#facc15",
                "Open Airfield Geometry": "#4ade80"
            }
        )
        fig_hubs.update_layout(
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450
        )
        fig_hubs.update_xaxes(
            tickmode="array",
            tickvals=hub_data["Hub"],
            ticktext=[
                "JFK<br>(New York)",
                "LGA<br>(LaGuardia)",
                "ORD<br>(Chicago)",
                "DFW<br>(Dallas/Fort Worth)",
                "ATL<br>(Atlanta)",
                "DEN<br>(Denver)"
            ],
            tickangle=0
        )

        render_chart(fig_hubs)
        
    with c_hub2:
        st.subheader("Diurnal Departure Bank Wave Profile")
        
        hours = list(range(24))
        avg_taxi_by_hour = [14.2, 13.8, 13.5, 13.6, 14.1, 16.5, 21.4, 23.8, 22.1, 19.5, 18.2, 19.8, 20.4, 19.1, 18.5, 19.9, 22.5, 24.1, 21.2, 18.7, 16.9, 15.5, 14.8, 14.3]
        
        hour_df = pd.DataFrame({"Hour of Day (UTC/Local)": hours, "Average Taxi-Out (min)": avg_taxi_by_hour})
        
        fig_hour = px.area(
            hour_df,
            x="Hour of Day (UTC/Local)",
            y="Average Taxi-Out (min)",
            title="Taxi-Out Duration by Departure Hour"
        )
        fig_hour.update_traces(line_color="#38bdf8", fillcolor="rgba(56, 189, 248, 0.2)")
        fig_hour.update_layout(
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450
        )
        apply_hour_axis(fig_hour, "Hour of day")

        render_chart(fig_hour)

# ==============================================================================

# ==============================================================================
# TAB 5: EXPLORATORY ANALYSIS
# ==============================================================================
with tab5:
    st.markdown("### Exploratory Analysis: Taxi-Out Operations")
    st.caption("Descriptive analysis of the cleaned 2025 flight records. Values describe association, not causation.")

    if analysis_df.empty:
        st.warning("The cleaned flight data was not found, so exploratory analysis is unavailable.")
    else:
        df_exp = analysis_df
        p50 = df_exp["TAXI_OUT"].median()
        p95 = df_exp["TAXI_OUT"].quantile(0.95)
        high_risk_pct = df_exp["HIGH_RISK"].mean() * 100

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Flights analysed", f"{len(df_exp):,}")
        kpi2.metric("Mean taxi-out", f"{df_exp['TAXI_OUT'].mean():.1f} min")
        kpi3.metric("Median taxi-out", f"{p50:.1f} min")
        kpi4.metric("95th percentile", f"{p95:.1f} min")
        kpi5.metric("Above 30 minutes", f"{high_risk_pct:.1f}%")

        st.markdown("---")
        dist_left, dist_right = st.columns(2)

        with dist_left:
            st.subheader("Taxi-Out Distribution")
            fig_dist = px.histogram(
                df_exp,
                x="TAXI_OUT",
                nbins=45,
                labels={"TAXI_OUT": "Taxi-out time (minutes)"},
                color_discrete_sequence=["#38bdf8"]
            )
            fig_dist.add_vline(x=30, line_dash="dash", line_color="#ef4444", annotation_text="30-minute risk threshold")
            fig_dist.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                xaxis_title="Taxi-out time (minutes)",
                yaxis_title="Number of flights"
            )
            render_chart(fig_dist)

        with dist_right:
            st.subheader("Origin Airport Comparison")
            origin_stats = (
                df_exp.groupby("ORIGIN", as_index=False)
                .agg(
                    Flights=("TAXI_OUT", "size"),
                    Mean_Taxi_Out=("TAXI_OUT", "mean"),
                    Median_Taxi_Out=("TAXI_OUT", "median"),
                    P95_Taxi_Out=("TAXI_OUT", lambda values: values.quantile(0.95))
                )
                .sort_values("Mean_Taxi_Out", ascending=True)
            )
            fig_origin_compare = px.bar(
                origin_stats,
                x="Mean_Taxi_Out",
                y="ORIGIN",
                orientation="h",
                text="Mean_Taxi_Out",
                color="P95_Taxi_Out",
                labels={
                    "Mean_Taxi_Out": "Mean taxi-out (minutes)",
                    "ORIGIN": "Origin airport",
                    "P95_Taxi_Out": "P95 taxi-out"
                },
                color_continuous_scale="YlOrRd",
                hover_data=["Flights", "Median_Taxi_Out", "P95_Taxi_Out"]
            )
            fig_origin_compare.update_traces(texttemplate="%{text:.1f}")
            fig_origin_compare.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                xaxis_title="Mean taxi-out (minutes)",
                yaxis_title="Origin airport"
            )
            render_chart(fig_origin_compare)

        st.markdown("---")
        heat_left, heat_right = st.columns(2)

        with heat_left:
            st.subheader("Congestion Heatmap by Day and Departure Hour")
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heat_df = (
                df_exp.pivot_table(
                    index="DAY_NAME",
                    columns="DEP_HOUR",
                    values="TAXI_OUT",
                    aggfunc="mean"
                )
                .reindex(day_order)
            )
            fig_heat = go.Figure(data=go.Heatmap(
                z=heat_df.values,
                x=heat_df.columns,
                y=heat_df.index,
                colorscale="YlOrRd",
                colorbar=dict(title="Mean min")
            ))
            fig_heat.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                xaxis_title="Scheduled departure hour",
                yaxis_title="Day of week"
            )
            apply_hour_axis(fig_heat, "Scheduled departure hour")
            render_chart(fig_heat)

        with heat_right:
            st.subheader("Busiest Routes")
            route_stats = (
                df_exp.groupby(["ORIGIN", "DEST"], as_index=False)
                .agg(
                    Flights=("TAXI_OUT", "size"),
                    Mean_Taxi_Out=("TAXI_OUT", "mean"),
                    P95_Taxi_Out=("TAXI_OUT", lambda values: values.quantile(0.95))
                )
            )
            route_stats["ROUTE"] = route_stats["ORIGIN"] + " → " + route_stats["DEST"]
            busiest_routes = route_stats.sort_values("Flights", ascending=False).head(12).sort_values("Flights")
            fig_routes = px.bar(
                busiest_routes,
                x="Flights",
                y="ROUTE",
                orientation="h",
                color="Mean_Taxi_Out",
                text="Flights",
                labels={
                    "Flights": "Number of flights",
                    "ROUTE": "Route",
                    "Mean_Taxi_Out": "Mean taxi-out"
                },
                color_continuous_scale="Blues",
                hover_data=["Mean_Taxi_Out", "P95_Taxi_Out"]
            )
            fig_routes.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                xaxis_title="Number of flights",
                yaxis_title="Route"
            )
            render_chart(fig_routes)

        st.markdown("---")
        risk_left, risk_right = st.columns(2)

        with risk_left:
            st.subheader("High-Risk Departures by Hour")
            risk_by_hour = (
                df_exp.groupby("DEP_HOUR", as_index=False)
                .agg(
                    Flights=("TAXI_OUT", "size"),
                    High_Risk_Rate=("HIGH_RISK", "mean")
                )
            )
            risk_by_hour["High_Risk_Rate"] *= 100
            fig_risk = px.bar(
                risk_by_hour,
                x="DEP_HOUR",
                y="High_Risk_Rate",
                text="High_Risk_Rate",
                labels={
                    "DEP_HOUR": "Scheduled departure hour",
                    "High_Risk_Rate": "Flights above 30 minutes (%)"
                },
                color="High_Risk_Rate",
                color_continuous_scale="YlOrRd",
                hover_data=["Flights"]
            )
            fig_risk.update_traces(texttemplate="%{text:.1f}%")
            fig_risk.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                xaxis_title="Scheduled departure hour",
                yaxis_title="High-risk flights (%)"
            )
            apply_hour_axis(fig_risk, "Scheduled departure hour")
            render_chart(fig_risk)

        with risk_right:
            delay_title_col, delay_filter_col = st.columns([3.2, 1.2])
            with delay_title_col:
                st.subheader("Departure Delay vs Taxi-Out")
            with delay_filter_col:
                origin_codes = sorted(df_exp["ORIGIN"].dropna().astype(str).unique().tolist())
                origin_labels = ["ALL"] + [airport_names.get(code, code).rsplit(" - ", 1)[0] for code in origin_codes]
                origin_label_to_code = {label: code for code, label in zip(origin_codes, origin_labels[1:])}
                selected_origin_label = st.selectbox(
                    "Origin airport",
                    origin_labels,
                    index=0,
                    key="delay_origin_filter",
                    label_visibility="collapsed"
                )
                selected_origin = origin_label_to_code.get(selected_origin_label)

            delay_data = df_exp if selected_origin is None else df_exp[df_exp["ORIGIN"] == selected_origin]
            scatter_df = delay_data.sample(min(5000, len(delay_data)), random_state=42)
            fig_delay = px.scatter(
                scatter_df,
                x="DEP_DELAY",
                y="TAXI_OUT",
                color="ORIGIN",
                opacity=0.55,
                labels={
                    "DEP_DELAY": "Departure delay (minutes)",
                    "TAXI_OUT": "Taxi-out time (minutes)",
                    "ORIGIN": "Origin airport"
                },
                hover_data=["OP_UNIQUE_CARRIER", "DEST", "DISTANCE"]
            )
            fig_delay.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                xaxis_title="Departure delay (minutes)",
                yaxis_title="Taxi-out time (minutes)"
            )
            delay_y_max = int(np.ceil(delay_data["TAXI_OUT"].max() / 10.0) * 10)
            fig_delay.update_yaxes(
                tickmode="linear",
                tick0=0,
                dtick=10,
                range=[0, delay_y_max]
            )
            render_chart(fig_delay)
            correlation = df_exp["DEP_DELAY"].corr(df_exp["TAXI_OUT"])
            st.caption(f"Linear correlation in this dataset: {correlation:.3f}. This suggests departure delay and taxi-out should be analysed as separate operational signals.")

        st.markdown("---")
        distance_left, distance_right = st.columns(2)

        with distance_left:
            st.subheader("Flight Distance vs Departure Delay")
            distance_scatter = df_exp.sample(min(5000, len(df_exp)), random_state=42)
            fig_distance = px.scatter(
                distance_scatter,
                x="DISTANCE",
                y="DEP_DELAY",
                color="ORIGIN",
                opacity=0.55,
                labels={
                    "DISTANCE": "Flight distance (miles)",
                    "DEP_DELAY": "Departure delay (minutes)",
                    "ORIGIN": "Origin airport"
                },
                hover_data=["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "TAXI_OUT"]
            )
            fig_distance.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                xaxis_title="Flight distance (miles)",
                yaxis_title="Departure delay (minutes)"
            )
            render_chart(fig_distance)

        with distance_right:
            st.subheader("Departure Delay by Distance Band")
            distance_bands = [0, 500, 1000, 1500, 2000, float("inf")]
            distance_labels = [
                "260–499 miles",
                "500–999 miles",
                "1,000–1,499 miles",
                "1,500–1,999 miles",
                "2,000+ miles"
            ]
            delay_band_df = df_exp.copy()
            delay_band_df["DISTANCE_BAND"] = pd.cut(
                delay_band_df["DISTANCE"],
                bins=distance_bands,
                labels=distance_labels,
                include_lowest=True
            )
            fig_delay_band = px.box(
                delay_band_df,
                x="DISTANCE_BAND",
                y="DEP_DELAY",
                color="DISTANCE_BAND",
                points=False,
                labels={
                    "DISTANCE_BAND": "Distance band",
                    "DEP_DELAY": "Departure delay (minutes)"
                }
            )
            fig_delay_band.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                showlegend=False,
                xaxis_title="Distance band",
                yaxis_title="Departure delay (minutes)"
            )
            render_chart(fig_delay_band)
        st.info("The dataset is concentrated around DFW and American Airlines, so airport and carrier comparisons should be presented with flight volume and interpreted within this sample.")
# TAB 4: A-CDM ARCHITECTURE DOCUMENTATION
# ==============================================================================
with tab4:
    doc_title_col, doc_search_col = st.columns([2.2, 1])
    with doc_title_col:
        st.markdown("### Airport Collaborative Decision Making (A-CDM) Documentation")
    with doc_search_col:
        help_query = st.text_input("Search documentation", placeholder="Search a term...", label_visibility="visible")

    st.markdown("""
    #### What this dashboard does
    This dashboard estimates how many minutes an aircraft will spend taxiing from pushback to becoming airborne. Use the search box to find plain-language explanations and examples for the terms used throughout the dashboard.

    #### Operational architecture
    The workflow supports Airport Collaborative Decision Making (A-CDM): airlines, airports, and air traffic control use a shared operational picture to coordinate pushback, surface movement, and departure sequencing.

    #### Model workflow
    1. Clean flight records and remove cancelled, diverted, invalid, or extreme target values.
    2. Build time, congestion, carrier, and airport features.
    3. Apply the selected prediction model.
    4. Present the predicted taxi-out time and operational interpretation.
    """)

    help_terms = [
        {"category": "Data", "term": "Taxi-Out Time", "definition": "The elapsed ground time between pushback and wheels-off. It is the target the models predict.", "example": "If pushback is 08:35 and wheels-off is 09:01, taxi-out time is about 26 minutes."},
        {"category": "Data", "term": "Raw Records", "definition": "Flight rows available before the cleaning rules are applied.", "example": "The project starts with 75,000 raw records."},
        {"category": "Data", "term": "Clean Records", "definition": "Rows remaining after cancelled, diverted, invalid, missing, and extreme records are removed.", "example": "A clean record has a usable taxi-out target and valid operational fields."},
        {"category": "Data", "term": "Retention Rate", "definition": "The percentage of raw records kept after cleaning.", "example": "A 96.36% retention rate means 96.36 out of every 100 raw records remain."},
        {"category": "Data", "term": "Cancelled Flight", "definition": "A flight that did not operate and should not contribute to taxi-out learning.", "example": "A cancelled row is removed before model training."},
        {"category": "Data", "term": "Diverted Flight", "definition": "A flight that did not complete its planned operation normally and may not have a comparable taxi-out process.", "example": "A diverted row is excluded from the standard departure-surface analysis."},
        {"category": "Data", "term": "Outlier", "definition": "An unusually extreme value that can distort model training.", "example": "A taxi-out value above the project threshold of 120 minutes is treated as an outlier."},
        {"category": "Data", "term": "P75, P90, P95", "definition": "Percentiles: the value below which 75%, 90%, or 95% of observations fall.", "example": "P90 of 29.5 minutes means nine out of ten inspected flights are at or below 29.5 minutes."},
        {"category": "Flight", "term": "Carrier", "definition": "The airline operating the flight. The interface shows the airline name while the model uses its code.", "example": "American Airlines is represented internally by AA."},
        {"category": "Flight", "term": "Departure / Origin", "definition": "The airport where the aircraft begins its taxi-out movement.", "example": "DALLAS/FORT WORTH - DFW is the origin airport code DFW."},
        {"category": "Flight", "term": "Arrival / Destination", "definition": "The planned airport where the flight will land. It is retained for route context.", "example": "CHICAGO O'HARE - ORD is the destination airport code ORD."},
        {"category": "Flight", "term": "Scheduled Departure", "definition": "The planned departure time in HHMM format.", "example": "830 means 08:30."},
        {"category": "Flight", "term": "Actual Pushback", "definition": "The time the aircraft leaves the gate and begins its surface movement.", "example": "835 means the aircraft pushed back at 08:35."},
        {"category": "Flight", "term": "Departure Delay", "definition": "The delay associated with departure before or around pushback.", "example": "A value of 25 means the flight is 25 minutes delayed."},
        {"category": "Flight", "term": "Stage Distance", "definition": "The planned flight distance in miles. It provides route-length context.", "example": "802 miles is the distance used by the DFW morning example."},
        {"category": "Congestion", "term": "Surface Traffic Density", "definition": "How many recent pushbacks are occurring at the departure airport.", "example": "18 pushbacks in the past hour indicates a busier departure surface than 5 pushbacks."},
        {"category": "Congestion", "term": "Origin Pushbacks Past 1 Hr", "definition": "The number of pushbacks from the origin airport during the previous hour.", "example": "A value of 22 is classified as severe surface queueing."},
        {"category": "Congestion", "term": "Origin Pushbacks Past 30 Min", "definition": "The shorter-term pushback count used to identify a recent surge in traffic.", "example": "12 pushbacks in 30 minutes suggests a concentrated departure bank."},
        {"category": "Congestion", "term": "Free Flow", "definition": "Low surface activity with little expected queueing.", "example": "Fewer than 8 pushbacks per hour is classified as Low (Free Flow)."},
        {"category": "Congestion", "term": "Peak Bank", "definition": "A concentrated period when many flights are scheduled to depart close together.", "example": "A high pushback count during a morning bank can increase taxi-out time."},
        {"category": "Congestion", "term": "Queue Delay", "definition": "The portion of taxi-out time associated with waiting or sequencing behind other aircraft.", "example": "If predicted taxi-out is 26 minutes and the unimpeded base is 18 minutes, the queue component is about 8 minutes."},
        {"category": "Congestion", "term": "Airfield Queue", "definition": "Aircraft waiting on taxiways or near runway departure points because of surface demand.", "example": "A growing airfield queue is a reason to delay pushback or meter departures."},
        {"category": "Model", "term": "Prediction Model Engine", "definition": "The model selected to estimate taxi-out duration.", "example": "Choose XGBoost when you want a boosted-tree alternative to Random Forest."},
        {"category": "Model", "term": "Random Forest Regressor", "definition": "An ensemble of decision trees that averages many tree predictions to model non-linear relationships.", "example": "It can learn that congestion has a stronger effect during busy departure hours."},
        {"category": "Model", "term": "XGBoost Regressor", "definition": "A gradient-boosted tree model that adds trees sequentially to correct earlier errors.", "example": "It can capture smaller improvements in prediction after the main congestion pattern is learned."},
        {"category": "Model", "term": "Linear Regression Baseline", "definition": "A simple reference model that estimates a linear relationship between features and taxi-out time.", "example": "It provides a benchmark to determine whether the ensemble models add value."},
        {"category": "Model", "term": "Feature", "definition": "An input variable supplied to a model.", "example": "DEP_DELAY and DISTANCE are continuous features."},
        {"category": "Model", "term": "Feature Importance", "definition": "An estimate of how strongly a feature contributes to a tree model prediction.", "example": "CONGESTION_1H ranks highly when surface demand is a major source of delay."},
        {"category": "Model", "term": "MAE", "definition": "Mean Absolute Error: the average absolute difference between predicted and actual taxi-out minutes.", "example": "MAE of 4.04 means the average error is about 4 minutes."},
        {"category": "Model", "term": "RMSE", "definition": "Root Mean Squared Error: an error measure that penalizes larger mistakes more strongly than MAE.", "example": "A model with a lower RMSE has fewer severe misses."},
        {"category": "Model", "term": "R-squared", "definition": "A score describing how much variation in the target is explained by the model relative to a simple baseline.", "example": "A score closer to 1 indicates stronger explained variation, although operational error metrics should also be considered."},
        {"category": "Model", "term": "Within +/- 3 or 5 Minutes", "definition": "The percentage of predictions whose absolute error is no more than the stated tolerance.", "example": "82.6% within +/-5 minutes means about 83 out of 100 predictions are within five minutes."},
        {"category": "Analytics", "term": "Exploratory Analysis", "definition": "Visual investigation used to understand patterns before relying on model output.", "example": "Compare mean taxi-out by carrier, hour, and origin to identify operational patterns."},
        {"category": "Analytics", "term": "Departure Hour", "definition": "The hour extracted from the scheduled departure time.", "example": "18:45 belongs to departure hour 18."},
        {"category": "Analytics", "term": "Hub Airport", "definition": "An airport with substantial airline or network activity and many connecting or scheduled flights.", "example": "DFW, ATL, and JFK are treated as major operational hubs in the dashboard."},
        {"category": "Analytics", "term": "Diurnal", "definition": "A pattern that changes across the hours of a day.", "example": "A diurnal wave profile can show higher taxi-out times during morning and evening banks."},
        {"category": "Operations", "term": "A-CDM", "definition": "Airport Collaborative Decision Making: shared coordination between airport, airline, and air traffic stakeholders.", "example": "A-CDM can use the prediction to coordinate pushback timing with runway and taxiway demand."},
        {"category": "Operations", "term": "ATC", "definition": "Air Traffic Control, which manages safe and orderly aircraft movement and departure sequencing.", "example": "ATC may use surface demand information when applying departure metering."},
        {"category": "Operations", "term": "Departure Metering Program (DMP)", "definition": "A procedure that regulates when aircraft push back or join the departure queue to reduce surface congestion.", "example": "If the queue is severe, DMP logic can delay pushback instead of holding aircraft with engines running."},
        {"category": "Operations", "term": "TBO", "definition": "Trajectory-Based Operations: managing flights using planned and updated trajectories across the airspace system.", "example": "A taxi-out estimate is one ground segment of a broader trajectory plan."}
    ]

    query = help_query.strip().lower()
    matching_terms = [
        item for item in help_terms
        if not query or query in " ".join(item.values()).lower()
    ]
    st.caption(f"Showing {len(matching_terms)} of {len(help_terms)} documentation terms.")
    if not matching_terms:
        st.warning("No matching term found. Try a simpler search such as taxi, delay, model, queue, or A-CDM.")
    for item in matching_terms:
        with st.expander(f'{item["term"]}  |  {item["category"]}'):
            st.markdown(f'**Meaning:** {item["definition"]}')
            st.markdown(f'**Simple example:** {item["example"]}')

    st.markdown("---")
    st.caption("JNTU Aviation Data Science & Operations Research Coursework Project | BTS Marketing Carrier On-Time Dataset 2025")
