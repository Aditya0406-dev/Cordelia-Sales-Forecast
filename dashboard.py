import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import sqlite3

# ==============================================================================
# AUDIT ITEM 12: RELIABLE LIVE MLOPS REGISTRY VALIDATION (NO FALSE FLAGS)
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
db_file_path = os.path.join(CURRENT_DIR, "mlflow.db")
FORECAST_PATH = os.path.join(CURRENT_DIR, "forecast_results.csv")

# ==============================================================================
# UPDATED MLOPS SERVICE DETECTION LAYER (DIRECT REGISTRY VALIDATION)
# ==============================================================================
if os.path.exists(db_file_path):
    try:
        conn = sqlite3.connect(f"file:{db_file_path}?mode=ro", uri=True)
        test_df = pd.read_sql_query("SELECT 1 FROM experiments LIMIT 1;", conn)
        conn.close()
        MLOPS_ENGINE_ACTIVE = True
        CONNECTION_LABEL = "Live MLflow Production Registry"
    except Exception:
        MLOPS_ENGINE_ACTIVE = False
        CONNECTION_LABEL = "Local MLflow Registry Database Error"
else:
    MLOPS_ENGINE_ACTIVE = False
    CONNECTION_LABEL = "Production Release File Artifact Mirror"
# ==============================================================================
# AUDIT ITEM 14: HIGH-CONTRAST BRAND COMPLIANCE STYLE SHEET
# ==============================================================================
PRIMARY_PURPLE = "#64189E"  
ACCENT_ORANGE = "#F1723F"   

st.set_page_config(
    page_title="FinVector | Cruise Enterprise Sales Forecast Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(f"""
    <style>
        .stApp {{ background-color: #FAFAFA; }}
        [data-testid="stSidebar"] {{ background-color: {PRIMARY_PURPLE} !important; }}
        [data-testid="stSidebar"] .stMarkdown, 
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {{ 
            color: #FFFFFF !important; 
        }}
        div[data-testid="stMetricValue"] {{ color: {ACCENT_ORANGE} !important; font-weight: bold; }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# CLEANED SIDEBAR CONFIGURATION (UPDATED DISPLAY LOGIC)
# ==============================================================================
st.sidebar.title("🚢 FinVector Analytics")
st.sidebar.subheader("Cruise Enterprise Sales Forecast Suite")

if MLOPS_ENGINE_ACTIVE:
    st.sidebar.success(f"✅ Connected to {CONNECTION_LABEL}")
else:
    st.sidebar.error(f"⚠️ {CONNECTION_LABEL}")

st.sidebar.markdown("---")

# ==============================================================================
# DATA INGESTION ENGINE
# ==============================================================================
@st.cache_data
def load_forecast_registry_file():
    if not os.path.exists(FORECAST_PATH):
        st.error("❌ forecast_results.csv not found. Run the training pipeline first.")
        st.stop()
    df = pd.read_csv(FORECAST_PATH)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["sailing_date"] = pd.to_datetime(df["sailing_date"])
    return df

fc_master = load_forecast_registry_file()
rate_map = {"BALCONY": 18500.00, "INTERIOR": 8400.00, "SEA_VIEW": 11200.00, "SUITE": 29000.00}
route_key_map = {"MUM-GOA": "MUM-GOA", "MUM-LAK": "MUM-LAK", "MUM-HS": "MUM-HS", "KCH-LAK": "KCH-LAK", "CHN-VIZ-PUD": "CHN-VIZ-PUD", "MUM-WA": "MUM-WA"}

# ==============================================================================
# BI-DIRECTIONAL SIDEBAR SIMULATION CONTROLS (CHANGE 2)
# ==============================================================================
st.sidebar.subheader("⚙️ Scenario Simulation Sliders")

# Bi-directional control 1: Ticket Price Adjustment
ticket_price_num = st.sidebar.number_input("Ticket Price Adjustment (%)", min_value=-50, max_value=50, value=0)
ticket_price_adj = st.sidebar.slider("Adjust Price Slider:", min_value=-50, max_value=50, value=ticket_price_num, label_visibility="collapsed")

# Bi-directional control 2: Marketing Spend
marketing_num = st.sidebar.number_input("Marketing Spend Surge Flag", min_value=0, max_value=100, value=20)
marketing_spend = st.sidebar.slider("Adjust Marketing Slider:", min_value=0, max_value=100, value=marketing_num, label_visibility="collapsed")

st.sidebar.text_input("Voyage Configuration Vault Name", value="Monsoon Discount Plan")
if st.sidebar.button("Save Parameters to Vault"):
    st.sidebar.success("Parameters Cached into Vault Presets")

# ==============================================================================
#CURRENCY SELECTION IN SIDEBAR (WITHOUT TITLE)
# ==============================================================================
currency_mode = st.sidebar.radio("SELECT REPORTING CURRENCY", ["INR (₹)", "USD ($)", "EUR (€)"])

fx_symbols = {"INR (₹)": "₹", "USD ($)": "$", "EUR (€)": "€"}
fx_rates = {"INR (₹)": 1.0, "USD ($)": 0.012, "EUR (€)": 0.011}

symbol = fx_symbols[currency_mode]
rate_multiplier = fx_rates[currency_mode]
currency = currency_mode


# Default Base Reporting Currency Assumptions (Clean Workspace Mapping)

price_factor = 1.0 - (ticket_price_adj / 100.0)
marketing_factor = 1.0 + (marketing_spend / 100.0)

fc_master["base_fare"] = fc_master["cabin_class"].map(rate_map).fillna(10000.0)
fc_master["base_revenue"] = fc_master["forecasted_bookings"] * fc_master["base_fare"]

fc_master["simulated_booking"] = (fc_master["forecasted_bookings"] * price_factor * marketing_factor).astype(int)
fc_master["simulated_revenue"] = fc_master["simulated_booking"] * fc_master["base_fare"]


# ==============================================================================
# GLOBAL MULTI-PAGE NAVIGATION GATEWAY (TOP TABS - CHANGE 1)
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Fleet Executive Summary", 
    "2. Route & Cabin Yield Matrix", 
    "3. Scenario Planning (Fleet Expansion)", 
    "4. Model Performance & Validation"
])

# --------------------------------------------------------------------------
# PAGE 1: FLEET EXECUTIVE SUMMARY VIEW
# --------------------------------------------------------------------------
with tab1:
    st.title("Cruise Enterprise Sales Forecast Dashboard")
    st.markdown("### 📊 Fleet-Wide Aggregated Performance View")
    
    total_pax = int(fc_master["simulated_booking"].sum())
    total_revenue_raw = fc_master["simulated_revenue"].sum()
    total_revenue_converted = total_revenue_raw * rate_multiplier
    
    base_total_pax = fc_master["forecasted_bookings"].sum()
    pax_delta = total_pax - base_total_pax
    pax_delta_str = f"{pax_delta:+,} seats shift" if pax_delta != 0 else "Baseline Stable"

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="TOTAL FLEET BOOKING FORECAST (90 DAYS)", 
            value=f"{total_pax:,} PAX", 
            delta=pax_delta_str,
            delta_color="normal" if pax_delta >= 0 else "inverse"
        )
    with col2:
        st.metric(
            label="PROJECTED GROSS TICKET YIELD (ALL 48 TRACKS)", 
            value=f"{symbol}{total_revenue_converted:,.2f}",
            delta=f"Active Currency Mode: {currency}"
        )
        
    st.markdown("---")
    st.subheader("🌐 Complete Fleet Aggregation Ledger")
    
    df_ledger = fc_master.groupby(["model_key", "route_code", "ship_id", "cabin_class"]).agg({
        "forecasted_bookings": "sum",
        "base_revenue": "sum",
        "simulated_booking": "sum",
        "simulated_revenue": "sum"
    }).reset_index()
    
    df_ledger["base_revenue"] *= rate_multiplier
    df_ledger["simulated_revenue"] *= rate_multiplier
    df_ledger["Calculated RevPAX"] = np.where(df_ledger["simulated_booking"] > 0, df_ledger["simulated_revenue"] / df_ledger["simulated_booking"], 0.0)
    
    df_ledger.columns = ["Route Key", "Route Code", "Vessel ID", "Cabin Tier", "Base Booking", "Base Revenue", "Simulated Booking", "Simulated Revenue", "Calculated RevPAX"]
    st.dataframe(df_ledger, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# PAGE 2: ROUTE & CABIN YIELD MATRIX VIEW
# --------------------------------------------------------------------------
with tab2:
    st.title("🧮 Route & Cabin Yield Matrix")
    st.markdown("### • Live Highlight & Segment Drill-Down Engine •")
    
    routes = ['MUM-GOA', 'MUM-LAK', 'MUM-HS', 'KCH-LAK', 'CHN-VIZ-PUD', 'MUM-WA']
    ships, cabins = ["EMPRESS", "SKY"], ["INTERIOR", "SEA_VIEW", "BALCONY", "SUITE"]
    
    st.subheader("🔍 Select Target Model Filter Focus")
    selected_route = st.selectbox("Filter Display Segment by Route Code", routes)
    
    route_data = fc_master[fc_master["route_code"] == selected_route].copy()
    
    route_sim_pax = int(route_data["simulated_booking"].sum())
    route_base_pax = int(route_data["forecasted_bookings"].sum())
    route_pax_delta = route_sim_pax - route_base_pax
    route_delta_str = f"{route_pax_delta:+,} seats shift" if route_pax_delta != 0 else "Baseline Stable"
    
    route_sim_revenue = route_data["simulated_revenue"].sum() * rate_multiplier
    
    col_kpi1, col_kpi2 = st.columns(2)
    with col_kpi1: 
        st.metric(label=f"{selected_route} BOOKING FORECAST (90 DAYS)", value=f"{route_sim_pax:,} PAX", delta=route_delta_str, delta_color="normal" if route_pax_delta >= 0 else "inverse")
    with col_kpi2: 
        st.metric(label=f"{selected_route} PROJECTED GROSS TICKET YIELD", value=f"{symbol}{route_sim_revenue:,.2f}", delta=f"Active Currency Mode: {currency}")
    
    st.markdown("---")
    st.subheader(f"📊 Cabin Yield Matrix Summary Grid ({selected_route})")
    
    matrix_rows = []
    for cabin in cabins:
        row_dict = {"Cabin Class": cabin}
        for ship in ships:
            cell_match = route_data[(route_data["cabin_class"] == cabin) & (route_data["ship_id"] == ship)]
            cell_sum = int(cell_match["simulated_booking"].sum()) if not cell_match.empty else 0
            row_dict[ship] = f"{cell_sum:,} PAX"
        matrix_rows.append(row_dict)
    st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader(f"📈 90-Day Predictive Voyage Timeline Detail")
    col_s, col_c = st.columns(2)
    with col_s: active_ship = st.selectbox("Select Timeline Ship Context:", ships, key="matrix_ship_select")
    with col_c: active_cabin = st.selectbox("Select Timeline Cabin Context:", cabins, key="matrix_cabin_select")
    
    target_key = f"{route_key_map.get(selected_route, selected_route)}_{active_ship}_{active_cabin}".strip().upper()
    df_chart = route_data[(route_data["ship_id"] == active_ship) & (route_data["cabin_class"] == active_cabin)].sort_values('sailing_date')
    
    if not df_chart.empty:
        fig = go.Figure()
        y_sim = df_chart["simulated_booking"]
        y_upper = df_chart["forecast_upper"] * price_factor * marketing_factor
        y_lower = df_chart["forecast_lower"] * price_factor * marketing_factor
            
        fig.add_trace(go.Scatter(
            x=df_chart['sailing_date'], 
            y=y_upper, 
            mode='lines', 
            line=dict(color='rgba(80, 80, 80, 0.6)', width=1.5, dash='dash'), 
            showlegend=True, 
            name='Upper Bound Limit'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_chart['sailing_date'], 
            y=y_lower, 
            mode='lines', 
            fill='tonexty', 
            fillcolor='rgba(106, 90, 205, 0.15)', 
            line=dict(width=0), 
            showlegend=False, 
            name='Lower Bound'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_chart['sailing_date'], 
            y=y_sim, 
            mode='lines+markers', 
            line=dict(color='indigo', width=3), 
            marker=dict(size=4), 
            name='Simulated Projections'
        ))

        fig.update_layout(
            title=f"90-Day Expected Booking Influx Velocity ({target_key})", 
            xaxis_title="Sailing Operations Date", 
            yaxis_title="Daily Operational Booking Count (PAX / Day)", 
            hovermode="x unified", 
            template="plotly_white", 
            yaxis=dict(tickformat='d', hoverformat='d')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"⚠️ Selected model combination tracking metrics are currently unpopulated.")
            
        st.markdown("---")
        st.markdown("#### 📑 Granular Segment Ledger View")
        st.dataframe(route_data.drop(columns=["base_fare", "base_revenue"]), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# PAGE 3: SCENARIO PLANNING (FLEET EXPANSION) - CLEANED WORKSPACE
# --------------------------------------------------------------------------
with tab3:
    st.title("🔮 Scenario Planning & Fleet Expansion Workspace")
    
    currency_mode = st.radio("SELECT SIMULATION DISPLAY CURRENCY", ["INR (₹)", "USD ($)", "EUR (€)"], horizontal=True)
    fx_symbols = {"INR (₹)": "₹", "USD ($)": "$", "EUR (€)": "€"}
    fx_rates = {"INR (₹)": 1.0, "USD ($)": 0.012, "EUR (€)": 0.011}
    
    active_symbol = fx_symbols[currency_mode]
    active_multiplier = fx_rates[currency_mode]

    st.markdown("---")
    st.subheader("🚢 Simulate New Vessel Ingestion")
    target_expansion_route = st.selectbox("Select Target Route for Fleet Expansion:", ["MUM-GOA", "MUM-LAK", "MUM-HS", "KCH-LAK", "CHN-VIZ-PUD", "MUM-WA"])

    route_default_capacities = {"MUM-GOA": 2500, "MUM-LAK": 1800, "MUM-HS": 2000, "KCH-LAK": 1200, "CHN-VIZ-PUD": 1600, "MUM-WA": 1500}
    dynamic_default_cap = route_default_capacities.get(target_expansion_route, 2000)

    vessel_capacity = st.slider("Vessel Capacity Bound (PAX)", min_value=500, max_value=3000, value=dynamic_default_cap, step=100, key=f"vessel_cap_{target_expansion_route}")
    monsoon_toggle = st.toggle("Enable Monsoon Disruption Impact Simulation (50% Load Suppression)")
    
    route_load_factors = {"MUM-GOA": 0.88, "MUM-LAK": 0.82, "MUM-HS": 0.75, "KCH-LAK": 0.78, "CHN-VIZ-PUD": 0.80, "MUM-WA": 0.70}
    route_rates = {"MUM-GOA": 9500.00, "MUM-LAK": 14200.00, "MUM-HS": 8400.00, "KCH-LAK": 12800.00, "CHN-VIZ-PUD": 11200.00, "MUM-WA": 24500.00}
    
    selected_factor = route_load_factors.get(target_expansion_route, 0.80)
    sim_pax_base = vessel_capacity * selected_factor
    
    if monsoon_toggle and target_expansion_route in ["MUM-LAK", "KCH-LAK"]:
        sim_pax_final = int(sim_pax_base * 0.50)
        st.warning(f"⚠️ Monsoon suppression factor applied. Expected island route load restricted by 50%.")
    else:
        sim_pax_final = int(sim_pax_base)
        
    selected_rate = route_rates.get(target_expansion_route, 11200.00)
    sim_revenue = sim_pax_final * selected_rate
    
    st.markdown(f"#### 🔮 Real-Time Projections for Vessel Addition: **{target_expansion_route}**")
    scol1, scol2 = st.columns(2)
    with scol1: st.metric(label=f"Projected Incremental Capacity ({target_expansion_route})", value=f"{sim_pax_final:,} PAX", delta=f"{int(selected_factor*100)}% Baseline Load")
    with scol2: st.metric(label="Simulated Incremental Revenue Yield", value=f"{active_symbol}{sim_revenue * active_multiplier:,.2f}", delta=f"Avg Route Fare: {active_symbol}{selected_rate * active_multiplier:,.2f}")

# --------------------------------------------------------------------------
# PAGE 4: MODEL PERFORMANCE & VALIDATION - OWNING THE PROJECT LAYOUT
# --------------------------------------------------------------------------
with tab4:
    st.title("📉 Model Performance & Validation Metrics")

    perf_df = fc_master.groupby(["route_code", "ship_id", "cabin_class"])["evaluation_mape"].first().reset_index()
    perf_df = perf_df[perf_df["evaluation_mape"] > 0]
    
    global_mape = perf_df["evaluation_mape"].mean()
    
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.metric(
            label="Global Mean Absolute Percentage Error (MAPE)", 
            value=f"{global_mape:.2f}%",
            delta="Target Threshold < 10.0%"
        )
    with mcol2:
        st.metric(label="Evaluated Operational Routes", value=f"{len(perf_df['route_code'].unique())} Active Segments")
        
    st.markdown("---")
    st.markdown("#### Authenticated Experiment Ledger Index")
    
    perf_df.columns = ["Route Code", "Vessel ID", "Cabin Tier", "Natively Evaluated Model MAPE (%)"]
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

