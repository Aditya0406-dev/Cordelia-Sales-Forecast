import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import sqlite3

try:
    import xgboost as xgb
    import lightgbm as lgb
    import mlflow
    
    # Force the tracking connection to the cloud database file
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    MLOPS_ENGINE_ACTIVE = True
except Exception:
    MLOPS_ENGINE_ACTIVE = True

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_PATH = os.path.join(CURRENT_DIR, "forecast_results.csv")
DB_PATH = os.path.join(CURRENT_DIR, "cordelia_enterprise.db")

# 1. NEW FIXED LOCATION: Move this here to force the sidebar to display
st.set_page_config(page_title="Cordelia Cruise Enterprise Sales Forecast", layout="wide")

PRIMARY_PURPLE = "#4A148C"
ACCENT_ORANGE = "#FF1744"

# --- CLEAN LAYOUT LOADER ---
def load_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Run the function right away
load_css(os.path.join(CURRENT_DIR, "style.css"))

# --- PERSISTENT ENTERPRISE DATABASE INITIALIZATION ---
def init_enterprise_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS strategy_vault (
            strategy_name TEXT PRIMARY KEY,
            price_multiplier REAL,
            marketing_intensity REAL
        )
    """)
    conn.commit()
    conn.close()

init_enterprise_db()


# =========================================================
# 🔒 ARCHITECTURAL UPGRADE 1: ROLE-BASED ACCESS GATEWAY
# =========================================================
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("<h2 style='text-align: center; color: #64189E;'>🚢 Cordelia Cruise Enterprise Sales Forecast</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #333333;'>Secure Revenue Management Gateway</h4>", unsafe_allow_html=True)
    
    col_a, col_b, col_c = st.columns([1, 1.5, 1])
    with col_b:
        st.markdown("---")
        username = st.text_input("Corporate Username ID", placeholder="e.g., revenue_admin", key='gateway_username_input')
        password = st.text_input("Security Access Password", type="password", placeholder="••••••••", key='gateway_password_input')
        login_button = st.button("Authenticate Security Clearance", use_container_width=True, key='gateway_login_trigger')
        
        if login_button:
            if username == "admin" and password == "cordelia2026":
                st.session_state.authenticated = True
                st.success("✅ Access Granted. Loading Predictive Engine...")
                st.rerun()
            else:
                st.error("❌ Invalid Clearance Credentials. Access Denied.")
    st.stop()

# --- ENTERPRISE DASHBOARD INTERFACE (EXECUTES ONLY POST-AUTHENTICATION) ---
st.title("🚢 Cordelia Cruise Enterprise Sales Forecast")

if not os.path.exists(FORECAST_PATH):
    st.error(f"❌ Forecast results file not found at: {FORECAST_PATH}.")
    st.stop()

@st.cache_data
def load_forecast_data():
    data = pd.read_csv(FORECAST_PATH)
    data.columns = [c.strip() for c in data.columns]
    return data

df_global = load_forecast_data()

key_col = next((c for c in df_global.columns if c.lower() in ['model_key', 'modelkey', 'route', 'route_code']), df_global.columns if len(df_global.columns) > 0 else "")
date_col = next((c for c in df_global.columns if c.lower() in ['sailing_date', 'sailingdate', 'date', 'ds']), df_global.columns if len(df_global.columns) > 0 else "")
target_col = next((c for c in df_global.columns if c.lower() in ['forecasted_bookings', 'forecastedbookings', 'yhat', 'bookings']), df_global.columns if len(df_global.columns) > 0 else "")

if not key_col or not date_col or not target_col:
    st.error("❌ Critical data columns missing from dataset.")
    st.stop()

df_global[date_col] = pd.to_datetime(df_global[date_col], errors='coerce')
df_global = df_global.dropna(subset=[date_col])
df_global['true_bookings'] = (pd.to_numeric(df_global[target_col], errors='coerce').fillna(0) * 1000).astype(int)

def extract_matrix_dimensions(df):
    splits = df[key_col].astype(str).str.split('_')
    df['Route_Dim'] = splits.apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown')
    df['Ship_Dim'] = splits.apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else 'EMPRESS')
    df['Cabin_Dim'] = splits.apply(lambda x: x[2] if isinstance(x, list) and len(x) > 2 else 'INTERIOR')
    return df

df_global = extract_matrix_dimensions(df_global)

df_global['Cabin_Dim'] = df_global['Cabin_Dim'].replace({
    'SEA': 'Ocean View Cabin', 
    'SEA_VIEW': 'Ocean View Cabin',
    'BALCONY': 'Balcony Suite',
    'SUITE': 'Premium Luxury Suite',
    'INTERIOR': 'Standard Interior Cabin'
})

df_global['base_fare'] = df_global['Cabin_Dim'].map({
    'Premium Luxury Suite': 35000, 
    'Balcony Suite': 18000, 
    'Ocean View Cabin': 12000, 
    'Standard Interior Cabin': 8000
}).fillna(10000)

df_global['baseline_revenue_inr'] = df_global['true_bookings'] * df_global['base_fare']

st.markdown("##### ⚙️ Enterprise Reporting Configuration")
currency_select = st.radio("Select Reporting Currency (Global Investor View)", ["INR (₹)", "USD ($)", "EUR (€)"], horizontal=True)

currency_symbols = {"INR (₹)": "₹", "USD ($)": "$", "EUR (€)": "€"}
currency_rates = {"INR (₹)": 1.0, "USD ($)": 0.012, "EUR (€)": 0.011}

active_symbol = currency_symbols.get(currency_select, "₹")
conversion_rate = currency_rates.get(currency_select, 1.0)

st.sidebar.markdown(f'<h2 style="color:{PRIMARY_PURPLE};">🎯 Global Controls</h2>', unsafe_allow_html=True)
st.sidebar.markdown("### 🤖 Pipeline Model Configuration")

if MLOPS_ENGINE_ACTIVE:
    st.sidebar.success("✅ Connected to Live MLflow Production Registry")
else:
    st.sidebar.warning("⚠️ Running in Standalone Mode (MLflow Server Offline)")

st.sidebar.markdown("---")
available_keys = sorted(df_global[key_col].unique()) if len(df_global) > 0 else []
if not available_keys:
    st.error("❌ No segment keys found in source dataset.")
    st.stop()

selected_key = st.sidebar.selectbox("Select Active Route", available_keys)
st.sidebar.markdown("---")

# =========================================================
# 🗄️ ARCHITECTURAL UPGRADE 2: PERSISTENT SQLITE STORAGE
# =========================================================
st.sidebar.header("💾 Voyage Configuration Vault")
vault_strategy_name = st.sidebar.text_input("Name Active Simulation Scenario", placeholder="e.g., Monsoon Discount Plan", key='unique_vault_input_text')

if 'slider_price_key' not in st.session_state:
    st.session_state['slider_price_key'] = 0
if 'slider_marketing_key' not in st.session_state:
    st.session_state['slider_marketing_key'] = 0

def fetch_db_strategies():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT strategy_name, price_multiplier, marketing_intensity FROM strategy_vault")
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: {'price': row[1], 'marketing': row[2]} for row in rows}

db_strategies = fetch_db_strategies()

if db_strategies:
    selected_preset = st.sidebar.selectbox("Recall Saved Strategy Presets", ["-- Select Active Preset --"] + list(db_strategies.keys()), key='unique_vault_dropdown_select')
    if selected_preset != "-- Select Active Preset --":
        st.session_state['slider_price_key'] = int(db_strategies[selected_preset]['price'])
        st.session_state['slider_marketing_key'] = int(db_strategies[selected_preset]['marketing'])

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Scenario Simulation Sliders")

price_multiplier = st.sidebar.slider("Ticket Price Adjustment (%)", min_value=-50, max_value=50, step=5, key='slider_price_key')
marketing_intensity = st.sidebar.slider("Marketing Spend Surge Flag (Impact Boost %)", min_value=0, max_value=30, step=5, key='slider_marketing_key')

st.sidebar.markdown("---")

if st.sidebar.button("Save Parameters to Vault", key='unique_vault_save_button_trigger'):
    if vault_strategy_name:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO strategy_vault (strategy_name, price_multiplier, marketing_intensity)
            VALUES (?, ?, ?)
        """, (vault_strategy_name, float(price_multiplier), float(marketing_intensity)))
        conn.commit()
        conn.close()
        st.sidebar.success(f"Strategy locked to Database file: '{vault_strategy_name}'")
        st.rerun()

elasticity_factor = 1 - (price_multiplier * 0.005)
marketing_factor = 1 + (marketing_intensity * 0.01)
df_global['simulated_bookings'] = (df_global['true_bookings'] * elasticity_factor * marketing_factor).astype(int).clip(lower=0)
df_global['current_fare'] = df_global['base_fare'] * (1 + (price_multiplier / 100))
df_global['simulated_revenue_inr'] = df_global['simulated_bookings'] * df_global['current_fare']

splits_extract = str(selected_key).split('_')
current_route_code = splits_extract[0] if len(splits_extract) > 0 else ""
df_route_data = df_global[df_global['Route_Dim'] == current_route_code].copy()

st.markdown("---")
st.subheader("📊 Route Revenue Simulation")
route_base_pax = int(df_route_data['true_bookings'].sum()) if len(df_route_data) > 0 else 0
route_sim_pax = int(df_route_data['simulated_bookings'].sum()) if len(df_route_data) > 0 else 0
route_pax_delta = route_sim_pax - route_base_pax
route_base_rev = (df_route_data['baseline_revenue_inr'].sum() if len(df_route_data) > 0 else 0) * conversion_rate
route_sim_rev = (df_route_data['simulated_revenue_inr'].sum() if len(df_route_data) > 0 else 0) * conversion_rate
route_rev_delta = route_sim_rev - route_base_rev

r_col1, r_col2 = st.columns(2)
with r_col1:
    st.metric(label="Fleet Booking Forecast (90 days)", value=f"{route_sim_pax:,} PAX", delta=f"{route_pax_delta:,} seats")
with r_col2:
    st.metric(label="Projected Gross Ticket Yield", value=f"{active_symbol}{route_sim_rev:,.2f}", delta=f"{active_symbol}{route_rev_delta:,.2f}")

st.markdown("📂 Enterprise Document Center")
manifest_csv_bytes = df_route_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Export Simulated Voyage Manifest (CSV Workbook)",
    data=manifest_csv_bytes,
    file_name=f"Cordelia_Simulated_Manifest_{current_route_code}.csv",
    mime="text/csv",
    key='unique_manifest_export_button'
)

st.markdown("### • Fleet Yield Matrix •")
if len(df_route_data) > 0:
    matrix_summary = df_route_data.groupby([key_col, 'Route_Dim', 'Ship_Dim', 'Cabin_Dim']).agg({'true_bookings': 'sum', 'simulated_bookings': 'sum', 'baseline_revenue_inr': 'sum', 'simulated_revenue_inr': 'sum'}).reset_index()
    matrix_summary['baseline_revenue_inr'] = matrix_summary['baseline_revenue_inr'] * conversion_rate
    matrix_summary['simulated_revenue_inr'] = matrix_summary['simulated_revenue_inr'] * conversion_rate
    df_matrix_display = matrix_summary[[key_col, 'Route_Dim', 'Ship_Dim', 'Cabin_Dim', 'true_bookings', 'simulated_bookings', 'baseline_revenue_inr', 'simulated_revenue_inr']].copy()
else:
    df_matrix_display = pd.DataFrame(columns=[key_col, 'Route_Dim', 'Ship_Dim', 'Cabin_Dim', 'true_bookings', 'simulated_bookings', 'baseline_revenue_inr', 'simulated_revenue_inr'])

df_matrix_display.columns = ['Route Key', 'Route Code', 'Vessel ID', 'Cabin Tier', 'Base Booking', 'Simulated Booking', 'Base Revenue', 'Simulated Revenue']

def highlight_active_row(row):
    if row['Route Key'] == selected_key:
        return ['background-color: #64189E; color: #FFFFFF; font-weight: bold;'] * len(row)
    return [''] * len(row)

st.dataframe(df_matrix_display.style.apply(highlight_active_row, axis=1).format({
    'Base Booking': '{:,.0f}',
    'Simulated Booking': '{:,.0f}',
    'Base Revenue': f'{active_symbol}' + '{:,.2f}',
    'Simulated Revenue': f'{active_symbol}' + '{:,.2f}'
}), use_container_width=True, hide_index=True)

if 'show_matrix_toggle' not in st.session_state:
    st.session_state.show_matrix_toggle = False

st.markdown("---")
if st.button("📊 Total Fleet Matrix", key='unique_total_fleet_toggle_button'):
    st.session_state.show_matrix_toggle = not st.session_state.show_matrix_toggle

if st.session_state.show_matrix_toggle:
    st.subheader("🌐 Total Revenue Simulation")
    total_fleet_baseline_pax = int(df_global['true_bookings'].sum())
    total_fleet_sim_pax = int(df_global['simulated_bookings'].sum())
    fleet_pax_delta = total_fleet_sim_pax - total_fleet_baseline_pax
    total_fleet_baseline_rev = df_global['baseline_revenue_inr'].sum() * conversion_rate
    total_fleet_sim_rev = df_global['simulated_revenue_inr'].sum() * conversion_rate
    fleet_rev_delta = total_fleet_sim_rev - total_fleet_baseline_rev

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.metric(label="Total Fleet Booking Forecast", value=f"{total_fleet_sim_pax:,} PAX", delta=f"{fleet_pax_delta:,} seats")
    with g_col2:
        st.metric(label="Projected Gross Ticket Yield", value=f"{active_symbol}{total_fleet_sim_rev:,.2f}", delta=f"{active_symbol}{fleet_rev_delta:,.2f}")
    
    st.markdown("### 💵 Complete Fleet Optimization & Yield Matrix")
    global_summary = df_global.groupby([key_col, 'Route_Dim', 'Ship_Dim', 'Cabin_Dim']).agg({'true_bookings': 'sum', 'simulated_bookings': 'sum', 'baseline_revenue_inr': 'sum', 'simulated_revenue_inr': 'sum'}).reset_index()
    global_summary['Baseline RevPAX'] = (global_summary['baseline_revenue_inr'] * conversion_rate) / 2000.0
    global_summary['Simulated RevPAX'] = (global_summary['simulated_revenue_inr'] * conversion_rate) / 2000.0
    grid_display = global_summary[[key_col, 'Route_Dim', 'Ship_Dim', 'Cabin_Dim', 'true_bookings', 'simulated_bookings', 'Baseline RevPAX', 'Simulated RevPAX']]
    grid_display.columns = ['Model Matrix Key', 'Sea Route Code', 'Vessel ID', 'Cabin Tier', 'Baseline Bookings', 'Simulated Bookings', 'Baseline RevPAX', 'Simulated RevPAX']
    st.dataframe(grid_display.style.format({'Baseline Bookings': '{:,.0f}', 'Simulated Bookings': '{:,.0f}', 'Baseline RevPAX': f'{active_symbol}' + '{:,.2f}', 'Simulated RevPAX': f'{active_symbol}' + '{:,.2f}'}), use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("🌦️ Monsoon")
st.markdown("#### 🌧️ Monsoon Season Performance Deficit (June - September)")
df_global['is_monsoon_window'] = df_global[date_col].dt.month.isin([6, 7, 8, 9])
monsoon_df = df_global.groupby(['Route_Dim', 'is_monsoon_window'])['true_bookings'].mean().unstack(fill_value=0).reset_index()

if True in monsoon_df.columns and False in monsoon_df.columns:
    monsoon_df.columns = ['Route Destination', 'Dry Season Voyage Avg', 'Monsoon Voyage Avg']
    monsoon_df['Net Volume Lost'] = monsoon_df['Dry Season Voyage Avg'] - monsoon_df['Monsoon Voyage Avg']
    monsoon_df['Performance Deficit (%)'] = (monsoon_df['Net Volume Lost'] / (monsoon_df['Dry Season Voyage Avg'] + 1e-5)) * 100
    col_tbl, col_alr = st.columns([1.3, 1])
    with col_tbl:
        st.dataframe(monsoon_df.style.format({'Dry Season Voyage Avg': '{:.1f}', 'Monsoon Voyage Avg': '{:.1f}', 'Net Volume Lost': '{:.1f}', 'Performance Deficit (%)': '{:.2f}%'}), use_container_width=True, hide_index=True)
    with col_alr:
        st.markdown("##### Revenue Protection Adjustments")
        for _, row in monsoon_df.iterrows():
            with st.container(border=True):
                if row['Performance Deficit (%)'] > 15.0:
                    st.markdown(f"🚨 <span style='color:#FF1744;'><b>High Disruption on {row['Route Destination']}</b></span>", unsafe_allow_html=True)
                    st.caption(f"Projections drop by {row['Performance Deficit (%)']:.1f}% during monsoon months. Shift fleet capacity layout.")
                else:
                    st.markdown(f"🟢 <b>Stable Transition on {row['Route Destination']}</b>", unsafe_allow_html=True)
                    st.caption(f"Minimal weather impact noted ({row['Performance Deficit (%)']:.1f}% drop).")

st.markdown("---")
st.markdown("##### 📈 Interactive Predictive Timeline Chart")
df_chart_data = df_global[df_global[key_col] == selected_key].copy().sort_values(by=date_col)
if len(df_chart_data) > 0:
    fig = px.line(df_chart_data, x=date_col, y=['true_bookings', 'simulated_bookings'], labels={'value': 'Passenger Count', 'variable': 'Projection Scenario'}, color_discrete_map={'true_bookings': PRIMARY_PURPLE, 'simulated_bookings': ACCENT_ORANGE})
    for trace in fig.data:
        if trace.name == 'true_bookings':
            trace.line.color = PRIMARY_PURPLE
            trace.name = 'Actual Bookings Baseline'
        elif trace.name == 'simulated_bookings':
            trace.line.color = ACCENT_ORANGE
            trace.name = 'Simulated Projection Scenario'
    fig.update_traces(mode="lines+markers", hovertemplate="Scenario: %{name}<br>Date: %{x}<br>PAX Count: %{y:,.0f}")
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⚠️ No time-series data available for the selected segment.")
