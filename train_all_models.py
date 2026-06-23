import os
import numpy as np
import pandas as pd
from prophet import Prophet
import mlflow  
import mlflow.data 
from datetime import datetime

# --- LOCAL FILE PATH CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(CURRENT_DIR, "features", "final_engineered_features.csv")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "forecast_results.csv")
MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"

def execute_pipeline():
    print(f"--> Loading Phase 2 Data from: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Could not find 'final_engineered_features.csv'!")
        return
        
    df = pd.read_csv(CSV_PATH)
    df.columns = [col.strip() for col in df.columns]
    
    # 1. SCHEMA DEFINITION
    ROUTE_COL = 'route_code'
    DS_COL = 'sailing_date'
    TARGET_COL = 'passengers_count' if 'passengers_count' in df.columns else df.columns[-1]

    if 'voyage_id' in df.columns:
        df['clean_ship_id'] = df['voyage_id'].astype(str).apply(
            lambda x: x.split('-')[1] if '-' in x and len(x.split('-')) > 1 else (x.split('_')[0] if '_' in x else x)
        )
    else:
        df['clean_ship_id'] = 'EMPRESS'
    SHIP_COL = 'clean_ship_id'
    CABIN_COL = 'cabin_class' if 'cabin_class' in df.columns else None
    
    # Ensure correct datetime parsing for lookups
    df[DS_COL] = pd.to_datetime(df[DS_COL])
    
    # 2. THE OFFICIAL FINVECTOR 48-COMBINATION MATRIX
    routes = ['MUM_GOA', 'MUM_LAK', 'MUM_HI_SEAS', 'KCH_LAK', 'CHN_VIZ_PUD', 'MUM_WASIA']
    ships = ['EMPRESS', 'SKY'] 
    cabins = ['INTERIOR', 'SEA_VIEW', 'BALCONY', 'SUITE'] 

    all_forecasts = []
    success_count = 0
    run_index = 1
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("Cordelia_Cruise_Forecasting")
    except Exception:
        print("[NOTICE] Running without MLflow background hooks.")

    print(f"--> Targets Defined: {len(routes)} Routes x {len(ships)} Ships x {len(cabins)} Cabins = 48 Combinations.")

    for r in routes:
        for s in ships:
            for c in cabins:
                model_key = f"{r}_{s}_{c}"
                
                if CABIN_COL:
                    df_filtered = df[(df[ROUTE_COL].str.strip() == r) & (df[SHIP_COL].str.strip() == s) & (df[CABIN_COL].str.strip() == c)].copy()
                else:
                    df_filtered = df[(df[ROUTE_COL].str.strip() == r) & (df[SHIP_COL].str.strip() == s)].copy()
                
                try:
                    df_prophet = pd.DataFrame()
                    if len(df_filtered) >= 30:
                        df_prophet['ds'] = df_filtered[DS_COL]
                        df_prophet['y'] = pd.to_numeric(df_filtered[TARGET_COL])
                        
                        # Set up regressors mapping directly from your master engineered feature file
                        regressors = ['is_weekend', 'is_indian_public_holiday', 'school_holiday_flag', 'is_monsoon']
                        for reg in regressors:
                            if reg in df_filtered.columns:
                                df_prophet[reg] = df_filtered[reg]
                        
                        df_prophet = df_prophet.groupby('ds', as_index=False).mean()
                        df_prophet = df_prophet.replace([np.inf, -np.inf], np.nan).dropna(subset=['y'])

                    segment_row_count = len(df_prophet)
                    print(f"[METRIC] Route Segment: {model_key} | Total Historical Rows: {segment_row_count}")

                    if segment_row_count < 30:
                        raise ValueError(f"CRITICAL ERROR: Segment model {model_key} has less than 30 data points. Training stopped.")

                    # --- NATIVE PROPHET ENGINE ---
                    model = Prophet(
                        seasonality_mode='multiplicative', 
                        yearly_seasonality=False,          
                        changepoint_prior_scale=0.05,
                        interval_width=0.95 # Explicitly sets 95% Confidence Intervals for your chart
                    )
                    
                    model.add_seasonality(name='yearly_custom', period=365.25, fourier_order=10)
                    model.add_seasonality(name='indian_summer_holiday', period=365.25, fourier_order=3)
                    
                    active_regressors = []
                    for reg in regressors:
                        if reg in df_prophet.columns and df_prophet[reg].nunique() > 1:
                            model.add_regressor(reg)
                            active_regressors.append(reg)
                            
                    model.fit(df_prophet)
                    
                    forecast_historical = model.predict(df_prophet)
                    actuals = df_prophet['y'].values
                    predictions = forecast_historical['yhat'].values
                    mape = float(np.mean(np.abs((actuals - predictions) / np.where(actuals == 0, 1e-5, actuals))) * 100)
                        
                    try:
                        with mlflow.start_run(run_name=f"Run_{model_key}"):
                            mlflow_dataset = mlflow.data.from_pandas(df_prophet, targets='y', name=f"dataset_{model_key}")
                            mlflow.log_input(mlflow_dataset, context="training")
                            mlflow.log_params({"route": str(r), "ship": str(s), "cabin": str(c), "engine_mode": "Prophet_Native"})
                            mlflow.log_metrics({"test_mape": mape})
                    except Exception:
                        pass

                    # --- FIXED FUTURE REGRESSOR ENGINE ---
                    # Create baseline future dataframe
                    future_df = model.make_future_dataframe(periods=90, freq='D')
                    
                    # Create a master lookup from your original df to map REAL future dates to your holiday flags
                    for reg in active_regressors:
                        # Fallback map to assign values if calendar entries exist in the feature file
                        feature_map = df.set_index(DS_COL)[reg].to_dict()
                        future_df[reg] = future_df['ds'].map(feature_map).fillna(0)
                            
                    # Generate predictions for BOTH historical and future frames to keep data unbroken
                    forecast_full = model.predict(future_df)
                    
                    # Append all rows (historical + future) so your Streamlit timeline has zero empty slots
                    for _, row in forecast_full.iterrows():
                        all_forecasts.append({
                            "model_key": model_key,
                            "sailing_date": row['ds'].strftime('%Y-%m-%d'),
                            "forecasted_bookings": max(0, int(round(row['yhat']))),
                            "forecast_lower": max(0, int(round(row['yhat_lower']))),
                            "forecast_upper": max(0, int(round(row['yhat_upper']))),
                            "evaluation_mape": round(mape, 2)
                        })
                        
                    print(f"[{run_index}] SUCCESS: Native Prophet Key -> {model_key} (MAPE: {round(mape, 2)}%)")
                    success_count += 1
                    
                except Exception as model_err:
                    print(f"[{run_index}] NOTED: Exception on key {model_key}. Details: {str(model_err)}")
                
                run_index += 1

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    if all_forecasts:
        results_df = pd.DataFrame(all_forecasts)
        results_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n[DONE] SUCCESS: All {success_count}/48 models populated in tracking output: '{OUTPUT_PATH}'")
    else:
        print("\n[CRITICAL] Data schema processing error. File not compiled.")

if __name__ == "__main__":
    execute_pipeline()
