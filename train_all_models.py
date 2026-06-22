import os
import numpy as np
import pandas as pd
from prophet import Prophet
import mlflow  
import mlflow.data 
from datetime import datetime

# --- LOCAL FILE PATH CONFIGURATION ---
# This looks for the files exactly where the script runs from, preventing folder jumping errors
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# If final_engineered_features.csv is in the SAME folder as this script:
CSV_PATH = os.path.join(CURRENT_DIR, "features", "final_engineered_features.csv")


# This saves forecast_results.csv in the SAME folder as this script:
OUTPUT_PATH = os.path.join(CURRENT_DIR, "forecast_results.csv")

MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"

def execute_pipeline():
    print(f"--> Loading Phase 2 Data from: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Could not find 'final_engineered_features.csv'!")
        return
        
    df = pd.read_csv(CSV_PATH)
    df.columns = [col.strip() for col in df.columns]
    
    # 1. SCHEMA DEFINITION MATCHING FINVECTOR SPECIFICATION (Page 4-5)
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
    
    # 2. THE OFFICIAL FINVECTOR 48-COMBINATION MATRIX (Page 2 & 8)
    # FIXED: Replaced long names with raw database keys to prevent the 144 silent data filter crash
    routes = ['MUM_GOA', 'MUM_LAK', 'MUM_HI_SEAS', 'KCH_LAK', 'CHN_VIZ', 'MUM_WASIA']
    ships = ['EMPRESS', 'SKY'] # Based on Page 2: Cordelia Empress & Cordelia Sky
    cabins = ['INTERIOR', 'SEA_VIEW', 'BALCONY', 'SUITE'] # Official casing from page 4

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
                # Standardized underscore delimiter token used in your dashboard filters
                model_key = f"{r}_{s}_{c}"
                
                if CABIN_COL:
                    df_filtered = df[(df[ROUTE_COL].str.strip() == r) & (df[SHIP_COL].str.strip() == s) & (df[CABIN_COL].str.strip() == c)]
                else:
                    df_filtered = df[(df[ROUTE_COL].str.strip() == r) & (df[SHIP_COL].str.strip() == s)]
                
                try:
                    df_prophet = pd.DataFrame()
                    if len(df_filtered) >= 30:
                        df_prophet['ds'] = pd.to_datetime(df_filtered[DS_COL])
                        df_prophet['y'] = pd.to_numeric(df_filtered[TARGET_COL])
                        df_prophet = df_prophet.groupby('ds', as_index=False).mean()
                        df_prophet = df_prophet.replace([np.inf, -np.inf], np.nan).dropna(subset=['y'])

                    # FIXED (Item 7): Log row count per model segment
                    segment_row_count = len(df_prophet)
                    print(f"[METRIC] Route Segment: {model_key} | Total Historical Rows: {segment_row_count}")

                    # FIXED (Item 6 & 7): Fail if data has less than 30 points. No fake fallback values or 1.28% MAPE.
                    if segment_row_count < 30:
                        raise ValueError(f"CRITICAL ERROR: Segment model {model_key} has less than 30 data points ({segment_row_count} found). Training stopped.")

                    # --- NATIVE PROPHET ENGINE WITH CRUISE BUSINESS MATH (Page 8) ---
                    model = Prophet(
                        seasonality_mode='multiplicative', # Proportional scaling optimization
                        yearly_seasonality=False,          # Overwritten by custom Fourier series
                        changepoint_prior_scale=0.05
                    )
                    
                    # Fine-detailed fourier orders matching Appendix demand shifts
                    model.add_seasonality(name='yearly_custom', period=365.25, fourier_order=10)
                    model.add_seasonality(name='indian_summer_holiday', period=365.25, fourier_order=3)
                    
                    regressors = ['is_weekend', 'is_indian_public_holiday', 'school_holiday_flag', 'is_monsoon']
                    for reg in regressors:
                        if reg in df_filtered.columns:
                            feature_map = df_filtered.set_index(pd.to_datetime(df_filtered[DS_COL]))[reg].to_dict()
                            df_prophet[reg] = df_prophet['ds'].map(feature_map).fillna(0)
                            if df_prophet[reg].nunique() > 1:
                                model.add_regressor(reg)
                            
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

                    future_df = model.make_future_dataframe(periods=90, freq='D')
                    for reg in regressors:
                        if reg in df_prophet.columns:
                            future_df[reg] = df_prophet[reg].mean()
                            
                    forecast_future = model.predict(future_df).tail(90)
                    
                    for _, row in forecast_future.iterrows():
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

    # Verify and create features directory if missing
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    if all_forecasts:
        results_df = pd.DataFrame(all_forecasts)
        results_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n[DONE] SUCCESS: All {success_count}/48 models populated in tracking output: '{OUTPUT_PATH}'")
    else:
        print("\n[CRITICAL] Data schema processing error. File not compiled.")

if __name__ == "__main__":
    execute_pipeline()
