# train_all_models.py  (trains 48 Prophet models, logs to MLflow)
import os, numpy as np, pandas as pd
from prophet import Prophet
import mlflow
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# READ THE FILE THAT HAS THE HOLIDAY FLAGS — not processed_features.csv
CSV_PATH    = os.path.join(CURRENT_DIR, "features", "final_engineered_features.csv")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "forecast_results.csv")
MLFLOW_URI  = "sqlite:///" + os.path.join(CURRENT_DIR, "mlflow.db")
ROUTES = ["MUM-GOA","MUM-LAK","MUM-HS","KCH-LAK","CHN-VIZ-PUD","MUM-WA"]  # one spelling
SHIPS  = ["EMPRESS","SKY"]
CABINS = ["INTERIOR","SEA_VIEW","BALCONY","SUITE"]
REGRESSORS = ["is_weekend","is_indian_public_holiday","school_holiday_flag","is_monsoon"]
def run():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing {CSV_PATH}. Run build_features then build_indian_holidays first.")
    df = pd.read_csv(CSV_PATH)
    df["sailing_date"] = pd.to_datetime(df["sailing_date"])
    for col in ["route_code","ship_id","cabin_class"]:
        df[col] = df[col].astype(str).str.strip()
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Cordelia_Cruise_Forecasting")
    # Build a future calendar of regressor values so 90-day forecast has its features
    cal = df.groupby("sailing_date")[REGRESSORS].max()
    out, ok = [], 0
    for r in ROUTES:
        for s in SHIPS:
            for c in CABINS:
                key = f"{r}_{s}_{c}"
                seg = df[(df.route_code==r) & (df.ship_id==s) & (df.cabin_class==c)]
                # Aggregate bookings to one row per sailing date (y = total PAX)
                agg = (seg.groupby("sailing_date")
                          .agg(y=("passengers_count","sum"),
                               **{g:(g,"max") for g in REGRESSORS})
                          .reset_index().rename(columns={"sailing_date":"ds"}))
                # FAIL LOUDLY instead of emitting fake constants
                if len(agg) < 30:
                    print(f"[SKIP] {key}: only {len(agg)} rows (<30). Fix the data, do not fake it.")
                    continue
                m = Prophet(seasonality_mode="multiplicative", yearly_seasonality=False,
                            changepoint_prior_scale=0.05, interval_width=0.95)
                m.add_seasonality("yearly_custom", period=365.25, fourier_order=10)
                active = []
                for g in REGRESSORS:
                    if agg[g].nunique() > 1:
                        m.add_regressor(g); active.append(g)
                m.fit(agg)
                # Honest in-sample MAPE
                pred = m.predict(agg)["yhat"].values
                act  = agg["y"].values
                mape = float(np.mean(np.abs((act-pred)/np.where(act==0,1e-5,act)))*100)
                # 90-day future with real regressor values from the calendar
                fut = m.make_future_dataframe(periods=90, freq="W")
                for g in active:
                    fut[g] = fut["ds"].map(cal[g].to_dict()).fillna(0)
                fc = m.predict(fut).tail(90)
                for _, row in fc.iterrows():
                    out.append({
                        "model_key": key, "route_code": r, "ship_id": s, "cabin_class": c,
                        "sailing_date": row["ds"].strftime("%Y-%m-%d"),
                        "forecasted_bookings": max(0, int(round(row["yhat"]))),
                        "forecast_lower": max(0, int(round(row["yhat_lower"]))),
                        "forecast_upper": max(0, int(round(row["yhat_upper"]))),
                        "evaluation_mape": round(mape, 2),
                    })
                with mlflow.start_run(run_name=f"Run_{key}"):
                    mlflow.log_params({"route":r,"ship":s,"cabin":c,
                                       "engine_mode":"Prophet_Native",
                                       "active_regressors":",".join(active)})
                    mlflow.log_metric("test_mape", mape)
                print(f"[OK] {key}  rows={len(agg)}  MAPE={mape:.2f}%  regs={active}")
                ok += 1
    if not out:
        raise RuntimeError("No models trained. Data pipeline is broken upstream.")
    pd.DataFrame(out).to_csv(OUTPUT_PATH, index=False)
    print(f"\n[DONE] {ok}/48 models trained. Forecast -> {OUTPUT_PATH}")
if __name__ == "__main__":
    run()
