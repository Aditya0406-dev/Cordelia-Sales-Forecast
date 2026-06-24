# build_features.py  (the REAL feature script — restore this)
# Reads bookings from the portable SQLite DB and engineers row-level features.
import pandas as pd, numpy as np, sqlite3, os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(CURRENT_DIR, "data", "cordelia_bookings.db")
OUT_DIR  = os.path.join(CURRENT_DIR, "features")
OUT_PATH = os.path.join(OUT_DIR, "processed_features.csv")
os.makedirs(OUT_DIR, exist_ok=True)
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM cordelia_bookings", conn)
conn.close()
print(f"[DB] Loaded {len(df):,} raw rows")
# 1. Drop cancellations
df["cancellation_flag"] = df["cancellation_flag"].astype(str).str.strip().str.upper()
df = df[df["cancellation_flag"].isin(["FALSE","0","N","NO",""])]
print(f"[CLEAN] {len(df):,} active rows after removing cancellations")
# 2. Dates
df["booking_date"] = pd.to_datetime(df["booking_date"])
df["sailing_date"] = pd.to_datetime(df["sailing_date"])
# 3. Features (EXACT names the trainer expects)
df["lead_time_days"] = (df["sailing_date"] - df["booking_date"]).dt.days
df["sailing_month"]  = df["sailing_date"].dt.month
df["is_weekend"]     = (df["sailing_date"].dt.dayofweek >= 5).astype(int)
df["is_monsoon"]     = df.apply(
 lambda r: 1 if (("LAK" in str(r["route_code"])) and r["sailing_month"] in [5,6,7,8,9]) else 0,
 axis=1)
df.to_csv(OUT_PATH, index=False)
print(f"[OK] Features -> {OUT_PATH}  | monsoon-flagged rows: {df['is_monsoon'].sum():,}")
