# build_indian_holidays.py  (adds holiday + school-break flags)
import pandas as pd, os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_PATH  = os.path.join(CURRENT_DIR, "features", "processed_features.csv")
OUT_PATH = os.path.join(CURRENT_DIR, "features", "final_engineered_features.csv")
df = pd.read_csv(IN_PATH)
df["sailing_date"] = pd.to_datetime(df["sailing_date"])
# Calendar must cover the full HISTORY + the 90-day forecast horizon (into 2026).
holiday_dates = [
    "2023-01-26","2023-03-08","2023-08-15","2023-11-12","2023-11-13","2023-12-25",
    "2024-01-26","2024-03-25","2024-08-15","2024-11-01","2024-11-02","2024-12-25",
    "2025-01-26","2025-03-14","2025-08-15","2025-10-20","2025-10-21","2025-12-25",
    "2026-01-26","2026-03-04","2026-08-15","2026-11-08","2026-11-09","2026-12-25",
]
hset = set(pd.to_datetime(holiday_dates).date)
df["is_indian_public_holiday"] = df["sailing_date"].dt.date.isin(hset).astype(int)
# School breaks: May, June (summer), October (Diwali), December (winter)
df["school_holiday_flag"] = df["sailing_date"].dt.month.isin([5,6,10,12]).astype(int)
df.to_csv(OUT_PATH, index=False)
print(f"[OK] Final features -> {OUT_PATH}")
print(f"  holiday rows: {df['is_indian_public_holiday'].sum():,} | "        f"school-break rows: {df['school_holiday_flag'].sum():,}")
