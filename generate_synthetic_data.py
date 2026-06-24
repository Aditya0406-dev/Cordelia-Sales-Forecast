# generate_synthetic_data.py
import pandas as pd, numpy as np, sqlite3, os
np.random.seed(42)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(CURRENT_DIR, "data", "cordelia_bookings.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
ROUTES = {
    "MUM-GOA":     {"load": 0.85, "is_island": False, "fare": 9500},
    "MUM-LAK":     {"load": 0.80, "is_island": True,  "fare": 14200},
    "MUM-HS":      {"load": 0.75, "is_island": False, "fare": 8400},
    "KCH-LAK":     {"load": 0.78, "is_island": True,  "fare": 12800},
    "CHN-VIZ-PUD": {"load": 0.80, "is_island": False, "fare": 11200},
    "MUM-WA":      {"load": 0.70, "is_island": False, "fare": 24500},
}
SHIPS  = {"EMPRESS": 800, "SKY": 2000}
CABINS = {"INTERIOR": (0.55, 8400), "SEA_VIEW": (0.25, 11200), "BALCONY": (0.15, 18500), "SUITE": (0.05, 29000)}
MONTH_MULT = {1:1.2, 2:1.0, 3:1.05, 4:1.0, 5:0.9, 6:0.7, 7:0.65, 8:0.7, 9:0.8, 10:1.1, 11:1.3, 12:1.5}
HOLIDAYS = set(pd.to_datetime(["2023-11-12","2024-11-01","2023-12-25","2024-12-25","2023-01-26","2024-01-26"]).date)
rows, bid = [], 1
for route, rm in ROUTES.items():
    for ship, cap in SHIPS.items():
        for base_sd in pd.date_range("2023-01-01", "2024-12-31", freq="W"):
            sd = base_sd + pd.Timedelta(days=int(np.random.randint(0, 7)))
            for cabin, (share, rate) in CABINS.items():
                seasonal = MONTH_MULT[sd.month]
                # FIX: Restored the true monsoon array mapping to close the syntax error
                if rm["is_island"] and sd.month in [5,6,7,8,9]:
                    seasonal *= 0.10
                if sd.date() in HOLIDAYS:  seasonal *= 1.5
                if sd.dayofweek >= 5:      seasonal *= 1.25
                expected = cap * share * rm["load"] * seasonal
                n_bookings = max(1, int(np.random.poisson(max(0.5, expected/8.0))))
                for _ in range(n_bookings):
                    lead = int(np.random.randint(5, 120))
                    pax  = int(np.random.randint(1, 5))
                    rows.append({
                        "booking_id": f"B{bid:07d}",
                        "voyage_id": f"{route}-{ship}-{sd.strftime('%Y%m%d')}",
                        "booking_date": (sd - pd.Timedelta(days=lead)).date(),
                        "sailing_date": sd.date(),
                        "route_code": route, "ship_id": ship, "cabin_class": cabin,
                        "passengers_count": pax,
                        "base_fare_inr": rate,
                        "ancillary_spend_inr": round(pax * np.random.uniform(500, 3000), 2),
                        "booking_channel": np.random.choice(["DIRECT","OTA","AGENT","CORPORATE"]),
                        "passenger_origin_city": np.random.choice(["Mumbai","Pune","Delhi","Kochi","Chennai"]),
                        "cancellation_flag": np.random.choice(["FALSE","TRUE"], p=[0.93, 0.07]),
                    })
                    bid += 1
df = pd.DataFrame(rows)
df["total_revenue_inr"] = df["passengers_count"]*df["base_fare_inr"] + df["ancillary_spend_inr"]
conn = sqlite3.connect(DB_PATH)
df.to_sql("cordelia_bookings", conn, if_exists="replace", index=False)
conn.close()
print(f"[OK] Wrote {len(df):,} booking rows to {DB_PATH}")
