import os
import pandas as pd
import pymysql
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) == "features":
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
else:
    PROJECT_ROOT = CURRENT_DIR

output_path = os.path.join(PROJECT_ROOT, "features", "processed_features.csv")

print("[PROCESS] Connecting to database to extract raw historical data...")
db_password = os.environ.get("DB_PASSWORD")

if not db_password:
    raise ValueError("CRITICAL COMPLIANCE FAILURE: 'DB_PASSWORD' environment context is unconfigured. Execution stopped.")

connection = pymysql.connect(
    host="localhost",
    user="root",
    password=db_password,
    database="sales_db",
    port=3306
)

try:
    query = """
        SELECT booking_id, voyage_id, booking_date, sailing_date, route_code, passengers_count, cancellation_flag, cabin_class
        FROM cordelia_bookings;
    """
    df = pd.read_sql(query, connection)
    print(f"[DATABASE] Total raw rows downloaded from table: {len(df)}")

    df['cancellation_flag'] = df['cancellation_flag'].fillna(0)
    df = df[df['cancellation_flag'].isin([0, '0', '0.0', 'FALSE', 'false', 'N', 'NO', 'no', 'No', ''])]
    print(f"[SUCCESS] Filtered active subset down to {len(df)} records for feature processing.")

    df['booking_date'] = pd.to_datetime(df['booking_date'])
    df['sailing_date'] = pd.to_datetime(df['sailing_date'])

    df['clean_ship_id'] = df['voyage_id'].astype(str).apply(
        lambda x: 'SKY' if 'SKY' in x or 'sky' in x else 'EMPRESS'
    )

    df['cabin_class'] = df['cabin_class'].astype(str).str.strip().str.upper()
    df['route_code'] = df['route_code'].astype(str).str.strip().str.upper()

    print("[FEATURE 1] Calculating Booking Lead Times...")
    df['lead_time_days'] = (df['sailing_date'] - df['booking_date']).dt.days

    print("[FEATURE 2] Generating Monsoon Route Disruption Flags...")
    df['sailing_month'] = df['sailing_date'].dt.month
    df['is_monsoon_disruption'] = df.apply(
        lambda row: 1 if ("LAK" in str(row['route_code']) and 5 <= row['sailing_month'] <= 9) else 0,
        axis=1
    )

    print("[FEATURE 3] Mapping Holiday Calendar Contexts...")
    # FIX: Restored the true monthly tracking array list to clear the SyntaxError completely
    df['is_indian_holiday'] = df['sailing_date'].apply(
        lambda d: 1 if (d.month in [10, 11, 12, 5] or (d.month == 1 and d.day == 26) or (d.month == 8 and d.day == 15)) else 0
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("🚀 DIRECT OVERRIDE VERIFICATION SUCCESSFUL")
    print(f"-> Target Absolute Path: {os.path.abspath(output_path)}")
    print(f"-> File Size Created: {os.path.getsize(output_path):,} bytes")
    print(f"-> Verified Total Row Output: {len(df):,}")
    print("="*60 + "\n")

finally:
    connection.close()
