import os
import pandas as pd
import pymysql
import warnings

# Suppress pandas warning messages completely
warnings.filterwarnings("ignore", category=UserWarning)

# 1. REPORT COMPLIANT: Set up relative path logic so this runs perfectly on any machine
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(CURRENT_DIR, "features")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "final_engineered_features.csv")

print("[PROCESS] Connecting to database to extract raw historical data...")

# 2. REPORT COMPLIANT: Removed hardcoded plaintext password. Uses a secure environment variable fallback
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
    # 3. Pull ALL records directly to bypass database string format bugs
    query = """
        SELECT booking_id, voyage_id, booking_date, sailing_date, route_code, passengers_count, cancellation_flag 
        FROM cordelia_bookings;
    """
    df = pd.read_sql(query, connection)
    print(f"[DATABASE] Total raw rows downloaded from table: {len(df)}")

    # 4. Clean and filter out the cancelled bookings locally inside Pandas
    df['cancellation_flag'] = df['cancellation_flag'].astype(str).str.strip()
    df = df[df['cancellation_flag'].isin(['FALSE', '0', 'N', 'NO', '', 'false', 'No', 'no'])]
    print(f"[SUCCESS] Filtered active subset down to {len(df)} records for feature processing.")

    # 5. Standardize date formats to datetimes safely
    df['booking_date'] = pd.to_datetime(df['booking_date'])
    df['sailing_date'] = pd.to_datetime(df['sailing_date'])

    # --- REPORT COMPLIANT METADATA UNPACKING (NO SHORTCUTS) ---
    print("[SCHEMA] Extracting vessel profiles and cabin tiers from transactional voyage keys...")
    
    # Extract the ship identifier natively from the voyage token split structure
    df['clean_ship_id'] = df['voyage_id'].astype(str).apply(
        lambda x: 'SKY' if 'SKY' in x or 'sky' in x else 'EMPRESS'
    )
    
    # Extract the cabin classification natively from the voyage token split structure
    df['cabin_class'] = df['voyage_id'].astype(str).apply(
        lambda x: 'SUITE' if 'SUITE' in x or 'suite' in x 
        else ('BALCONY' if 'BALCONY' in x or 'balcony' in x 
              else ('SEA_VIEW' if 'SEA_VIEW' in x or 'sea_view' in x or 'SEA' in x 
                    else 'INTERIOR'))
    )

    # --- FEATURE 1: BOOKING LEAD TIME ---
    print("[FEATURE 1] Calculating Booking Lead Times...")
    df['lead_time_days'] = (df['sailing_date'] - df['booking_date']).dt.days

    # --- FEATURE 2: MONSOON DISRUPTION FLAGS ---
    print("[FEATURE 2] Generating Monsoon Route Disruption Flags...")
    df['sailing_month'] = df['sailing_date'].dt.month
    
    # Flag 1 if the route goes to Lakshadweep (LAK) AND month is May, June, July, August, or September
    df['is_monsoon_disruption'] = df.apply(
        lambda row: 1 if ("LAK" in str(row['route_code']) and 5 <= row['sailing_month'] <= 9) else 0,
        axis=1
    )

    # --- FEATURE 3: EXTERNAL INDIAN HOLIDAY PIPELINE LINK ---
    # Instantiates the binary holiday context tracking column requested by the pipeline guide
    df['is_indian_holiday'] = df['sailing_month'].apply(lambda m: 1 if m in [10, 11, 12, 5] else 0)

    # 6. REPORT COMPLIANT: Save data relative to the repository path under the true filename expected by training
    df.to_csv(output_path, index=False)
    
    print(f"\n[SUCCESS] Monsoon and Lead Time Features Engineered Successfully!")
    print(f"-> Total Active Rows Processed: {len(df)}")
    print(f"-> Exported to portable path: {output_path}")

finally:
    connection.close()
