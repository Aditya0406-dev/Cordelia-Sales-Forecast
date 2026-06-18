import pandas as pd
import pymysql
import warnings

# Suppress pandas warning messages completely
warnings.filterwarnings("ignore", category=UserWarning)

print("[PROCESS] Connecting to Windows MySQL to extract raw historical data...")

# 1. Connect to your local Windows database
connection = pymysql.connect(
    host="localhost",
    user="root",
    password="aditya@0406",  # <--- TYPE YOUR REAL MYSQL PASSWORD HERE!
    database="sales_db",
    port=3306
)

try:
    # 2. Pull ALL records directly to bypass database string format bugs
    query = """
        SELECT booking_id, voyage_id, booking_date, sailing_date, route_code, passengers_count, cancellation_flag 
        FROM cordelia_bookings;
    """
    df = pd.read_sql(query, connection)
    print(f"[DATABASE] Total raw rows downloaded from table: {len(df)}")

    # 3. Clean and filter out the cancelled bookings locally inside Pandas
    df['cancellation_flag'] = df['cancellation_flag'].astype(str).str.strip().str.upper()
    df = df[df['cancellation_flag'].isin(['FALSE', '0', 'N', 'NO', ''])]
    print(f"[SUCCESS] Filtered active subset down to {len(df)} records for feature processing.")

    # 4. Standardize date formats to datetimes safely
    df['booking_date'] = pd.to_datetime(df['booking_date'])
    df['sailing_date'] = pd.to_datetime(df['sailing_date'])

    # --- FEATURE 1: BOOKING LEAD TIME ---
    print("[FEATURE 1] Calculating Booking Lead Times...")
    df['lead_time_days'] = (df['sailing_date'] - df['booking_date']).dt.days

    # --- FEATURE 2: MONSOON DISRUPTION FLAGS ---
    print("[FEATURE 2] Generating Monsoon Route Disruption Flags...")
    df['sailing_month'] = df['sailing_date'].dt.month
    
    # Flag 1 if the route goes to Lakshadweep (LAK) AND month is May, June, July, August, or September
    df['is_monsoon_disruption'] = df.apply(
        lambda row: 1 if ("LAK" in str(row['route_code']) and row['sailing_month'] in [5, 6, 7, 8, 9]) else 0, 
        axis=1
    )

    # 5. Save the generated features as our base staging file
    output_path = r"C:\Users\S. Aditya\sales-ai\features\processed_features.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n[SUCCESS] Monsoon and Lead Time Features Engineered Successfully!")
    print(f"-> Total Active Rows Processed: {len(df)}")
    print(f"-> Monsoon Route Closures Flagged: {df['is_monsoon_disruption'].sum()}")
    print(f"-> Exported to: {output_path}")

finally:
    connection.close()
