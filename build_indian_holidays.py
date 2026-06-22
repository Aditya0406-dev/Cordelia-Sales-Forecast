import os
import pandas as pd

# 1. FIXED (Item 4): Establish strict B2B relative path variables to prevent laptop crash errors
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(CURRENT_DIR, "features", "processed_features.csv")
output_path = os.path.join(CURRENT_DIR, "features", "final_engineered_features.csv")

print(f"[PROCESS] Reading baseline feature matrix from: {input_path}")
df = pd.read_csv(input_path)

print("[FEATURE 3] Extracting Comprehensive Indian Holiday & Festival Peak Matrix (2022-2026)...")

# --- MASTER EXHAUSTIVE INDIAN FESTIVAL & HOLIDAY CALENDAR MATRIX ---
# FIXED (Item 16): Extended calendar through 2026 to cover the entire future forecast horizon
holiday_dates = [
    # === 2022 HOLIDAYS & LONG WEEKENDS ===
    "2022-01-26", "2022-03-18", "2022-04-15", "2022-05-03", 
    "2022-08-15", "2022-08-31", "2022-10-05", "2022-10-24", 
    "2022-10-25", "2022-11-08", "2022-12-24", "2022-12-25", "2022-12-31",

    # === 2023 HOLIDAYS & LONG WEEKENDS ===
    "2023-01-26", "2023-03-08", "2023-04-07", "2023-04-22", 
    "2023-06-29", "2023-08-15", "2023-09-19", "2023-10-02", 
    "2023-10-24", "2023-11-12", "2023-11-13", "2023-12-24", "2023-12-25", "2023-12-31",

    # === 2024 HOLIDAYS & LONG WEEKENDS ===
    "2024-01-26", "2024-03-25", "2024-03-29", "2024-04-11", 
    "2024-06-17", "2024-08-15", "2024-09-07", "2024-10-02", 
    "2024-10-12", "2024-11-01", "2024-11-02", "2024-12-24", "2024-12-25", "2024-12-31",

    # === 2025 HOLIDAYS & LONG WEEKENDS ===
    "2025-01-26", "2025-03-14", "2025-04-18", "2025-10-02", 
    "2025-10-18", "2025-10-19", "2025-10-20", "2025-10-22", "2025-10-23", # Full Diwali Festive Window (Dhanteras to Bhai Dooj)
    "2025-12-24", "2025-12-25", "2025-12-31", # Christmas & New Year Week 2025

    # === 2026 HOLIDAYS & LONG WEEKENDS (CRITICAL FOR FORWARD PROJECTIONS) ===
    "2026-01-01", # New Year's Day
    "2026-01-26", # Republic Day
    "2026-03-04", # Holi Festival Peak
    "2026-03-21", # Id-ul-Fitr
    "2026-04-03", # Good Friday
    "2026-05-01", # Buddha Purnima
    "2026-05-27", # Id-ul-Zuha (Bakrid)
    "2026-06-26", # Muharram
    "2026-08-15", # Independence Day
    "2026-10-02", # Mahatma Gandhi Jayanti
    "2026-10-20", # Dussehra / Vijay Dashmi
    "2026-11-08", # Diwali / Deepavali Peak Day
    "2026-11-09", "2026-11-11", # Govardhan Puja & Bhai Dooj 2026
    "2026-11-24", # Guru Nanak Jayanti
    "2026-12-24", "2026-12-25", "2026-12-31"  # Christmas & New Year's Eve 2026
]

# Convert calendar list into a high-performance hash set
holiday_set = set(pd.to_datetime(holiday_dates).date)

# Parse your dataset's sailing timestamps down to pure date objects
df['sailing_date_parsed'] = pd.to_datetime(df['sailing_date']).dt.date

# Row Evaluation mapping loop: 1 if matching any festival/holiday date, else 0
df['is_indian_holiday'] = df['sailing_date_parsed'].isin(holiday_set).astype(int)

# Clean up temporary lookup column before output file export
df = df.drop(columns=['sailing_date_parsed'])

# Save our unified master feature matrix
df.to_csv(output_path, index=False)

print(f"\n[SUCCESS] Phase 2 Extended Indian Holiday Mapping Completed!")
print(f"-> Total Dataset Records Evaluated: {len(df)}")
print(f"-> Expanded Festive & Holiday Sailings Flagged: {df['is_indian_holiday'].sum()}")
print(f"-> Master Core Feature Dataset Safely Exported to: {output_path}")
