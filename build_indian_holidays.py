import pandas as pd
import os

input_path = r"C:\Users\S. Aditya\sales-ai\features\processed_features.csv"
output_path = r"C:\Users\S. Aditya\sales-ai\features\final_engineered_features.csv"

print(f"[PROCESS] Reading baseline feature matrix from: {input_path}")
df = pd.read_csv(input_path)

print("[FEATURE 3] Extracting Comprehensive Indian Holiday & Festival Peak Matrix (2022-2025)...")

# --- MASTER EXHAUSTIVE INDIAN FESTIVAL & HOLIDAY CALENDAR MATRIX ---
holiday_dates = [
    # === 2022 HOLIDAYS & LONG WEEKENDS ===
    "2022-01-26", # Republic Day
    "2022-03-18", # Holi Long Weekend
    "2022-04-15", # Good Friday Long Weekend
    "2022-05-03", # Eid-ul-Fitr
    "2022-08-15", # Independence Day (Long Weekend)
    "2022-08-31", # Ganesh Chaturthi
    "2022-10-05", # Dussehra
    "2022-10-24", # Diwali Peak
    "2022-10-25", # Diwali Extended
    "2022-11-08", # Guru Nanak Jayanti
    "2022-12-24", "2022-12-25", "2022-12-31", # Christmas & New Year Week

    # === 2023 HOLIDAYS & LONG WEEKENDS ===
    "2023-01-26", # Republic Day
    "2023-03-08", # Holi
    "2023-04-07", # Good Friday Long Weekend
    "2023-04-22", # Eid-ul-Fitr Long Weekend
    "2023-06-29", # Bakrid Long Weekend
    "2023-08-15", # Independence Day Mid-Week Peak
    "2023-09-19", # Ganesh Chaturthi
    "2023-10-02", # Gandhi Jayanti (Long Weekend)
    "2023-10-24", # Dussehra
    "2023-11-12", # Diwali Peak
    "2023-11-13", # Diwali Extended
    "2023-12-24", "2023-12-25", "2023-12-31", # Christmas & New Year Week

    # === 2024 HOLIDAYS & LONG WEEKENDS ===
    "2024-01-26", # Republic Day (Long Weekend)
    "2024-03-25", # Holi (Long Weekend)
    "2024-03-29", # Good Friday Long Weekend
    "2024-04-11", # Eid-ul-Fitr
    "2024-06-17", # Bakrid Long Weekend
    "2024-08-15", # Independence Day Extended Peak
    "2024-09-07", # Ganesh Chaturthi (Long Weekend)
    "2024-10-02", # Gandhi Jayanti
    "2024-10-12", # Dussehra
    "2024-11-01", # Diwali Peak
    "2024-11-02", # Diwali Extended
    "2024-12-24", "2024-12-25", "2024-12-31", # Christmas & New Year Week

    # === 2025 HOLIDAYS & LONG WEEKENDS ===
    "2025-01-26", # Republic Day
    "2025-03-14", # Holi Long Weekend
    "2025-04-18", # Good Friday Long Weekend
    "2025-10-02", # Gandhi Jayanti Long Weekend
    "2025-10-20"  # Diwali / Festive Peak window kickoff
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
