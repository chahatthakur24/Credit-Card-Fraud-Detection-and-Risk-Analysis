# ============================================================
# Credit Card Fraud Detection & Transaction Risk Analysis
# Script 1 of 2: Data Loading, Cleaning & Feature Engineering
# ============================================================
#
# Dataset  : ULB Credit Card Fraud Detection (Kaggle)
# Link     : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Rows     : 284,807 transactions over 2 days (September 2013)
# Fraud    : 492 cases (0.17%) — highly imbalanced dataset
#
# What this script does:
#   1. Loads the raw CSV and inspects its shape and columns
#   2. Checks for missing values and duplicate rows
#   3. Engineers new features from Time and Amount
#   4. Prints a basic statistical summary
#   5. Saves the cleaned file for use in Script 2
#
# STUDY NOTE:
#   The dataset columns V1 to V28 are PCA-transformed (Principal
#   Component Analysis). The bank anonymised the original features
#   for privacy. We cannot name them, but we can still analyse
#   which ones separate fraud from legitimate transactions.
# ============================================================

import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

# ---------- file paths ----------
DATA_PATH  = "../data/creditcard.csv"
CLEAN_PATH = "../data/creditcard_clean.csv"
os.makedirs("../data", exist_ok=True)


# ============================================================
# STEP 1: LOAD THE DATA
# ============================================================
# pd.read_csv reads the CSV file into a DataFrame.
# A DataFrame is a table-like structure with rows and columns.

print("=" * 55)
print("STEP 1 — Loading the dataset")
print("=" * 55)

df = pd.read_csv(DATA_PATH)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")
print(f"\nColumn names:\n{list(df.columns)}")
print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")

# Class column: 0 = legitimate, 1 = fraud
total_fraud = df["Class"].sum()
total_legit = (df["Class"] == 0).sum()
fraud_pct   = df["Class"].mean() * 100

print(f"\nFraud transactions : {total_fraud:,}  ({fraud_pct:.3f}%)")
print(f"Legit transactions : {total_legit:,}")


# ============================================================
# STEP 2: DATA QUALITY CHECKS
# ============================================================
# Before any analysis, always verify the data is clean.
# Two common problems: missing values and duplicate rows.

print("\n" + "=" * 55)
print("STEP 2 — Data Quality Checks")
print("=" * 55)

# --- Missing values ---
# isnull() returns True wherever a cell is empty (NaN).
# sum() counts those Trues per column.
missing = df.isnull().sum()
print(f"\nMissing values per column:")
if missing.sum() == 0:
    print("  None found — dataset is complete.")
else:
    print(missing[missing > 0])

# --- Duplicate rows ---
# duplicated() marks rows that are exact copies of an earlier row.
dup_count = df.duplicated().sum()
print(f"\nDuplicate rows: {dup_count}")

if dup_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    # reset_index(drop=True) gives the cleaned dataframe fresh 0-based row numbers
    print(f"  Removed duplicates. New row count: {df.shape[0]:,}")
else:
    print("  No duplicates — no rows removed.")

# --- Data types ---
print(f"\nData types:\n{df.dtypes}")


# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================
# Feature engineering means creating new, more useful columns
# from existing ones. Here we extract time-based features from
# the raw 'Time' column (which is just seconds elapsed).

print("\n" + "=" * 55)
print("STEP 3 — Feature Engineering")
print("=" * 55)

# --- Hour of day ---
# Time is in seconds since the first transaction.
# Dividing by 3600 gives hours. The % 24 wraps it into 0–23.
# This tells us what hour of day each transaction happened.
df["Hour"] = (df["Time"] // 3600 % 24).astype(int)
print("Created 'Hour' — hour of day (0 to 23)")

# --- Day number (1 or 2) ---
# The dataset covers exactly 2 days. Dividing by 86400 (seconds
# in a day) tells us which day a transaction belongs to.
df["Day"] = (df["Time"] // 86400 + 1).astype(int).clip(upper=2)
print("Created 'Day'  — Day 1 or Day 2 of the observation window")

# --- Log-transformed Amount ---
# The Amount column is very skewed (most txns are small, a few
# are very large). np.log1p applies log(x + 1) which compresses
# large values and makes the distribution easier to visualise.
# We use +1 so that log(0) doesn't cause an error.
df["Amount_Log"] = np.log1p(df["Amount"]).round(6)
print("Created 'Amount_Log' — log(Amount + 1) to reduce skew")

# --- Amount bins (buckets) ---
# pd.cut divides the Amount column into labelled ranges.
# This helps us see which amount brackets have more fraud.
bins   = [0, 10, 50, 100, 250, 500, 1000, df["Amount"].max() + 1]
labels = ["<10", "10-50", "50-100", "100-250", "250-500", "500-1K", ">1K"]
df["Amount_Bin"] = pd.cut(df["Amount"], bins=bins, labels=labels, right=False)
print("Created 'Amount_Bin' — transaction amount range category")

# --- Time segment ---
# Converts the hour into a readable time-of-day label.
def get_time_segment(hour):
    if 6  <= hour < 12:  return "Morning"
    if 12 <= hour < 18:  return "Afternoon"
    if 18 <= hour < 22:  return "Evening"
    return "Night"

df["Time_Segment"] = df["Hour"].apply(get_time_segment)
print("Created 'Time_Segment' — Morning / Afternoon / Evening / Night")

# --- Readable fraud label ---
# Maps 0/1 to readable text for charts and Power BI.
df["Fraud_Label"] = df["Class"].map({0: "Legitimate", 1: "Fraud"})
print("Created 'Fraud_Label' — 'Legitimate' or 'Fraud'")

# Preview the new columns
print(f"\nSample of engineered columns (first 5 rows):")
print(df[["Time", "Hour", "Day", "Amount", "Amount_Bin", "Time_Segment", "Class"]].head(5).to_string())


# ============================================================
# STEP 4: STATISTICAL SUMMARY
# ============================================================
# Now we compare fraud vs legitimate across key measures.
# This helps us understand patterns before we visualise.

print("\n" + "=" * 55)
print("STEP 4 — Statistical Summary")
print("=" * 55)

# --- Amount stats by class ---
# groupby groups rows by Fraud_Label, then describe() computes
# count, mean, std, min, quartiles, max for each group.
print("\nTransaction Amount — by class:")
print(df.groupby("Fraud_Label")["Amount"].describe().round(2))

# --- Fraud rate by time segment ---
# For each segment, count total txns, sum up fraud cases,
# and divide to get the fraud rate.
print("\nFraud rate by time segment:")
seg_summary = (
    df.groupby("Time_Segment")["Class"]
    .agg(total="count", fraud_cases="sum")
    .assign(fraud_rate_pct=lambda x: (x["fraud_cases"] / x["total"] * 100).round(4))
    .sort_values("fraud_rate_pct", ascending=False)
)
print(seg_summary)

# --- Amount distribution for fraud transactions ---
fraud_df = df[df["Class"] == 1]
print(f"\nFraud amount breakdown:")
print(f"  Minimum  : €{fraud_df['Amount'].min():.2f}")
print(f"  Median   : €{fraud_df['Amount'].median():.2f}")
print(f"  Mean     : €{fraud_df['Amount'].mean():.2f}")
print(f"  Maximum  : €{fraud_df['Amount'].max():.2f}")
print(f"  Under €10 (card testing) : {(fraud_df['Amount'] < 10).sum()} cases")


# ============================================================
# STEP 5: SAVE CLEANED DATA
# ============================================================

df.to_csv(CLEAN_PATH, index=False)

print("\n" + "=" * 55)
print("STEP 5 — Saved")
print("=" * 55)
print(f"Clean file : {CLEAN_PATH}")
print(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"New columns: Hour, Day, Amount_Log, Amount_Bin, Time_Segment, Fraud_Label")
print("\nRun script 02_eda.py next.")
