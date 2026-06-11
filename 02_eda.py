# ============================================================
# Credit Card Fraud Detection & Transaction Risk Analysis
# Script 2 of 2: Exploratory Data Analysis, Risk Scoring
#                & Power BI Export
# ============================================================
#
# What this script does:
#   1. Visualises the class imbalance and transaction patterns
#   2. Analyses amount distributions for fraud vs legitimate
#   3. Shows which PCA features best separate fraud from legit
#   4. Detects card-testing anomalies (micro transactions)
#   5. Builds a rule-based risk score (no ML needed)
#   6. Exports the scored dataset for Power BI
#
# STUDY NOTE — Why no machine learning here?
#   This is an analytics project, not a modelling project.
#   We use EDA (Exploratory Data Analysis) to understand the
#   data, find patterns, and build an interpretable risk score
#   based on the features most correlated with fraud.
#   This is exactly what a fraud risk analyst does day-to-day.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "figure.dpi": 130,
})

CLEAN_PATH = "../data/creditcard_clean.csv"
OUT_DIR    = "../outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Colours used consistently across all charts
FRAUD_RED  = "#E53935"   # red  — fraud / high risk
LEGIT_BLUE = "#1E88E5"   # blue — legitimate
PURPLE     = "#7B1FA2"   # purple — neutral/accent
AMBER      = "#F9A825"   # amber — highlight / peak value


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

print("=" * 55)
print("STEP 1 — Loading cleaned data")
print("=" * 55)

df    = pd.read_csv(CLEAN_PATH)
fraud = df[df["Class"] == 1].copy()   # only fraud rows
legit = df[df["Class"] == 0].copy()   # only legitimate rows

print(f"Total rows : {len(df):,}")
print(f"Fraud      : {len(fraud):,}  ({len(fraud)/len(df)*100:.3f}%)")
print(f"Legitimate : {len(legit):,}")


# ============================================================
# STEP 2: CHART 1 — EDA OVERVIEW (8 panels)
# ============================================================
# gridspec lets us create a grid layout of subplots with
# different sizes. Here we use a 3-row × 3-column grid.

print("\n" + "=" * 55)
print("STEP 2 — Building EDA overview chart")
print("=" * 55)

fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor("#F8F9FA")
fig.suptitle(
    "Credit Card Fraud Detection  ·  EDA Overview",
    fontsize=17, fontweight="bold", y=0.99, color="#1A1A2E"
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)


# --- Panel A: Class imbalance bar chart ---
# The most important thing to show first — how rare fraud is.
ax = fig.add_subplot(gs[0, 0])
counts = [len(legit), len(fraud)]
bars   = ax.bar(
    ["Legitimate", "Fraud"], counts,
    color=[LEGIT_BLUE, FRAUD_RED], edgecolor="white", linewidth=1.5, width=0.5
)
# Label each bar with its exact count
for bar, val in zip(bars, counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2000,
        f"{val:,}", ha="center", fontsize=10, fontweight="bold"
    )
ax.set_title("Transaction Class Distribution")
ax.set_ylabel("Count")
# Format y-axis as "284K" instead of "284000"
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))


# --- Panel B: Amount distribution (overlapping histograms) ---
# density=True normalises both to the same scale so we can
# compare their shapes even though one has 578x more rows.
ax = fig.add_subplot(gs[0, 1])
ax.hist(legit["Amount"].clip(upper=600), bins=60, alpha=0.65,
        color=LEGIT_BLUE, label="Legitimate", density=True)
ax.hist(fraud["Amount"].clip(upper=600), bins=40, alpha=0.70,
        color=FRAUD_RED,  label="Fraud",      density=True)
ax.set_xlabel("Transaction Amount (€)")
ax.set_ylabel("Density")
ax.set_title("Transaction Amount Distribution\n(clipped at €600 for readability)")
ax.legend()


# --- Panel C: How fraud is spread across amount buckets ---
ax = fig.add_subplot(gs[0, 2])
# Reindex ensures all 7 bins appear even if some have 0 fraud
fraud_bins = (
    fraud["Amount_Bin"]
    .value_counts()
    .reindex(["<10","10-50","50-100","100-250","250-500","500-1K",">1K"], fill_value=0)
)
bars = ax.bar(fraud_bins.index, fraud_bins.values,
              color=FRAUD_RED, alpha=0.85, edgecolor="white")
ax.set_title("Fraud Cases by Amount Range")
ax.set_xlabel("Amount Range (€)")
ax.set_ylabel("Number of Fraud Cases")
ax.tick_params(axis="x", rotation=30)
for bar in bars:
    if bar.get_height() > 0:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(int(bar.get_height())), ha="center", fontsize=8
        )


# --- Panel D: Hourly volume (dual axis) ---
# twinx() adds a second y-axis on the right side of the same plot.
# This lets us show two different scales (volume vs fraud count)
# on the same x-axis (hours).
ax = fig.add_subplot(gs[1, 0])
legit_by_hour = legit.groupby("Hour").size()
fraud_by_hour = fraud.groupby("Hour").size()

ax.fill_between(legit_by_hour.index, legit_by_hour.values,
                alpha=0.30, color=LEGIT_BLUE)
ax.plot(legit_by_hour.index, legit_by_hour.values,
        color=LEGIT_BLUE, linewidth=2, label="Legit")

ax_right = ax.twinx()
ax_right.plot(fraud_by_hour.index, fraud_by_hour.values,
              color=FRAUD_RED, linewidth=2.5, marker="o", markersize=4, label="Fraud")

ax.set_xlabel("Hour of Day")
ax.set_ylabel("Legit Volume", color=LEGIT_BLUE)
ax_right.set_ylabel("Fraud Count", color=FRAUD_RED)
ax.set_title("Transaction Volume by Hour")
lines1, l1 = ax.get_legend_handles_labels()
lines2, l2 = ax_right.get_legend_handles_labels()
ax.legend(lines1 + lines2, l1 + l2, loc="upper right", fontsize=8)


# --- Panel E: Fraud rate by time segment ---
ax = fig.add_subplot(gs[1, 1])
seg_order  = ["Morning", "Afternoon", "Evening", "Night"]
seg_rates  = df.groupby("Time_Segment")["Class"].mean() * 100
seg_rates  = seg_rates.reindex(seg_order)
# Highlight the peak segment in amber, the rest in purple
bar_colors = [AMBER if v == seg_rates.max() else PURPLE for v in seg_rates.values]
bars = ax.bar(seg_rates.index, seg_rates.values,
              color=bar_colors, alpha=0.85, edgecolor="white")
ax.set_ylabel("Fraud Rate (%)")
ax.set_title("Fraud Rate by Time of Day")
for bar, val in zip(bars, seg_rates.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.002,
        f"{val:.3f}%", ha="center", fontsize=9, fontweight="bold"
    )


# --- Panel F: Day 1 vs Day 2 ---
ax = fig.add_subplot(gs[1, 2])
day_stats = (
    df.groupby("Day")
    .agg(total=("Class","count"), fraud=("Class","sum"))
    .reset_index()
    .assign(fraud_rate=lambda x: x["fraud"] / x["total"] * 100)
)
x = np.arange(len(day_stats))
ax.bar(x, day_stats["total"] / 1000, color=LEGIT_BLUE, alpha=0.6, label="Total (K)")
ax_r = ax.twinx()
ax_r.plot(x, day_stats["fraud_rate"], color=FRAUD_RED,
          marker="D", linewidth=2.5, markersize=8, label="Fraud Rate %")
ax.set_xticks(x)
ax.set_xticklabels([f"Day {d}" for d in day_stats["Day"]])
ax.set_ylabel("Total Transactions (K)", color=LEGIT_BLUE)
ax_r.set_ylabel("Fraud Rate (%)", color=FRAUD_RED)
ax.set_title("Day 1 vs Day 2 Comparison")


# --- Panel G: Log-scale boxplot ---
# A boxplot shows the median (centre line), IQR (box), and
# whiskers (1.5x IQR). Dots beyond whiskers are outliers.
# Using the log-transformed amount makes the boxes readable.
ax = fig.add_subplot(gs[2, 0])
bp = ax.boxplot(
    [legit["Amount_Log"].values, fraud["Amount_Log"].values],
    labels=["Legitimate", "Fraud"],
    patch_artist=True,
    medianprops={"color": "black", "linewidth": 2}
)
bp["boxes"][0].set_facecolor(LEGIT_BLUE)
bp["boxes"][0].set_alpha(0.6)
bp["boxes"][1].set_facecolor(FRAUD_RED)
bp["boxes"][1].set_alpha(0.6)
ax.set_ylabel("log(Amount + 1)")
ax.set_title("Amount Distribution — Log Scale\n(Boxplot)")


# --- Panel H: Heatmap — fraud rate by Day × Hour ---
# A heatmap shows a 2D grid where colour intensity = value.
# Here darker red = higher fraud rate in that hour/day cell.
ax = fig.add_subplot(gs[2, 1:])
pivot_data = (
    df.groupby(["Day", "Hour"])["Class"].mean() * 100
)
pivot_df = pivot_data.reset_index().pivot(
    index="Day", columns="Hour", values="Class"
)
sns.heatmap(
    pivot_df, ax=ax,
    cmap="YlOrRd", annot=False,
    linewidths=0.3, linecolor="white",
    cbar_kws={"label": "Fraud Rate (%)", "shrink": 0.7}
)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Day")
ax.set_title("Fraud Rate Heatmap  (Day × Hour)")

plt.savefig(f"{OUT_DIR}/01_eda_overview.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("Saved: 01_eda_overview.png")


# ============================================================
# STEP 3: CHART 2 — RISK INDICATORS (6 panels)
# ============================================================
# The PCA features (V1–V28) are anonymised but we can still
# measure their correlation with fraud and plot their
# distributions to understand which ones matter most.

print("\n" + "=" * 55)
print("STEP 3 — Building risk indicators chart")
print("=" * 55)

fig2, axes = plt.subplots(2, 3, figsize=(18, 11))
fig2.patch.set_facecolor("#F8F9FA")
fig2.suptitle("Transaction Risk Indicators — Feature Analysis",
              fontsize=16, fontweight="bold", y=1.01, color="#1A1A2E")


# --- Panel A: V14 distribution ---
# V14 has the strongest negative correlation with fraud.
# When V14 is very negative, there is a much higher chance
# the transaction is fraudulent.
ax = axes[0, 0]
ax.hist(legit["V14"].clip(-15, 10), bins=60, alpha=0.60,
        color=LEGIT_BLUE, label="Legitimate", density=True)
ax.hist(fraud["V14"].clip(-15, 10), bins=40, alpha=0.70,
        color=FRAUD_RED,  label="Fraud",      density=True)
ax.set_xlabel("V14  (anonymised PCA feature)")
ax.set_ylabel("Density")
ax.set_title("V14 Distribution\n(Strongest fraud signal — most separation)")
ax.legend()


# --- Panel B: V3 distribution ---
ax = axes[0, 1]
ax.hist(legit["V3"].clip(-15, 10), bins=60, alpha=0.60,
        color=LEGIT_BLUE, density=True)
ax.hist(fraud["V3"].clip(-15, 10), bins=40, alpha=0.70,
        color=FRAUD_RED,  density=True)
ax.set_xlabel("V3  (anonymised PCA feature)")
ax.set_title("V3 Distribution\n(Second strongest fraud signal)")
ax.legend(["Legitimate", "Fraud"])


# --- Panel C: V10 distribution ---
ax = axes[0, 2]
ax.hist(legit["V10"].clip(-15, 10), bins=60, alpha=0.60,
        color=LEGIT_BLUE, density=True)
ax.hist(fraud["V10"].clip(-15, 10), bins=40, alpha=0.70,
        color=FRAUD_RED,  density=True)
ax.set_xlabel("V10  (anonymised PCA feature)")
ax.set_title("V10 Distribution\n(Third strongest fraud signal)")
ax.legend(["Legitimate", "Fraud"])


# --- Panel D: Correlation bar chart ---
# Pearson correlation measures how linearly related each feature
# is with the Class column.
#   -1 = perfect negative correlation (very fraud-predictive)
#    0 = no relationship
#   +1 = perfect positive correlation
ax = axes[1, 0]
pca_cols = [f"V{i}" for i in range(1, 29)] + ["Amount_Log"]
correlations = (
    df[pca_cols + ["Class"]]
    .corr()["Class"]
    .drop("Class")
    .sort_values()
)
# Show top 8 most negative and top 8 most positive
top_features = pd.concat([correlations.head(8), correlations.tail(8)])
bar_cols = [FRAUD_RED if v < 0 else LEGIT_BLUE for v in top_features.values]
ax.barh(top_features.index, top_features.values,
        color=bar_cols, alpha=0.85, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson Correlation with Class (Fraud)")
ax.set_title("Feature Correlations with Fraud\n(Red = fraud-associated, Blue = legit-associated)")


# --- Panel E: Summary statistics table ---
# ax.table() draws a formatted table inside a subplot.
# We turn off the axes (ticks, borders) so only the table shows.
ax = axes[1, 1]
ax.axis("off")
table_rows = [
    ["Metric",         "Legitimate (€)",              "Fraud (€)"],
    ["Mean Amount",    f"{legit.Amount.mean():.2f}",   f"{fraud.Amount.mean():.2f}"],
    ["Median Amount",  f"{legit.Amount.median():.2f}", f"{fraud.Amount.median():.2f}"],
    ["Max Amount",     f"{legit.Amount.max():.2f}",    f"{fraud.Amount.max():.2f}"],
    ["Std Deviation",  f"{legit.Amount.std():.2f}",    f"{fraud.Amount.std():.2f}"],
    ["Count",          f"{len(legit):,}",               f"{len(fraud):,}"],
]
tbl = ax.table(
    cellText=table_rows[1:], colLabels=table_rows[0],
    loc="center", cellLoc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.8)
# Style the header row
for col in range(3):
    tbl[0, col].set_facecolor("#1A1A2E")
    tbl[0, col].set_text_props(color="white", fontweight="bold")
# Alternate row shading
for row in range(1, len(table_rows)):
    for col in range(3):
        tbl[row, col].set_facecolor("#F0F4F8" if row % 2 == 0 else "white")
ax.set_title("Amount Statistics: Fraud vs Legitimate",
             pad=20, fontweight="bold")


# --- Panel F: Percentile comparison bar chart ---
# Percentiles split data into ranked segments.
# P50 = median (half above, half below).
# Comparing percentiles shows if fraud amounts tend to be
# higher or lower than legitimate at each percentile rank.
ax = axes[1, 2]
percentile_points = [10, 25, 50, 75, 90, 95, 99]
legit_pcts = [np.percentile(legit["Amount"], p) for p in percentile_points]
fraud_pcts = [np.percentile(fraud["Amount"], p) for p in percentile_points]
x = np.arange(len(percentile_points))
ax.bar(x - 0.2, legit_pcts, 0.38, label="Legitimate", color=LEGIT_BLUE, alpha=0.85)
ax.bar(x + 0.2, fraud_pcts, 0.38, label="Fraud",      color=FRAUD_RED,  alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f"P{p}" for p in percentile_points])
ax.set_ylabel("Amount (€)")
ax.set_title("Amount Percentile Comparison\n(How fraud amounts rank vs legitimate)")
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_risk_indicators.png", dpi=150,
            bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close()
print("Saved: 02_risk_indicators.png")


# ============================================================
# STEP 4: CHART 3 — ANOMALY PATTERNS (3 panels)
# ============================================================

print("\n" + "=" * 55)
print("STEP 4 — Building anomaly patterns chart")
print("=" * 55)

fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
fig3.patch.set_facecolor("#F8F9FA")
fig3.suptitle("Anomaly & Pattern Analysis",
              fontsize=16, fontweight="bold", y=1.02, color="#1A1A2E")


# --- Panel A: Micro-transaction fraud (card testing) ---
# A common fraud pattern is testing a stolen card with a tiny
# purchase first (under €10) to check it works before making
# a large fraudulent transaction. This is called "card testing".
ax = axes3[0]
micro_fraud = fraud[fraud["Amount"] < 10]
ax.hist(micro_fraud["Amount"], bins=25, color=FRAUD_RED,
        alpha=0.85, edgecolor="white")
ax.set_xlabel("Amount (€)")
ax.set_ylabel("Number of Transactions")
ax.set_title(
    f"Micro-Transaction Fraud  (Amount < €10)\n"
    f"{len(micro_fraud)} cases  —  "
    f"{len(micro_fraud)/len(fraud)*100:.1f}% of all fraud"
)


# --- Panel B: V14 vs V3 scatter (fraud cluster) ---
# Plotting two features against each other lets us see if fraud
# forms a cluster that is visually separate from legitimate txns.
# We sample 5,000 legit rows so the chart doesn't take forever.
ax = axes3[1]
sample_legit = legit.sample(5000, random_state=42)

ax.scatter(
    sample_legit["V14"].clip(-15, 10),
    sample_legit["V3"].clip(-15, 10),
    alpha=0.15, s=8, color=LEGIT_BLUE, label="Legitimate (sample)"
)
ax.scatter(
    fraud["V14"].clip(-15, 10),
    fraud["V3"].clip(-15, 10),
    alpha=0.60, s=20, color=FRAUD_RED, label="Fraud (all)", zorder=5
)
ax.set_xlabel("V14")
ax.set_ylabel("V3")
ax.set_title("V14 vs V3 Scatter Plot\n(Fraud clusters in bottom-left region)")
ax.legend()


# --- Panel C: Fraud rate over time (rolling average) ---
# We bucket transactions into 1-hour windows and calculate the
# fraud rate in each bucket. Then we smooth it with a 3-period
# rolling average to reduce noise and show the trend.
ax = axes3[2]
df_sorted = df.sort_values("Time").copy()
df_sorted["Time_Bucket"] = (df_sorted["Time"] // 3600).astype(int)

bucket_stats  = df_sorted.groupby("Time_Bucket")["Class"].agg(["sum", "count"])
bucket_rate   = (bucket_stats["sum"] / bucket_stats["count"] * 100)
smoothed_rate = bucket_rate.rolling(window=3, min_periods=1).mean()

ax.fill_between(smoothed_rate.index, smoothed_rate.values,
                alpha=0.30, color=FRAUD_RED)
ax.plot(smoothed_rate.index, smoothed_rate.values,
        color=FRAUD_RED, linewidth=1.8)
ax.set_xlabel("Hours Elapsed from First Transaction")
ax.set_ylabel("Fraud Rate (%)  — 3-Hour Rolling Avg")
ax.set_title("Fraud Rate Over Time\n(Rolling average smooths hourly noise)")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_anomaly_patterns.png", dpi=150,
            bbox_inches="tight", facecolor=fig3.get_facecolor())
plt.close()
print("Saved: 03_anomaly_patterns.png")


# ============================================================
# STEP 5: RISK SCORING
# ============================================================
# We build a simple, interpretable risk score using the features
# most correlated with fraud. No machine learning required —
# just weighted feature signals.
#
# HOW IT WORKS:
#   1. For each fraud-associated feature (V14, V12, V10, V3, V17),
#      we compute how far the value deviates below the mean
#      (fraud tends to have very negative values on these features)
#   2. We normalise each deviation by the standard deviation
#      so all features are on the same scale (z-scores)
#   3. We apply weights based on correlation strength:
#         V14 → 30%  (strongest signal)
#         V12 → 25%
#         V10 → 20%
#         V3  → 15%
#         V17 → 10%
#   4. We normalise the final score to 0–1 range
#   5. We assign a risk tier based on the score

print("\n" + "=" * 55)
print("STEP 5 — Building risk scores")
print("=" * 55)

# z-score: how many standard deviations away from the mean
# A more negative V14 than average → higher z → higher risk
def compute_risk_contribution(series):
    mean = series.mean()
    std  = series.std()
    z    = (series - mean) / std
    return -z    # flip sign: more negative original → higher score

v14_signal = compute_risk_contribution(df["V14"])
v12_signal = compute_risk_contribution(df["V12"])
v10_signal = compute_risk_contribution(df["V10"])
v3_signal  = compute_risk_contribution(df["V3"])
v17_signal = compute_risk_contribution(df["V17"])

# Weighted sum
raw_score = (
    0.30 * v14_signal +
    0.25 * v12_signal +
    0.20 * v10_signal +
    0.15 * v3_signal  +
    0.10 * v17_signal
)

# Normalise to 0–1
score_min = raw_score.min()
score_max = raw_score.max()
df["Risk_Score"] = ((raw_score - score_min) / (score_max - score_min)).round(4)

# Assign risk tier
def assign_risk_tier(score):
    if score >= 0.80:  return "High Risk"
    if score >= 0.60:  return "Medium Risk"
    return "Low Risk"

df["Risk_Tier"] = df["Risk_Score"].apply(assign_risk_tier)

# How well does the risk score separate actual fraud?
print("\nRisk tier breakdown:")
print(df["Risk_Tier"].value_counts())

print("\nFraud captured by risk tier:")
fraud_by_tier = df[df["Class"] == 1]["Risk_Tier"].value_counts()
print(fraud_by_tier)

total_fraud_count = df["Class"].sum()
high_risk_fraud   = (df["Risk_Tier"] == "High Risk") & (df["Class"] == 1)
print(f"\nDetection: {high_risk_fraud.sum()} of {total_fraud_count} fraud cases "
      f"flagged as High Risk "
      f"({high_risk_fraud.sum()/total_fraud_count*100:.1f}%)")


# ============================================================
# STEP 6: EXPORT FOR POWER BI
# ============================================================
# We select the columns that are useful for Power BI dashboards
# and export them to a CSV. Power BI can then import this file
# and build visuals directly without running any Python.

print("\n" + "=" * 55)
print("STEP 6 — Exporting for Power BI")
print("=" * 55)

export_columns = [
    "Time", "Hour", "Day", "Time_Segment",
    "Amount", "Amount_Log", "Amount_Bin",
    "V3", "V10", "V12", "V14", "V17",
    "Class", "Fraud_Label",
    "Risk_Score", "Risk_Tier"
]

export_df = df[export_columns].copy()
export_df.to_csv("../data/creditcard_powerbi.csv", index=False)

print(f"File saved : ../data/creditcard_powerbi.csv")
print(f"Rows       : {len(export_df):,}")
print(f"Columns    : {len(export_df.columns)}")


# ============================================================
# STEP 7: FINAL SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("PROJECT SUMMARY")
print("=" * 55)
print(f"Total transactions analysed : {len(df):,}")
print(f"Fraud cases identified       : {df['Class'].sum():,}  ({df['Class'].mean()*100:.3f}%)")
print(f"Legitimate transactions      : {(df['Class']==0).sum():,}")
print(f"Average fraud amount         : €{fraud['Amount'].mean():.2f}")
print(f"Average legitimate amount    : €{legit['Amount'].mean():.2f}")
print(f"Peak fraud time segment      : {df.groupby('Time_Segment')['Class'].mean().idxmax()}")
print(f"Card testing cases (<€10)    : {(fraud['Amount']<10).sum()}")
print(f"High Risk alerts flagged     : {(df['Risk_Tier']=='High Risk').sum():,}")
print(f"Fraud caught in High Risk    : {high_risk_fraud.sum()} / {total_fraud_count}")
print(f"\nCharts saved to : {OUT_DIR}/")
print(f"Power BI file   : ../data/creditcard_powerbi.csv")
