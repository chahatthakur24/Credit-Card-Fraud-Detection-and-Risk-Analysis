# Credit Card Fraud Detection & Transaction Risk Analysis

**Tools:** Python · PostgreSQL · Power BI  
**Dataset:** ULB Machine Learning Group — [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**Transactions:** 284,807 over 48 hours · 492 fraud cases (0.17%)

---

## About the Dataset

The dataset contains credit card transactions made by European cardholders over two days in September 2013. It was compiled by the Machine Learning Group at Université Libre de Bruxelles (ULB). Features V1 through V28 are PCA-transformed to protect cardholder identity. The only non-transformed features are `Time` (seconds elapsed), `Amount`, and `Class` (1 = fraud, 0 = legitimate).



## What Was Done

### 1. Data Cleaning & Feature Engineering (`01_data_cleaning.py`)
- Checked for missing values (none found) and duplicates (none found)
- Converted `Time` (seconds) → `Hour` and `Day` features
- Created `Time_Segment` (Morning / Afternoon / Evening / Night)
- Applied log transformation to `Amount` to reduce skew
- Bucketed transaction amounts into 7 ranges for categorical analysis

### 2. Exploratory Data Analysis (`02_eda.py`)
Key findings from the EDA:
- **Evening (6pm–10pm)** has the highest fraud rate at 0.192%
- Fraud transactions have a **higher average amount (€113)** vs legitimate (€88)
- **38 fraud cases under €10** — a classic card testing / carding pattern
- V14 shows the clearest separation between fraud and legitimate distributions — the most discriminative PCA feature
- Fraud is spread across both days but shows localized hourly spikes

### 3. Risk Scoring 
Built an interpretable rule-based risk score using the top PCA features most correlated with fraud (V14, V12, V10, V3, V17) — weighted by their Pearson correlation with the fraud label. This produces a 0–1 risk score per transaction and assigns each to a risk tier:

| Risk Tier | Threshold | Transactions | Fraud Captured |
|-----------|-----------|-------------|----------------|
| High Risk | Score ≥ 0.80 | 477 | 477 (97%) |
| Medium Risk | Score 0.60–0.80 | 15 | 15 (3%) |
| Low Risk | Score < 0.60 | 284,315 | 0 |

### 4. SQL Analysis (`fraud_analysis.sql`)
17 analytical queries covering:
- Dataset overview & data quality checks
- Fraud rate by time segment, hour, day, and amount range
- High-value transaction analysis (90th percentile)
- Risk tier validation against actual fraud labels
- Card testing anomaly detection (micro-transactions)
- Statistical outlier identification (z-score > 3)
- Feature-based anomaly flagging using V14 thresholds
- Financial impact: total fraud loss, exposure by segment
- Three exportable views for Power BI consumption

### 5. Power BI Dashboard (4 pages)
- **Executive Overview** — KPI cards, class distribution, fraud rate summary
- **Fraud Risk Analysis** — breakdown by time, hour, day, amount range
- **Risk Scoring & Anomalies** — risk tier distribution, V14 scatter, high-risk table
- **Financial Impact** — volume vs fraud loss, segment-level exposure

---

## Key Insights

- Only 0.17% of transactions are fraudulent — severe class imbalance that requires careful analysis
- Despite low volume, fraud accounts for a disproportionate share of high-value transactions
- Evening transactions carry the highest fraud rate; Night the lowest despite common assumptions
- Card testing behaviour (sub-€10 fraud) represents 7.7% of all fraud cases
- V14 below -7 corresponds to a 40%+ fraud concentration — a strong rule-based signal
- 97% of actual fraud cases fall into the High Risk tier using the weighted feature score

---



---

