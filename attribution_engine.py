"""
Marketing Attribution Analysis Engine
======================================
Project: Comparative Analysis of Marketing Attribution Models
Dataset: marketing_campaign_performance_10000.csv
Author:  Data Analytics Project
Models:  First-Touch, Last-Touch, Linear, Time-Decay, Position-Based
Extras:  ML Segmentation (K-Means), Data-Driven Attribution (Logistic Regression)
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
#  1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """Load CSV, clean, and engineer features."""
    df = pd.read_csv(filepath)

    # Normalize channel names (strip whitespace, title-case)
    df["Channel"] = df["Channel"].str.strip().str.title()

    # Parse dates
    df["StartDate"] = pd.to_datetime(df["StartDate"])
    df["EndDate"]   = pd.to_datetime(df["EndDate"])

    # Remove exact duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[Preprocessing] Removed {before - len(df)} duplicate rows.")

    # Fill missing numeric values with column medians
    num_cols = ["Impressions", "Clicks", "Leads", "Conversions", "Cost_USD", "Revenue_USD", "ROI"]
    for col in num_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Derived metrics
    df["Duration_Days"] = (df["EndDate"] - df["StartDate"]).dt.days + 1
    df["CTR"]           = (df["Clicks"] / df["Impressions"] * 100).round(4)
    df["CVR"]           = (df["Conversions"] / df["Clicks"]  * 100).round(4)
    df["CPC"]           = (df["Cost_USD"]  / df["Clicks"]).round(4)
    df["CPL"]           = (df["Cost_USD"]  / df["Leads"]).round(4)
    df["CPA"]           = (df["Cost_USD"]  / df["Conversions"]).round(4)
    df["ROAS"]          = (df["Revenue_USD"] / df["Cost_USD"]).round(4)
    df["Profit"]        = (df["Revenue_USD"] - df["Cost_USD"]).round(2)

    # Sort by StartDate for journey ordering
    df.sort_values("StartDate", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[Preprocessing] Dataset shape after cleaning: {df.shape}")
    print(f"[Preprocessing] Channels found: {sorted(df['Channel'].unique().tolist())}")
    return df


# ─────────────────────────────────────────────
#  2. CUSTOMER JOURNEY SIMULATION
# ─────────────────────────────────────────────

def simulate_journeys(df: pd.DataFrame, n_journeys: int = 5000) -> list:
    """
    Since the dataset is campaign-level (not user-level), simulate
    multi-touch customer journeys using channel weight distributions.
    Each journey = list of channel touchpoints, converted flag, revenue.
    """
    channels = df["Channel"].unique().tolist()
    weights_series = df.groupby("Channel")["Conversions"].sum()
    weights = (weights_series / weights_series.sum()).reindex(channels).values

    journeys = []
    for i in range(n_journeys):
        n_touches = np.random.choice([1, 2, 3, 4, 5, 6],
                                     p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05])
        touchpoints = np.random.choice(channels, size=n_touches, p=weights).tolist()
        converted   = int(np.random.choice([0, 1], p=[0.30, 0.70]))
        revenue     = round(float(np.random.uniform(50, 500)), 2) if converted else 0.0
        journeys.append({
            "journey_id":  f"J{i:05d}",
            "touchpoints": touchpoints,
            "n_touches":   n_touches,
            "converted":   converted,
            "revenue":     revenue,
        })

    converted_count = sum(j["converted"] for j in journeys)
    print(f"[Journeys] Simulated {n_journeys} journeys · {converted_count} converted "
          f"({converted_count/n_journeys*100:.1f}%)")
    return journeys


# ─────────────────────────────────────────────
#  3. ATTRIBUTION ENGINE
# ─────────────────────────────────────────────

class AttributionEngine:
    """Implements 5 standard marketing attribution models."""

    def __init__(self, channels: list):
        self.channels = channels

    def _empty(self) -> dict:
        return {c: 0.0 for c in self.channels}

    # --- 3a. First-Touch ---
    def first_touch(self, journey: dict) -> dict:
        """100% credit to the first touchpoint."""
        credits = self._empty()
        if journey["converted"]:
            credits[journey["touchpoints"][0]] += journey["revenue"]
        return credits

    # --- 3b. Last-Touch ---
    def last_touch(self, journey: dict) -> dict:
        """100% credit to the last touchpoint."""
        credits = self._empty()
        if journey["converted"]:
            credits[journey["touchpoints"][-1]] += journey["revenue"]
        return credits

    # --- 3c. Linear ---
    def linear(self, journey: dict) -> dict:
        """Equal credit distributed across all touchpoints."""
        credits = self._empty()
        if journey["converted"]:
            share = journey["revenue"] / journey["n_touches"]
            for t in journey["touchpoints"]:
                credits[t] += share
        return credits

    # --- 3d. Time Decay ---
    def time_decay(self, journey: dict) -> dict:
        """Exponentially increasing credit toward the conversion event."""
        credits = self._empty()
        if journey["converted"]:
            n = journey["n_touches"]
            raw_weights = np.array([2 ** (i / (n - 1)) if n > 1 else 1.0 for i in range(n)])
            weights = raw_weights / raw_weights.sum()
            for t, w in zip(journey["touchpoints"], weights):
                credits[t] += journey["revenue"] * w
        return credits

    # --- 3e. Position-Based (U-Shaped) ---
    def position_based(self, journey: dict) -> dict:
        """40% first touch, 40% last touch, 20% shared among middle touches."""
        credits = self._empty()
        if journey["converted"]:
            n = journey["n_touches"]
            tp = journey["touchpoints"]
            rev = journey["revenue"]
            if n == 1:
                credits[tp[0]] += rev
            elif n == 2:
                credits[tp[0]]  += rev * 0.4
                credits[tp[-1]] += rev * 0.4
                # remaining 20% split between them
                credits[tp[0]]  += rev * 0.1
                credits[tp[-1]] += rev * 0.1
            else:
                mid_share = 0.20 / (n - 2)
                credits[tp[0]]  += rev * 0.40
                credits[tp[-1]] += rev * 0.40
                for t in tp[1:-1]:
                    credits[t] += rev * mid_share
        return credits

    def run_all(self, journeys: list) -> dict:
        """Run all models over all journeys and aggregate results."""
        model_funcs = {
            "First Touch":     self.first_touch,
            "Last Touch":      self.last_touch,
            "Linear":          self.linear,
            "Time Decay":      self.time_decay,
            "Position Based":  self.position_based,
        }
        results = {}
        for name, fn in model_funcs.items():
            totals = {c: 0.0 for c in self.channels}
            for j in journeys:
                cr = fn(j)
                for c in self.channels:
                    totals[c] += cr[c]
            total_rev = sum(totals.values())
            pct = {c: round(v / total_rev * 100, 2) if total_rev > 0 else 0
                   for c, v in totals.items()}
            results[name] = {
                "revenue":    {c: round(v, 2) for c, v in totals.items()},
                "percentage": pct,
                "total":      round(total_rev, 2),
            }
        return results


# ─────────────────────────────────────────────
#  4. ML SEGMENTATION
# ─────────────────────────────────────────────

def segment_campaigns(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """K-Means clustering on campaign performance metrics."""
    features = ["CTR", "CVR", "ROI", "ROAS", "CPA"]
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, df["Cluster"])
    print(f"[ML Segmentation] Silhouette score: {sil:.4f}")

    cluster_summary = (df.groupby("Cluster")
                         .agg(Count=("CampaignID", "count"),
                              Avg_CTR=("CTR", "mean"),
                              Avg_CVR=("CVR", "mean"),
                              Avg_ROI=("ROI", "mean"),
                              Avg_ROAS=("ROAS", "mean"),
                              Avg_CPA=("CPA", "mean"),
                              Avg_Revenue=("Revenue_USD", "mean"),
                              Avg_Cost=("Cost_USD", "mean"))
                         .round(3)
                         .reset_index())

    def label(row):
        if row["Avg_ROI"] > 1.2 and row["Avg_CVR"] > 11:
            return "High Performers"
        elif row["Avg_ROAS"] > 1.8:
            return "Steady Growth"
        elif row["Avg_CPA"] < cluster_summary["Avg_CPA"].median():
            return "Cost Efficient"
        else:
            return "Underperformers"

    cluster_summary["Label"] = cluster_summary.apply(label, axis=1)
    print("[ML Segmentation] Cluster summary:")
    print(cluster_summary[["Cluster", "Label", "Count", "Avg_ROI", "Avg_CVR", "Avg_ROAS"]])
    return df, cluster_summary


# ─────────────────────────────────────────────
#  5. DATA-DRIVEN ATTRIBUTION (ML)
# ─────────────────────────────────────────────

def ml_attribution(df: pd.DataFrame) -> dict:
    """
    Logistic Regression to identify which performance metrics
    best distinguish channels — proxy for data-driven attribution.
    """
    feat_cols = ["CTR", "CVR", "CPC", "ROAS", "ROI", "Impressions", "Clicks"]
    X = df[feat_cols].fillna(0)
    le = LabelEncoder()
    y = le.fit_transform(df["Channel"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, y)

    importance = np.abs(lr.coef_).mean(axis=0)
    result = dict(zip(feat_cols, importance.round(5)))
    result = dict(sorted(result.items(), key=lambda x: -x[1]))
    print("[ML Attribution] Feature importance:")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
    return result


# ─────────────────────────────────────────────
#  6. REPORTING & COMPARISON
# ─────────────────────────────────────────────

def print_attribution_report(attribution_results: dict, channels: list):
    """Print a formatted comparison table."""
    sep = "─" * 70
    print(f"\n{sep}")
    print("  ATTRIBUTION MODEL COMPARISON REPORT")
    print(sep)
    header = f"{'Channel':<14}" + "".join(f"{m[:10]:>12}" for m in attribution_results)
    print(header)
    print("─" * len(header))
    for ch in channels:
        row = f"{ch:<14}"
        for m, data in attribution_results.items():
            p = data["percentage"][ch]
            row += f"{p:>11.2f}%"
        print(row)
    print(sep)
    print("  * Values show % of total conversion credit assigned to each channel.")
    print(sep + "\n")


def channel_performance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate channel KPIs."""
    summary = (df.groupby("Channel")
                 .agg(Campaigns=("CampaignID", "count"),
                      Total_Revenue=("Revenue_USD", "sum"),
                      Total_Cost=("Cost_USD", "sum"),
                      Total_Conversions=("Conversions", "sum"),
                      Total_Clicks=("Clicks", "sum"),
                      Total_Impressions=("Impressions", "sum"),
                      Avg_ROI=("ROI", "mean"),
                      Avg_CTR=("CTR", "mean"),
                      Avg_CVR=("CVR", "mean"))
                 .round(3)
                 .reset_index())
    summary["ROAS"]       = (summary["Total_Revenue"] / summary["Total_Cost"]).round(3)
    summary["CPA"]        = (summary["Total_Cost"] / summary["Total_Conversions"]).round(4)
    summary["Profit"]     = (summary["Total_Revenue"] - summary["Total_Cost"]).round(2)
    summary["ROI_Pct"]    = ((summary["Total_Revenue"] - summary["Total_Cost"])
                              / summary["Total_Cost"] * 100).round(2)
    return summary


def top_conversion_paths(journeys: list, top_n: int = 10) -> list:
    """Find the most frequent multi-channel conversion paths."""
    counter = Counter()
    for j in journeys:
        if j["converted"]:
            path = " → ".join(j["touchpoints"])
            counter[path] += 1
    return [{"path": p, "count": c} for p, c in counter.most_common(top_n)]


# ─────────────────────────────────────────────
#  7. MAIN PIPELINE
# ─────────────────────────────────────────────

def main(data_path: str = "marketing_campaign_performance_10000.csv"):
    print("\n" + "═" * 70)
    print("  MARKETING ATTRIBUTION ANALYSIS — FULL PIPELINE")
    print("═" * 70 + "\n")

    # Step 1 — Load
    df = load_and_preprocess(data_path)

    # Step 2 — Channel summary
    ch_summary = channel_performance_summary(df)
    print("\n[Channel Summary]\n", ch_summary[["Channel","Total_Revenue","ROAS","CPA","ROI_Pct"]].to_string(index=False))

    # Step 3 — Journeys
    journeys = simulate_journeys(df, n_journeys=5000)
    channels = df["Channel"].unique().tolist()

    # Step 4 — Attribution models
    engine  = AttributionEngine(channels)
    results = engine.run_all(journeys)
    print_attribution_report(results, channels)

    # Step 5 — Top paths
    paths = top_conversion_paths(journeys, top_n=10)
    print("[Top Conversion Paths]")
    for p in paths[:5]:
        print(f"  {p['path']}: {p['count']} journeys")

    # Step 6 — ML segmentation
    df, cluster_summary = segment_campaigns(df, k=4)

    # Step 7 — Data-driven attribution
    ml_feats = ml_attribution(df)

    # Step 8 — Save outputs
    output = {
        "channel_summary":      ch_summary.to_dict("records"),
        "attribution_results":  results,
        "top_conversion_paths": paths,
        "cluster_summary":      cluster_summary.to_dict("records"),
        "ml_feature_importance":ml_feats,
    }
    out_path = Path(data_path).stem + "_attribution_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✅ Results saved to: {out_path}")

    df.to_csv(Path(data_path).stem + "_processed.csv", index=False)
    print(f"✅ Processed dataset saved.")
    print("\n" + "═" * 70 + "\n")
    return df, results, cluster_summary


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "marketing_campaign_performance_10000.csv"
    main(path)
