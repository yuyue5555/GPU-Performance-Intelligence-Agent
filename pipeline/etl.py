"""
pipeline/etl.py
ETL pipeline: Extract → Transform → Load benchmark data.
Run this first to populate the database.
"""
import json
import sqlite3
from pathlib import Path

import pandas as pd

from pipeline.generator import generate_benchmark_records, seed_database, DB_PATH, DATA_DIR

METRICS = [
    "llm_inference_tokens_per_sec",
    "cnn_training_images_per_sec",
    "render_fps_4k",
    "memory_bandwidth_gbps",
]


# ─── Extract ──────────────────────────────────────────────────────────────────

def extract() -> list[dict]:
    """Generate or reload raw benchmark records."""
    print("🔍 [Extract] Generating synthetic benchmark data...")
    records = generate_benchmark_records(n_days=90)
    print(f"   {len(records)} raw records generated.")
    return records


# ─── Transform ────────────────────────────────────────────────────────────────

def transform(records: list[dict]) -> pd.DataFrame:
    """Clean, enrich, and compute derived metrics."""
    print("⚙️  [Transform] Cleaning and enriching records...")
    df = pd.DataFrame(records)
    df["run_date"] = pd.to_datetime(df["run_date"])

    # Derived: performance-per-watt
    df["llm_perf_per_watt"] = df["llm_inference_tokens_per_sec"] / df["tdp_w"]

    # Derived: performance-per-dollar (LLM focused)
    df["llm_perf_per_dollar"] = (
        df["llm_inference_tokens_per_sec"] / df["msrp_usd"] * 1000
    )

    # Regression flag: if latest run is >10% below gpu's 30-day avg, flag it
    df = df.sort_values("run_date")
    df["regression_flag"] = False
    for gpu_name, group in df.groupby("gpu_name"):
        for metric in METRICS:
            if metric not in group.columns:
                continue
            rolling_avg = group[metric].rolling(window=5, min_periods=1).mean().shift(1)
            regression = group[metric] < (rolling_avg * 0.90)
            df.loc[regression[regression].index, "regression_flag"] = True

    print(f"   Transformed {len(df)} records. Regressions flagged: {df['regression_flag'].sum()}")
    return df


# ─── Load ──────────────────────────────────────────────────────────────────────

def load(df: pd.DataFrame) -> None:
    """Load transformed data into SQLite and export summary JSON."""
    print("💾 [Load] Writing to database...")
    records = df.to_dict(orient="records")
    # Convert timestamps back to strings for SQLite
    for r in records:
        if hasattr(r["run_date"], "isoformat"):
            r["run_date"] = r["run_date"].isoformat()
    seed_database(records)

    # Export aggregated summary
    summary = []
    for gpu_name, group in df.groupby("gpu_name"):
        row = {"gpu_name": gpu_name, "tier": group["tier"].iloc[0],
               "vram_gb": int(group["vram_gb"].iloc[0]),
               "tdp_w": int(group["tdp_w"].iloc[0]),
               "msrp_usd": int(group["msrp_usd"].iloc[0])}
        for m in METRICS + ["llm_perf_per_watt", "llm_perf_per_dollar"]:
            if m in group.columns:
                row[f"{m}_avg"] = round(group[m].mean(), 2) if group[m].notna().any() else None
                row[f"{m}_latest"] = round(group.sort_values("run_date")[m].iloc[-1], 2) \
                    if group[m].notna().any() else None
        row["regression_count"] = int(group["regression_flag"].sum())
        summary.append(row)

    summary_path = DATA_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"✅ Summary written to {summary_path}")


# ─── Run pipeline ─────────────────────────────────────────────────────────────

def run_pipeline():
    DATA_DIR.mkdir(exist_ok=True)
    raw = extract()
    transformed = transform(raw)
    load(transformed)
    print("\n🎉 ETL pipeline complete!")


if __name__ == "__main__":
    run_pipeline()
