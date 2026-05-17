"""
pipeline/generator.py
Generates realistic synthetic GPU benchmark data.
Simulates workloads: LLM inference, CNN training, image rendering.
"""
import json
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "benchmarks.db"

GPUS = [
    {"name": "NVIDIA H100 80GB", "tier": "datacenter", "vram_gb": 80,  "tdp_w": 700, "msrp_usd": 30000},
    {"name": "NVIDIA A100 80GB", "tier": "datacenter", "vram_gb": 80,  "tdp_w": 400, "msrp_usd": 15000},
    {"name": "NVIDIA RTX 4090",  "tier": "consumer",   "vram_gb": 24,  "tdp_w": 450, "msrp_usd": 1599},
    {"name": "NVIDIA RTX 4080",  "tier": "consumer",   "vram_gb": 16,  "tdp_w": 320, "msrp_usd": 1199},
    {"name": "NVIDIA RTX 4070",  "tier": "consumer",   "vram_gb": 12,  "tdp_w": 200, "msrp_usd": 599},
    {"name": "NVIDIA RTX 3090",  "tier": "consumer",   "vram_gb": 24,  "tdp_w": 350, "msrp_usd": 1499},
    {"name": "AMD RX 7900 XTX",  "tier": "consumer",   "vram_gb": 24,  "tdp_w": 355, "msrp_usd": 999},
    {"name": "AMD MI300X",       "tier": "datacenter", "vram_gb": 192, "tdp_w": 750, "msrp_usd": 20000},
]

WORKLOADS = {
    "llm_inference_tokens_per_sec": {
        "H100 80GB": (4500, 200), "A100 80GB": (2800, 150),
        "RTX 4090":  (850,  40),  "RTX 4080":  (600,  35),
        "RTX 4070":  (380,  20),  "RTX 3090":  (720,  38),
        "RX 7900 XTX": (420, 25), "MI300X":    (5200, 250),
    },
    "cnn_training_images_per_sec": {
        "H100 80GB": (12000, 400), "A100 80GB": (7500, 300),
        "RTX 4090":  (2200,  80),  "RTX 4080":  (1600,  70),
        "RTX 4070":  (950,   45),  "RTX 3090":  (1800,  75),
        "RX 7900 XTX": (1100, 55), "MI300X":    (13500, 500),
    },
    "render_fps_4k": {
        "H100 80GB": (None, None), "A100 80GB": (None, None),  # not applicable
        "RTX 4090":  (145, 8),  "RTX 4080":  (112, 6),
        "RTX 4070":  (78,  5),  "RTX 3090":  (95,  6),
        "RX 7900 XTX": (108, 7), "MI300X":   (None, None),
    },
    "memory_bandwidth_gbps": {
        "H100 80GB": (3350, 50), "A100 80GB": (2000, 30),
        "RTX 4090":  (1008, 20), "RTX 4080":  (717,  15),
        "RTX 4070":  (504,  10), "RTX 3090":  (936,  18),
        "RX 7900 XTX": (960, 20), "MI300X":   (5300, 80),
    },
}


def _gpu_key(gpu_name: str) -> str:
    """Extract short key from full GPU name."""
    for key in list(WORKLOADS["llm_inference_tokens_per_sec"].keys()):
        if key in gpu_name:
            return key
    return gpu_name


def generate_benchmark_records(n_days: int = 90) -> list[dict]:
    """Generate n_days of benchmark records per GPU."""
    records = []
    base_date = datetime.now() - timedelta(days=n_days)

    for gpu in GPUS:
        key = _gpu_key(gpu["name"])
        for day_offset in range(0, n_days, random.randint(3, 7)):
            run_date = base_date + timedelta(days=day_offset)
            record = {
                "gpu_name": gpu["name"],
                "tier": gpu["tier"],
                "vram_gb": gpu["vram_gb"],
                "tdp_w": gpu["tdp_w"],
                "msrp_usd": gpu["msrp_usd"],
                "run_date": run_date.isoformat(),
                "driver_version": f"55{random.randint(0,5)}.{random.randint(10,99)}",
            }
            for metric, gpu_stats in WORKLOADS.items():
                stats = gpu_stats.get(key)
                if stats and stats[0] is not None:
                    mu, sigma = stats
                    val = max(0, np.random.normal(mu, sigma))
                    record[metric] = round(val, 2)
                else:
                    record[metric] = None
            records.append(record)
    return records


def seed_database(records: list[dict]) -> None:
    """Write records to SQLite."""
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS benchmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gpu_name TEXT,
            tier TEXT,
            vram_gb INTEGER,
            tdp_w INTEGER,
            msrp_usd INTEGER,
            run_date TEXT,
            driver_version TEXT,
            llm_inference_tokens_per_sec REAL,
            cnn_training_images_per_sec REAL,
            render_fps_4k REAL,
            memory_bandwidth_gbps REAL
        )
    """)
    cur.execute("DELETE FROM benchmarks")  # fresh seed

    for r in records:
        cur.execute("""
            INSERT INTO benchmarks (
                gpu_name, tier, vram_gb, tdp_w, msrp_usd, run_date, driver_version,
                llm_inference_tokens_per_sec, cnn_training_images_per_sec,
                render_fps_4k, memory_bandwidth_gbps
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["gpu_name"], r["tier"], r["vram_gb"], r["tdp_w"], r["msrp_usd"],
            r["run_date"], r["driver_version"],
            r.get("llm_inference_tokens_per_sec"),
            r.get("cnn_training_images_per_sec"),
            r.get("render_fps_4k"),
            r.get("memory_bandwidth_gbps"),
        ))

    conn.commit()
    conn.close()
    print(f"✅ Seeded {len(records)} benchmark records into {DB_PATH}")

    # Also dump latest snapshot to JSON for fast API reads
    snapshot = {}
    for gpu in GPUS:
        gpu_records = [r for r in records if r["gpu_name"] == gpu["name"]]
        if gpu_records:
            latest = sorted(gpu_records, key=lambda x: x["run_date"])[-1]
            snapshot[gpu["name"]] = latest
    json_path = DATA_DIR / "latest_snapshot.json"
    json_path.write_text(json.dumps(snapshot, indent=2))
    print(f"✅ Snapshot written to {json_path}")


if __name__ == "__main__":
    records = generate_benchmark_records(n_days=90)
    seed_database(records)
