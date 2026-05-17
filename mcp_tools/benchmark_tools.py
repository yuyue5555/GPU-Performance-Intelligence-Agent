"""
mcp_tools/benchmark_tools.py

MCP-style tool definitions for the GPU benchmark agent.
Each tool follows the MCP pattern: name, description, input_schema, and a callable handler.

In a real MCP setup these would be registered with an MCP server.
Here we define them as LangChain-compatible tools that the LangGraph agent can invoke.
"""
import json
import sqlite3
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

DB_PATH = Path(__file__).parent.parent / "data" / "benchmarks.db"
SUMMARY_PATH = Path(__file__).parent.parent / "data" / "summary.json"

METRICS_MAP = {
    "llm": "llm_inference_tokens_per_sec",
    "cnn": "cnn_training_images_per_sec",
    "render": "render_fps_4k",
    "memory": "memory_bandwidth_gbps",
    "perf_per_watt": "llm_perf_per_watt",
    "price_performance": "llm_perf_per_dollar",
}


def _load_summary() -> list[dict]:
    if SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text())
    return []


def _query_db(sql: str, params: tuple = ()) -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ─── Tool 1: Query Benchmarks ─────────────────────────────────────────────────

@tool
def query_benchmarks(gpu_name: Optional[str] = None, tier: Optional[str] = None) -> str:
    """
    Query the benchmark database for GPU performance data.
    
    Args:
        gpu_name: Optional GPU name filter (e.g. 'RTX 4090', 'H100')
        tier: Optional tier filter: 'consumer' or 'datacenter'
    
    Returns:
        JSON string with benchmark summary data for matching GPUs.
    """
    summary = _load_summary()
    if gpu_name:
        summary = [s for s in summary if gpu_name.lower() in s["gpu_name"].lower()]
    if tier:
        summary = [s for s in summary if s.get("tier", "").lower() == tier.lower()]
    if not summary:
        return json.dumps({"error": "No matching GPUs found.", "available": [s["gpu_name"] for s in _load_summary()]})
    return json.dumps(summary, indent=2)


# ─── Tool 2: Compare GPUs ─────────────────────────────────────────────────────

@tool
def compare_gpus(gpu_names: list[str], metric: str = "llm") -> str:
    """
    Side-by-side comparison of specific GPUs on a chosen metric.
    
    Args:
        gpu_names: List of GPU names to compare (e.g. ['RTX 4090', 'RX 7900 XTX'])
        metric: One of 'llm', 'cnn', 'render', 'memory', 'perf_per_watt', 'price_performance'
    
    Returns:
        JSON with ranked comparison results.
    """
    db_metric = METRICS_MAP.get(metric, metric)
    summary = _load_summary()

    results = []
    for s in summary:
        match = any(name.lower() in s["gpu_name"].lower() for name in gpu_names)
        if not match:
            continue
        val = s.get(f"{db_metric}_avg") or s.get(f"{db_metric}_latest")
        results.append({
            "gpu_name": s["gpu_name"],
            "tier": s["tier"],
            "msrp_usd": s["msrp_usd"],
            "metric": metric,
            "value": val,
            "unit": _metric_unit(db_metric),
        })

    results.sort(key=lambda x: (x["value"] or 0), reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return json.dumps(results, indent=2)


# ─── Tool 3: Detect Regressions ───────────────────────────────────────────────

@tool
def detect_regressions(gpu_name: Optional[str] = None, days: int = 30) -> str:
    """
    Scan recent benchmark runs for performance regressions (>10% drop vs prior average).
    
    Args:
        gpu_name: Optional GPU name to check (checks all if omitted)
        days: Number of recent days to inspect (default 30)
    
    Returns:
        JSON list of regression events with details.
    """
    where_clause = "WHERE regression_flag = 1"
    params: list = []
    if gpu_name:
        where_clause += " AND gpu_name LIKE ?"
        params.append(f"%{gpu_name}%")

    rows = _query_db(
        f"""
        SELECT gpu_name, run_date, driver_version,
               llm_inference_tokens_per_sec, cnn_training_images_per_sec,
               memory_bandwidth_gbps, regression_flag
        FROM benchmarks
        {where_clause}
        ORDER BY run_date DESC
        LIMIT 20
        """,
        tuple(params),
    )

    if not rows:
        return json.dumps({"message": "No regressions detected in the specified period. ✅"})

    return json.dumps({
        "regression_count": len(rows),
        "regressions": rows,
    }, indent=2)


# ─── Tool 4: Generate Report ──────────────────────────────────────────────────

@tool
def generate_report(focus: str = "overview") -> str:
    """
    Generate a structured benchmark insights report.
    
    Args:
        focus: Report focus area — 'overview', 'llm', 'datacenter', 'consumer', 'efficiency'
    
    Returns:
        Markdown-formatted report string.
    """
    summary = _load_summary()
    if not summary:
        return "No data available. Run the ETL pipeline first."

    lines = [f"# GPU Benchmark Report — {focus.upper()}", ""]

    if focus in ("overview", "llm"):
        llm_ranked = sorted(
            [s for s in summary if s.get("llm_inference_tokens_per_sec_avg")],
            key=lambda x: x["llm_inference_tokens_per_sec_avg"], reverse=True
        )
        lines += ["## LLM Inference Performance (tokens/sec)", ""]
        for r in llm_ranked:
            lines.append(
                f"- **{r['gpu_name']}**: {r['llm_inference_tokens_per_sec_avg']:.0f} tok/s "
                f"(${r['msrp_usd']:,})"
            )
        lines.append("")

    if focus in ("overview", "efficiency"):
        eff_ranked = sorted(
            [s for s in summary if s.get("llm_perf_per_watt_avg")],
            key=lambda x: x["llm_perf_per_watt_avg"], reverse=True
        )
        lines += ["## Best Performance-per-Watt", ""]
        for r in eff_ranked[:5]:
            lines.append(
                f"- **{r['gpu_name']}**: {r['llm_perf_per_watt_avg']:.2f} tok/s/W"
            )
        lines.append("")

    if focus in ("overview", "consumer"):
        consumer = [s for s in summary if s.get("tier") == "consumer"]
        pp_ranked = sorted(
            [s for s in consumer if s.get("llm_perf_per_dollar_avg")],
            key=lambda x: x["llm_perf_per_dollar_avg"], reverse=True
        )
        lines += ["## Best Price-Performance (Consumer)", ""]
        for r in pp_ranked:
            lines.append(
                f"- **{r['gpu_name']}**: {r['llm_perf_per_dollar_avg']:.3f} tok/s per $1k"
            )

    return "\n".join(lines)


# ─── Helper ───────────────────────────────────────────────────────────────────

def _metric_unit(metric: str) -> str:
    units = {
        "llm_inference_tokens_per_sec": "tokens/sec",
        "cnn_training_images_per_sec": "images/sec",
        "render_fps_4k": "FPS",
        "memory_bandwidth_gbps": "GB/s",
        "llm_perf_per_watt": "tok/s/W",
        "llm_perf_per_dollar": "tok/s per $1k",
    }
    return units.get(metric, "")


# ─── Tool registry (MCP-style) ────────────────────────────────────────────────

ALL_TOOLS = [query_benchmarks, compare_gpus, detect_regressions, generate_report]
