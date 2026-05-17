"""
api/main.py

FastAPI REST API for the GPU benchmark platform.

Endpoints:
  GET  /benchmarks          — List all benchmark summaries
  GET  /benchmarks/{gpu}    — Get data for a specific GPU
  POST /analyze             — Run analysis with specific tools
  POST /agent/chat          — Chat with the LangGraph agent
  GET  /health              — Health check
"""
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

SUMMARY_PATH = Path(__file__).parent.parent / "data" / "summary.json"
SNAPSHOT_PATH = Path(__file__).parent.parent / "data" / "latest_snapshot.json"

app = FastAPI(
    title="GPU Performance Intelligence API",
    description="REST API for GPU benchmark data and AI-powered analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Models ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class AnalyzeRequest(BaseModel):
    tool: str  # query_benchmarks | compare_gpus | detect_regressions | generate_report
    params: dict = {}


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "data_ready": SUMMARY_PATH.exists()}


@app.get("/benchmarks")
def list_benchmarks(
    tier: Optional[str] = Query(None, description="Filter by tier: consumer or datacenter"),
    gpu: Optional[str] = Query(None, description="Filter by partial GPU name"),
):
    """Return all GPU benchmark summaries."""
    if not SUMMARY_PATH.exists():
        raise HTTPException(503, "Data not ready. Run `python -m pipeline.etl` first.")
    data = json.loads(SUMMARY_PATH.read_text())
    if tier:
        data = [d for d in data if d.get("tier", "").lower() == tier.lower()]
    if gpu:
        data = [d for d in data if gpu.lower() in d["gpu_name"].lower()]
    return {"count": len(data), "benchmarks": data}


@app.get("/benchmarks/{gpu_name:path}")
def get_gpu(gpu_name: str):
    """Return benchmark data for a specific GPU."""
    if not SUMMARY_PATH.exists():
        raise HTTPException(503, "Data not ready.")
    data = json.loads(SUMMARY_PATH.read_text())
    matches = [d for d in data if gpu_name.lower() in d["gpu_name"].lower()]
    if not matches:
        raise HTTPException(404, f"GPU '{gpu_name}' not found.")
    return matches[0] if len(matches) == 1 else matches


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Invoke a specific MCP tool directly."""
    from mcp_tools.benchmark_tools import (
        compare_gpus, detect_regressions, generate_report, query_benchmarks,
    )
    tools = {
        "query_benchmarks": query_benchmarks,
        "compare_gpus": compare_gpus,
        "detect_regressions": detect_regressions,
        "generate_report": generate_report,
    }
    if req.tool not in tools:
        raise HTTPException(400, f"Unknown tool '{req.tool}'. Available: {list(tools)}")
    try:
        result = tools[req.tool].invoke(req.params)
        return {"tool": req.tool, "result": json.loads(result) if isinstance(result, str) else result}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/agent/chat")
def agent_chat(req: ChatRequest):
    """Chat with the LangGraph AI agent."""
    try:
        from agent.graph import run_agent
        response = run_agent(req.message, history=req.history)
        return {"response": response}
    except Exception as e:
        raise HTTPException(500, f"Agent error: {e}")


@app.get("/metrics")
def available_metrics():
    """Return metadata about available benchmark metrics."""
    return {
        "metrics": [
            {"key": "llm_inference_tokens_per_sec", "label": "LLM Inference", "unit": "tokens/sec"},
            {"key": "cnn_training_images_per_sec",  "label": "CNN Training",  "unit": "images/sec"},
            {"key": "render_fps_4k",                "label": "4K Rendering",  "unit": "FPS"},
            {"key": "memory_bandwidth_gbps",        "label": "Memory Bandwidth","unit": "GB/s"},
            {"key": "llm_perf_per_watt_avg",        "label": "Perf/Watt",     "unit": "tok/s/W"},
            {"key": "llm_perf_per_dollar_avg",      "label": "Price-Performance","unit": "tok/s per $1k"},
        ]
    }
