# GPU Performance Intelligence Agent

A full-stack AI agent platform that automates GPU benchmark collection, analysis, and visualization — built to demonstrate skills relevant to NVIDIA's PerfLab team.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                   │
│              (Interactive Benchmark Insights)            │
└──────────────────────┬──────────────────────────────────┘
                       │ REST API
┌──────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend                        │
│         /benchmarks  /analyze  /agent/chat               │
└────────┬─────────────────────────┬──────────────────────┘
         │                         │
┌────────▼────────┐    ┌───────────▼──────────────────────┐
│  Data Pipeline  │    │       LangGraph AI Agent          │
│  (ETL: collect  │    │  ┌─────────────────────────────┐ │
│   transform     │    │  │  MCP Tools:                 │ │
│   store)        │    │  │  - query_benchmarks         │ │
└────────┬────────┘    │  │  - compare_gpus             │ │
         │             │  │  - detect_regressions       │ │
┌────────▼────────┐    │  │  - generate_report          │ │
│  SQLite / JSON  │◄───┤  └─────────────────────────────┘ │
│  Data Store     │    └──────────────────────────────────┘
└─────────────────┘

Deployed via GitLab CI → Docker → Kubernetes (perflab namespace)
```

## Tech Stack

| Layer | Technology |
|---|---|
| AI Agent | LangGraph + LangChain (ReAct loop) |
| MCP Tools | Custom Python MCP tool definitions |
| REST API | FastAPI |
| Dashboard | Streamlit + Plotly |
| Data Pipeline | Python ETL (pandas + SQLite) |
| Containerization | Docker + docker-compose |
| Orchestration | Kubernetes + Helm-ready manifests |
| CI/CD | GitLab CI (lint → test → build → deploy) |

## Project Structure

```
gpu-perf-agent/
├── agent/
│   ├── graph.py          # LangGraph ReAct agent
│   ├── cli.py            # Typer CLI (single query + interactive)
│   └── prompts.py        # PerfBot system prompt
├── mcp_tools/
│   └── benchmark_tools.py  # 4 MCP-style tools
├── api/
│   └── main.py           # FastAPI (5 endpoints)
├── pipeline/
│   ├── etl.py            # ETL pipeline (Extract → Transform → Load)
│   └── generator.py      # Synthetic benchmark data generator
├── dashboard/
│   └── app.py            # Streamlit dashboard (5 pages)
├── k8s/
│   ├── configmap.yaml    # Environment configuration
│   ├── secret.yaml       # API key secret (template)
│   ├── pvc.yaml          # Persistent volume for SQLite data
│   ├── deployment-api.yaml       # FastAPI deployment (2 replicas)
│   ├── deployment-dashboard.yaml # Streamlit deployment
│   ├── service.yaml      # ClusterIP services
│   └── ingress.yaml      # NGINX ingress (perflab.internal)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .gitlab-ci.yml        # CI/CD: lint → test → build → deploy
├── requirements.txt
└── data/                 # SQLite DB + JSON outputs (git-ignored)
```

## Quick Start (Local)

```bash
# 1. Clone and create virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env

# 4. Seed the database (run once)
python -m pipeline.etl

# 5. Start services (three terminals)
uvicorn api.main:app --reload --port 8000   # API  → http://localhost:8000/docs
streamlit run dashboard/app.py              # UI   → http://localhost:8501
python -m agent.cli --interactive           # CLI  → chat in terminal
```

## Docker Compose

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
docker compose -f docker/docker-compose.yml up
```

Services start in order: `etl` (seed) → `api` (:8000) → `dashboard` (:8501).

## Kubernetes Deployment

```bash
# Create namespace and secret
kubectl create namespace perflab
kubectl create secret generic gpu-perflab-secrets \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-your-key-here \
  --namespace=perflab

# Build and push image
docker build -f docker/Dockerfile -t gpu-perflab:latest .
# docker tag / push to your registry...

# Apply all manifests
kubectl apply -f k8s/ --namespace=perflab

# Verify rollout
kubectl rollout status deployment/gpu-perflab-api --namespace=perflab
kubectl rollout status deployment/gpu-perflab-dashboard --namespace=perflab
```

Access via `http://perflab.internal` (configure DNS or `/etc/hosts` for local clusters).

## CI/CD Pipeline (GitLab)

`.gitlab-ci.yml` runs four stages on every merge request and main branch push:

| Stage | Jobs | What it does |
|---|---|---|
| **lint** | `ruff-lint`, `type-check` | Style and type checks |
| **test** | `unit-tests`, `etl-smoke-test` | Pytest + coverage, ETL end-to-end |
| **build** | `build-image` | Docker build + push to GitLab registry |
| **deploy** | `deploy-staging` | `kubectl apply` + rolling restart |

## MCP Tools

| Tool | Description |
|---|---|
| `query_benchmarks` | Filter GPU benchmark data by name or tier |
| `compare_gpus` | Side-by-side comparison on any metric |
| `detect_regressions` | Scan for >10% performance drops vs rolling baseline |
| `generate_report` | Structured Markdown report (overview / efficiency / consumer) |
| `query_time_series` | Trend data for a specific GPU over a configurable time window, with min/max/avg and trend direction (improving / stable / declining) |

## Dashboard Pages

| Page | Content |
|---|---|
| Overview | KPI cards, LLM performance bar chart, price-vs-perf scatter, radar chart |
| Time Series | Per-metric trend lines with regression markers |
| GPU Compare | Head-to-head on all metrics + specs table |
| AI Agent | Natural language chat powered by PerfBot |
| Regressions | Flagged performance drops by GPU and date |

## Benchmark Coverage

8 GPUs across consumer and datacenter tiers (NVIDIA H100, A100, RTX 4090/4080/4070/3090, AMD RX 7900 XTX, MI300X) across 4 workloads:

- **LLM inference** — tokens/sec (Transformer workload)
- **CNN training** — images/sec (ResNet-scale batch training)
- **4K rendering** — FPS (consumer GPUs only)
- **Memory bandwidth** — GB/s

Derived metrics: performance-per-watt, price-performance (tok/s per $1k MSRP).
