# Neural Volatility Surface Forecasting System — Architecture

## System Overview

This document describes the architecture of the Neural Volatility Surface Forecasting
System: a production-grade platform that ingests live options market data, constructs
implied-volatility (IV) surfaces, forecasts next-day surface evolution with deep neural
networks, and emits actionable trading signals.

---

## Component Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL DATA SOURCES                            │
│   Market Data Vendor API        Exchange Feeds        RV Data Feed      │
└──────────────┬──────────────────────────┬──────────────────┬────────────┘
               │                          │                  │
               ▼                          ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                            │
│                                                                         │
│  ┌──────────────┐   ┌────────────────┐   ┌──────────────────────────┐  │
│  │ Options Chain│   │  Tick Adapter  │   │  Realised-Vol Calculator  │  │
│  │   Fetcher    │   │ (Kafka Producer)│   │  (EWMA / Parkinson)      │  │
│  └──────┬───────┘   └───────┬────────┘   └────────────┬─────────────┘  │
│         │                   │                          │                │
└─────────┼───────────────────┼──────────────────────────┼───────────────┘
          │                   │                          │
          ▼                   ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            KAFKA BUS                                    │
│                                                                         │
│  options.raw.SPX  │  options.clean.SPX  │  surface.snapshots.SPX  │    │
│  options.iv.SPX   │                     │  signals.vol.SPX         │    │
└──────────┬──────────────────────────────────────┬──────────────────────┘
           │                                      │
           ▼                                      ▼
┌──────────────────────────┐          ┌───────────────────────────────────┐
│   IV SURFACE ENGINE      │          │        SIGNAL CONSUMER            │
│                          │          │                                   │
│  ┌─────────────────────┐ │          │  ┌────────────────────────────┐   │
│  │  Newton-Raphson IV  │ │          │  │  Signal Generator          │   │
│  │  Solver             │ │          │  │  (vol carry / skew / ts)   │   │
│  └──────────┬──────────┘ │          │  └──────────────┬─────────────┘   │
│             │             │          │                 │                 │
│  ┌──────────▼──────────┐ │          │  ┌──────────────▼─────────────┐   │
│  │  SVI Surface Fitter │ │          │  │  Execution Model           │   │
│  │  (per-expiry SSVI)  │ │          │  │  (bid-ask, market impact)  │   │
│  └──────────┬──────────┘ │          │  └────────────────────────────┘   │
│             │             │          └───────────────────────────────────┘
│  ┌──────────▼──────────┐ │
│  │  Feature Extractor  │ │
│  │  (60+ features)     │ │
│  └──────────┬──────────┘ │
└─────────────┼────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        NEURAL FORECAST ENGINE                           │
│                                                                         │
│  ┌─────────────────────────┐      ┌─────────────────────────────────┐   │
│  │  ConvLSTM Surface Model │      │  Surface Transformer (Attn.)    │   │
│  │  (spatial + temporal)   │      │  (alt. architecture)            │   │
│  └───────────┬─────────────┘      └──────────────┬──────────────────┘   │
│              │                                    │                     │
│              └──────────────┬─────────────────────┘                     │
│                             │                                           │
│                    ┌────────▼─────────┐                                 │
│                    │  Ensemble Layer  │                                 │
│                    │  + Uncertainty   │                                 │
│                    │  Quantification  │                                 │
│                    └────────┬─────────┘                                 │
└─────────────────────────────┼───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    MONITORING & SERVING LAYER                           │
│                                                                         │
│  ┌────────────────┐  ┌──────────────────┐  ┌──────────────────────┐    │
│  │  FastAPI REST  │  │  Streamlit       │  │  Prometheus + Grafana │    │
│  │  (signals,     │  │  Dashboard       │  │  (latency, drift,     │    │
│  │   forecasts)   │  │  (live surfaces) │  │   model metrics)      │    │
│  └────────────────┘  └──────────────────┘  └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                              │
│                                                                         │
│  Airflow DAGs:                                                          │
│    daily_vol_surface_pipeline  (0 17 * * 1-5)                          │
│    weekly_model_retraining     (0 2 * * 0)                             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Description

1. **Ingestion**: The options chain fetcher pulls bid/ask/last/OI/volume from
   the market data vendor API at 5 PM ET each weekday. Tick data arrives
   continuously via the Kafka-based tick adapter.

2. **IV Computation**: A Newton–Raphson solver computes implied volatility for
   each option contract. Contracts with anomalous prices (zero bid, crossed
   markets) are filtered before solving.

3. **Surface Construction**: An SVI (Stochastic Volatility Inspired) model is
   fit to each expiry slice and evaluated on a fixed (strike × expiry) grid,
   producing a surface tensor of shape `(H=15, W=25)`.

4. **Feature Extraction**: ~60 features are derived from the surface and its
   rolling history: ATM IV level, term-structure slope, 25Δ skew, risk-reversal,
   butterfly, realised-vol ratios, PCA scores, and regime indicators.

5. **Neural Forecast**: The ConvLSTM or Transformer model ingests a 30-day
   rolling window of feature tensors and outputs a 1-day-ahead surface
   forecast plus per-point prediction intervals.

6. **Signal Generation**: The SignalGenerator converts the forecast vs. current
   surface into four signals (vol carry, skew, term structure, inversion) which
   are combined into a composite signal in [-1, 1].

7. **Publishing**: Signals are serialised to JSON and published to the
   `signals.vol.SPX` Kafka topic. The REST API exposes the latest signals to
   downstream consumers.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **SVI parameterisation** | Guarantees no static arbitrage (calendar / butterfly) by construction while offering enough flexibility to fit realistic smiles. |
| **ConvLSTM architecture** | Exploits both the 2-D spatial structure of the surface (strike × expiry) and the temporal auto-correlation of IV dynamics. |
| **Walk-forward validation** | Prevents look-ahead bias; mirrors the live production inference regime exactly. |
| **Champion–challenger registry** | Ensures only statistically significant improvements are promoted to production, preventing model degradation. |
| **Kafka for real-time streaming** | Decouples producers from consumers; enables replay for backtesting without touching the production pipeline. |
| **Airflow for orchestration** | Provides dependency management, retry logic, SLA monitoring and an audit trail for all daily runs. |
| **Uncertainty quantification** | Conformal prediction intervals around surface forecasts allow downstream risk systems to size positions proportionally to model confidence. |

---

## Technology Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.11+ |
| Deep learning | PyTorch + PyTorch Lightning | 2.x |
| Surface fitting | Custom SVI (scipy.optimize) | — |
| Feature engineering | NumPy, pandas | latest |
| Orchestration | Apache Airflow | 2.9+ |
| Streaming | Apache Kafka | 3.7+ |
| REST API | FastAPI + Uvicorn | 0.110+ |
| Dashboard | Streamlit | 1.34+ |
| Monitoring | Prometheus + Grafana | latest |
| Containerisation | Docker + docker-compose | 24+ |
| Experiment tracking | MLflow | 2.x |
| Storage | PostgreSQL (metadata), S3/GCS (artefacts) | — |
| CI/CD | GitHub Actions | — |

---

## Getting Started

### Prerequisites

- Docker Engine 24+ and docker-compose v2
- Python 3.11+ (for local development without Docker)
- 8 GB RAM minimum (16 GB recommended for GPU training)

### Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/nareh-avag/neural-volatility.git
cd neural-volatility

# 2. Copy and edit the environment file
cp .env.example .env
# Edit .env: set MARKET_DATA_API_KEY, DATABASE_URL, etc.

# 3. Start all services (Kafka, Airflow, API, Dashboard)
docker-compose up -d

# 4. Wait ~60 seconds for services to initialise, then check health
curl http://localhost:8000/health

# 5. Open the live dashboard
open http://localhost:8501
```

The Airflow UI is available at `http://localhost:8080` (user: `admin`, password: see `.env`).

### Running Backtests Locally

```bash
pip install -e ".[dev]"
python -m backtesting.backtest_engine --config configs/backtest_config.yaml
```
