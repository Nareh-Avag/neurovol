# Neural Volatility Surface Forecasting System

A production-grade deep learning platform that forecasts implied-volatility surfaces for equity index options, generates actionable trading signals, and provides a complete research and backtesting framework.

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│              MARKET DATA  (options chain, ticks)           │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│              KAFKA STREAMING BUS                           │
│  options.raw  │  options.iv  │  surface.snapshots  │ ...   │
└───────────────────────┬────────────────────────────────────┘
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
┌──────────────────┐        ┌──────────────────────────┐
│ IV Surface Engine │        │  Feature Extractor       │
│ (SVI fitting,    │        │  (60+ features, PCA,     │
│  Newton IV solve)│        │   regime detection)      │
└────────┬─────────┘        └────────────┬─────────────┘
         └──────────────┬────────────────┘
                        ▼
┌────────────────────────────────────────────────────────────┐
│              NEURAL FORECAST ENGINE                        │
│    ConvLSTM (spatial+temporal)  │  Transformer (attn.)    │
│    Ensemble + Uncertainty Quantification                   │
└───────────────────────┬────────────────────────────────────┘
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
┌──────────────────┐        ┌──────────────────────────┐
│ Signal Generator  │        │  FastAPI REST + Streamlit│
│ (carry/skew/TS)  │        │  Dashboard               │
└──────────────────┘        └──────────────────────────┘
```

---

## Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| Python | 3.11+ | Use pyenv or conda |
| Docker + docker-compose | 24 + v2 | Required for all services |
| CUDA toolkit | 11.8+ | Optional, for GPU training |
| 8 GB RAM | 16 GB recommended | For full pipeline |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/nareh-avag/neural-volatility.git
cd neural-volatility

# 2. Install Python dependencies (for local development)
pip install -e ".[dev]"

# 3. Copy environment template and add your API keys
cp .env.example .env

# 4. Start all services (Kafka, Airflow, API, Dashboard, Postgres)
docker-compose up -d

# 5. Open the live Streamlit dashboard
open http://localhost:8501        # macOS
# xdg-open http://localhost:8501  # Linux
# start http://localhost:8501     # Windows
```

**Check service health:**
```bash
curl http://localhost:8000/health
# → {"status": "ok", "model": "champion_v3", "uptime_s": 42}
```

**Airflow UI:** http://localhost:8080 (user: `admin`, password: from `.env`)

---

## Project Structure

```
neural-volatility/
├── airflow/
│   └── dags/
│       ├── daily_surface_pipeline.py   # 5 PM ET weekday pipeline
│       └── weekly_retraining.py        # Sunday champion-challenger training
├── backtesting/
│   ├── backtest_engine.py              # Event-driven backtest loop
│   ├── execution_model.py              # Transaction cost modelling
│   ├── pnl_attribution.py              # Greek P&L breakdown
│   └── signal_generator.py            # Vol carry / skew / TS signals
├── configs/
│   ├── model_config.yaml               # Architecture hyperparameters
│   └── backtest_config.yaml            # Backtest parameters
├── data/                               # Local data cache (gitignored)
├── docs/
│   ├── architecture.md                 # System design & decisions
│   └── data_dictionary.md             # Full field definitions
├── features/                           # Feature extraction modules
├── kafka/
│   └── create_topics.sh                # Kafka topic bootstrap script
├── models/                             # Model architecture implementations
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_iv_calculation_demo.ipynb
│   └── 05_model_training_demo.ipynb
├── research/
│   └── analysis/
│       ├── regime_analysis.py          # HMM-based regime classification
│       └── surface_pca.py             # PCA of IV surfaces
├── training/                           # Walk-forward training utilities
├── vol_surface_engine/                 # Core IV computation library
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Configuration Guide

All runtime configuration lives in the `configs/` directory and in environment variables.

### Environment Variables (`.env`)

```bash
# Market data
MARKET_DATA_API_KEY=your_key_here
UNDERLYING=SPX

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/vol_surface

# Object storage
S3_BUCKET=vol-surface-data
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# Airflow
AIRFLOW_ADMIN_PASSWORD=changeme
AIRFLOW_SECRET_KEY=changeme

# Kafka
KAFKA_HOST=localhost:9092

# Model settings
MODEL_REGISTRY_PATH=s3://vol-surface-models/registry
ENABLE_GPU=false
```

### Model Configuration (`configs/model_config.yaml`)

```yaml
architecture: conv_lstm      # conv_lstm | transformer | ensemble
in_channels: 1
hidden_channels: 32
n_lstm_layers: 2
lookback_days: 30
H: 15                        # expiry nodes
W: 25                        # strike nodes
learning_rate: 1.0e-3
batch_size: 32
max_epochs: 100
early_stopping_patience: 10
```

### Backtest Configuration (`configs/backtest_config.yaml`)

```yaml
signal_config:
  vol_carry_threshold: 0.02
  skew_scale: 0.05
  term_structure_scale: 0.03
vega_notional: 10000
annualisation: 252
min_signal_threshold: 0.10
```

---

## Running Backtests

```bash
# Run full backtest with default config
python -m backtesting.backtest_engine \
    --config configs/backtest_config.yaml \
    --start 2020-01-01 \
    --end   2024-12-31

# Run with custom vega notional
python -m backtesting.backtest_engine \
    --vega-notional 50000 \
    --underlying SPX
```

The backtest engine prints a formatted results table (using `rich`) and saves
daily P&L to `data/backtest_results_{date}.parquet`.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `GET` | `/surface/latest` | Latest IV surface tensor |
| `GET` | `/surface/{date}` | Historical surface for a date |
| `GET` | `/forecast/latest` | Latest 1-day-ahead surface forecast |
| `GET` | `/signals/latest` | Current trading signals |
| `GET` | `/signals/history?days=30` | Signal history |
| `POST` | `/surface/compute` | Compute surface from custom chain |
| `GET` | `/model/metrics` | Champion model performance metrics |
| `GET` | `/model/champion` | Champion model metadata |
| `GET` | `/regimes/current` | Current market regime |
| `GET` | `/pca/components` | PCA factor loadings |

Interactive API docs: http://localhost:8000/docs

---

## Contributing

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

3. Make your changes with tests:
   ```bash
   pytest tests/ -v --cov=vol_surface_engine --cov=backtesting
   ```

4. Ensure code quality passes:
   ```bash
   ruff check .
   mypy backtesting/ vol_surface_engine/ --ignore-missing-imports
   ```

5. Submit a pull request with a clear description of the change and its motivation.

### Coding Standards

- Python 3.11+ type hints throughout
- Docstrings on all public classes and methods
- Unit tests for all numerical functions
- No hard-coded credentials or paths

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Nareh Avagyan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
