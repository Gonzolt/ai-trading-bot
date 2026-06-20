# Universal AI Trading Bot

Production-style scaffold for a multi-asset algorithmic trading system with data collection,
strategy signals, asset ranking, professional risk controls, paper/live execution adapters,
backtesting, scheduled retraining, and a secured web dashboard.

The stack starts in paper mode with demo-safe defaults. Do not enable live trading until API
keys, broker permissions, monitoring, and risk limits have been reviewed by a qualified human.

## Quick Start

```bash
cp .env.example .env
docker compose up -d --build
```

Open `http://localhost/` and sign in with the credentials in `.env`.

## Directory Tree

```text
.
|-- config/settings.yaml              # Strategy, universe, risk, broker, dashboard config
|-- db/migrations/001_init.sql         # TimescaleDB schema and seed assets
|-- deploy/trading-bot.service         # systemd unit for Ubuntu boot startup
|-- docker/python.Dockerfile           # Shared Python runtime for backend services
|-- docker/nginx/nginx.conf            # Reverse proxy for UI, API, and WebSocket
|-- docker-compose.yml                 # db, redis, mongo, collectors, engines, API, UI, nginx
|-- dashboard-ui/                      # React, MUI, Plotly dashboard
|-- install.sh                         # Ubuntu installer for Docker and stack startup
|-- src/trading_bot/
|   |-- data/                          # yfinance, Polygon/Finnhub clients, MT5 adapter, cleaning
|   |-- strategies/                    # Trend, momentum, mean reversion, LightGBM strategy classes
|   |-- selection/                     # Universal daily 0-100 composite score
|   |-- risk/                          # Daily loss, sizing, leverage, drawdown, correlation checks
|   |-- backtest/                      # Vectorized backtester, metrics, walk-forward validation
|   |-- execution/                     # PaperBroker and guarded Alpaca live adapter
|   |-- ml/                            # Gaussian HMM regime detector
|   |-- dashboard/api/                 # FastAPI REST + JWT + WebSocket
|   `-- services/                      # Container entrypoints and Celery beat tasks
`-- tests/                             # Focused pytest coverage for math and guardrails
```

## Services

`docker-compose.yml` defines the deployable stack:

- `db`: PostgreSQL + TimescaleDB for OHLCV, signals, trades, rankings, equity.
- `redis`: tick cache, live updates, Celery broker.
- `mongo`: news/sentiment article storage target.
- `data_collector`: collects daily history through `YFinanceProvider` and extension clients.
- `signal_engine`: ranks assets and runs all strategy classes.
- `risk_manager`: central risk service skeleton with configured limits.
- `broker`: paper broker service by default.
- `scheduler`: Celery worker with beat for nightly model retraining and weekly walk-forward jobs.
- `dashboard_api`: FastAPI REST and WebSocket backend.
- `dashboard_ui`: React dashboard served by Nginx.
- `nginx`: browser entrypoint and reverse proxy.

## Core Components

Every strategy implements:

```python
def generate_signal(self, symbol: str, asset_df: pd.DataFrame) -> StrategySignal:
    ...
```

Implemented strategies:

- `TrendFollowingStrategy`: EMA 20/50 crossover, ADX filter, ATR stop/target.
- `MomentumStrategy`: RSI oversold exit plus MACD histogram flip.
- `MeanReversionStrategy`: Bollinger Band and z-score entry.
- `MLDirectionStrategy`: LightGBM next-day direction probability with engineered features.

The scoring engine in `src/trading_bot/selection/scoring.py` uses the requested weights:

- 20-day momentum: 25%
- 5-day volume growth: 15%
- ADX trend strength: 10%
- inverse ATR/close: 10%
- sentiment: 15%
- 60-day Sharpe: 25%

The risk manager in `src/trading_bot/risk/manager.py` enforces daily loss, drawdown,
position sizing, max exposure, leverage, and correlation limits. Live adapters are explicitly
guarded by `ENABLE_LIVE_TRADING=true`.

## API

The dashboard API exposes:

- `POST /api/token`
- `GET /api/portfolio`
- `GET /api/risk`
- `GET /api/rankings`
- `GET /api/trades`
- `WS /ws/live`

JWT credentials are controlled by:

```env
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=change-me
JWT_SECRET=change-me-before-live
```

## Deployment on Ubuntu

On a fresh Ubuntu 22.04 or 24.04 server:

```bash
sudo REPO_URL=https://github.com/you/universal-ai-trading-bot.git bash install.sh
```

Then edit `/opt/universal-ai-trading-bot/.env`, restart with:

```bash
cd /opt/universal-ai-trading-bot
sudo docker compose up -d --build
```

The systemd unit is installed as `trading-bot.service`.

## Data and Broker Notes

- Stocks/ETFs can run through yfinance without keys for initial paper mode.
- Polygon, Alpha Vantage, Finnhub, and NewsAPI keys are optional environment variables.
- MetaTrader 5 is kept as an optional Python extra because MT5 support on Linux usually
  requires broker-specific terminal setup or Wine. Install with `pip install -e ".[mt5]"` on
  a compatible host and set `broker.metatrader5.enabled`.
- Alpaca live trading requires `ENABLE_LIVE_TRADING=true`, Alpaca credentials, paper/live
  account review, and startup reconciliation before use.

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[runtime,dev]"
pytest
```

Frontend:

```bash
cd dashboard-ui
npm install
npm run dev
```

## Professional Pitfalls and Solutions

- Survivorship bias: store the historical universe and metadata snapshots instead of only
  today's tradable symbols.
- Vendor outages: collectors should persist raw provider payloads and mark source quality.
- Split/dividend drift: use adjusted prices for research and raw prices for execution checks.
- Overfitting: keep grids small and require out-of-sample Sharpe to retain at least 50% of
  in-sample Sharpe.
- Broker mismatch: reconcile positions and cash on startup before submitting new orders.
- MT5 on Linux: isolate broker terminal setup from the main stack and treat it as an adapter.
- Live risk failure: daily loss, drawdown, leverage, and correlation limits are central, not
  per-strategy suggestions.
- Secret leakage: never commit `.env`; rotate keys before enabling live trading.
- Dashboard exposure: put SSL and firewall rules in front of Nginx for non-local deployment.

## Next Production Hardening Steps

This repo is ready to clone and boot as a paper-trading analytics stack. Before real capital,
add broker-specific integration tests, provider data quality checks, persistent order-state
machines, alerting, SSL certificate automation, and manual approval gates for live mode.
