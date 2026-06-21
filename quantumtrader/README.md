# QuantumTrader – Production Algorithmic Trading Platform

**QuantumTrader** is a containerized, multi-asset algorithmic trading system supporting stocks, ETFs, forex, crypto, and commodities. It combines four systematic strategies with ML augmentation, professional risk management, walk-forward backtesting, and a secure React dashboard.

Deploy on a fresh **Ubuntu 22.04/24.04 LTS** server with a single command:

```bash
curl -fsSL https://raw.githubusercontent.com/YOUR_ORG/QuantumTrader/main/install.sh | sudo bash
# or, from a cloned repo:
sudo ./install.sh
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ data_collector│──▶│ TimescaleDB  │◀──│ asset_scorer │
└─────────────┘     └──────────────┘     └─────────────┘
       │                    ▲                    │
       ▼                    │                    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   MongoDB   │     │ signal_engine│────▶│ risk_manager│
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                    ┌──────────────┐             ▼
                    │    Redis     │◀────┌─────────────┐
                    └──────────────┘     │   broker    │
                           ▲             └─────────────┘
                           │                    │
                    ┌──────────────┐     ┌─────────────┐
                    │  dashboard   │◀────│   nginx     │
                    └──────────────┘     └─────────────┘
```

### Services

| Service | Description |
|---------|-------------|
| `db` | PostgreSQL + TimescaleDB (OHLCV hypertables) |
| `redis` | Pub/sub for signals, orders, live updates |
| `mongo` | News and sentiment articles |
| `data_collector` | yfinance, Polygon, Alpha Vantage, Finnhub, MT5 |
| `signal_engine` | Trend, Momentum, Mean Reversion, LightGBM |
| `asset_scorer` | Daily composite ranking (top 15 assets) |
| `risk_manager` | Pre-trade validation (drawdown, correlation, sector limits) |
| `broker` | Paper + Alpaca execution gateway |
| `backtester` | Vectorized backtests + walk-forward validation |
| `celery_worker` / `scheduler` | Nightly ML retrain, weekly backtests |
| `dashboard_api` | FastAPI REST + WebSocket |
| `dashboard_ui` | React + MUI + Plotly dark-theme dashboard |
| `nginx` | Reverse proxy with optional SSL |

## Quick Start (Development)

```bash
git clone https://github.com/YOUR_ORG/QuantumTrader.git
cd QuantumTrader
cp config/.env.example .env
# Edit .env – set API keys and passwords
docker compose up -d --build
```

Open **http://localhost/** and sign in with credentials from `.env` (default `admin` / `QuantumTrader2026!`).

## Production Install

```bash
export REPO_URL=https://github.com/YOUR_ORG/QuantumTrader.git
export DOMAIN=trading.example.com          # optional
export LETSENCRYPT_EMAIL=you@example.com   # optional
sudo -E ./install.sh
```

The installer will:
1. Update the system and install Docker + docker-compose
2. Clone the repo (or use the current directory)
3. Create `.env` from `config/.env.example`
4. Build and start all containers
5. Optionally configure systemd auto-start on boot
6. Print the dashboard URL and login credentials

## Configuration

- **`config/settings.yaml`** – strategy parameters, risk limits, universe, scheduler cron
- **`.env`** – secrets and infrastructure URLs (never commit this file)

### Required for live trading

```env
ENABLE_LIVE_TRADING=true
TRADING_MODE=live
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

### Optional data APIs

| Provider | Purpose | Env var |
|----------|---------|---------|
| Polygon.io | Reference/metadata | `POLYGON_API_KEY` |
| Alpha Vantage | FX/crypto | `ALPHA_VANTAGE_API_KEY` |
| Finnhub | News sentiment | `FINNHUB_API_KEY` |
| MetaTrader5 | Forex/CFD ticks | `MT5_ENABLED=true` |

yfinance works without API keys for daily OHLCV.

## Strategies

1. **TrendFollowing** – EMA(20/50) cross + ADX > 25; stop 2×ATR, target 3×ATR
2. **Momentum** – RSI cross above 30 + MACD histogram positive; stop 1.5×ATR
3. **MeanReversion** – Bollinger lower band + Z < -2; exit at mid-band
4. **MLPredictor** – LightGBM classifier retrained nightly via Celery

## Risk Management

- Max daily loss: 2% → halt trading
- Position size: (1% equity) / (2 × ATR)
- Max 25% per asset, max 2 per sector
- Global drawdown 20% → switch to paper
- Correlation filter: reject if ρ > 0.8 (60-day)

## Testing

```bash
pip install -e ".[runtime,dev]"
pytest tests/ -v
```

## Project Structure

```
quantumtrader/
├── docker/                  # Dockerfiles per service
├── services/                # Service entry points
│   ├── data_collector/
│   ├── signal_engine/
│   ├── asset_scorer/
│   ├── risk_manager/
│   ├── broker/
│   ├── backtester/
│   └── dashboard/ui/        # React frontend
├── src/quantumtrader/       # Shared Python library
├── database/migrations/     # SQL schema + TimescaleDB hypertables
├── config/                  # settings.yaml, .env.example
├── tests/
├── docker-compose.yml
├── install.sh
└── README.md
```

## Common Pitfalls

| Pitfall | How QuantumTrader addresses it |
|---------|-------------------------------|
| Look-ahead bias in backtests | Walk-forward 60/20/20 split; OOS Sharpe must be ≥ 50% of IS |
| Stale/missing data | Multi-provider fallback chain; UTC alignment; outlier removal |
| Over-leveraging | Central risk manager validates every order before broker |
| Secret leakage | All credentials in `.env`; `.gitignore` excludes secrets |
| Container startup race | `depends_on` with healthchecks for db/redis |
| Live trading accidents | `ENABLE_LIVE_TRADING=false` by default; drawdown → paper mode |

## License

MIT – use at your own risk. Algorithmic trading involves substantial risk of loss.
