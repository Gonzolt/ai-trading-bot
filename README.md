# Korvax TradeMind

Korvax TradeMind is a single-file, CPU-only Windows desktop application for market research, neural-network training, and Alpaca paper execution. It coordinates four daemon agents through a local SQLite database while Tkinter remains on the main UI thread.

This is research software, not investment advice. Alpaca is always created with `paper=True`; no live-money endpoint exists in the application.

## Architecture

- **Researcher** polls Alpaca IEX stock bars and keyless Binance crypto bars, gathers Finnhub or Yahoo RSS news, and scores headlines with a lazily downloaded local `ProsusAI/finbert` model.
- **Analyst** uses Polars/Pandas and `ta` to calculate RSI, MACD, ATR, ADX, momentum, volume ratio and rolling Sharpe. It publishes a weighted 0-100 ranking for the top ten assets.
- **Investor** loads the local `20 -> 32 -> 16 -> 3` PyTorch model, evaluates the top three assets, and records ATR-derived decisions.
- **Cashier** is started only after explicit confirmation. It submits paper orders while enforcing 1% equity risk per trade, a 25% per-asset cap, a 2% daily-loss halt, sector concentration limits, and a 20% global drawdown halt.
- **Monitor** is the Tkinter UI, live charts, logs, rankings, P&L table, and animated network trace.

Agents communicate through `trademind.db` in SQLite WAL mode using `live_prices`, `research_findings`, `asset_rankings`, `investment_decisions`, `portfolio_log`, `training_log`, and `agent_logs` tables.

## Setup

1. Install 64-bit Python 3.11.
2. Run `setup.bat`.
3. Optionally copy `.env.example` to `.env` and enter keys locally.
4. Run `run.bat`.

API keys are never committed:

- `MASSIVE_API_KEY`: optional free Massive Stocks Basic key for historical stock training data.
- `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`: required for Alpaca stock streaming and paper execution.
- `FINNHUB_API_KEY`: optional; Yahoo Finance RSS is the news fallback.

Binance public crypto downloads require no key. If no Massive key exists, the app starts with `BTCUSDT`, `ETHUSDT`, and `SOLUSDT`.

## Workflow

1. Use **Load Symbols** to select stocks or crypto and daily or one-minute bars.
2. Select **Download Data**. Historical bars are cached in `market_cache/` as Parquet or CSV.
3. Select **Start Training**. Missing data is downloaded automatically.
4. Watch training/validation loss and accuracy or open the animated network trace.
5. Inspect live scores in **Rankings**.
6. Add Alpaca paper keys and select **Start Paper Trading** to enable the Investor -> Cashier pipeline.

## Model and data safety

Each sample uses a 30-bar lookback summarized into 20 explainable features: return horizons and volatility, SMA gaps, RSI, MACD, ATR, volume change/ratio, candle range/body, momentum, rolling Sharpe, and sentiment. The network is `20 -> 32 -> 16 -> 3` for hold, buy, and sell.

Every symbol is split chronologically before concatenation. Scaler statistics are fitted only on training windows. Training uses class-weighted cross entropy, gradient clipping, validation-based early stopping, and saves the best checkpoint to `models/trade_model.pt`. Checkpoints are validated and atomically replaced so the Investor never reads a partially written model. Training metrics are also written to SQLite.

FinBERT is downloaded only when news is first processed and is cached under `models/finbert/`. If transformers or the model service is unavailable, the Researcher records a warning and uses a small deterministic lexical fallback instead of stopping the other agents.

Generated caches, SQLite files, models, logs, `.env`, and the virtual environment are excluded from Git.

## Troubleshooting

- Training automatically switches to the Training tab. `STATE MODEL READY / BEST EPOCH N` means the run succeeded. `EARLY STOP (NORMAL)` is expected: the best validation checkpoint was restored, not discarded.
- If cache preparation fails, the failed symbols and provider errors are shown in the event log. The controls unlock after the error is reported.
- `https://paper-api.alpaca.markets/v2` is the paper endpoint, not an API key. Paper execution requires both values from the Alpaca paper dashboard in a local `.env` file. Restart the app after creating it.
- Stock decisions wait while the US market is closed. Crypto decisions can execute continuously when the Alpaca paper account supports crypto.
