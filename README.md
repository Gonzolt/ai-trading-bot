# TradeMind Trainer

A single-file Windows desktop application that downloads free Massive US-stock bars or free Binance crypto bars, builds leakage-safe market-feature windows, trains a shallow PyTorch classifier on CPU, and can execute US-stock signals against Alpaca's paper-money endpoint.

This is research software, not investment advice. The model is intentionally simple and paper trading is permanently enabled in the source. It cannot connect to Alpaca's live-money endpoint.

## Setup

1. Install 64-bit Python 3.11.
2. Run `setup.bat`.
3. Copy `.env.example` to `.env`.
4. Add a free Massive Stocks Basic key as `MASSIVE_API_KEY` for stock downloads.
5. Add Alpaca paper-account values only if paper execution is needed.
6. Run `run.bat`.

Binance crypto downloads need no key. Alpaca credentials do not block downloads or training; they are only checked before paper execution. Recent free IEX bars are overlaid in memory for stock paper decisions and are never written into the Massive training cache.

## Workflow

1. **Load Symbols** selects `stocks` (Massive) or `crypto` (Binance), comma-separated symbols, and daily or one-minute bars. Use symbols such as `AAPL` or `BTCUSDT`.
2. **Download Data** incrementally synchronizes OHLCV bars into `market_cache/`. Existing Parquet or CSV data is reused.
3. **Start Training** engineers indicators, creates chronological windows and trains the CPU network.
4. **Open Simulation** shows actual batch-majority labels moving through the 300→32→16→3 topology.
5. **Start Paper Trading** requires an Alpaca stock-paper account, explicit confirmation, and a stock-trained model. Crypto order submission is intentionally disabled pending a separate crypto execution safety pass.

When no Massive key is configured, the app starts in keyless crypto mode with `BTCUSDT`, `ETHUSDT`, and `SOLUSDT`. With a Massive key present, it starts in stock mode with `AAPL`, `MSFT`, and `GOOGL`. Both providers request roughly two years of daily data or 90 days of minute data. Downloads are cached in Parquet with CSV fallback.

## Dataset and model

Each sample contains the prior 30 bars of ten features:

- log return
- 20-bar volatility
- normalized 14-bar ATR
- normalized 14-bar RSI
- normalized MACD histogram
- volume change
- high-low range
- open-close body
- distance from 10-bar SMA
- distance from 30-bar SMA

The target uses the close five bars after the sample endpoint. The threshold is scaled to the market and timeframe: 0.2% for one-minute data, 0.5% for daily stocks, and 2% for daily crypto.

- `BUY (1)` when the future return is above the configured positive threshold
- `SELL (2)` when it is below the negative threshold
- `HOLD (0)` otherwise

Windows never contain future bars. Every symbol is split chronologically before concatenation, and normalization statistics are fitted only on training windows. The model is an explainable multilayer perceptron: 300 inputs, 32 hidden units, 16 hidden units, and three logits. Training stops after five validation epochs without improvement and `trade_model.pt` contains the best validation checkpoint rather than the final overfit epoch.

## Paper execution

Paper mode polls once per minute and uses the first selected symbol. It checks Alpaca's market clock, computes the latest feature window, and performs long-only actions:

- BUY with no position: submit a one-share paper market order.
- SELL with a position: submit a paper market order for the held quantity.
- HOLD: no order.

The paper-account tab records signal confidence, price, position, account equity, session P&L and the submitted action. API calls retry transient rate-limit and server failures with exponential backoff.
