CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS assets (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    asset_class TEXT NOT NULL,
    exchange TEXT,
    sector TEXT,
    market_cap NUMERIC,
    currency TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ohlcv_1d (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL REFERENCES assets(symbol) ON DELETE CASCADE,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    adjusted_close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    source TEXT NOT NULL,
    PRIMARY KEY (time, symbol)
);
SELECT create_hypertable('ohlcv_1d', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ohlcv_1d_symbol_time ON ohlcv_1d(symbol, time DESC);

CREATE TABLE IF NOT EXISTS order_book_snapshots (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL REFERENCES assets(symbol) ON DELETE CASCADE,
    bid DOUBLE PRECISION,
    ask DOUBLE PRECISION,
    bid_size DOUBLE PRECISION,
    ask_size DOUBLE PRECISION,
    source TEXT NOT NULL,
    PRIMARY KEY (time, symbol)
);
SELECT create_hypertable('order_book_snapshots', 'time', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS news_sentiment (
    id BIGSERIAL PRIMARY KEY,
    published_at TIMESTAMPTZ NOT NULL,
    symbol TEXT REFERENCES assets(symbol) ON DELETE CASCADE,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT,
    sentiment_score DOUBLE PRECISION NOT NULL CHECK (sentiment_score >= -1 AND sentiment_score <= 1),
    raw JSONB DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_news_symbol_time ON news_sentiment(symbol, published_at DESC);

CREATE TABLE IF NOT EXISTS asset_rankings (
    ranking_date DATE NOT NULL,
    symbol TEXT NOT NULL REFERENCES assets(symbol) ON DELETE CASCADE,
    score DOUBLE PRECISION NOT NULL,
    components JSONB NOT NULL,
    rank INTEGER NOT NULL,
    PRIMARY KEY (ranking_date, symbol)
);

CREATE TABLE IF NOT EXISTS signals (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol TEXT NOT NULL REFERENCES assets(symbol) ON DELETE CASCADE,
    strategy TEXT NOT NULL,
    signal TEXT NOT NULL CHECK (signal IN ('buy', 'sell', 'hold')),
    confidence DOUBLE PRECISION DEFAULT 0,
    stop DOUBLE PRECISION,
    target DOUBLE PRECISION,
    metadata JSONB DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at DESC);

CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol TEXT NOT NULL REFERENCES assets(symbol) ON DELETE CASCADE,
    strategy TEXT,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity DOUBLE PRECISION NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    commission DOUBLE PRECISION NOT NULL DEFAULT 0,
    slippage DOUBLE PRECISION NOT NULL DEFAULT 0,
    realized_pnl DOUBLE PRECISION DEFAULT 0,
    mode TEXT NOT NULL DEFAULT 'paper',
    order_id TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_trades_created ON trades(created_at DESC);

CREATE TABLE IF NOT EXISTS equity_curve (
    time TIMESTAMPTZ PRIMARY KEY,
    equity DOUBLE PRECISION NOT NULL,
    cash DOUBLE PRECISION NOT NULL,
    drawdown DOUBLE PRECISION NOT NULL,
    daily_pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
    mode TEXT NOT NULL DEFAULT 'paper'
);
SELECT create_hypertable('equity_curve', 'time', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY REFERENCES assets(symbol) ON DELETE CASCADE,
    quantity DOUBLE PRECISION NOT NULL,
    average_price DOUBLE PRECISION NOT NULL,
    mark_price DOUBLE PRECISION NOT NULL,
    unrealized_pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
    stop DOUBLE PRECISION,
    target DOUBLE PRECISION,
    strategy TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id BIGSERIAL PRIMARY KEY,
    run_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    sharpe DOUBLE PRECISION,
    sortino DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    profit_factor DOUBLE PRECISION,
    calmar DOUBLE PRECISION,
    passed_walk_forward BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'::jsonb
);

INSERT INTO assets(symbol, name, asset_class, exchange, sector, currency)
VALUES
('SPY', 'SPDR S&P 500 ETF Trust', 'etf', 'NYSEARCA', 'ETF', 'USD'),
('QQQ', 'Invesco QQQ Trust', 'etf', 'NASDAQ', 'ETF', 'USD'),
('IWM', 'iShares Russell 2000 ETF', 'etf', 'NYSEARCA', 'ETF', 'USD'),
('GLD', 'SPDR Gold Shares', 'commodity', 'NYSEARCA', 'Commodity', 'USD'),
('TLT', 'iShares 20+ Year Treasury Bond ETF', 'etf', 'NASDAQ', 'Fixed Income', 'USD'),
('AAPL', 'Apple Inc.', 'stock', 'NASDAQ', 'Technology', 'USD'),
('MSFT', 'Microsoft Corporation', 'stock', 'NASDAQ', 'Technology', 'USD'),
('NVDA', 'NVIDIA Corporation', 'stock', 'NASDAQ', 'Technology', 'USD'),
('EURUSD=X', 'EUR/USD', 'forex', 'FX', 'Forex', 'USD'),
('GBPUSD=X', 'GBP/USD', 'forex', 'FX', 'Forex', 'USD'),
('BTC-USD', 'Bitcoin USD', 'crypto', 'CRYPTO', 'Crypto', 'USD'),
('ETH-USD', 'Ethereum USD', 'crypto', 'CRYPTO', 'Crypto', 'USD')
ON CONFLICT (symbol) DO NOTHING;
