export type Position = {
  symbol: string;
  quantity: number;
  average_price: number;
  mark_price: number;
  unrealized_pnl: number;
  stop?: number;
  target?: number;
  strategy?: string;
};

export type EquityPoint = {
  time: string;
  equity: number;
  drawdown: number;
  daily_pnl: number;
};

export type Ranking = {
  rank: number;
  symbol: string;
  score: number;
  components: Record<string, number>;
};

export type Trade = {
  created_at: string;
  symbol: string;
  strategy?: string;
  side: "buy" | "sell";
  quantity: number;
  price: number;
  commission: number;
  slippage: number;
  realized_pnl: number;
  mode: string;
};

export type RiskSnapshot = {
  mode: string;
  live_trading_enabled: boolean;
  daily_loss_limit_pct: number;
  max_drawdown_pct: number;
  risk_per_trade_pct: number;
  current_drawdown: number;
  gross_leverage: number;
  status: string;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";
const WS_BASE = import.meta.env.VITE_WS_BASE_URL ?? "/ws/live";

export function wsUrl() {
  if (WS_BASE.startsWith("ws")) {
    return WS_BASE;
  }
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}${WS_BASE}`;
}

export async function login(username: string, password: string) {
  const response = await fetch(`${API_BASE}/token`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!response.ok) {
    throw new Error("Invalid credentials");
  }
  return (await response.json()) as { access_token: string };
}

export async function apiGet<T>(path: string, token: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    throw new Error(`API ${path} failed`);
  }
  return (await response.json()) as T;
}
