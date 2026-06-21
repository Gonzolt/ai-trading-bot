const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";

export type EquityPoint = {
  time: string;
  equity: number;
  drawdown: number;
  daily_pnl: number;
};

export async function login(username: string, password: string): Promise<string> {
  const res = await fetch(`${API_BASE}/token`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) throw new Error("Login failed");
  const data = await res.json();
  return data.access_token;
}

export async function fetchPortfolio(token: string) {
  const res = await fetch(`${API_BASE}/portfolio`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error("Failed to load portfolio");
  return res.json();
}

export async function fetchRisk(token: string) {
  const res = await fetch(`${API_BASE}/risk`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error("Failed to load risk");
  return res.json();
}

export async function fetchRankings(token: string) {
  const res = await fetch(`${API_BASE}/rankings`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error("Failed to load rankings");
  return res.json();
}

export async function fetchTrades(token: string) {
  const res = await fetch(`${API_BASE}/trades`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error("Failed to load trades");
  return res.json();
}
