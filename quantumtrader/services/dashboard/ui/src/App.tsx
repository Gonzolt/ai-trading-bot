import { useCallback, useEffect, useState } from "react";
import { Box, Container, Grid, Typography } from "@mui/material";
import { LoginPanel } from "./components/LoginPanel";
import { EquityChart } from "./components/EquityChart";
import { PositionsTable } from "./components/PositionsTable";
import { RankingsTable } from "./components/RankingsTable";
import { MetricTile } from "./components/MetricTile";
import * as api from "./lib/api";

export default function App() {
  const [token, setToken] = useState<string | null>(localStorage.getItem("qt_token"));
  const [error, setError] = useState<string>();
  const [portfolio, setPortfolio] = useState<{ positions: []; equity_curve: api.EquityPoint[] } | null>(null);
  const [risk, setRisk] = useState<Record<string, unknown> | null>(null);
  const [rankings, setRankings] = useState([]);

  const load = useCallback(async (t: string) => {
    const [p, r, k] = await Promise.all([
      api.fetchPortfolio(t),
      api.fetchRisk(t),
      api.fetchRankings(t),
    ]);
    setPortfolio(p);
    setRisk(r);
    setRankings(k);
  }, []);

  useEffect(() => {
    if (token) load(token).catch(() => setToken(null));
  }, [token, load]);

  const handleLogin = async (username: string, password: string) => {
    try {
      const t = await api.login(username, password);
      localStorage.setItem("qt_token", t);
      setToken(t);
      setError(undefined);
    } catch {
      setError("Invalid credentials");
    }
  };

  if (!token) return <LoginPanel onLogin={handleLogin} error={error} />;

  const latest = portfolio?.equity_curve?.slice(-1)[0];

  return (
    <Container maxWidth="xl" sx={{ py: 3 }}>
      <Typography variant="h4" gutterBottom>QuantumTrader</Typography>
      <Grid container spacing={2} mb={2}>
        <Grid item xs={6} md={3}><MetricTile label="Equity" value={`$${latest?.equity?.toLocaleString() ?? "—"}`} /></Grid>
        <Grid item xs={6} md={3}><MetricTile label="Daily P&amp;L" value={`$${latest?.daily_pnl?.toFixed(0) ?? "—"}`} accent={(latest?.daily_pnl ?? 0) >= 0 ? "#6abf69" : "#f85149"} /></Grid>
        <Grid item xs={6} md={3}><MetricTile label="Drawdown" value={`${((latest?.drawdown ?? 0) * 100).toFixed(2)}%`} /></Grid>
        <Grid item xs={6} md={3}><MetricTile label="Mode" value={String(risk?.status ?? "paper")} /></Grid>
      </Grid>
      <Grid container spacing={2}>
        <Grid item xs={12} md={8}>
          {portfolio && <EquityChart data={portfolio.equity_curve} />}
        </Grid>
        <Grid item xs={12} md={4}>
          <RankingsTable rankings={rankings} />
        </Grid>
        <Grid item xs={12}>
          {portfolio && <PositionsTable positions={portfolio.positions} />}
        </Grid>
      </Grid>
    </Container>
  );
}
