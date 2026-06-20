import AccountBalanceWalletIcon from "@mui/icons-material/AccountBalanceWallet";
import LogoutIcon from "@mui/icons-material/Logout";
import SecurityIcon from "@mui/icons-material/Security";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import { AppBar, Box, Button, Container, Grid, Stack, Toolbar, Typography } from "@mui/material";
import { useEffect, useMemo, useState } from "react";
import { EquityChart } from "./components/EquityChart";
import { LoginPanel } from "./components/LoginPanel";
import { MetricTile } from "./components/MetricTile";
import { PositionsTable } from "./components/PositionsTable";
import { RankingsTable } from "./components/RankingsTable";
import { TradesTable } from "./components/TradesTable";
import { apiGet, EquityPoint, Position, Ranking, RiskSnapshot, Trade, wsUrl } from "./lib/api";

type Portfolio = {
  positions: Position[];
  equity_curve: EquityPoint[];
};

export default function App() {
  const [token, setToken] = useState(() => localStorage.getItem("token") ?? "");
  const [portfolio, setPortfolio] = useState<Portfolio>({ positions: [], equity_curve: [] });
  const [rankings, setRankings] = useState<Ranking[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [risk, setRisk] = useState<RiskSnapshot | null>(null);
  const [live, setLive] = useState<EquityPoint | null>(null);

  function saveToken(nextToken: string) {
    localStorage.setItem("token", nextToken);
    setToken(nextToken);
  }

  function logout() {
    localStorage.removeItem("token");
    setToken("");
  }

  useEffect(() => {
    if (!token) return;
    async function refresh() {
      const [portfolioData, riskData, rankingData, tradeData] = await Promise.all([
        apiGet<Portfolio>("/portfolio", token),
        apiGet<RiskSnapshot>("/risk", token),
        apiGet<Ranking[]>("/rankings", token),
        apiGet<Trade[]>("/trades", token),
      ]);
      setPortfolio(portfolioData);
      setRisk(riskData);
      setRankings(rankingData);
      setTrades(tradeData);
    }
    refresh().catch(logout);
    const interval = window.setInterval(() => refresh().catch(console.error), 5000);
    return () => window.clearInterval(interval);
  }, [token]);

  useEffect(() => {
    if (!token) return;
    const socket = new WebSocket(wsUrl());
    socket.onmessage = (event) => {
      const point = JSON.parse(event.data) as EquityPoint;
      setLive(point);
      setPortfolio((current) => ({
        ...current,
        equity_curve: [...current.equity_curve.slice(-119), point],
      }));
    };
    return () => socket.close();
  }, [token]);

  const latestEquityPoint = portfolio.equity_curve[portfolio.equity_curve.length - 1];
  const equity = live?.equity ?? latestEquityPoint?.equity ?? 0;
  const pnl = live?.daily_pnl ?? latestEquityPoint?.daily_pnl ?? 0;
  const exposure = useMemo(
    () => portfolio.positions.reduce((sum, item) => sum + item.quantity * item.mark_price, 0),
    [portfolio.positions],
  );

  if (!token) {
    return <LoginPanel onToken={saveToken} />;
  }

  return (
    <Box>
      <AppBar position="sticky" elevation={0} className="topbar">
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Universal AI Trading Bot
          </Typography>
          <Button color="inherit" startIcon={<LogoutIcon />} onClick={logout}>
            Sign out
          </Button>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" className="dashboard">
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} lg={3}>
            <MetricTile label="Equity" value={`$${equity.toLocaleString()}`} icon={<AccountBalanceWalletIcon />} />
          </Grid>
          <Grid item xs={12} sm={6} lg={3}>
            <MetricTile
              label="Daily P&L"
              value={`$${pnl.toLocaleString()}`}
              icon={<TrendingUpIcon />}
              tone={pnl >= 0 ? "good" : "bad"}
            />
          </Grid>
          <Grid item xs={12} sm={6} lg={3}>
            <MetricTile
              label="Exposure"
              value={`$${Math.round(exposure).toLocaleString()}`}
              icon={<SecurityIcon />}
              tone="warn"
            />
          </Grid>
          <Grid item xs={12} sm={6} lg={3}>
            <MetricTile
              label="Risk Status"
              value={risk?.status ?? "loading"}
              icon={<WarningAmberIcon />}
              tone={risk?.live_trading_enabled ? "warn" : "good"}
            />
          </Grid>

          <Grid item xs={12} lg={8}>
            <EquityChart data={portfolio.equity_curve} />
          </Grid>
          <Grid item xs={12} lg={4}>
            <Stack spacing={2}>
              <RankingsTable rankings={rankings} />
            </Stack>
          </Grid>
          <Grid item xs={12} lg={6}>
            <PositionsTable positions={portfolio.positions} />
          </Grid>
          <Grid item xs={12} lg={6}>
            <TradesTable trades={trades} />
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}
