import { Paper, Stack, Typography } from "@mui/material";
import type { ReactNode } from "react";

type Props = {
  label: string;
  value: string;
  icon: ReactNode;
  tone?: "good" | "warn" | "bad";
};

export function MetricTile({ label, value, icon, tone = "good" }: Props) {
  return (
    <Paper className={`metric-tile metric-${tone}`} elevation={0}>
      <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={2}>
        <Stack spacing={0.5}>
          <Typography variant="caption" color="text.secondary">
            {label}
          </Typography>
          <Typography variant="h5">{value}</Typography>
        </Stack>
        <span className="metric-icon">{icon}</span>
      </Stack>
    </Paper>
  );
}
