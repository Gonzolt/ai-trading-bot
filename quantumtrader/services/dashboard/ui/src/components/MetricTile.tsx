import { Box, Typography } from "@mui/material";

type Props = { label: string; value: string; accent?: string };

export function MetricTile({ label, value, accent }: Props) {
  return (
    <Box className="metric-tile">
      <Typography variant="caption" color="text.secondary">{label}</Typography>
      <Typography variant="h5" sx={{ color: accent || "text.primary" }}>{value}</Typography>
    </Box>
  );
}
