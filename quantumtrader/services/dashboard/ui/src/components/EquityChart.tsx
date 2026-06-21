import { Paper, Typography } from "@mui/material";
import Plot from "react-plotly.js";
import type { EquityPoint } from "../lib/api";

type Props = {
  data: EquityPoint[];
};

/** Plotly equity curve with dark-theme styling. */
export function EquityChart({ data }: Props) {
  return (
    <Paper className="panel" elevation={0}>
      <Typography variant="h6" gutterBottom>
        Equity Curve
      </Typography>
      <Plot
        data={[
          {
            x: data.map((point) => point.time),
            y: data.map((point) => point.equity),
            type: "scatter",
            mode: "lines",
            line: { color: "#6abf69", width: 3 },
            fill: "tozeroy",
            fillcolor: "rgba(106,191,105,0.12)",
            name: "Equity",
          },
        ]}
        layout={{
          autosize: true,
          height: 330,
          margin: { l: 48, r: 20, t: 20, b: 40 },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "#d9e1dc" },
          xaxis: { gridcolor: "#263034" },
          yaxis: { gridcolor: "#263034", tickprefix: "$" },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </Paper>
  );
}
