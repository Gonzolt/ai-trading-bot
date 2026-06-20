import {
  Chip,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import type { Trade } from "../lib/api";

type Props = {
  trades: Trade[];
};

export function TradesTable({ trades }: Props) {
  return (
    <Paper className="panel table-panel" elevation={0}>
      <Typography variant="h6">Trade Log</Typography>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Time</TableCell>
            <TableCell>Symbol</TableCell>
            <TableCell>Side</TableCell>
            <TableCell align="right">Qty</TableCell>
            <TableCell align="right">Price</TableCell>
            <TableCell align="right">Realized</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {trades.map((trade, index) => (
            <TableRow key={`${trade.created_at}-${index}`}>
              <TableCell>{new Date(trade.created_at).toLocaleString()}</TableCell>
              <TableCell>{trade.symbol}</TableCell>
              <TableCell>
                <Chip size="small" color={trade.side === "buy" ? "primary" : "warning"} label={trade.side} />
              </TableCell>
              <TableCell align="right">{trade.quantity.toFixed(2)}</TableCell>
              <TableCell align="right">${trade.price.toFixed(2)}</TableCell>
              <TableCell align="right" className={trade.realized_pnl >= 0 ? "up" : "down"}>
                ${trade.realized_pnl.toFixed(2)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Paper>
  );
}
