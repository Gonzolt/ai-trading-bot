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
import type { Position } from "../lib/api";

type Props = {
  positions: Position[];
};

export function PositionsTable({ positions }: Props) {
  return (
    <Paper className="panel table-panel" elevation={0}>
      <Typography variant="h6">Open Positions</Typography>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Symbol</TableCell>
            <TableCell align="right">Size</TableCell>
            <TableCell align="right">Mark</TableCell>
            <TableCell align="right">P&L</TableCell>
            <TableCell>Strategy</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {positions.map((position) => (
            <TableRow key={position.symbol}>
              <TableCell>{position.symbol}</TableCell>
              <TableCell align="right">{position.quantity.toFixed(2)}</TableCell>
              <TableCell align="right">${position.mark_price.toFixed(2)}</TableCell>
              <TableCell align="right" className={position.unrealized_pnl >= 0 ? "up" : "down"}>
                ${position.unrealized_pnl.toFixed(2)}
              </TableCell>
              <TableCell>
                <Chip size="small" label={position.strategy ?? "manual"} />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Paper>
  );
}
