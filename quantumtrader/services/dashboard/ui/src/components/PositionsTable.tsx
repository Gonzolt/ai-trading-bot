import { Paper, Table, TableBody, TableCell, TableHead, TableRow, Typography } from "@mui/material";

type Position = {
  symbol: string;
  quantity: number;
  average_price: number;
  mark_price: number;
  unrealized_pnl: number;
  strategy: string;
};

export function PositionsTable({ positions }: { positions: Position[] }) {
  return (
    <Paper className="panel" elevation={0}>
      <Typography variant="h6" gutterBottom>Open Positions</Typography>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Symbol</TableCell>
            <TableCell align="right">Qty</TableCell>
            <TableCell align="right">Avg</TableCell>
            <TableCell align="right">Mark</TableCell>
            <TableCell align="right">P&amp;L</TableCell>
            <TableCell>Strategy</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {positions.map((p) => (
            <TableRow key={p.symbol}>
              <TableCell>{p.symbol}</TableCell>
              <TableCell align="right">{p.quantity}</TableCell>
              <TableCell align="right">${p.average_price.toFixed(2)}</TableCell>
              <TableCell align="right">${p.mark_price.toFixed(2)}</TableCell>
              <TableCell align="right" sx={{ color: p.unrealized_pnl >= 0 ? "#6abf69" : "#f85149" }}>
                ${p.unrealized_pnl.toFixed(2)}
              </TableCell>
              <TableCell>{p.strategy}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Paper>
  );
}
