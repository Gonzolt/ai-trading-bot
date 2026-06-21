import { Paper, Table, TableBody, TableCell, TableHead, TableRow, Typography } from "@mui/material";

type Ranking = { rank: number; symbol: string; score: number };

export function RankingsTable({ rankings }: { rankings: Ranking[] }) {
  return (
    <Paper className="panel" elevation={0}>
      <Typography variant="h6" gutterBottom>Asset Rankings</Typography>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>#</TableCell>
            <TableCell>Symbol</TableCell>
            <TableCell align="right">Score</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rankings.map((r) => (
            <TableRow key={r.symbol}>
              <TableCell>{r.rank}</TableCell>
              <TableCell>{r.symbol}</TableCell>
              <TableCell align="right">{r.score.toFixed(1)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Paper>
  );
}
