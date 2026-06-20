import {
  LinearProgress,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import type { Ranking } from "../lib/api";

type Props = {
  rankings: Ranking[];
};

export function RankingsTable({ rankings }: Props) {
  return (
    <Paper className="panel table-panel" elevation={0}>
      <Typography variant="h6">Asset Ranking</Typography>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Rank</TableCell>
            <TableCell>Symbol</TableCell>
            <TableCell align="right">Score</TableCell>
            <TableCell>Composite</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rankings.map((row) => (
            <TableRow key={row.symbol}>
              <TableCell>{row.rank}</TableCell>
              <TableCell>{row.symbol}</TableCell>
              <TableCell align="right">{row.score.toFixed(1)}</TableCell>
              <TableCell>
                <LinearProgress variant="determinate" value={row.score} />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Paper>
  );
}
