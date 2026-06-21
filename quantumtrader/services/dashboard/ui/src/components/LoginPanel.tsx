import { Box, Button, Paper, TextField, Typography } from "@mui/material";

type Props = {
  onLogin: (username: string, password: string) => void;
  error?: string;
};

export function LoginPanel({ onLogin, error }: Props) {
  return (
    <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
      <Paper className="panel" sx={{ width: 360 }}>
        <Typography variant="h5" gutterBottom>
          QuantumTrader
        </Typography>
        <Typography variant="body2" color="text.secondary" mb={2}>
          Sign in to monitor and control your trading bot.
        </Typography>
        <Box component="form" onSubmit={(e) => {
          e.preventDefault();
          const fd = new FormData(e.currentTarget);
          onLogin(String(fd.get("username")), String(fd.get("password")));
        }}>
          <TextField fullWidth name="username" label="Username" margin="normal" defaultValue="admin" />
          <TextField fullWidth name="password" label="Password" type="password" margin="normal" />
          {error && <Typography color="error" variant="body2">{error}</Typography>}
          <Button fullWidth type="submit" variant="contained" sx={{ mt: 2 }}>
            Sign In
          </Button>
        </Box>
      </Paper>
    </Box>
  );
}
