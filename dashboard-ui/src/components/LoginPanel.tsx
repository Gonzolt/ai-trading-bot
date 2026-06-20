import LockIcon from "@mui/icons-material/Lock";
import LoginIcon from "@mui/icons-material/Login";
import { Alert, Box, Button, Paper, Stack, TextField, Typography } from "@mui/material";
import { FormEvent, useState } from "react";
import { login } from "../lib/api";

type Props = {
  onToken: (token: string) => void;
};

export function LoginPanel({ onToken }: Props) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("change-me");
  const [error, setError] = useState("");

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    try {
      const response = await login(username, password);
      onToken(response.access_token);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    }
  }

  return (
    <Box className="login-shell">
      <Paper className="login-panel" elevation={0}>
        <Stack spacing={3} component="form" onSubmit={submit}>
          <Stack direction="row" spacing={1.5} alignItems="center">
            <LockIcon color="primary" />
            <Typography variant="h5">Trading Dashboard</Typography>
          </Stack>
          {error && <Alert severity="error">{error}</Alert>}
          <TextField label="Username" value={username} onChange={(e) => setUsername(e.target.value)} />
          <TextField
            label="Password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <Button type="submit" variant="contained" endIcon={<LoginIcon />} size="large">
            Sign in
          </Button>
        </Stack>
      </Paper>
    </Box>
  );
}
