import { createTheme } from "@mui/material/styles";

export const theme = createTheme({
  palette: {
    mode: "dark",
    background: {
      default: "#101314",
      paper: "#171b1e",
    },
    primary: {
      main: "#6abf69",
    },
    secondary: {
      main: "#3aa7a3",
    },
    warning: {
      main: "#e6b84a",
    },
    error: {
      main: "#e66b5b",
    },
  },
  shape: {
    borderRadius: 8,
  },
  typography: {
    fontFamily:
      'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    h4: {
      fontWeight: 700,
      letterSpacing: 0,
    },
    h6: {
      fontWeight: 700,
      letterSpacing: 0,
    },
    button: {
      textTransform: "none",
      letterSpacing: 0,
    },
  },
});
