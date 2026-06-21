import { createTheme } from "@mui/material/styles";

export const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: { main: "#6abf69" },
    background: { default: "#0d1117", paper: "#161b22" },
    text: { primary: "#d9e1dc" },
  },
  typography: { fontFamily: "'Inter', 'Segoe UI', sans-serif" },
});
