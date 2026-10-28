import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev server on :3000 to match CORS_ORIGINS and LOCAL_RUN.md.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 3000, host: "127.0.0.1", fs: { allow: [".."] } },
  preview: { port: 3000, host: "127.0.0.1" },
});
