#!/usr/bin/env bash
# Starts the FastAPI backend (port 8000) and Vite dev server (port 5173)
# together. Ctrl-C kills both.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
VENV="$ROOT/.venv/bin/python"

# Start API server in background
"$VENV" -m uvicorn server:app --port 8000 --reload --app-dir "$ROOT" &
SERVER_PID=$!

# Kill server when this script exits for any reason
trap 'echo "Stopping server (pid $SERVER_PID)..."; kill "$SERVER_PID" 2>/dev/null' EXIT INT TERM

echo "API server started on http://localhost:8000 (pid $SERVER_PID)"
echo "Starting player..."

# Start Vite in foreground (Ctrl-C here triggers EXIT trap above)
cd "$ROOT/player" && npm run dev
