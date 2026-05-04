#!/usr/bin/env bash
# Pings both the API and the frontend every 4 minutes to prevent the dev repl from hibernating.
API_PORT="${API_PORT:-3001}"
FRONTEND_PORT="${FRONTEND_PORT:-5000}"
INTERVAL=240

echo "[keepalive] Starting — pinging API :${API_PORT} and frontend :${FRONTEND_PORT} every ${INTERVAL}s"

wait_for_port() {
  local port=$1
  local max_attempts=15
  local attempt=0
  while ! curl -sf --max-time 3 -o /dev/null "http://localhost:${port}/" 2>/dev/null && \
        ! curl -sf --max-time 3 -o /dev/null "http://localhost:${port}/api/healthz" 2>/dev/null; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "[keepalive] Timed out waiting for port ${port}"
      return 1
    fi
    sleep 5
  done
  echo "[keepalive] Port ${port} is ready"
}

# Wait for both services to be up before starting the ping loop
wait_for_port "$API_PORT"
wait_for_port "$FRONTEND_PORT"

while true; do
  sleep "$INTERVAL"
  API_STATUS=$(curl -sf --max-time 5 -o /dev/null -w "%{http_code}" "http://localhost:${API_PORT}/api/healthz" 2>/dev/null || echo "ERR")
  FRONTEND_STATUS=$(curl -sf --max-time 5 -o /dev/null -w "%{http_code}" "http://localhost:${FRONTEND_PORT}/" 2>/dev/null || echo "ERR")
  echo "[keepalive] api=${API_STATUS}  frontend=${FRONTEND_STATUS}"
done
