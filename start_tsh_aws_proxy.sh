#!/usr/bin/env bash
set -euo pipefail

APP_NAME=${1:-aws-betler-prod}
ENV_FILE="${HOME}/.tsh/aws_env_$APP_NAME"
PID_FILE="${ENV_FILE}.pid"

mkdir -p "$(dirname "$ENV_FILE")"
rm -f "$ENV_FILE" "$PID_FILE"

# Kill any existing AWS proxy processes
pkill -f "tsh proxy aws" 2>/dev/null || true

# Start proxy, save env vars + proxy PID
{
  tsh proxy aws --app "$APP_NAME" |
    awk '/export/ { sub(/^.*export /, "", $0); print $0; fflush() }' > "$ENV_FILE"
} >/dev/null 2>&1 &
PROXY_PID=$!
echo "$PROXY_PID" > "$PID_FILE"

# Wait a moment for the proxy to start and env vars to be written
sleep 2

# Add certificate-related environment variables if AWS_CA_BUNDLE is available
if [[ -f "$ENV_FILE" ]]; then
  # Source the env file to get AWS_CA_BUNDLE
  source "$ENV_FILE"

  # Add certificate exports if AWS_CA_BUNDLE is set
  if [[ -n "${AWS_CA_BUNDLE:-}" ]]; then
    {
      echo "SSL_CERT_FILE=$AWS_CA_BUNDLE"
      echo "REQUESTS_CA_BUNDLE=$AWS_CA_BUNDLE"
      echo "CURL_CA_BUNDLE=$AWS_CA_BUNDLE"
      echo "NODE_EXTRA_CA_CERTS=$AWS_CA_BUNDLE"
      echo "NODE_OPTIONS=--use-system-ca"
      echo "NO_PROXY=169.254.169.254,169.254.170.2,mcp.atlassian.com,mcp.notion.com"
    } >> "$ENV_FILE"
  fi
fi

# Return both PID and env file path
echo "$PROXY_PID $ENV_FILE"

