#!/bin/bash
# Expose HDFS WebHDFS (HTTP) via ngrok for Colab access.
# Requires: ngrok installed and authenticated.
# Usage: ./scripts/start_hdfs_tunnel.sh

set -euo pipefail

if ! command -v ngrok &> /dev/null; then
  echo "Error: ngrok not installed. Install via: brew install ngrok"
  exit 1
fi

echo "Starting single-endpoint WebHDFS proxy (for ngrok free)..."
echo ""
echo "Starting HttpFS (single endpoint, full WebHDFS API; no datanode redirects)..."
docker compose up -d httpfs
echo ""
echo "Start ONE ngrok HTTP tunnel to HttpFS:"
echo "  ngrok http ${HTTPFS_PORT:-14000}"
echo ""
echo "Then in Colab set:"
echo "  HDFS_WEBHDFS_URL = \"https://...ngrok-free.app\""
echo ""
echo "Health check (local):"
echo "  curl -sS \"http://localhost:${HTTPFS_PORT:-14000}/webhdfs/v1/?op=LISTSTATUS&user.name=root\" | head"
echo ""
echo "Leave ngrok running while Colab runs."

