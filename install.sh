#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/universal-ai-trading-bot}"
REPO_URL="${REPO_URL:-}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root or with sudo."
  exit 1
fi

apt-get update
apt-get install -y ca-certificates curl gnupg git
install -m 0755 -d /etc/apt/keyrings
if [[ ! -f /etc/apt/keyrings/docker.gpg ]]; then
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
fi

. /etc/os-release
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

if [[ ! -d "${APP_DIR}/.git" ]]; then
  if [[ -z "${REPO_URL}" ]]; then
    echo "Set REPO_URL=https://github.com/you/universal-ai-trading-bot.git or clone the repo to ${APP_DIR} first."
    exit 1
  fi
  git clone "${REPO_URL}" "${APP_DIR}"
fi

cd "${APP_DIR}"
if [[ ! -f .env ]]; then
  cp .env.example .env
  chmod 600 .env
  echo "Created .env. Edit secrets before enabling live trading."
fi

docker compose pull || true
docker compose up -d --build

cp deploy/trading-bot.service /etc/systemd/system/trading-bot.service
systemctl daemon-reload
systemctl enable trading-bot.service

echo "Trading bot stack is starting. Open http://SERVER_IP/ after containers become healthy."
