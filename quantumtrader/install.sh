#!/usr/bin/env bash
# QuantumTrader – one-command deployment for Ubuntu 22.04/24.04 LTS
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/quantumtrader}"
REPO_URL="${REPO_URL:-}"
INSTALL_SYSTEMD="${INSTALL_SYSTEMD:-true}"
DOMAIN="${DOMAIN:-}"
LETSENCRYPT_EMAIL="${LETSENCRYPT_EMAIL:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[QuantumTrader]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*" >&2; }

cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        err "Installation failed (exit $exit_code). Check logs above."
    fi
}
trap cleanup EXIT

if [[ "$(id -u)" -ne 0 ]]; then
    err "Run as root: sudo ./install.sh"
    exit 1
fi

if ! grep -qiE 'ubuntu (22|24)\.' /etc/os-release 2>/dev/null; then
    warn "Tested on Ubuntu 22.04/24.04 LTS. Proceeding on $(grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '\"')"
fi

log "Updating system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq ca-certificates curl gnupg git ufw jq

log "Installing Docker..."
install -m 0755 -d /etc/apt/keyrings
if [[ ! -f /etc/apt/keyrings/docker.gpg ]]; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
fi

. /etc/os-release
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list

apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

systemctl enable docker
systemctl start docker

log "Configuring firewall (UFW)..."
ufw --force enable || true
ufw allow OpenSSH || true
ufw allow 80/tcp || true
ufw allow 443/tcp || true

if [[ ! -d "${APP_DIR}/.git" ]] && [[ ! -f "${APP_DIR}/docker-compose.yml" ]]; then
    if [[ -z "${REPO_URL}" ]]; then
        if [[ -f "./docker-compose.yml" ]]; then
            APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
            log "Using current directory: ${APP_DIR}"
        else
            err "Set REPO_URL=https://github.com/YOUR_ORG/QuantumTrader.git or run from cloned repo."
            exit 1
        fi
    else
        log "Cloning repository from ${REPO_URL}..."
        git clone "${REPO_URL}" "${APP_DIR}"
    fi
fi

cd "${APP_DIR}"

if [[ ! -f .env ]]; then
    if [[ -f config/.env.example ]]; then
        cp config/.env.example .env
    elif [[ -f .env.example ]]; then
        cp .env.example .env
    else
        err "No .env.example found."
        exit 1
    fi
    chmod 600 .env
    log "Created .env from template. Review secrets before live trading."
fi

# shellcheck disable=SC1091
set -a && source .env && set +a

log "Building and starting Docker services (this may take several minutes)..."
docker compose pull --ignore-buildable || true
docker compose up -d --build

log "Waiting for database health..."
for i in $(seq 1 30); do
    if docker compose exec -T db pg_isready -U "${POSTGRES_USER:-trader}" -d "${POSTGRES_DB:-quantumtrader}" &>/dev/null; then
        break
    fi
    sleep 2
done

if [[ -n "${DOMAIN}" && -n "${LETSENCRYPT_EMAIL}" ]]; then
    log "Installing Certbot for SSL on ${DOMAIN}..."
    apt-get install -y -qq certbot
    certbot certonly --standalone -d "${DOMAIN}" --email "${LETSENCRYPT_EMAIL}" --agree-tos --non-interactive || warn "SSL setup failed; using HTTP only."
fi

if [[ "${INSTALL_SYSTEMD}" == "true" ]] && [[ -f deploy/quantumtrader.service ]]; then
    log "Installing systemd service for auto-start on boot..."
    sed "s|/opt/quantumtrader|${APP_DIR}|g" deploy/quantumtrader.service > /etc/systemd/system/quantumtrader.service
    systemctl daemon-reload
    systemctl enable quantumtrader.service
fi

SERVER_IP="$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')"
DASH_USER="${DASHBOARD_USERNAME:-admin}"
DASH_PASS="${DASHBOARD_PASSWORD:-QuantumTrader2026!}"

echo ""
echo "============================================================"
echo -e "${GREEN}  QuantumTrader deployment complete!${NC}"
echo "============================================================"
echo ""
echo "  Dashboard URL:  http://${SERVER_IP}/"
if [[ -n "${DOMAIN}" ]]; then
    echo "  Domain URL:     https://${DOMAIN}/"
fi
echo ""
echo "  Login username: ${DASH_USER}"
echo "  Login password: ${DASH_PASS}"
echo ""
echo "  Trading mode:   ${TRADING_MODE:-paper}"
echo "  App directory:  ${APP_DIR}"
echo ""
warn "Change DASHBOARD_PASSWORD and JWT_SECRET in .env before production use."
echo ""
echo "  Useful commands:"
echo "    docker compose -f ${APP_DIR}/docker-compose.yml logs -f"
echo "    docker compose -f ${APP_DIR}/docker-compose.yml ps"
echo "============================================================"

trap - EXIT
