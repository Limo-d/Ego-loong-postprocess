#!/usr/bin/env bash
set -euo pipefail

MIN_DRIVER="${MIN_DRIVER:-580.65.06}"
NVIDIA_REPO_BASE="${NVIDIA_REPO_BASE:-https://raw.githubusercontent.com/NVIDIA/libnvidia-container/gh-pages}"

version_ge() {
  dpkg --compare-versions "$1" ge "$2"
}

printf '[host] OS: '
. /etc/os-release
printf '%s %s\n' "${NAME}" "${VERSION_ID}"

if command -v nvidia-smi >/dev/null 2>&1; then
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d ' ')"
  printf '[host] NVIDIA driver: %s\n' "${driver_version}"
  if ! version_ge "${driver_version}" "${MIN_DRIVER}"; then
    printf '[ERROR] CUDA 13.0 requires NVIDIA driver >= %s\n' "${MIN_DRIVER}" >&2
    printf 'Upgrade the host driver before running the image; this script does not replace kernel drivers automatically.\n' >&2
    exit 1
  fi
else
  printf '[ERROR] nvidia-smi not found. Install an NVIDIA R580-or-newer host driver first.\n' >&2
  exit 1
fi

if [[ "${INSTALL:-0}" == "1" ]]; then
  [[ "$(id -u)" == "0" ]] || { printf 'Run with sudo when INSTALL=1\n' >&2; exit 2; }
  apt-get -o Acquire::ForceIPv4=true update
  apt-get install -y ca-certificates curl gnupg
  if ! command -v docker >/dev/null 2>&1; then
    apt-get install -y docker.io
  fi
  apt-get install -y docker-buildx docker-compose-v2
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL --connect-timeout 10 --max-time 60 --retry 2 \
    "${NVIDIA_REPO_BASE}/gpgkey" \
    | gpg --batch --yes --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg
  repo_arch="$(dpkg --print-architecture)"
  printf 'deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg] %s/stable/deb/%s /\n' \
    "${NVIDIA_REPO_BASE}" "${repo_arch}" \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get -o Acquire::ForceIPv4=true update
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

command -v docker >/dev/null 2>&1 || { printf '[ERROR] Docker not found. Re-run with sudo INSTALL=1.\n' >&2; exit 1; }
docker info >/dev/null
printf '[host] Docker: %s\n' "$(docker version --format '{{.Server.Version}}')"
if docker buildx version >/dev/null 2>&1; then
  printf '[host] Buildx: %s\n' "$(docker buildx version | head -n 1)"
else
  printf '[host] Buildx: missing (INSTALL=1 installs it)\n'
fi
if docker compose version >/dev/null 2>&1; then
  printf '[host] Compose: %s\n' "$(docker compose version --short)"
else
  printf '[host] Compose: missing (INSTALL=1 installs it)\n'
fi
printf '[host] Run GPU test after the image is built: docker run --rm --gpus all <image> smoke\n'
