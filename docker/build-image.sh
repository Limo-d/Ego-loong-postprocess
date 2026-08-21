#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-ego-loong-postprocess:ubuntu24.04-cuda13.0}"
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04@sha256:ae7f650405a3964972dacfa889273bf8e3fbe9709899afd187da01c4cdff3105}"
UBUNTU_MIRROR="${UBUNTU_MIRROR:-https://mirrors.ustc.edu.cn/ubuntu}"
ROS_MIRROR="${ROS_MIRROR:-https://mirrors.ustc.edu.cn/ros2/ubuntu}"
LOCATE_ENV_PREFIX="${LOCATE_ENV_PREFIX:-/home/lenovo/miniconda3/envs/locate_anything}"
HAMER_ENV_PREFIX="${HAMER_ENV_PREFIX:-/home/lenovo/miniconda3/envs/hamer}"
CONDA_PACK_BIN="${CONDA_PACK_BIN:-${HAMER_ENV_PREFIX}/bin/conda-pack}"
RETARGET_SOURCE="${RETARGET_SOURCE:-/home/lenovo/Retarget}"
ROS_KEY_SOURCE="${ROS_KEY_SOURCE:-/usr/share/keyrings/ros2-archive-keyring.gpg}"
ASSET_ROOT="${ROOT}/.docker-assets"
ENV_ASSET_DIR="${ASSET_ROOT}/envs"
RETARGET_ASSET_DIR="${ASSET_ROOT}/retarget"
ROS_ASSET_DIR="${ASSET_ROOT}/ros"

fail() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
info() { printf '[INFO] %s\n' "$*"; }

[[ -x "${CONDA_PACK_BIN}" ]] || fail "conda-pack not found: ${CONDA_PACK_BIN}"
[[ -x "${LOCATE_ENV_PREFIX}/bin/python" ]] || fail "Locate environment not found: ${LOCATE_ENV_PREFIX}"
[[ -x "${HAMER_ENV_PREFIX}/bin/python" ]] || fail "HaMeR environment not found: ${HAMER_ENV_PREFIX}"
[[ -d "${RETARGET_SOURCE}/retarget" ]] || fail "Retarget source not found: ${RETARGET_SOURCE}/retarget"
[[ -f "${RETARGET_SOURCE}/host/hand_config.json" ]] || fail "hand_config.json not found"
[[ -x "${RETARGET_SOURCE}/data/ros_ws/resolve_check/resolve_driver" ]] || fail "resolve_driver not found"
[[ -f "${ROS_KEY_SOURCE}" ]] || fail "ROS archive key not found: ${ROS_KEY_SOURCE}"
[[ -f "${ROOT}/models--nvidia--LocateAnything-3B/resolved/model-00001-of-00002.safetensors" ]] || fail "LocateAnything model shard missing"
[[ -f "${ROOT}/hamer/_DATA/hamer_ckpts/checkpoints/new_hamer_weights.ckpt" ]] || fail "HaMeR checkpoint missing"
[[ -f "${ROOT}/hamer/_DATA/data/mano/MANO_RIGHT.pkl" ]] || fail "MANO_RIGHT.pkl missing"

mkdir -p "${ENV_ASSET_DIR}" "${RETARGET_ASSET_DIR}" "${ROS_ASSET_DIR}"

pack_env() {
  local prefix="$1"
  local output="$2"
  local label="$3"
  if [[ -s "${output}" && "${FORCE_PACK:-0}" != "1" ]]; then
    if python3 -c 'import sys, tarfile; t=tarfile.open(sys.argv[1], "r:gz"); sum(1 for _ in t)' "${output}"; then
      info "Reuse verified ${label}: ${output}"
      return
    fi
    info "Discard invalid packed ${label}: ${output}"
    mv "${output}" "${output}.invalid-$(date +%Y%m%dT%H%M%S)"
  fi
  info "Packing ${label} from ${prefix}"
  rm -f "${output}"
  "${HAMER_ENV_PREFIX}/bin/python" "${CONDA_PACK_BIN}" \
    --prefix "${prefix}" --output "${output}" \
    --ignore-editable-packages --ignore-missing-files
  python3 -c 'import sys, tarfile; t=tarfile.open(sys.argv[1], "r:gz"); sum(1 for _ in t)' "${output}"
  info "Verified packed ${label}: ${output}"
}

pack_env "${LOCATE_ENV_PREFIX}" "${ENV_ASSET_DIR}/locate_anything.tar.gz" LocateAnything
pack_env "${HAMER_ENV_PREFIX}" "${ENV_ASSET_DIR}/hamer.tar.gz" HaMeR

if [[ "${PACK_ONLY:-0}" == "1" ]]; then
  info "PACK_ONLY=1: environment archives are ready"
  exit 0
fi

info "Staging Retarget runtime"
mkdir -p "${RETARGET_ASSET_DIR}/retarget"
rsync -a "${RETARGET_SOURCE}/retarget/" "${RETARGET_ASSET_DIR}/retarget/"
install -m 0644 "${RETARGET_SOURCE}/host/hand_config.json" "${RETARGET_ASSET_DIR}/hand_config.json"
install -m 0755 "${RETARGET_SOURCE}/data/ros_ws/resolve_check/resolve_driver" "${RETARGET_ASSET_DIR}/resolve_driver"
install -m 0644 "${ROS_KEY_SOURCE}" "${ROS_ASSET_DIR}/ros2-archive-keyring.gpg"

git_commit="$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || printf unknown)"
build_date="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

info "Building ${IMAGE_TAG}"
build_command=(docker build)
if docker buildx version >/dev/null 2>&1; then
  build_command=(docker buildx build --load)
  info "Using Buildx/BuildKit incremental cache"
fi
exec "${build_command[@]}" \
  --file "${ROOT}/docker/Dockerfile" \
  --target full \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "UBUNTU_MIRROR=${UBUNTU_MIRROR}" \
  --build-arg "ROS_MIRROR=${ROS_MIRROR}" \
  --build-arg "GIT_COMMIT=${git_commit}" \
  --build-arg "BUILD_DATE=${build_date}" \
  --tag "${IMAGE_TAG}" \
  "${ROOT}"
