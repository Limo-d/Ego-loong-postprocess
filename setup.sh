#!/usr/bin/env bash
set -euo pipefail

# Setup for the current Ego-loong postprocess pipeline.
#
# Pipeline covered:
#   ROS2 bag -> RGBD/hand_frame extract -> LocateAnything bbox -> stable bbox
#   -> HaMeR from bbox -> depth-root correction -> visual+glove fusion
#   -> glove FK/calibration/root tracking -> trajectory JSONL/MP4.
#
# This script intentionally does not install removed training, web, Project Aria,
# WiLoR, ViTPose, SAM, CoTracker, or robot-learning dependencies.
#
# Defaults match scripts/run_sampler_bag_to_glove_trajectory.sh.
# Override with env vars when needed:
#   LOCATE_ENV=locate_anything
#   HAMER_ENV=hamer
#   CUDA_WHEEL_INDEX=https://download.pytorch.org/whl/cu130
#   SKIP_INSTALL=1        only verify existing envs and paths
#   SKIP_ROS_BUILD=1      skip hand_msg_ws colcon build
#   RUN_SMOKE=1           run a 1-frame end-to-end smoke test after setup
#   SMOKE_BAG_SESSION=... required when RUN_SMOKE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

LOCATE_ENV="${LOCATE_ENV:-locate_anything}"
HAMER_ENV="${HAMER_ENV:-hamer}"
CONDA_BIN="${CONDA_BIN:-conda}"
CUDA_WHEEL_INDEX="${CUDA_WHEEL_INDEX:-https://download.pytorch.org/whl/cu130}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
HAND_MSG_WS="${HAND_MSG_WS:-${ROOT}/hand_msg_ws}"
LOCATE_MODEL="${LOCATE_MODEL:-${ROOT}/models--nvidia--LocateAnything-3B/resolved}"
HAMER_CACHE="${HAMER_CACHE:-${HOME}/.cache/hamer}"
BASE_HAND_CONFIG="${BASE_HAND_CONFIG:-/home/lenovo/Retarget/host/hand_config.json}"
RETARGET_ROOT="${RETARGET_ROOT:-/home/lenovo/Retarget/retarget}"
RESOLVE_DRIVER="${RESOLVE_DRIVER:-/home/lenovo/Retarget/data/ros_ws/resolve_check/resolve_driver}"
SMOKE_BAG_SESSION="${SMOKE_BAG_SESSION:-}"

info() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
fail() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

have_cmd() { command -v "$1" >/dev/null 2>&1; }

conda_env_exists() {
  "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "$1"
}

conda_run() {
  local env_name="$1"
  shift
  "${CONDA_BIN}" run -n "${env_name}" "$@"
}

pip_install() {
  local env_name="$1"
  shift
  conda_run "${env_name}" python -m pip install "$@"
}

check_python_import() {
  local env_name="$1"
  local label="$2"
  local code="$3"
  if conda_run "${env_name}" python -c "${code}" >/dev/null 2>&1; then
    printf '  [OK]   %-28s (%s)\n' "${label}" "${env_name}"
  else
    printf '  [MISS] %-28s (%s)\n' "${label}" "${env_name}"
    return 1
  fi
}

ensure_conda_env() {
  local env_name="$1"
  if conda_env_exists "${env_name}"; then
    info "Conda env exists: ${env_name}"
  else
    info "Creating conda env: ${env_name} (python=3.10)"
    "${CONDA_BIN}" create -y -n "${env_name}" python=3.10 pip
  fi
}

install_locate_env() {
  ensure_conda_env "${LOCATE_ENV}"
  if [[ "${SKIP_INSTALL:-0}" == "1" ]]; then
    info "SKIP_INSTALL=1: not installing LocateAnything packages"
    return
  fi

  info "Installing LocateAnything runtime packages into ${LOCATE_ENV}"
  pip_install "${LOCATE_ENV}" --upgrade pip setuptools wheel
  pip_install "${LOCATE_ENV}" --index-url "${CUDA_WHEEL_INDEX}" torch torchvision
  pip_install "${LOCATE_ENV}" \
    'numpy>=1.26,<2.0' \
    'opencv-python>=4.8' \
    'pillow>=10' \
    'tqdm>=4.65' \
    'transformers>=4.57' \
    'huggingface-hub>=0.36' \
    'accelerate>=1.5' \
    'safetensors>=0.8'
}

install_hamer_env() {
  ensure_conda_env "${HAMER_ENV}"
  if [[ "${SKIP_INSTALL:-0}" == "1" ]]; then
    info "SKIP_INSTALL=1: not installing HaMeR/FK packages"
    return
  fi

  info "Installing HaMeR/FK runtime packages into ${HAMER_ENV}"
  pip_install "${HAMER_ENV}" --upgrade pip setuptools wheel
  pip_install "${HAMER_ENV}" --index-url "${CUDA_WHEEL_INDEX}" torch torchvision
  pip_install "${HAMER_ENV}" \
    'numpy>=1.23,<2.0' \
    'scipy>=1.10' \
    'opencv-python>=4.8' \
    'pillow>=10' \
    'tqdm>=4.65' \
    'pyyaml>=6' \
    'python-box>=7' \
    'matplotlib>=3.7' \
    'imageio>=2.30' \
    'imageio-ffmpeg>=0.5' \
    'pytorch-lightning>=2.6' \
    'timm>=1.0' \
    'einops>=0.8' \
    'yacs>=0.1.8' \
    'smplx>=0.1.28' \
    'pyrender>=0.1.45' \
    'trimesh>=4.0' \
    'scikit-image>=0.25' \
    'braceexpand>=0.1' \
    'webdataset>=1.0' \
    'mediapipe>=0.10' \
    'huggingface-hub>=0.36' \
    'chumpy>=0.70'
}

build_ros_messages() {
  if [[ "${SKIP_ROS_BUILD:-0}" == "1" ]]; then
    info "SKIP_ROS_BUILD=1: not building hand_msg_ws"
    return
  fi
  [[ -f "${ROS_SETUP}" ]] || fail "ROS setup not found: ${ROS_SETUP}"
  [[ -d "${HAND_MSG_WS}/src/hand_frame" ]] || fail "hand_frame package not found under ${HAND_MSG_WS}/src"
  have_cmd colcon || fail "colcon not found. Install ROS 2 colcon tools first."

  info "Building ROS hand_frame messages in ${HAND_MSG_WS}"
  (
    cd "${HAND_MSG_WS}"
    # Force system Python. Using conda Python can fail ROS interface generation.
    set +u
    source "${ROS_SETUP}"
    set -u
    colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
  )
}

verify_paths() {
  info "Checking required local paths"
  [[ -f "${ROOT}/scripts/run_sampler_bag_to_glove_trajectory.sh" ]] || fail "missing main pipeline script"
  [[ -f "${ROOT}/scripts/run_glove_fk_visual_bones_pipeline.sh" ]] || fail "missing glove FK pipeline script"
  [[ -f "${ROOT}/preprocess/ExtractRosbagSampler.py" ]] || fail "missing preprocess scripts"
  [[ -d "${ROOT}/hamer/hamer" ]] || fail "missing local hamer package"
  [[ -d "${LOCATE_MODEL}" ]] || fail "missing LocateAnything model dir: ${LOCATE_MODEL}"
  [[ -f "${LOCATE_MODEL}/config.json" ]] || fail "LocateAnything config not found under ${LOCATE_MODEL}"
  [[ -f "${LOCATE_MODEL}/model-00001-of-00002.safetensors" ]] || fail "LocateAnything shard 1 not found"
  [[ -f "${LOCATE_MODEL}/model-00002-of-00002.safetensors" ]] || fail "LocateAnything shard 2 not found"
  [[ -f "${BASE_HAND_CONFIG}" ]] || fail "Base hand config not found: ${BASE_HAND_CONFIG}"
  [[ -f "${RETARGET_ROOT}/hand_retarget/config.py" ]] || fail "Retarget FK config module not found under: ${RETARGET_ROOT}"
  [[ -f "${RETARGET_ROOT}/hand_retarget/layout.py" ]] || fail "Retarget FK layout module not found under: ${RETARGET_ROOT}"
  [[ -f "${RETARGET_ROOT}/hand_retarget/human_fk.py" ]] || fail "Retarget FK module not found under: ${RETARGET_ROOT}"
  if [[ -x "${RESOLVE_DRIVER}" ]]; then
    info "Found legacy /glove resolve_driver: ${RESOLVE_DRIVER}"
  else
    warn "Legacy /glove resolve_driver not executable: ${RESOLVE_DRIVER}; /hand_frame bags are unaffected."
  fi
  [[ -f "${HAND_MSG_WS}/install/setup.bash" ]] || fail "hand_msg_ws/install/setup.bash missing; run setup without SKIP_ROS_BUILD"
}

verify_ros() {
  info "Checking ROS Python imports"
  set +u
  source "${ROS_SETUP}"
  source "${HAND_MSG_WS}/install/setup.bash"
  set -u
  /usr/bin/python3 - <<'PY'
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
msg = get_message('hand_frame/msg/HandFrame')
print(msg)
PY
}

verify_locate_env() {
  info "Checking ${LOCATE_ENV} imports"
  local ok=0
  check_python_import "${LOCATE_ENV}" torch 'import torch; assert torch.cuda.is_available()' || ok=1
  check_python_import "${LOCATE_ENV}" transformers 'import transformers' || ok=1
  check_python_import "${LOCATE_ENV}" huggingface_hub 'import huggingface_hub' || ok=1
  check_python_import "${LOCATE_ENV}" cv2 'import cv2' || ok=1
  check_python_import "${LOCATE_ENV}" PIL 'from PIL import Image' || ok=1
  check_python_import "${LOCATE_ENV}" safetensors 'import safetensors' || ok=1
  return "${ok}"
}

verify_hamer_env() {
  info "Checking ${HAMER_ENV} imports"
  local ok=0
  check_python_import "${HAMER_ENV}" torch 'import torch; assert torch.cuda.is_available()' || ok=1
  check_python_import "${HAMER_ENV}" local_hamer "import sys; sys.path[:0]=['${ROOT}/hamer','${ROOT}']; import hamer; from hamer.models import load_hamer" || ok=1
  check_python_import "${HAMER_ENV}" smplx 'import smplx' || ok=1
  check_python_import "${HAMER_ENV}" timm 'import timm' || ok=1
  check_python_import "${HAMER_ENV}" yacs 'import yacs' || ok=1
  check_python_import "${HAMER_ENV}" pytorch_lightning 'import pytorch_lightning' || ok=1
  check_python_import "${HAMER_ENV}" cv2 'import cv2' || ok=1
  check_python_import "${HAMER_ENV}" scipy 'import scipy' || ok=1
  check_python_import "${HAMER_ENV}" matplotlib 'import matplotlib' || ok=1
  check_python_import "${HAMER_ENV}" pyrender 'import pyrender' || ok=1
  check_python_import "${HAMER_ENV}" project_scripts "import sys; sys.path[:0]=['${ROOT}/hamer','${ROOT}']; import preprocess.run_hamer_from_locate_bboxes; import preprocess.BuildHandFusionInput" || ok=1
  return "${ok}"
}

verify_hamer_cache() {
  info "Checking HaMeR cached model files"
  local ckpt="${HAMER_CACHE}/hamer_ckpts/checkpoints/hamer.ckpt"
  local mano="${HAMER_CACHE}/data/mano/MANO_RIGHT.pkl"
  if [[ -e "${ckpt}" ]]; then
    info "Found HaMeR checkpoint: ${ckpt}"
  else
    warn "HaMeR checkpoint not found at ${ckpt}; first HaMeR run may download it via HuggingFace."
  fi
  if [[ -e "${mano}" ]]; then
    info "Found MANO_RIGHT.pkl: ${mano}"
  else
    warn "MANO_RIGHT.pkl not found at ${mano}; first HaMeR run may download it."
  fi
}

run_smoke() {
  if [[ "${RUN_SMOKE:-0}" != "1" ]]; then
    return
  fi
  [[ -n "${SMOKE_BAG_SESSION}" ]] || fail "RUN_SMOKE=1 requires SMOKE_BAG_SESSION=/path/to/session-or-data"
  if [[ ! -d "${SMOKE_BAG_SESSION}/bag" && ! -d "${SMOKE_BAG_SESSION}/data/bag" ]]; then
    fail "Smoke ROS2 bag not found under SMOKE_BAG_SESSION: ${SMOKE_BAG_SESSION}"
  fi
  info "Running 1-frame smoke test"
  BAG_SESSION="${SMOKE_BAG_SESSION}" \
    BASE_HAND_CONFIG="${BASE_HAND_CONFIG}" \
    RETARGET_ROOT="${RETARGET_ROOT}" \
    RESOLVE_DRIVER="${RESOLVE_DRIVER}" \
    MAX_FRAMES=1 OVERWRITE=1 FRAME_START=0 FRAME_END=0 \
    SESSION="${ROOT}/postprocess_data/setup_smoke" \
    bash "${ROOT}/scripts/run_sampler_bag_to_glove_trajectory.sh"
  info "Smoke output: ${ROOT}/postprocess_data/setup_smoke"
}

main() {
  have_cmd "${CONDA_BIN}" || fail "conda not found. Set CONDA_BIN=/path/to/conda if needed."

  info "Project root: ${ROOT}"
  info "Locate env:   ${LOCATE_ENV}"
  info "HaMeR env:    ${HAMER_ENV}"

  install_locate_env
  install_hamer_env
  build_ros_messages
  verify_paths
  verify_ros

  local verify_failed=0
  verify_locate_env || verify_failed=1
  verify_hamer_env || verify_failed=1
  verify_hamer_cache

  if [[ "${verify_failed}" != "0" ]]; then
    fail "One or more environment checks failed. Review [MISS] lines above."
  fi

  run_smoke

  info "Setup complete. Main command:"
  printf '  bash scripts/run_sampler_bag_to_glove_trajectory.sh\n'
}

main "$@"
