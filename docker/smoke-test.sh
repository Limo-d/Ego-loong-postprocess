#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/opt/ego-loong-postprocess}"
ROS_PYTHON="${ROS_PYTHON:-/usr/bin/python3}"
LOCATE_PYTHON="${LOCATE_PYTHON:-/opt/conda/envs/locate_anything/bin/python}"
HAMER_PYTHON="${HAMER_PYTHON:-/opt/conda/envs/hamer/bin/python}"

printf '[smoke] GPU\n'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

printf '[smoke] ROS 2 / hand_frame\n'
"${ROS_PYTHON}" -c 'import rosbag2_py; from hand_frame.msg import HandFrame, HandImuSample; print(HandFrame, HandImuSample)'

printf '[smoke] LocateAnything environment\n'
"${LOCATE_PYTHON}" -c 'import torch, transformers, cv2; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, transformers.__version__, cv2.__version__)'

printf '[smoke] HaMeR / Retarget environment\n'
"${HAMER_PYTHON}" -c "import sys, torch; from pathlib import Path; sys.path[:0]=['${ROOT}/hamer','${ROOT}']; import hamer; from hamer.models import load_hamer; from preprocess.BuildGloveFk21FromFusion import load_retarget_fk; HandConfig, parse_state27, reconstruct = load_retarget_fk(Path('${RETARGET_ROOT}')); assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, HandConfig, parse_state27, reconstruct)"

for required in \
  "${LOCATE_MODEL}/config.json" \
  "${LOCATE_MODEL}/model-00001-of-00002.safetensors" \
  "${LOCATE_MODEL}/model-00002-of-00002.safetensors" \
  "${ROOT}/hamer/_DATA/hamer_ckpts/checkpoints/new_hamer_weights.ckpt" \
  "${ROOT}/hamer/_DATA/data/mano/MANO_RIGHT.pkl"; do
  [[ -s "${required}" ]] || { printf '[smoke] missing: %s\n' "${required}" >&2; exit 1; }
done

printf '[smoke] static checks passed\n'

if [[ "${FULL_SMOKE:-0}" == "1" ]]; then
  bag_session="${SMOKE_BAG_SESSION:-${1:-}}"
  [[ -n "${bag_session}" ]] || { printf 'FULL_SMOKE=1 requires SMOKE_BAG_SESSION or first argument\n' >&2; exit 2; }
  output_session="${SMOKE_OUTPUT_SESSION:-${ROOT}/postprocess_data/docker_smoke}"
  printf '[smoke] one-frame pipeline: %s\n' "${bag_session}"
  BAG_SESSION="${bag_session}" SESSION="${output_session}" MAX_FRAMES=1 \
    FRAME_START=0 FRAME_END=0 OVERWRITE=1 RUN_QUALITY_CHECK=0 COMPACT_OUTPUTS=0 \
    "${ROOT}/scripts/run_sampler_bag_to_glove_trajectory.sh"
fi
