#!/usr/bin/env bash
set -euo pipefail

# Full sampler-bag pipeline:
# ROS2 bag -> RGBD/hand_frame extract -> RTAB-Map camera pose -> LocateAnything bbox
# -> stable bbox -> HaMeR wrist visual prior -> depth-root correction
# -> visual+glove fusion -> glove FK/calibration/root tracking/trajectory videos
# -> review web.

ROOT="${ROOT:-/home/lenovo/Ego-loong-postprocess}"
BAG_SESSION="${BAG_SESSION:-${ROOT}/datatsets/sampler_ros2_bags/rotation/data_20260628_123410}"
# Accept either the acquisition root (.../<session>) or its data directory.
# New batch datasets share <batch>/calibrations while keeping bag/map per session.
# Older datasets may keep bag/calibration directly under the acquisition root.
if [[ -d "${BAG_SESSION}/bag" ]]; then
  BAG_DATA_DIR="${BAG_SESSION}"
elif [[ -d "${BAG_SESSION}/data/bag" ]]; then
  BAG_DATA_DIR="${BAG_SESSION}/data"
else
  BAG_DATA_DIR="${BAG_SESSION}"
fi
if [[ "$(basename "${BAG_DATA_DIR%/}")" == "data" ]]; then
  BAG_ACQUISITION_ROOT="$(dirname "${BAG_DATA_DIR%/}")"
else
  BAG_ACQUISITION_ROOT="${BAG_DATA_DIR%/}"
fi
if [[ -z "${BATCH_ROOT:-}" ]]; then
  if [[ -d "${BAG_ACQUISITION_ROOT}/calibrations" ]]; then
    BATCH_ROOT="${BAG_ACQUISITION_ROOT}"
  elif [[ -d "$(dirname "${BAG_ACQUISITION_ROOT}")/calibrations" ]]; then
    BATCH_ROOT="$(dirname "${BAG_ACQUISITION_ROOT}")"
  else
    BATCH_ROOT="${BAG_ACQUISITION_ROOT}"
  fi
fi
if [[ -z "${CALIBRATION_DIR:-}" ]]; then
  if [[ -d "${BATCH_ROOT}/calibrations" ]]; then
    CALIBRATION_DIR="${BATCH_ROOT}/calibrations"
  elif [[ -d "${BAG_DATA_DIR}/calibration" ]]; then
    CALIBRATION_DIR="${BAG_DATA_DIR}/calibration"
  else
    CALIBRATION_DIR="${BAG_ACQUISITION_ROOT}/calibration"
  fi
fi
if [[ -z "${BAG_DIR:-}" ]]; then
  BAG_DIR="${BAG_DATA_DIR}/bag"
fi
if [[ -z "${HAND_CALIBRATION_FILE:-}" ]]; then
  if [[ -f "${CALIBRATION_DIR}/hand_calibration.txt" ]]; then
    HAND_CALIBRATION_FILE="${CALIBRATION_DIR}/hand_calibration.txt"
  else
    HAND_CALIBRATION_FILE="${CALIBRATION_DIR}/handcal.txt"
  fi
fi
if [[ -z "${CAMERA_EXTRINSICS_FILE:-}" ]]; then
  if [[ -f "${CALIBRATION_DIR}/camera_extrinsics.json" ]]; then
    CAMERA_EXTRINSICS_FILE="${CALIBRATION_DIR}/camera_extrinsics.json"
  else
    CAMERA_EXTRINSICS_FILE="${CALIBRATION_DIR}/oak_extrinsics.json"
  fi
fi
if [[ -z "${SESSION_NAME:-}" ]]; then
  BAG_SESSION_CLEAN="${BAG_ACQUISITION_ROOT%/}"
  BAG_SESSION_BASE="$(basename "${BAG_SESSION_CLEAN}")"
  BAG_SESSION_GROUP="$(basename "$(dirname "${BAG_SESSION_CLEAN}")")"
  if [[ "${BAG_SESSION_BASE}" =~ ^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2})([0-9]{2})\.([0-9]{2})$ ]]; then
    SESSION_NAME="${BAG_SESSION_GROUP}_${BASH_REMATCH[1]}${BASH_REMATCH[2]}${BASH_REMATCH[3]}T${BASH_REMATCH[4]}${BASH_REMATCH[5]}${BASH_REMATCH[6]}"
  else
    SESSION_NAME="${BAG_SESSION_BASE}"
  fi
fi
SESSION="${SESSION:-${ROOT}/postprocess_data/${SESSION_NAME}}"
AUTO_CALIB_VIDEO="${AUTO_CALIB_VIDEO:-1}"
CALIB_BAG_SESSION="${CALIB_BAG_SESSION:-}"
if [[ -z "${CALIB_BAG_SESSION}" && "${AUTO_CALIB_VIDEO}" == "1" && -d "${CALIBRATION_DIR}" ]]; then
  if [[ -d "${CALIBRATION_DIR}/calibration_video/bag" ]]; then
    CALIB_BAG_SESSION="${CALIBRATION_DIR}/calibration_video"
  else
    CALIB_BAG_SESSION="$(find "${CALIBRATION_DIR}" -maxdepth 1 -type d -name 'calib_video_*' | sort | tail -n 1 || true)"
  fi
fi
if [[ -n "${CALIB_BAG_SESSION}" ]]; then
  if [[ -z "${CALIB_BAG_DIR:-}" ]]; then
    CALIB_BAG_DIR="${CALIB_BAG_SESSION}/bag"
  fi
  USE_CALIB_VIDEO="${USE_CALIB_VIDEO:-1}"
else
  CALIB_BAG_DIR="${CALIB_BAG_DIR:-}"
  USE_CALIB_VIDEO="${USE_CALIB_VIDEO:-0}"
fi

ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
HAND_MSG_SETUP="${HAND_MSG_SETUP:-${ROOT}/hand_msg_ws/install/setup.bash}"
ROS_PYTHON="${ROS_PYTHON:-/usr/bin/python3}"
LOCATE_PYTHON="${LOCATE_PYTHON:-/home/lenovo/miniconda3/envs/locate_anything/bin/python}"
HAMER_PYTHON="${HAMER_PYTHON:-/home/lenovo/miniconda3/envs/hamer/bin/python}"
export PYTHONPATH="${ROOT}/hamer:${ROOT}:${PYTHONPATH:-}"

LOCATE_MODEL="${LOCATE_MODEL:-${ROOT}/models--nvidia--LocateAnything-3B/resolved}"
PROMPT="${PROMPT:-white tactile glove with an IMU module worn on a hand}"
PROMPT_TAG="${PROMPT_TAG:-white_tactile_glove_with_imu}"
LOCATE_DTYPE="${LOCATE_DTYPE:-bf16}"
LOCATE_ATTN_IMPLEMENTATION="${LOCATE_ATTN_IMPLEMENTATION:-sdpa}"
LOCATE_BATCH_SIZE="${LOCATE_BATCH_SIZE:-16}"
LOCATE_DEVICE="${LOCATE_DEVICE:-cuda}"
HAMER_DEVICE="${HAMER_DEVICE:-cuda}"
HAMER_BATCH_SIZE="${HAMER_BATCH_SIZE:-32}"
HAMER_HANDEDNESS="${HAMER_HANDEDNESS:-track}"
VISUAL_SIDE="${VISUAL_SIDE:-hand_l}"
GLOVE_SIDE="${GLOVE_SIDE:-left}"
IMAGE_LEFT_PHYSICAL_SIDE="${IMAGE_LEFT_PHYSICAL_SIDE:-left}"
HAND_FRAME_SWAP_LR="${HAND_FRAME_SWAP_LR:-0}"
REQUESTED_FPS="${FPS:-}"
FPS=""
MAX_FRAMES="${MAX_FRAMES:-}"
EXTRACT_IMAGE_WRITE_WORKERS="${EXTRACT_IMAGE_WRITE_WORKERS:-8}"
OVERWRITE="${OVERWRITE:-0}"
SAVE_BBOX_FRAMES="${SAVE_BBOX_FRAMES:-0}"
VISUAL_2D_SMOOTH_ALPHA="${VISUAL_2D_SMOOTH_ALPHA:-0.35}"
VISUAL_2D_MAX_INTERP_GAP="${VISUAL_2D_MAX_INTERP_GAP:-3}"
HAMER_BRANCH_JUMP_THRESHOLD_DEG="${HAMER_BRANCH_JUMP_THRESHOLD_DEG:-75}"
HAMER_BRANCH_BRIDGE_GAP_FRAMES="${HAMER_BRANCH_BRIDGE_GAP_FRAMES:-3}"
HAMER_BRANCH_MAX_REJECT_FRAMES="${HAMER_BRANCH_MAX_REJECT_FRAMES:-60}"
TIME_FILTER_REFERENCE_FPS="${TIME_FILTER_REFERENCE_FPS:-30}"
USE_RTABMAP_POSE="${USE_RTABMAP_POSE:-1}"
RTABMAP_RENDER_VIDEOS="${RTABMAP_RENDER_VIDEOS:-0}"
if [[ -z "${RTABMAP_DB:-}" ]]; then
  if [[ -f "${BAG_DATA_DIR}/map/rtabmap.db" ]]; then
    RTABMAP_DB="${BAG_DATA_DIR}/map/rtabmap.db"
  else
    RTABMAP_DB="${CALIBRATION_DIR}/rtabmap.db"
  fi
fi
RTABMAP_POSE_DIR="${RTABMAP_POSE_DIR:-${SESSION}/rtabmap_pose}"
RTABMAP_MAX_INTERP_GAP_SEC="${RTABMAP_MAX_INTERP_GAP_SEC:-0.25}"
DEPTH_RADIUS="${DEPTH_RADIUS:-8}"
DEPTH_METHOD="${DEPTH_METHOD:-robust}"
DEPTH_ROBUST_INDICES="${DEPTH_ROBUST_INDICES:-0,5,9,13,17}"
DEPTH_PALM_INDICES="${DEPTH_PALM_INDICES:-0,5,9,13,17}"
DEPTH_MIN_CANDIDATES="${DEPTH_MIN_CANDIDATES:-2}"
DEPTH_ROBUST_INLIER_M="${DEPTH_ROBUST_INLIER_M:-0.055}"
CALIB_BBOX_MAX_CENTER_JUMP_PX="${CALIB_BBOX_MAX_CENTER_JUMP_PX:-260}"
CALIB_BBOX_LOST_JUMP_PX="${CALIB_BBOX_LOST_JUMP_PX:-80}"
CALIB_BBOX_MAX_AREA_RATIO="${CALIB_BBOX_MAX_AREA_RATIO:-20}"
CALIB_BBOX_MIN_IOU="${CALIB_BBOX_MIN_IOU:-0}"
CALIB_BBOX_MIN_IOU_CENTER_PX="${CALIB_BBOX_MIN_IOU_CENTER_PX:-9999}"
CALIB_BBOX_MAX_GAP="${CALIB_BBOX_MAX_GAP:-30}"
BASE_HAND_CONFIG="${BASE_HAND_CONFIG:-/home/lenovo/Retarget/host/hand_config.json}"
RETARGET_ROOT="${RETARGET_ROOT:-/home/lenovo/Retarget/retarget}"
RESOLVE_DRIVER="${RESOLVE_DRIVER:-/home/lenovo/Retarget/data/ros_ws/resolve_check/resolve_driver}"
FRAME_START="${FRAME_START:-120}"
FRAME_END="${FRAME_END:-170}"
CALIB_FRAME_START="${CALIB_FRAME_START:-}"
CALIB_FRAME_END="${CALIB_FRAME_END:-}"
ALPHA_ANGLE="${ALPHA_ANGLE:-0.45}"
ALPHA_QUAT="${ALPHA_QUAT:-0.45}"
GLOVE_NEUTRAL_FRAMES="${GLOVE_NEUTRAL_FRAMES:-30}"
PALM_LEVEL_FRAMES="${PALM_LEVEL_FRAMES:-30}"
HAMER_GLOBAL_ROOT_ITERATIONS="${HAMER_GLOBAL_ROOT_ITERATIONS:-120}"
HAMER_GLOBAL_SMOOTH_ITERATIONS="${HAMER_GLOBAL_SMOOTH_ITERATIONS:-70}"
HAMER_GLOBAL_MAX_TRANSLATION_STEP_M="${HAMER_GLOBAL_MAX_TRANSLATION_STEP_M:-0.02}"
HAMER_GLOBAL_W_TRANSLATION_SPEED="${HAMER_GLOBAL_W_TRANSLATION_SPEED:-4000.0}"
HAMER_GLOBAL_W_TRANSLATION_JERK="${HAMER_GLOBAL_W_TRANSLATION_JERK:-120.0}"
HAMER_GLOBAL_TRANSLATION_OUTLIER_THRESHOLD_M="${HAMER_GLOBAL_TRANSLATION_OUTLIER_THRESHOLD_M:-0.025}"
HAMER_GLOBAL_MIN_ROOT_OBSERVATION_WEIGHT="${HAMER_GLOBAL_MIN_ROOT_OBSERVATION_WEIGHT:-0.1}"
MOTION_FILTER_MIN_TRACK_LENGTH="${MOTION_FILTER_MIN_TRACK_LENGTH:-15}"
MOTION_FILTER_MIN_HAND_VALID_RATIO="${MOTION_FILTER_MIN_HAND_VALID_RATIO:-0.90}"
MOTION_FILTER_MAX_TERMINAL_INVALID_FRAMES="${MOTION_FILTER_MAX_TERMINAL_INVALID_FRAMES:-5}"
MOTION_FILTER_TERMINAL_TRIM_LOOKBACK_FRAMES="${MOTION_FILTER_TERMINAL_TRIM_LOOKBACK_FRAMES:-30}"
MOTION_FILTER_TERMINAL_TRIM_PRE_ROLL_FRAMES="${MOTION_FILTER_TERMINAL_TRIM_PRE_ROLL_FRAMES:-15}"
MOTION_FILTER_TERMINAL_FAST_TRANSLATION_M="${MOTION_FILTER_TERMINAL_FAST_TRANSLATION_M:-0.012}"
MOTION_FILTER_TERMINAL_FAST_ROTATION_DEG="${MOTION_FILTER_TERMINAL_FAST_ROTATION_DEG:-5.0}"
MOTION_FILTER_QUATERNION_TOLERANCE="${MOTION_FILTER_QUATERNION_TOLERANCE:-0.001}"
MOTION_FILTER_SPIKE_SIGMA_MULTIPLIER="${MOTION_FILTER_SPIKE_SIGMA_MULTIPLIER:-3.0}"
MOTION_FILTER_MAX_SPIKE_FRAME_FRACTION="${MOTION_FILTER_MAX_SPIKE_FRAME_FRACTION:-0.05}"
MOTION_FILTER_STATIC_ENERGY_THRESHOLD_M="${MOTION_FILTER_STATIC_ENERGY_THRESHOLD_M:-0.002}"
MOTION_FILTER_STATIC_EPISODE_FRACTION="${MOTION_FILTER_STATIC_EPISODE_FRACTION:-0.90}"
WRIST_TRACK_ALPHA="${WRIST_TRACK_ALPHA:-0.25}"
WRIST_TRACK_ACCEPT_STEP_M="${WRIST_TRACK_ACCEPT_STEP_M:-0.035}"
WRIST_TRACK_PENDING_RADIUS_M="${WRIST_TRACK_PENDING_RADIUS_M:-0.035}"
WRIST_TRACK_CONFIRM_FRAMES="${WRIST_TRACK_CONFIRM_FRAMES:-8}"
WRIST_TRACK_MAX_STEP_M="${WRIST_TRACK_MAX_STEP_M:-0.007}"
USE_CAMERA_OPTICAL_FIX="${USE_CAMERA_OPTICAL_FIX:-1}"
CONSTRAINT_PREALIGN="${CONSTRAINT_PREALIGN:-1}"
ESTIMATE_SCALE="${ESTIMATE_SCALE:-1}"
APPLY_PALM_QUAT="${APPLY_PALM_QUAT:-1}"
CONSTRAINT_MIDDLE_IDX="${CONSTRAINT_MIDDLE_IDX:-9}"
CONSTRAINT_THUMB_IDX="${CONSTRAINT_THUMB_IDX:-2}"
FLIP_PALM_NORMAL="${FLIP_PALM_NORMAL:-0}"
WRIST_TRACK_SNAP_ROOT="${WRIST_TRACK_SNAP_ROOT:-0}"
WRIST_TRACK_DEPTH_ONLY="${WRIST_TRACK_DEPTH_ONLY:-1}"
REVIEW_HAND_DISPLAY_ROTATE_DEG="${REVIEW_HAND_DISPLAY_ROTATE_DEG:-45}"
REVIEW_RGB_WORKERS="${REVIEW_RGB_WORKERS:-8}"
RUN_QUALITY_CHECK="${RUN_QUALITY_CHECK:-1}"
COMPACT_OUTPUTS="${COMPACT_OUTPUTS:-0}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"
RENDER_DEBUG_VIDEOS="${RENDER_DEBUG_VIDEOS:-0}"
RENDER_STABLE_BBOX_VIDEO="${RENDER_STABLE_BBOX_VIDEO:-1}"
RENDER_HAMER_SMOOTH_VIDEO="${RENDER_HAMER_SMOOTH_VIDEO:-1}"
RENDER_FINAL_VIDEO="${RENDER_FINAL_VIDEO:-0}"
PARALLEL_HANDS="${PARALLEL_HANDS:-1}"
CALIB_OVERWRITE="${CALIB_OVERWRITE:-${OVERWRITE}}"

if [[ "${COMPACT_OUTPUTS}" == "1" && "${RUN_QUALITY_CHECK}" != "1" ]]; then
  printf 'Error: COMPACT_OUTPUTS=1 requires RUN_QUALITY_CHECK=1.\n' >&2
  exit 2
fi

RGBD_DIR="${SESSION}/preprocess"
LOCATE_DIR="${SESSION}/locateanything_${PROMPT_TAG}"
STABLE_BBOX_DIR="${SESSION}/locateanything_${PROMPT_TAG}_stable"
HAMER_DIR="${SESSION}/hamer_from_stable_locateanything_${PROMPT_TAG}_force_right"
DEPTH_DIR="${SESSION}/depth_correct_hamer_force_right"
FUSION_LEFT_DIR="${SESSION}/fusion_input_left_depthroot"
FUSION_RIGHT_DIR="${SESSION}/fusion_input_right_depthroot"
VISUAL_SMOOTH_LEFT_DIR="${SESSION}/visual_2d_smooth_left"
VISUAL_SMOOTH_RIGHT_DIR="${SESSION}/visual_2d_smooth_right"
VISUAL_SMOOTH_DUAL_DIR="${SESSION}/visual_2d_smooth_dual"
OUTPUT_DIR="${SESSION}/outputs"
CALIB_SESSION="${CALIB_SESSION:-${SESSION}/calibration_handeye}"
CALIB_RGBD_DIR="${CALIB_SESSION}/preprocess"
CALIB_LOCATE_DIR="${CALIB_SESSION}/locateanything_${PROMPT_TAG}"
CALIB_STABLE_BBOX_DIR="${CALIB_SESSION}/locateanything_${PROMPT_TAG}_stable"
CALIB_HAMER_DIR="${CALIB_SESSION}/hamer_from_stable_locateanything_${PROMPT_TAG}_force_right"
CALIB_DEPTH_DIR="${CALIB_SESSION}/depth_correct_hamer_force_right"
CALIB_FUSION_LEFT_DIR="${CALIB_SESSION}/fusion_input_left_depthroot"
CALIB_FUSION_RIGHT_DIR="${CALIB_SESSION}/fusion_input_right_depthroot"

LOCATE_JSON="${LOCATE_DIR}/bboxes.json"
LOCATE_MP4="${LOCATE_DIR}/bboxes.mp4"
STABLE_BBOX_JSON="${STABLE_BBOX_DIR}/bboxes_stable.json"
STABLE_BBOX_MP4="${STABLE_BBOX_DIR}/bboxes_stable.mp4"
HAMER_JSON_NAME="hamer_${PROMPT_TAG}_stablebbox_force_right.json"
HAMER_DEPTH_JSON_NAME="hamer_${PROMPT_TAG}_stablebbox_force_right_depthroot.json"
HAMER_AGG_JSON="${HAMER_DIR}/hamer_${PROMPT_TAG}_stablebbox_force_right_aggregate.json"
HAMER_MP4="${HAMER_DIR}/hamer_21kpts_stablebbox_force_right.mp4"
DEPTH_SUMMARY="${DEPTH_DIR}/depthroot_summary.json"
FUSION_LEFT_JSONL="${FUSION_LEFT_DIR}/fusion_frames.jsonl"
FUSION_RIGHT_JSONL="${FUSION_RIGHT_DIR}/fusion_frames.jsonl"
FUSION_LEFT_SUMMARY="${FUSION_LEFT_DIR}/fusion_summary.json"
FUSION_RIGHT_SUMMARY="${FUSION_RIGHT_DIR}/fusion_summary.json"
VISUAL_SMOOTH_LEFT_MP4="${VISUAL_SMOOTH_LEFT_DIR}/visual_21kpts_2d_smooth.mp4"
VISUAL_SMOOTH_RIGHT_MP4="${VISUAL_SMOOTH_RIGHT_DIR}/visual_21kpts_2d_smooth.mp4"
VISUAL_SMOOTH_LEFT_JSONL="${VISUAL_SMOOTH_LEFT_DIR}/visual_2d_smooth.jsonl"
VISUAL_SMOOTH_RIGHT_JSONL="${VISUAL_SMOOTH_RIGHT_DIR}/visual_2d_smooth.jsonl"
VISUAL_SMOOTH_LEFT_SUMMARY="${VISUAL_SMOOTH_LEFT_DIR}/visual_2d_smooth_summary.json"
VISUAL_SMOOTH_RIGHT_SUMMARY="${VISUAL_SMOOTH_RIGHT_DIR}/visual_2d_smooth_summary.json"
VISUAL_SMOOTH_DUAL_MP4="${VISUAL_SMOOTH_DUAL_DIR}/visual_21kpts_2d_smooth.mp4"
VISUAL_SMOOTH_DUAL_SUMMARY="${VISUAL_SMOOTH_DUAL_DIR}/visual_2d_smooth_video_summary.json"
RTABMAP_TRAJECTORY_JSONL="${RTABMAP_POSE_DIR}/rtabmap_camera_pose_rgb30_interp.jsonl"
RTABMAP_TRAJECTORY_SUMMARY="${RTABMAP_POSE_DIR}/rtabmap_camera_pose_rgb30_interp_summary.json"
RTABMAP_APPLY_SUMMARY="${RTABMAP_POSE_DIR}/apply_rtabmap_pose_to_preprocess_summary.json"
RTABMAP_CAMERA_DIR="${RTABMAP_POSE_DIR}/camera_frames"
CAMERA_JSON_DIR="${RGBD_DIR}/all_data"
if [[ "${USE_RTABMAP_POSE}" == "1" ]]; then
  CAMERA_JSON_DIR="${RTABMAP_CAMERA_DIR}"
fi
HAMER_FRAME_DIR="${HAMER_DIR}/per_frame"
DEPTH_FRAME_DIR="${DEPTH_DIR}/per_frame"
REVIEW_WEB_HTML="${OUTPUT_DIR}/web/index.html"
COMPACT_SUMMARY="${OUTPUT_DIR}/compact_summary.json"
CALIB_LOCATE_JSON="${CALIB_LOCATE_DIR}/bboxes.json"
CALIB_LOCATE_MP4="${CALIB_LOCATE_DIR}/bboxes.mp4"
CALIB_STABLE_BBOX_JSON="${CALIB_STABLE_BBOX_DIR}/bboxes_stable.json"
CALIB_STABLE_BBOX_MP4="${CALIB_STABLE_BBOX_DIR}/bboxes_stable.mp4"
CALIB_HAMER_AGG_JSON="${CALIB_HAMER_DIR}/hamer_${PROMPT_TAG}_stablebbox_force_right_aggregate.json"
CALIB_HAMER_MP4="${CALIB_HAMER_DIR}/hamer_21kpts_stablebbox_force_right.mp4"
CALIB_DEPTH_SUMMARY="${CALIB_DEPTH_DIR}/depthroot_summary.json"
CALIB_FUSION_LEFT_JSONL="${CALIB_FUSION_LEFT_DIR}/fusion_frames.jsonl"
CALIB_FUSION_RIGHT_JSONL="${CALIB_FUSION_RIGHT_DIR}/fusion_frames.jsonl"
CALIB_FUSION_LEFT_SUMMARY="${CALIB_FUSION_LEFT_DIR}/fusion_summary.json"
CALIB_FUSION_RIGHT_SUMMARY="${CALIB_FUSION_RIGHT_DIR}/fusion_summary.json"
CALIB_CACHE_DIR="${CALIB_CACHE_DIR:-${CALIB_SESSION}/.pipeline_cache}"
CALIB_STAGE_MANIFEST="${CALIB_CACHE_DIR}/calibration_fusion.json"
TF_STATIC_JSONL="${TF_STATIC_JSONL:-${SESSION}/preprocess/tf_static.jsonl}"
CACHE_DIR="${CACHE_DIR:-${SESSION}/.pipeline_cache}"
CACHE_TOOL="${CACHE_TOOL:-${ROOT}/scripts/pipeline_stage_cache.py}"
CACHE_PYTHON="${CACHE_PYTHON:-${ROS_PYTHON}}"
QUALITY_REPORT="${OUTPUT_DIR}/quality_report.json"

stage_manifest() {
  printf '%s/%s.json' "${CACHE_DIR}" "$1"
}

stage_cache_hit() {
  local stage="$1"
  shift
  if [[ "${OVERWRITE}" == "1" ]]; then
    printf '[cache] MISS %s: OVERWRITE=1\n' "${stage}"
    return 1
  fi
  "${CACHE_PYTHON}" "${CACHE_TOOL}" check \
    --stage "${stage}" \
    --manifest "$(stage_manifest "${stage}")" \
    "$@"
}

stage_cache_write() {
  local stage="$1"
  shift
  "${CACHE_PYTHON}" "${CACHE_TOOL}" write \
    --stage "${stage}" \
    --manifest "$(stage_manifest "${stage}")" \
    "$@"
}

calibration_cache_hit() {
  if [[ "${CALIB_OVERWRITE}" == "1" ]]; then
    printf '[cache] MISS calibration_fusion: CALIB_OVERWRITE=1\n'
    return 1
  fi
  "${CACHE_PYTHON}" "${CACHE_TOOL}" check \
    --stage calibration_fusion \
    --manifest "${CALIB_STAGE_MANIFEST}" \
    "$@"
}

calibration_cache_write() {
  "${CACHE_PYTHON}" "${CACHE_TOOL}" write \
    --stage calibration_fusion \
    --manifest "${CALIB_STAGE_MANIFEST}" \
    "$@"
}

fusion_has_calibration_data() {
  local summary_json="$1"
  local side="$2"
  "${HAMER_PYTHON}" -c '
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
side = sys.argv[2]
try:
    stats = json.loads(path.read_text(encoding="utf-8")).get("stats", {})
    solve_valid = int(stats.get(f"{side}_solve_valid", 0) or 0)
    visual_valid = int(stats.get("visual_hand_present", 0) or 0)
except (OSError, ValueError, TypeError):
    raise SystemExit(1)
raise SystemExit(0 if solve_valid >= 3 and visual_valid >= 3 else 1)
' "${summary_json}" "${side}"
}

max_frames_args=()
if [[ -n "${MAX_FRAMES}" ]]; then
  max_frames_args=(--max_frames "${MAX_FRAMES}")
fi
rtabmap_strict_args=(--require_full_coverage)
quality_rtabmap_args=()
if [[ "${USE_RTABMAP_POSE}" == "1" ]]; then
  quality_rtabmap_args=(--require_rtabmap)
fi

printf '\n[0/11] Config\n'
printf '  BAG_SESSION: %s\n' "${BAG_SESSION}"
printf '  BATCH_ROOT:  %s\n' "${BATCH_ROOT}"
printf '  BAG_DATA_DIR: %s\n' "${BAG_DATA_DIR}"
printf '  BAG_DIR:     %s\n' "${BAG_DIR}"
printf '  CALIBRATION: %s\n' "${CALIBRATION_DIR}"
printf '  HAND_CALIB:  %s\n' "${HAND_CALIBRATION_FILE}"
printf '  CAMERA_EXT:  %s\n' "${CAMERA_EXTRINSICS_FILE}"
printf '  SESSION:     %s\n' "${SESSION}"
printf '  PROMPT:      %s\n' "${PROMPT}"
printf '  Locate:      dtype=%s attn=%s batch=%s\n' "${LOCATE_DTYPE}" "${LOCATE_ATTN_IMPLEMENTATION}" "${LOCATE_BATCH_SIZE}"
printf '  HaMeR:       device=%s batch=%s handedness=%s\n' "${HAMER_DEVICE}" "${HAMER_BATCH_SIZE}" "${HAMER_HANDEDNESS}"
printf '  Extract workers:%s\n' "${EXTRACT_IMAGE_WRITE_WORKERS}"
printf '  OUTPUT_DIR:  %s\n' "${OUTPUT_DIR}"
printf '  RTABMAP_DB:  %s\n' "${RTABMAP_DB}"
printf '  RTABMAP:     %s\n' "${USE_RTABMAP_POSE}"
printf '  RTAB videos: %s\n' "${RTABMAP_RENDER_VIDEOS}"
printf '  Debug videos:%s\n' "${RENDER_DEBUG_VIDEOS}"
printf '  Stable bbox video:%s\n' "${RENDER_STABLE_BBOX_VIDEO}"
printf '  HaMeR smooth video:%s\n' "${RENDER_HAMER_SMOOTH_VIDEO}"
printf '  Final video: %s\n' "${RENDER_FINAL_VIDEO}"
printf '  Parallel hands:%s\n' "${PARALLEL_HANDS}"
printf '  CALIB_SESSION:%s\n' "${CALIB_SESSION}"
if [[ "${CONFIG_ONLY}" == "1" ]]; then
  exit 0
fi

mkdir -p "${SESSION}" "${LOCATE_DIR}" "${STABLE_BBOX_DIR}" "${HAMER_DIR}" "${DEPTH_DIR}" \
  "${FUSION_LEFT_DIR}" "${FUSION_RIGHT_DIR}" "${VISUAL_SMOOTH_LEFT_DIR}" "${VISUAL_SMOOTH_RIGHT_DIR}" \
  "${OUTPUT_DIR}" "${RTABMAP_POSE_DIR}" "${CACHE_DIR}" "${CALIB_CACHE_DIR}"

printf '\n[1/11] Extract ROS2 bag RGBD/hand_frame\n'
extract_cache_args=(
  --input "${BAG_DIR}"
  --input "${HAND_CALIBRATION_FILE}"
  --input "${CAMERA_EXTRINSICS_FILE}"
  --input "${RESOLVE_DRIVER}"
  --code "${ROOT}/preprocess/ExtractRosbagSampler.py"
  --code "${ROOT}/preprocess/native/rvl_decode.cpp"
  --param "max_frames=${MAX_FRAMES}"
  --param "bag_dir=${BAG_DIR}"
  --param "resolve_driver=${RESOLVE_DRIVER}"
  --param "image_write_workers=${EXTRACT_IMAGE_WRITE_WORKERS}"
  --output "${RGBD_DIR}/timestamps.jsonl"
  --output "${RGBD_DIR}/extract_summary.json"
  --output "${RGBD_DIR}/all_data"
)
if stage_cache_hit extract "${extract_cache_args[@]}"; then
  printf '  skip valid cache: %s\n' "$(stage_manifest extract)"
else
  set +u
  source "${ROS_SETUP}"
  source "${HAND_MSG_SETUP}"
  set -u
  extract_args=(
    --session_path "${BAG_DATA_DIR}"
    --bag_dir "${BAG_DIR}"
    --output_dir "${RGBD_DIR}"
    --resolve_driver "${RESOLVE_DRIVER}"
    --image_write_workers "${EXTRACT_IMAGE_WRITE_WORKERS}"
    --overwrite
  )
  if [[ -f "${HAND_CALIBRATION_FILE}" ]]; then
    extract_args+=(--handcal_path "${HAND_CALIBRATION_FILE}")
  fi
  if [[ -f "${CAMERA_EXTRINSICS_FILE}" ]]; then
    extract_args+=(--camera_extrinsics "${CAMERA_EXTRINSICS_FILE}")
  fi
  if [[ -n "${MAX_FRAMES}" ]]; then
    extract_args+=(--max_frames "${MAX_FRAMES}")
  fi
  "${ROS_PYTHON}" "${ROOT}/preprocess/ExtractRosbagSampler.py" "${extract_args[@]}"
  stage_cache_write extract "${extract_cache_args[@]}"
fi

FPS="$("${ROS_PYTHON}" -c 'import json,sys; value=float(json.load(open(sys.argv[1]))["fps"]); assert value > 0; print(f"{value:.9f}")' "${RGBD_DIR}/extract_summary.json")"
printf '  RGB timebase: %.6f fps from rgb_stamp_ns\n' "${FPS}"
if [[ -n "${REQUESTED_FPS}" ]]; then
  printf '  note: ignoring requested FPS=%s; real RGB timebase is authoritative\n' "${REQUESTED_FPS}"
fi

printf '\n[2/11] Build/apply RTAB-Map camera pose\n'
if [[ "${USE_RTABMAP_POSE}" == "1" ]]; then
  if [[ ! -s "${RTABMAP_DB}" ]]; then
    printf '  ERROR: RTABMAP_DB not found or empty: %s\n' "${RTABMAP_DB}" >&2
    exit 1
  fi
  rtabmap_cache_args=(
    --input "$(stage_manifest extract)"
    --input "${RTABMAP_DB}"
    --code "${ROOT}/preprocess/BuildRtabmapCameraTrajectory.py"
    --code "${ROOT}/preprocess/ApplyCameraTrajectoryToPreprocess.py"
    --code "${ROOT}/preprocess/Timebase.py"
    --param "fps=${FPS}"
    --param "max_interp_gap_sec=${RTABMAP_MAX_INTERP_GAP_SEC}"
    --param "require_full_coverage=1"
    --param "render_videos=${RTABMAP_RENDER_VIDEOS}"
    --output "${RTABMAP_TRAJECTORY_JSONL}"
    --output "${RTABMAP_TRAJECTORY_SUMMARY}"
    --output "${RTABMAP_APPLY_SUMMARY}"
    --output "${RTABMAP_CAMERA_DIR}"
  )
  if stage_cache_hit rtabmap_pose "${rtabmap_cache_args[@]}"; then
    printf '  skip valid cache: %s\n' "$(stage_manifest rtabmap_pose)"
  else
    rtabmap_render_args=()
    if [[ "${RTABMAP_RENDER_VIDEOS}" == "1" ]]; then
      rtabmap_render_args=(--render_videos)
    fi
    "${HAMER_PYTHON}" "${ROOT}/preprocess/BuildRtabmapCameraTrajectory.py" \
      --rtabmap_db "${RTABMAP_DB}" \
      --timestamps_jsonl "${RGBD_DIR}/timestamps.jsonl" \
      --out_dir "${RTABMAP_POSE_DIR}" \
      --fps "${FPS}" \
      --max_interp_gap_sec "${RTABMAP_MAX_INTERP_GAP_SEC}" \
      "${rtabmap_render_args[@]}" \
      "${rtabmap_strict_args[@]}"
    "${HAMER_PYTHON}" "${ROOT}/preprocess/ApplyCameraTrajectoryToPreprocess.py" \
      --all_data_dir "${RGBD_DIR}/all_data" \
      --output_dir "${RTABMAP_CAMERA_DIR}" \
      --trajectory_jsonl "${RTABMAP_TRAJECTORY_JSONL}" \
      --summary_json "${RTABMAP_APPLY_SUMMARY}" \
      "${rtabmap_strict_args[@]}"
    stage_cache_write rtabmap_pose "${rtabmap_cache_args[@]}"
  fi
else
  printf '  skip: USE_RTABMAP_POSE=%s\n' "${USE_RTABMAP_POSE}"
fi

printf '\n[3/11] LocateAnything bbox detector\n'
locate_cache_args=(
  --input "$(stage_manifest extract)"
  --input "${LOCATE_MODEL}/config.json"
  --code "${ROOT}/preprocess/VisualizeLocateAnythingBboxes.py"
  --code "${ROOT}/third_party/nvidia_locateanything_batch/batch_utils/engine_hybrid.py"
  --code "${ROOT}/third_party/nvidia_locateanything_batch/batch_utils/hybrid_runtime.py"
  --code "${ROOT}/preprocess/Timebase.py"
  --param "prompt=${PROMPT}"
  --param "model=${LOCATE_MODEL}"
  --param "device=${LOCATE_DEVICE}"
  --param "dtype=${LOCATE_DTYPE}"
  --param "attn_implementation=${LOCATE_ATTN_IMPLEMENTATION}"
  --param "batch_size=${LOCATE_BATCH_SIZE}"
  --param "fps=${FPS}"
  --param "max_frames=${MAX_FRAMES}"
  --param "save_bbox_frames=${SAVE_BBOX_FRAMES}"
  --output "${LOCATE_JSON}"
  --param "render_debug_videos=${RENDER_DEBUG_VIDEOS}"
)
if [[ "${RENDER_DEBUG_VIDEOS}" == "1" ]]; then
  locate_cache_args+=(--output "${LOCATE_MP4}")
fi
if stage_cache_hit locate "${locate_cache_args[@]}"; then
  printf '  skip valid cache: %s\n' "$(stage_manifest locate)"
else
  locate_frame_args=(--no_save_frames)
  if [[ "${SAVE_BBOX_FRAMES}" == "1" ]]; then
    locate_frame_args=(--out_frames_dir "${LOCATE_DIR}/bbox_frames")
  fi
  if [[ "${RENDER_DEBUG_VIDEOS}" != "1" ]]; then
    locate_frame_args+=(--no_video)
  fi
  "${LOCATE_PYTHON}" "${ROOT}/preprocess/VisualizeLocateAnythingBboxes.py" \
    --session_path "${SESSION}" \
    --prompt "${PROMPT}" \
    --model_path "${LOCATE_MODEL}" \
    --device "${LOCATE_DEVICE}" \
    --dtype "${LOCATE_DTYPE}" \
    --attn_implementation "${LOCATE_ATTN_IMPLEMENTATION}" \
    --batch_size "${LOCATE_BATCH_SIZE}" \
    --fps "${FPS}" \
    --out_json "${LOCATE_JSON}" \
    --out_video "${LOCATE_MP4}" \
    "${locate_frame_args[@]}" \
    "${max_frames_args[@]}"
  stage_cache_write locate "${locate_cache_args[@]}"
fi

printf '\n[4/11] Track/stabilize left and right hand bboxes\n'
track_bbox_cache_args=(
  --input "$(stage_manifest locate)"
  --code "${ROOT}/preprocess/TrackDualHandBboxes.py"
  --code "${ROOT}/preprocess/TrackSingleHandBboxes.py"
  --code "${ROOT}/preprocess/Timebase.py"
  --param "fps=${FPS}"
  --param "reference_fps=${TIME_FILTER_REFERENCE_FPS}"
  --param "image_left_physical_side=${IMAGE_LEFT_PHYSICAL_SIDE}"
  --output "${STABLE_BBOX_JSON}"
  --output "${STABLE_BBOX_DIR}/tracking_summary.json"
  --param "render_video=${RENDER_STABLE_BBOX_VIDEO}:${RENDER_DEBUG_VIDEOS}"
)
if [[ "${RENDER_STABLE_BBOX_VIDEO}" == "1" || "${RENDER_DEBUG_VIDEOS}" == "1" ]]; then
  track_bbox_cache_args+=(--output "${STABLE_BBOX_MP4}")
fi
if stage_cache_hit track_bbox "${track_bbox_cache_args[@]}"; then
  printf '  skip valid cache: %s\n' "$(stage_manifest track_bbox)"
else
  track_video_args=()
  if [[ "${RENDER_STABLE_BBOX_VIDEO}" == "1" || "${RENDER_DEBUG_VIDEOS}" == "1" ]]; then
    track_video_args=(--out_video "${STABLE_BBOX_MP4}")
  fi
  "${HAMER_PYTHON}" "${ROOT}/preprocess/TrackDualHandBboxes.py" \
    --session_path "${SESSION}" \
    --input_json "${LOCATE_JSON}" \
    --output_json "${STABLE_BBOX_JSON}" \
    --summary_json "${STABLE_BBOX_DIR}/tracking_summary.json" \
    "${track_video_args[@]}" \
    --fps "${FPS}" \
    --reference_fps "${TIME_FILTER_REFERENCE_FPS}" \
    --image_left_side "${IMAGE_LEFT_PHYSICAL_SIDE}"
  stage_cache_write track_bbox "${track_bbox_cache_args[@]}"
fi

printf '\n[5/11] HaMeR from stable bbox, handedness=%s\n' "${HAMER_HANDEDNESS}"
POSE_STAGE_MANIFEST="$(stage_manifest extract)"
if [[ "${USE_RTABMAP_POSE}" == "1" ]]; then
  POSE_STAGE_MANIFEST="$(stage_manifest rtabmap_pose)"
fi
hamer_cache_args=(
  --input "$(stage_manifest track_bbox)"
  --input "${POSE_STAGE_MANIFEST}"
  --input "${ROOT}/hamer/_DATA/hamer_ckpts/checkpoints/new_hamer_weights.ckpt"
  --code "${ROOT}/preprocess/run_hamer_from_locate_bboxes.py"
  --code "${ROOT}/preprocess/HaMeRHands.py"
  --code "${ROOT}/preprocess/MediaPipeHands.py"
  --code "${ROOT}/hamer/hamer/datasets/vitdet_dataset.py"
  --code "${ROOT}/preprocess/VisualizeHandKpts.py"
  --code "${ROOT}/preprocess/Timebase.py"
  --param "device=${HAMER_DEVICE}"
  --param "batch_size=${HAMER_BATCH_SIZE}"
  --param "handedness=${HAMER_HANDEDNESS}"
  --param "max_frames=${MAX_FRAMES}"
  --param "fps=${FPS}"
  --param "camera_json_dir=${CAMERA_JSON_DIR}"
  --output "${HAMER_FRAME_DIR}"
  --output "${HAMER_AGG_JSON}"
  --param "render_debug_videos=${RENDER_DEBUG_VIDEOS}"
)
if [[ "${RENDER_DEBUG_VIDEOS}" == "1" ]]; then
  hamer_cache_args+=(--output "${HAMER_MP4}")
fi
if stage_cache_hit hamer "${hamer_cache_args[@]}"; then
  printf '  skip valid cache: %s\n' "$(stage_manifest hamer)"
else
  hamer_video_args=()
  if [[ "${RENDER_DEBUG_VIDEOS}" != "1" ]]; then
    hamer_video_args=(--no_video)
  fi
  "${HAMER_PYTHON}" "${ROOT}/preprocess/run_hamer_from_locate_bboxes.py" \
    --session_path "${SESSION}" \
    --bbox_json "${STABLE_BBOX_JSON}" \
    --fallback_bbox_json "" \
    --aggregate_json "${HAMER_AGG_JSON}" \
    --per_frame_output_dir "${HAMER_FRAME_DIR}" \
    --camera_json_dir "${CAMERA_JSON_DIR}" \
    --out_json_name "${HAMER_JSON_NAME}" \
    --out_video "${HAMER_MP4}" \
    --device "${HAMER_DEVICE}" \
    --batch_size "${HAMER_BATCH_SIZE}" \
    --max_boxes 2 \
    --handedness "${HAMER_HANDEDNESS}" \
    --fps "${FPS}" \
    "${hamer_video_args[@]}" \
    "${max_frames_args[@]}"
  stage_cache_write hamer "${hamer_cache_args[@]}"
fi

printf '\n[6/11] Correct HaMeR wrist root with aligned depth\n'
depth_cache_args=(
  --input "$(stage_manifest hamer)"
  --input "$(stage_manifest extract)"
  --code "${ROOT}/preprocess/DepthCorrectHandKpts.py"
  --param "depth_radius=${DEPTH_RADIUS}"
  --param "depth_method=${DEPTH_METHOD}"
  --param "robust_indices=${DEPTH_ROBUST_INDICES}"
  --param "palm_indices=${DEPTH_PALM_INDICES}"
  --param "min_candidates=${DEPTH_MIN_CANDIDATES}"
  --param "inlier_m=${DEPTH_ROBUST_INLIER_M}"
  --param "max_frames=${MAX_FRAMES}"
  --output "${DEPTH_FRAME_DIR}"
  --output "${DEPTH_SUMMARY}"
)
if stage_cache_hit depth_root "${depth_cache_args[@]}"; then
  printf '  skip valid cache: %s\n' "$(stage_manifest depth_root)"
else
  "${HAMER_PYTHON}" "${ROOT}/preprocess/DepthCorrectHandKpts.py" \
    --session_path "${SESSION}" \
    --input_json_name "${HAMER_JSON_NAME}" \
    --output_json_name "${HAMER_DEPTH_JSON_NAME}" \
    --input_json_dir "${HAMER_FRAME_DIR}" \
    --output_json_dir "${DEPTH_FRAME_DIR}" \
    --summary_json "${DEPTH_SUMMARY}" \
    --depth_name depth_aligned.png \
    --root_idx 0 \
    --depth_radius "${DEPTH_RADIUS}" \
    --method "${DEPTH_METHOD}" \
    --robust_indices "${DEPTH_ROBUST_INDICES}" \
    --palm_indices "${DEPTH_PALM_INDICES}" \
    --min_depth_candidates "${DEPTH_MIN_CANDIDATES}" \
    --robust_inlier_m "${DEPTH_ROBUST_INLIER_M}" \
    "${max_frames_args[@]}"
  stage_cache_write depth_root "${depth_cache_args[@]}"
fi

printf '\n[7/11] Build left/right visual + /hand_frame fusion inputs\n'
fusion_cache_args=(
  --input "$(stage_manifest depth_root)"
  --input "${POSE_STAGE_MANIFEST}"
  --code "${ROOT}/preprocess/BuildHandFusionInput.py"
  --param "visual_sides=hand_l,hand_r"
  --param "hand_frame_swap_lr=${HAND_FRAME_SWAP_LR}"
  --param "glove_sides=left,right"
  --param "hand_sync_key=bag_time_ns"
  --param "camera_json_dir=${CAMERA_JSON_DIR}"
  --param "visual_json_dir=${DEPTH_FRAME_DIR}"
  --output "${FUSION_LEFT_JSONL}"
  --output "${FUSION_RIGHT_JSONL}"
  --output "${FUSION_LEFT_SUMMARY}"
  --output "${FUSION_RIGHT_SUMMARY}"
)
if stage_cache_hit fusion "${fusion_cache_args[@]}"; then
  printf '  skip valid cache: %s\n' "$(stage_manifest fusion)"
else
  fusion_swap_args=()
  if [[ "${HAND_FRAME_SWAP_LR}" == "1" ]]; then
    fusion_swap_args+=(--swap_hand_frame_lr)
  fi
  fusion_pids=()
  fusion_sides=()
  for side in left right; do
    if [[ "${side}" == "left" ]]; then
      visual_side=hand_l; fusion_jsonl="${FUSION_LEFT_JSONL}"; fusion_summary="${FUSION_LEFT_SUMMARY}"
    else
      visual_side=hand_r; fusion_jsonl="${FUSION_RIGHT_JSONL}"; fusion_summary="${FUSION_RIGHT_SUMMARY}"
    fi
    fusion_cmd=(
      "${HAMER_PYTHON}" "${ROOT}/preprocess/BuildHandFusionInput.py"
      --session_path "${SESSION}"
      --rgbd_subdir preprocess
      --visual_json_name "${HAMER_DEPTH_JSON_NAME}"
      --visual_json_dir "${DEPTH_FRAME_DIR}"
      --camera_json_dir "${CAMERA_JSON_DIR}"
      --output_jsonl "${fusion_jsonl}"
      --summary_json "${fusion_summary}"
      --visual_side "${visual_side}"
      --glove_side "${side}"
      --hand_sync_key bag_time_ns
      "${fusion_swap_args[@]}"
    )
    if [[ "${PARALLEL_HANDS}" == "1" ]]; then
      "${fusion_cmd[@]}" &
      fusion_pids+=("$!")
      fusion_sides+=("${side}")
    else
      "${fusion_cmd[@]}"
    fi
  done
  fusion_failed=0
  for i in "${!fusion_pids[@]}"; do
    if ! wait "${fusion_pids[$i]}"; then
      printf 'Error: %s main fusion failed\n' "${fusion_sides[$i]}" >&2
      fusion_failed=1
    fi
  done
  if [[ "${fusion_failed}" == "1" ]]; then
    exit 1
  fi
  stage_cache_write fusion "${fusion_cache_args[@]}"
fi


CALIB_INPUT_FUSION_LEFT_FOR_FK=""
CALIB_INPUT_FUSION_RIGHT_FOR_FK=""
if [[ "${USE_CALIB_VIDEO}" == "1" ]]; then
  printf '\n[calib] Build dedicated hand-eye calibration fusion from calib_video bag\n'
  calib_cache_args=(
    --input "${CALIB_BAG_DIR}"
    --input "${HAND_CALIBRATION_FILE}"
    --input "${CAMERA_EXTRINSICS_FILE}"
    --input "${LOCATE_MODEL}/config.json"
    --code "${ROOT}/scripts/build_calib_video_fusion.sh"
    --code "${ROOT}/preprocess/ExtractRosbagSampler.py"
    --code "${ROOT}/preprocess/native/rvl_decode.cpp"
    --code "${ROOT}/preprocess/VisualizeLocateAnythingBboxes.py"
    --code "${ROOT}/third_party/nvidia_locateanything_batch/batch_utils/engine_hybrid.py"
    --code "${ROOT}/third_party/nvidia_locateanything_batch/batch_utils/hybrid_runtime.py"
    --code "${ROOT}/preprocess/TrackDualHandBboxes.py"
    --code "${ROOT}/preprocess/TrackSingleHandBboxes.py"
    --code "${ROOT}/preprocess/run_hamer_from_locate_bboxes.py"
    --code "${ROOT}/preprocess/HaMeRHands.py"
    --code "${ROOT}/preprocess/MediaPipeHands.py"
    --code "${ROOT}/hamer/hamer/datasets/vitdet_dataset.py"
    --code "${ROOT}/preprocess/VisualizeHandKpts.py"
    --param "hand_frame_swap_lr=${HAND_FRAME_SWAP_LR}"
    --code "${ROOT}/preprocess/DepthCorrectHandKpts.py"
    --code "${ROOT}/preprocess/BuildHandFusionInput.py"
    --code "${ROOT}/preprocess/Timebase.py"
    --param "prompt=${PROMPT}"
    --param "handedness=${HAMER_HANDEDNESS}"
    --param "visual_sides=hand_l,hand_r"
    --param "glove_sides=left,right"
    --param "max_frames=${MAX_FRAMES}"
    --param "depth_radius=${DEPTH_RADIUS}"
    --param "depth_method=${DEPTH_METHOD}"
    --param "robust_indices=${DEPTH_ROBUST_INDICES}"
    --param "palm_indices=${DEPTH_PALM_INDICES}"
    --param "min_candidates=${DEPTH_MIN_CANDIDATES}"
    --param "inlier_m=${DEPTH_ROBUST_INLIER_M}"
    --param "locate_device=${LOCATE_DEVICE}"
    --param "locate_dtype=${LOCATE_DTYPE}"
    --param "locate_attn_implementation=${LOCATE_ATTN_IMPLEMENTATION}"
    --param "locate_batch_size=${LOCATE_BATCH_SIZE}"
    --param "hamer_device=${HAMER_DEVICE}"
    --param "hamer_batch_size=${HAMER_BATCH_SIZE}"
    --param "render_debug_videos=${RENDER_DEBUG_VIDEOS}"
    --param "save_bbox_frames=${SAVE_BBOX_FRAMES}"
    --param "bbox_max_center_jump_px=${CALIB_BBOX_MAX_CENTER_JUMP_PX}"
    --param "bbox_lost_jump_px=${CALIB_BBOX_LOST_JUMP_PX}"
    --param "bbox_max_area_ratio=${CALIB_BBOX_MAX_AREA_RATIO}"
    --param "bbox_min_iou=${CALIB_BBOX_MIN_IOU}"
    --param "bbox_min_iou_center_px=${CALIB_BBOX_MIN_IOU_CENTER_PX}"
    --param "bbox_max_gap=${CALIB_BBOX_MAX_GAP}"
    --param "reference_fps=${TIME_FILTER_REFERENCE_FPS}"
    --param "image_left_physical_side=${IMAGE_LEFT_PHYSICAL_SIDE}"
    --output "${CALIB_FUSION_LEFT_JSONL}"
    --output "${CALIB_FUSION_RIGHT_JSONL}"
    --output "${CALIB_FUSION_LEFT_SUMMARY}"
    --output "${CALIB_FUSION_RIGHT_SUMMARY}"
  )
  if calibration_cache_hit "${calib_cache_args[@]}"; then
    printf '  skip shared calibration cache: %s\n' "${CALIB_STAGE_MANIFEST}"
  else
    ROOT="${ROOT}" \
    CALIB_BAG_SESSION="${CALIB_BAG_SESSION}" \
    CALIB_BAG_DIR="${CALIB_BAG_DIR}" \
    CALIB_SESSION="${CALIB_SESSION}" \
    HAND_CALIBRATION_FILE="${HAND_CALIBRATION_FILE}" \
    CAMERA_EXTRINSICS_FILE="${CAMERA_EXTRINSICS_FILE}" \
    ROS_SETUP="${ROS_SETUP}" \
    HAND_MSG_SETUP="${HAND_MSG_SETUP}" \
    ROS_PYTHON="${ROS_PYTHON}" \
    LOCATE_PYTHON="${LOCATE_PYTHON}" \
    HAMER_PYTHON="${HAMER_PYTHON}" \
    LOCATE_MODEL="${LOCATE_MODEL}" \
    PROMPT="${PROMPT}" \
    PROMPT_TAG="${PROMPT_TAG}" \
    LOCATE_DTYPE="${LOCATE_DTYPE}" \
    HAND_FRAME_SWAP_LR="${HAND_FRAME_SWAP_LR}" \
    LOCATE_ATTN_IMPLEMENTATION="${LOCATE_ATTN_IMPLEMENTATION}" \
    LOCATE_BATCH_SIZE="${LOCATE_BATCH_SIZE}" \
    EXTRACT_IMAGE_WRITE_WORKERS="${EXTRACT_IMAGE_WRITE_WORKERS}" \
    LOCATE_DEVICE="${LOCATE_DEVICE}" \
    HAMER_DEVICE="${HAMER_DEVICE}" \
    HAMER_BATCH_SIZE="${HAMER_BATCH_SIZE}" \
    HAMER_HANDEDNESS="${HAMER_HANDEDNESS}" \
    VISUAL_SIDE="${VISUAL_SIDE}" \
    GLOVE_SIDE="${GLOVE_SIDE}" \
    IMAGE_LEFT_PHYSICAL_SIDE="${IMAGE_LEFT_PHYSICAL_SIDE}" \
    FPS="${FPS}" \
    MAX_FRAMES="${MAX_FRAMES}" \
    OVERWRITE=1 \
    SAVE_BBOX_FRAMES="${SAVE_BBOX_FRAMES}" \
    DEPTH_RADIUS="${DEPTH_RADIUS}" \
    DEPTH_METHOD="${DEPTH_METHOD}" \
    DEPTH_ROBUST_INDICES="${DEPTH_ROBUST_INDICES}" \
    DEPTH_PALM_INDICES="${DEPTH_PALM_INDICES}" \
    DEPTH_MIN_CANDIDATES="${DEPTH_MIN_CANDIDATES}" \
    DEPTH_ROBUST_INLIER_M="${DEPTH_ROBUST_INLIER_M}" \
    CALIB_BBOX_MAX_CENTER_JUMP_PX="${CALIB_BBOX_MAX_CENTER_JUMP_PX}" \
    CALIB_BBOX_LOST_JUMP_PX="${CALIB_BBOX_LOST_JUMP_PX}" \
    CALIB_BBOX_MAX_AREA_RATIO="${CALIB_BBOX_MAX_AREA_RATIO}" \
    CALIB_BBOX_MIN_IOU="${CALIB_BBOX_MIN_IOU}" \
    CALIB_BBOX_MIN_IOU_CENTER_PX="${CALIB_BBOX_MIN_IOU_CENTER_PX}" \
    CALIB_BBOX_MAX_GAP="${CALIB_BBOX_MAX_GAP}" \
    RESOLVE_DRIVER="${RESOLVE_DRIVER}" \
    TIME_FILTER_REFERENCE_FPS="${TIME_FILTER_REFERENCE_FPS}" \
    RENDER_DEBUG_VIDEOS="${RENDER_DEBUG_VIDEOS}" \
    PARALLEL_HANDS="${PARALLEL_HANDS}" \
    "${ROOT}/scripts/build_calib_video_fusion.sh"
    calibration_cache_write "${calib_cache_args[@]}"
  fi
  CALIB_INPUT_FUSION_LEFT_FOR_FK="${CALIB_FUSION_LEFT_JSONL}"
  CALIB_INPUT_FUSION_RIGHT_FOR_FK="${CALIB_FUSION_RIGHT_JSONL}"
  if ! fusion_has_calibration_data "${CALIB_FUSION_LEFT_SUMMARY}" left; then
    printf '  warning: dedicated calibration has no usable left solve/visual pairs; use main sequence window %s-%s\n' "${FRAME_START}" "${FRAME_END}"
    CALIB_INPUT_FUSION_LEFT_FOR_FK=""
  fi
  if ! fusion_has_calibration_data "${CALIB_FUSION_RIGHT_SUMMARY}" right; then
    printf '  warning: dedicated calibration has no usable right solve/visual pairs; use main sequence window %s-%s\n' "${FRAME_START}" "${FRAME_END}"
    CALIB_INPUT_FUSION_RIGHT_FOR_FK=""
  fi
else
  printf '\n[calib] skip dedicated calibration video: USE_CALIB_VIDEO=%s\n' "${USE_CALIB_VIDEO}"
fi

printf '\n[8/11] Smooth left/right visual 21 keypoints\n'
visual_smooth_cache_args=(
  --input "$(stage_manifest fusion)"
  --code "${ROOT}/preprocess/VisualizeVisual2DSmooth.py"
  --code "${ROOT}/preprocess/Timebase.py"
  --param "fps=${FPS}"
  --param "alpha=${VISUAL_2D_SMOOTH_ALPHA}"
  --param "max_interp_gap=${VISUAL_2D_MAX_INTERP_GAP}"
  --param "reference_fps=${TIME_FILTER_REFERENCE_FPS}"
  --param "render_debug_videos=${RENDER_DEBUG_VIDEOS}"
  --output "${VISUAL_SMOOTH_LEFT_JSONL}"
  --output "${VISUAL_SMOOTH_RIGHT_JSONL}"
  --output "${VISUAL_SMOOTH_LEFT_SUMMARY}"
  --output "${VISUAL_SMOOTH_RIGHT_SUMMARY}"
)
if [[ "${RENDER_DEBUG_VIDEOS}" == "1" ]]; then
  visual_smooth_cache_args+=(--output "${VISUAL_SMOOTH_LEFT_MP4}" --output "${VISUAL_SMOOTH_RIGHT_MP4}")
fi
if stage_cache_hit visual_smooth "${visual_smooth_cache_args[@]}"; then
  printf '  skip valid cache: %s\n' "$(stage_manifest visual_smooth)"
else
  smooth_pids=()
  smooth_sides=()
  for side in left right; do
    if [[ "${side}" == "left" ]]; then
      fusion_jsonl="${FUSION_LEFT_JSONL}"; smooth_mp4="${VISUAL_SMOOTH_LEFT_MP4}"; smooth_jsonl="${VISUAL_SMOOTH_LEFT_JSONL}"; smooth_summary="${VISUAL_SMOOTH_LEFT_SUMMARY}"
    else
      fusion_jsonl="${FUSION_RIGHT_JSONL}"; smooth_mp4="${VISUAL_SMOOTH_RIGHT_MP4}"; smooth_jsonl="${VISUAL_SMOOTH_RIGHT_JSONL}"; smooth_summary="${VISUAL_SMOOTH_RIGHT_SUMMARY}"
    fi
    smooth_render_args=()
    if [[ "${RENDER_DEBUG_VIDEOS}" != "1" ]]; then
      smooth_render_args=(--no_render)
    fi
    smooth_cmd=(
      "${HAMER_PYTHON}" "${ROOT}/preprocess/VisualizeVisual2DSmooth.py"
      --input_jsonl "${fusion_jsonl}"
      --out_path "${smooth_mp4}"
      --output_jsonl "${smooth_jsonl}"
      --summary_json "${smooth_summary}"
      --fps "${FPS}"
      --alpha "${VISUAL_2D_SMOOTH_ALPHA}"
      --max_interp_gap "${VISUAL_2D_MAX_INTERP_GAP}"
      --reference_fps "${TIME_FILTER_REFERENCE_FPS}"
      "${smooth_render_args[@]}"
    )
    if [[ "${PARALLEL_HANDS}" == "1" ]]; then
      "${smooth_cmd[@]}" &
      smooth_pids+=("$!")
      smooth_sides+=("${side}")
    else
      "${smooth_cmd[@]}"
    fi
  done
  smooth_failed=0
  for i in "${!smooth_pids[@]}"; do
    if ! wait "${smooth_pids[$i]}"; then
      printf 'Error: %s visual smoothing failed\n' "${smooth_sides[$i]}" >&2
      smooth_failed=1
    fi
  done
  if [[ "${smooth_failed}" == "1" ]]; then
    exit 1
  fi
  stage_cache_write visual_smooth "${visual_smooth_cache_args[@]}"
fi

printf '\n[9/11] Glove FK + visual-bone calibration + wristroot tracking\n'
ACTIVE_GLOVE_SIDES=()
if fusion_has_calibration_data "${FUSION_LEFT_SUMMARY}" left; then
  ACTIVE_GLOVE_SIDES+=(left)
else
  printf '  warning: main sequence has no usable left solve state; skip left glove FK/trajectory\n'
fi
if fusion_has_calibration_data "${FUSION_RIGHT_SUMMARY}" right; then
  ACTIVE_GLOVE_SIDES+=(right)
else
  printf '  warning: main sequence has no usable right solve state; skip right glove FK/trajectory\n'
fi
if [[ "${#ACTIVE_GLOVE_SIDES[@]}" == "0" ]]; then
  printf 'Error: neither glove side has usable solve state and visual observations.\n' >&2
  exit 2
fi
ACTIVE_GLOVE_SIDES_CSV="$(IFS=,; printf '%s' "${ACTIVE_GLOVE_SIDES[*]}")"
printf '  active glove sides: %s\n' "${ACTIVE_GLOVE_SIDES_CSV}"
LEFT_FK_DIR="${SESSION}/glove_fk21_visual_bones_smooth_solve045_left"
RIGHT_FK_DIR="${SESSION}/glove_fk21_visual_bones_smooth_solve045_right"
DUAL_FK_DIR="${SESSION}/glove_fk21_visual_bones_smooth_solve045_dual"
LEFT_TRAJECTORY="${LEFT_FK_DIR}/trajectory_wristroot_track_cameraoptical.jsonl"
RIGHT_TRAJECTORY="${RIGHT_FK_DIR}/trajectory_wristroot_track_cameraoptical.jsonl"
LEFT_TRAJECTORY_SUMMARY="${LEFT_FK_DIR}/trajectory_wristroot_track_cameraoptical_summary.json"
RIGHT_TRAJECTORY_SUMMARY="${RIGHT_FK_DIR}/trajectory_wristroot_track_cameraoptical_summary.json"
LEFT_CALIBRATION="${LEFT_FK_DIR}/glove_fk_to_camera_calib_smooth_solve045.json"
RIGHT_CALIBRATION="${RIGHT_FK_DIR}/glove_fk_to_camera_calib_smooth_solve045.json"
FINAL_TRAJECTORY="${DUAL_FK_DIR}/trajectory_wristroot_track_cameraoptical.jsonl"
FINAL_TRAJECTORY_SUMMARY="${DUAL_FK_DIR}/trajectory_wristroot_track_cameraoptical_summary.json"
PALM_LEVEL_SUMMARY="${DUAL_FK_DIR}/palm_plane_level_summary.json"
WORLD_REBASE_SUMMARY="${DUAL_FK_DIR}/world_rebase_first_camera_summary.json"
HAMER_GLOBAL_SUMMARY="${DUAL_FK_DIR}/hamer_global_trajectory_summary.json"
PALM_FRAME_SUMMARY="${DUAL_FK_DIR}/stable_palm_frame_summary.json"
MOTION_FILTER_SUMMARY="${DUAL_FK_DIR}/motion_filter_summary.json"
FINAL_TRAJECTORY_MP4="${DUAL_FK_DIR}/trajectory_3d_world_wristroot_track_cameraoptical.mp4"
FINAL_TRAJECTORY_VIDEO_SUMMARY="${DUAL_FK_DIR}/trajectory_3d_world_wristroot_track_cameraoptical_summary.json"
glove_cache_args=(
  --input "$(stage_manifest fusion)"
  --input "${BASE_HAND_CONFIG}"
  --input "${RETARGET_ROOT}/hand_retarget"
  --input "${TF_STATIC_JSONL}"
  --code "${ROOT}/scripts/run_glove_fk_visual_bones_pipeline.sh"
  --code "${ROOT}/preprocess/EstimateHandConfigFromVisual.py"
  --code "${ROOT}/preprocess/SmoothGloveSolveState.py"
  --code "${ROOT}/preprocess/BuildGloveFk21FromFusion.py"
  --code "${ROOT}/preprocess/CalibrateGloveFkToCamera.py"
  --code "${ROOT}/preprocess/ApplyGloveFkCalibration.py"
  --code "${ROOT}/preprocess/TrackGloveWristRoot.py"
  --code "${ROOT}/preprocess/BuildUnifiedTrajectory.py"
  --code "${ROOT}/preprocess/MergeDualHandTrajectories.py"
  --code "${ROOT}/preprocess/FixTrajectoryCameraOpticalWorld.py"
  --code "${ROOT}/preprocess/Timebase.py"
  --code "${ROOT}/preprocess/VisualizeGloveFkVsVisual.py"
  --code "${ROOT}/preprocess/VisualizeTrajectory3D.py"
  --code "${ROOT}/preprocess/RenderDualVisual2DSmooth.py"
  --code "${ROOT}/preprocess/LevelDualHandPalmPlane.py"
  --code "${ROOT}/preprocess/RebaseTrajectoryWorldToFirstCamera.py"
  --code "${ROOT}/preprocess/OptimizeHamerGlobalTrajectory.py"
  --code "${ROOT}/preprocess/BuildStablePalmFrames.py"
  --code "${ROOT}/preprocess/FilterTrajectoryQuality.py"
  --code "${ROOT}/preprocess/VisualizeVisual2DSmooth.py"
  --param "fps=${FPS}"
  --param "glove_sides=${ACTIVE_GLOVE_SIDES_CSV}"
  --param "base_hand_config=${BASE_HAND_CONFIG}"
  --param "retarget_root=${RETARGET_ROOT}"
  --param "use_camera_optical_fix=${USE_CAMERA_OPTICAL_FIX}"
  --param "frame_start=${FRAME_START}"
  --param "frame_end=${FRAME_END}"
  --param "calib_frame_start=${CALIB_FRAME_START}"
  --param "calib_frame_end=${CALIB_FRAME_END}"
  --param "alpha_angle=${ALPHA_ANGLE}"
  --param "alpha_quat=${ALPHA_QUAT}"
  --param "glove_neutral_frames=${GLOVE_NEUTRAL_FRAMES}"
  --param "palm_level_frames=${PALM_LEVEL_FRAMES}"
  --param "hamer_global_root_iterations=${HAMER_GLOBAL_ROOT_ITERATIONS}"
  --param "hamer_global_smooth_iterations=${HAMER_GLOBAL_SMOOTH_ITERATIONS}"
  --param "hamer_global_max_translation_step_m=${HAMER_GLOBAL_MAX_TRANSLATION_STEP_M}"
  --param "hamer_global_w_translation_speed=${HAMER_GLOBAL_W_TRANSLATION_SPEED}"
  --param "hamer_global_w_translation_jerk=${HAMER_GLOBAL_W_TRANSLATION_JERK}"
  --param "hamer_global_translation_outlier_threshold_m=${HAMER_GLOBAL_TRANSLATION_OUTLIER_THRESHOLD_M}"
  --param "hamer_global_min_root_observation_weight=${HAMER_GLOBAL_MIN_ROOT_OBSERVATION_WEIGHT}"
  --param "motion_filter_min_track_length=${MOTION_FILTER_MIN_TRACK_LENGTH}"
  --param "motion_filter_min_hand_valid_ratio=${MOTION_FILTER_MIN_HAND_VALID_RATIO}"
  --param "motion_filter_max_terminal_invalid_frames=${MOTION_FILTER_MAX_TERMINAL_INVALID_FRAMES}"
  --param "motion_filter_terminal_trim_lookback_frames=${MOTION_FILTER_TERMINAL_TRIM_LOOKBACK_FRAMES}"
  --param "motion_filter_terminal_trim_pre_roll_frames=${MOTION_FILTER_TERMINAL_TRIM_PRE_ROLL_FRAMES}"
  --param "motion_filter_terminal_fast_translation_m=${MOTION_FILTER_TERMINAL_FAST_TRANSLATION_M}"
  --param "motion_filter_terminal_fast_rotation_deg=${MOTION_FILTER_TERMINAL_FAST_ROTATION_DEG}"
  --param "motion_filter_quaternion_tolerance=${MOTION_FILTER_QUATERNION_TOLERANCE}"
  --param "motion_filter_spike_sigma_multiplier=${MOTION_FILTER_SPIKE_SIGMA_MULTIPLIER}"
  --param "motion_filter_max_spike_frame_fraction=${MOTION_FILTER_MAX_SPIKE_FRAME_FRACTION}"
  --param "motion_filter_static_energy_threshold_m=${MOTION_FILTER_STATIC_ENERGY_THRESHOLD_M}"
  --param "motion_filter_static_episode_fraction=${MOTION_FILTER_STATIC_EPISODE_FRACTION}"
  --param "wrist_track_alpha=${WRIST_TRACK_ALPHA}"
  --param "wrist_track_accept_step_m=${WRIST_TRACK_ACCEPT_STEP_M}"
  --param "wrist_track_pending_radius_m=${WRIST_TRACK_PENDING_RADIUS_M}"
  --param "wrist_track_confirm_frames=${WRIST_TRACK_CONFIRM_FRAMES}"
  --param "wrist_track_max_step_m=${WRIST_TRACK_MAX_STEP_M}"
  --param "apply_palm_quat=${APPLY_PALM_QUAT}"
  --param "constraint_prealign=${CONSTRAINT_PREALIGN}"
  --param "estimate_scale=${ESTIMATE_SCALE}"
  --param "constraint_middle_idx=${CONSTRAINT_MIDDLE_IDX}"
  --param "constraint_thumb_idx=${CONSTRAINT_THUMB_IDX}"
  --param "flip_palm_normal=${FLIP_PALM_NORMAL}"
  --param "wrist_track_snap_root=${WRIST_TRACK_SNAP_ROOT}"
  --param "wrist_track_depth_only=${WRIST_TRACK_DEPTH_ONLY}"
  --param "reference_fps=${TIME_FILTER_REFERENCE_FPS}"
  --param "parallel_hands=${PARALLEL_HANDS}"
  --param "render_debug_videos=${RENDER_DEBUG_VIDEOS}"
  --param "render_hamer_smooth_video=${RENDER_HAMER_SMOOTH_VIDEO}"
  --param "hamer_smooth_alpha=${VISUAL_2D_SMOOTH_ALPHA}"
  --param "hamer_smooth_max_interp_gap=${VISUAL_2D_MAX_INTERP_GAP}"
  --param "hamer_branch_jump_threshold_deg=${HAMER_BRANCH_JUMP_THRESHOLD_DEG}"
  --param "hamer_branch_bridge_gap_frames=${HAMER_BRANCH_BRIDGE_GAP_FRAMES}"
  --param "hamer_branch_max_reject_frames=${HAMER_BRANCH_MAX_REJECT_FRAMES}"
  --param "render_final_video=${RENDER_FINAL_VIDEO}"
  --output "${FINAL_TRAJECTORY}"
  --output "${FINAL_TRAJECTORY_SUMMARY}"
  --output "${WORLD_REBASE_SUMMARY}"
  --output "${HAMER_GLOBAL_SUMMARY}"
  --output "${PALM_FRAME_SUMMARY}"
  --output "${MOTION_FILTER_SUMMARY}"
)
if [[ "${PALM_LEVEL_FRAMES}" -gt 0 ]]; then
  glove_cache_args+=(--output "${PALM_LEVEL_SUMMARY}")
fi
if [[ "${RENDER_HAMER_SMOOTH_VIDEO}" == "1" ]]; then
  glove_cache_args+=(--output "${VISUAL_SMOOTH_DUAL_MP4}" --output "${VISUAL_SMOOTH_DUAL_SUMMARY}")
fi
if [[ "${RENDER_FINAL_VIDEO}" == "1" ]]; then
  glove_cache_args+=(--output "${FINAL_TRAJECTORY_MP4}" --output "${FINAL_TRAJECTORY_VIDEO_SUMMARY}")
fi
for side in "${ACTIVE_GLOVE_SIDES[@]}"; do
  if [[ "${side}" == "left" ]]; then
    glove_cache_args+=(--output "${LEFT_TRAJECTORY}" --output "${LEFT_TRAJECTORY_SUMMARY}" --output "${LEFT_CALIBRATION}")
  else
    glove_cache_args+=(--output "${RIGHT_TRAJECTORY}" --output "${RIGHT_TRAJECTORY_SUMMARY}" --output "${RIGHT_CALIBRATION}")
  fi
done
if [[ -n "${CALIB_INPUT_FUSION_LEFT_FOR_FK}" || -n "${CALIB_INPUT_FUSION_RIGHT_FOR_FK}" ]]; then
  glove_cache_args+=(--input "${CALIB_STAGE_MANIFEST}")
fi
if stage_cache_hit glove_fk_trajectory "${glove_cache_args[@]}"; then
  printf '  skip valid cache: %s\n' "$(stage_manifest glove_fk_trajectory)"
else
  run_glove_side() {
    local side="$1"
    local input_fusion="$2"
    local calib_input_fusion="$3"
    env \
      SESSION="${SESSION}" \
      INPUT_FUSION="${input_fusion}" \
      CALIB_INPUT_FUSION="${calib_input_fusion}" \
      PYTHON="${HAMER_PYTHON}" \
      FPS="${FPS}" \
      GLOVE_SIDE="${side}" \
      BASE_HAND_CONFIG="${BASE_HAND_CONFIG}" \
      RETARGET_ROOT="${RETARGET_ROOT}" \
      FRAME_START="${FRAME_START}" \
      FRAME_END="${FRAME_END}" \
      CALIB_FRAME_START="${CALIB_FRAME_START}" \
      CALIB_FRAME_END="${CALIB_FRAME_END}" \
      ALPHA_ANGLE="${ALPHA_ANGLE}" \
      ALPHA_QUAT="${ALPHA_QUAT}" \
      GLOVE_NEUTRAL_FRAMES="${GLOVE_NEUTRAL_FRAMES}" \
      WRIST_TRACK_ALPHA="${WRIST_TRACK_ALPHA}" \
      WRIST_TRACK_ACCEPT_STEP_M="${WRIST_TRACK_ACCEPT_STEP_M}" \
      WRIST_TRACK_PENDING_RADIUS_M="${WRIST_TRACK_PENDING_RADIUS_M}" \
      WRIST_TRACK_CONFIRM_FRAMES="${WRIST_TRACK_CONFIRM_FRAMES}" \
      WRIST_TRACK_MAX_STEP_M="${WRIST_TRACK_MAX_STEP_M}" \
      USE_CAMERA_OPTICAL_FIX="${USE_CAMERA_OPTICAL_FIX}" \
      CONSTRAINT_PREALIGN="${CONSTRAINT_PREALIGN}" \
      ESTIMATE_SCALE="${ESTIMATE_SCALE}" \
      APPLY_PALM_QUAT="${APPLY_PALM_QUAT}" \
      CONSTRAINT_MIDDLE_IDX="${CONSTRAINT_MIDDLE_IDX}" \
      CONSTRAINT_THUMB_IDX="${CONSTRAINT_THUMB_IDX}" \
      FLIP_PALM_NORMAL="${FLIP_PALM_NORMAL}" \
      WRIST_TRACK_SNAP_ROOT="${WRIST_TRACK_SNAP_ROOT}" \
      WRIST_TRACK_DEPTH_ONLY="${WRIST_TRACK_DEPTH_ONLY}" \
      TF_STATIC_JSONL="${TF_STATIC_JSONL}" \
      TIME_FILTER_REFERENCE_FPS="${TIME_FILTER_REFERENCE_FPS}" \
      RENDER_DEBUG_VIDEOS="${RENDER_DEBUG_VIDEOS}" \
      "${ROOT}/scripts/run_glove_fk_visual_bones_pipeline.sh"
  }
  glove_pids=()
  glove_pid_sides=()
  for side in "${ACTIVE_GLOVE_SIDES[@]}"; do
    if [[ "${side}" == "left" ]]; then
      input_fusion="${FUSION_LEFT_JSONL}"
      calib_input_fusion="${CALIB_INPUT_FUSION_LEFT_FOR_FK}"
    else
      input_fusion="${FUSION_RIGHT_JSONL}"
      calib_input_fusion="${CALIB_INPUT_FUSION_RIGHT_FOR_FK}"
    fi
    printf '\n  [9/11:%s] FK, calibration and wrist tracking\n' "${side}"
    if [[ "${PARALLEL_HANDS}" == "1" && "${#ACTIVE_GLOVE_SIDES[@]}" -gt 1 ]]; then
      run_glove_side "${side}" "${input_fusion}" "${calib_input_fusion}" &
      glove_pids+=("$!")
      glove_pid_sides+=("${side}")
    else
      run_glove_side "${side}" "${input_fusion}" "${calib_input_fusion}"
    fi
  done
  glove_failed=0
  for i in "${!glove_pids[@]}"; do
    if ! wait "${glove_pids[$i]}"; then
      printf 'Error: %s glove FK/wrist tracking failed\n' "${glove_pid_sides[$i]}" >&2
      glove_failed=1
    fi
  done
  if [[ "${glove_failed}" == "1" ]]; then
    exit 1
  fi
  merge_args=(--output_jsonl "${FINAL_TRAJECTORY}" --summary_json "${FINAL_TRAJECTORY_SUMMARY}")
  for side in "${ACTIVE_GLOVE_SIDES[@]}"; do
    if [[ "${side}" == "left" ]]; then
      merge_args+=(--left_jsonl "${LEFT_TRAJECTORY}")
    else
      merge_args+=(--right_jsonl "${RIGHT_TRAJECTORY}")
    fi
  done
  "${HAMER_PYTHON}" "${ROOT}/preprocess/MergeDualHandTrajectories.py" "${merge_args[@]}"
  if [[ "${PALM_LEVEL_FRAMES}" -gt 0 ]]; then
    printf '\n  [9/11:dual-level] Level initial left/right palm planes in world coordinates\n'
    "${HAMER_PYTHON}" "${ROOT}/preprocess/LevelDualHandPalmPlane.py" \
      --input_jsonl "${FINAL_TRAJECTORY}" \
      --output_jsonl "${FINAL_TRAJECTORY}" \
      --summary_json "${PALM_LEVEL_SUMMARY}" \
      --level_frames "${PALM_LEVEL_FRAMES}"
  fi
  printf '\n  [9/11:dual-rebase] Rebase world coordinates to the first camera optical frame\n'
  "${HAMER_PYTHON}" "${ROOT}/preprocess/RebaseTrajectoryWorldToFirstCamera.py" \
    --input_jsonl "${FINAL_TRAJECTORY}" \
    --output_jsonl "${FINAL_TRAJECTORY}" \
    --summary_json "${WORLD_REBASE_SUMMARY}"
  printf '\n  [9/11:dual-hand-global] Smooth FK wrist translation/orientation in world frame\n'
  hamer_global_args=(
    --input_jsonl "${FINAL_TRAJECTORY}"
    --output_jsonl "${FINAL_TRAJECTORY}"
    --summary_json "${HAMER_GLOBAL_SUMMARY}"
    --sides "${ACTIVE_GLOVE_SIDES_CSV}"
    --root_iterations "${HAMER_GLOBAL_ROOT_ITERATIONS}"
    --smooth_iterations "${HAMER_GLOBAL_SMOOTH_ITERATIONS}"
    --max_translation_step_m "${HAMER_GLOBAL_MAX_TRANSLATION_STEP_M}"
    --w_translation_speed "${HAMER_GLOBAL_W_TRANSLATION_SPEED}"
    --w_translation_jerk "${HAMER_GLOBAL_W_TRANSLATION_JERK}"
    --translation_outlier_threshold_m "${HAMER_GLOBAL_TRANSLATION_OUTLIER_THRESHOLD_M}"
    --min_root_observation_weight "${HAMER_GLOBAL_MIN_ROOT_OBSERVATION_WEIGHT}"
  )
  "${HAMER_PYTHON}" "${ROOT}/preprocess/OptimizeHamerGlobalTrajectory.py" "${hamer_global_args[@]}"
  printf '\n  [9/11:dual-palm-frame] Build stable camera/world wrist and palm action frames\n'
  "${HAMER_PYTHON}" "${ROOT}/preprocess/BuildStablePalmFrames.py" \
    --input_jsonl "${FINAL_TRAJECTORY}" \
    --output_jsonl "${FINAL_TRAJECTORY}" \
    --summary_json "${PALM_FRAME_SUMMARY}"
  printf '\n  [9/11:quality-filter] Label frame/episode motion quality (no duration/chunk filtering)\n'
  "${HAMER_PYTHON}" "${ROOT}/preprocess/FilterTrajectoryQuality.py" \
    --input_jsonl "${FINAL_TRAJECTORY}" \
    --output_jsonl "${FINAL_TRAJECTORY}" \
    --summary_json "${MOTION_FILTER_SUMMARY}" \
    --min_track_length "${MOTION_FILTER_MIN_TRACK_LENGTH}" \
    --min_hand_valid_ratio "${MOTION_FILTER_MIN_HAND_VALID_RATIO}" \
    --max_terminal_invalid_frames "${MOTION_FILTER_MAX_TERMINAL_INVALID_FRAMES}" \
    --terminal_trim_lookback_frames "${MOTION_FILTER_TERMINAL_TRIM_LOOKBACK_FRAMES}" \
    --terminal_trim_pre_roll_frames "${MOTION_FILTER_TERMINAL_TRIM_PRE_ROLL_FRAMES}" \
    --terminal_fast_translation_m "${MOTION_FILTER_TERMINAL_FAST_TRANSLATION_M}" \
    --terminal_fast_rotation_deg "${MOTION_FILTER_TERMINAL_FAST_ROTATION_DEG}" \
    --quaternion_tolerance "${MOTION_FILTER_QUATERNION_TOLERANCE}" \
    --spike_sigma_multiplier "${MOTION_FILTER_SPIKE_SIGMA_MULTIPLIER}" \
    --max_spike_frame_fraction "${MOTION_FILTER_MAX_SPIKE_FRAME_FRACTION}" \
    --static_energy_threshold_m "${MOTION_FILTER_STATIC_ENERGY_THRESHOLD_M}" \
    --static_episode_fraction "${MOTION_FILTER_STATIC_EPISODE_FRACTION}"
  if [[ "${RENDER_HAMER_SMOOTH_VIDEO}" == "1" ]]; then
    printf '\n  [9/11:dual-2d] Render branch-corrected left/right HaMeR smooth video\n'
    "${HAMER_PYTHON}" "${ROOT}/preprocess/RenderDualVisual2DSmooth.py" \
      --trajectory_jsonl "${FINAL_TRAJECTORY}" \
      --out_path "${VISUAL_SMOOTH_DUAL_MP4}" \
      --summary_json "${VISUAL_SMOOTH_DUAL_SUMMARY}" \
      --fps "${FPS}" \
      --alpha "${VISUAL_2D_SMOOTH_ALPHA}" \
      --max_interp_gap "${VISUAL_2D_MAX_INTERP_GAP}" \
      --reference_fps "${TIME_FILTER_REFERENCE_FPS}" \
      --branch_jump_threshold_deg "${HAMER_BRANCH_JUMP_THRESHOLD_DEG}" \
      --branch_bridge_gap_frames "${HAMER_BRANCH_BRIDGE_GAP_FRAMES}" \
      --branch_max_reject_frames "${HAMER_BRANCH_MAX_REJECT_FRAMES}"
  fi
  if [[ "${RENDER_FINAL_VIDEO}" == "1" ]]; then
    printf '\n  [9/11:dual] Render final dual-hand 3D trajectory once\n'
    "${HAMER_PYTHON}" "${ROOT}/preprocess/VisualizeTrajectory3D.py" \
      --trajectory_jsonl "${FINAL_TRAJECTORY}" \
      --out_path "${FINAL_TRAJECTORY_MP4}" \
      --summary_json "${FINAL_TRAJECTORY_VIDEO_SUMMARY}" \
      --fps "${FPS}"
  fi
  stage_cache_write glove_fk_trajectory "${glove_cache_args[@]}"
fi

printf '\n[10/11] Collect user-facing outputs\n'
collect_cache_args=(
  --input "$(stage_manifest locate)"
  --input "$(stage_manifest track_bbox)"
  --input "$(stage_manifest hamer)"
  --input "$(stage_manifest depth_root)"
  --input "$(stage_manifest fusion)"
  --input "$(stage_manifest visual_smooth)"
  --input "$(stage_manifest glove_fk_trajectory)"
  --code "${ROOT}/preprocess/CollectPipelineOutputs.py"
  --param "prompt_tag=${PROMPT_TAG}"
  --param "out_tag=visual_bones_smooth_solve045"
  --param "active_glove_sides=${ACTIVE_GLOVE_SIDES_CSV}"
  --param "render_debug_videos=${RENDER_DEBUG_VIDEOS}"
  --param "render_stable_bbox_video=${RENDER_STABLE_BBOX_VIDEO}"
  --param "render_hamer_smooth_video=${RENDER_HAMER_SMOOTH_VIDEO}"
  --param "render_final_video=${RENDER_FINAL_VIDEO}"
  --output "${OUTPUT_DIR}/manifest.json"
  --output "${OUTPUT_DIR}/data/trajectory_wristroot_track_cameraoptical.jsonl"
  --output "${OUTPUT_DIR}/summaries/world_rebase_first_camera_summary.json"
  --output "${OUTPUT_DIR}/summaries/hamer_global_trajectory_summary.json"
  --output "${OUTPUT_DIR}/summaries/stable_palm_frame_summary.json"
  --output "${OUTPUT_DIR}/summaries/motion_filter_summary.json"
)
if [[ "${RENDER_STABLE_BBOX_VIDEO}" == "1" ]]; then
  collect_cache_args+=(--output "${OUTPUT_DIR}/videos/02_stable_bbox.mp4")
fi
if [[ "${RENDER_HAMER_SMOOTH_VIDEO}" == "1" ]]; then
  collect_cache_args+=(--output "${OUTPUT_DIR}/videos/04_dual_visual_21kpts_2d_smooth.mp4")
fi
if [[ "${RENDER_FINAL_VIDEO}" == "1" ]]; then
  collect_cache_args+=(--output "${OUTPUT_DIR}/videos/07_dual_trajectory_3d_world.mp4")
fi
if [[ "${USE_RTABMAP_POSE}" == "1" ]]; then
  collect_cache_args+=(
    --input "$(stage_manifest rtabmap_pose)"
    --output "${OUTPUT_DIR}/summaries/rtabmap_trajectory_summary.json"
    --output "${OUTPUT_DIR}/summaries/rtabmap_apply_summary.json"
  )
fi
if stage_cache_hit collect_outputs "${collect_cache_args[@]}"; then
  printf '  skip valid cache: %s\n' "$(stage_manifest collect_outputs)"
else
  collect_debug_args=()
  if [[ "${RENDER_DEBUG_VIDEOS}" == "1" ]]; then
    collect_debug_args=(--include_debug_videos)
  fi
  collect_video_args=()
  if [[ "${RENDER_STABLE_BBOX_VIDEO}" != "1" ]]; then
    collect_video_args+=(--no_stable_bbox_video)
  fi
  if [[ "${RENDER_HAMER_SMOOTH_VIDEO}" != "1" ]]; then
    collect_video_args+=(--no_hamer_smooth_video)
  fi
  if [[ "${RENDER_FINAL_VIDEO}" == "1" ]]; then
    collect_video_args+=(--include_final_video)
  fi
  "${HAMER_PYTHON}" "${ROOT}/preprocess/CollectPipelineOutputs.py" \
    --session_path "${SESSION}" \
    --output_dir "${OUTPUT_DIR}" \
    --prompt_tag "${PROMPT_TAG}" \
    --out_tag visual_bones_smooth_solve045 \
    --active_glove_sides "${ACTIVE_GLOVE_SIDES_CSV}" \
    "${collect_video_args[@]}" \
    "${collect_debug_args[@]}"
  stage_cache_write collect_outputs "${collect_cache_args[@]}"
fi


printf '\n[11/11] Generate review web visualization\n'
web_cache_args=(
  --input "$(stage_manifest collect_outputs)"
  --input "${OUTPUT_DIR}/data/trajectory_wristroot_track_cameraoptical.jsonl"
  --input "${ROOT}/tactile/assets/hand_live.png"
  --code "${ROOT}/scripts/generate_review_web.py"
  --code "${ROOT}/preprocess/Timebase.py"
  --param "fps=${FPS}"
  --param "hand_display_rotate_deg=${REVIEW_HAND_DISPLAY_ROTATE_DEG}"
  --param "rgb_workers=${REVIEW_RGB_WORKERS}"
  --output "${REVIEW_WEB_HTML}"
  --output "${OUTPUT_DIR}/web/rgb_frames"
  --output "${OUTPUT_DIR}/web/tactile_hand.png"
)
if stage_cache_hit review_web "${web_cache_args[@]}"; then
  printf '  skip valid cache: %s\n' "$(stage_manifest review_web)"
else
  "${HAMER_PYTHON}" "${ROOT}/scripts/generate_review_web.py" \
    --session "${SESSION}" \
    --fps "${FPS}" \
    --rgb_workers "${REVIEW_RGB_WORKERS}" \
    --hand_display_rotate_deg "${REVIEW_HAND_DISPLAY_ROTATE_DEG}"
  stage_cache_write review_web "${web_cache_args[@]}"
fi

if [[ "${RUN_QUALITY_CHECK}" == "1" ]]; then
  printf '\n[quality] Validate pipeline outputs\n'
  "${HAMER_PYTHON}" "${ROOT}/scripts/validate_pipeline_quality.py" \
    --session "${SESSION}" \
    --report "${QUALITY_REPORT}" \
    --min_hand_match_ratio "${QUALITY_MIN_HAND_MATCH_RATIO:-0.95}" \
    --min_visual_ratio "${QUALITY_MIN_VISUAL_RATIO:-0.90}" \
    --min_depth_applied_ratio "${QUALITY_MIN_DEPTH_APPLIED_RATIO:-0.85}" \
    --max_calibration_median_m "${QUALITY_MAX_CALIBRATION_MEDIAN_M:-0.030}" \
    --max_calibration_p95_m "${QUALITY_MAX_CALIBRATION_P95_M:-0.060}" \
    --max_wrist_residual_p95_m "${QUALITY_MAX_WRIST_RESIDUAL_P95_M:-0.070}" \
    --min_rtabmap_coverage_ratio "${QUALITY_MIN_RTABMAP_COVERAGE_RATIO:-1.0}" \
    --max_rtabmap_interp_gap_sec "${QUALITY_MAX_RTABMAP_INTERP_GAP_SEC:-${RTABMAP_MAX_INTERP_GAP_SEC}}" \
    "${quality_rtabmap_args[@]}"
else
  printf '\n[quality] skip: RUN_QUALITY_CHECK=%s\n' "${RUN_QUALITY_CHECK}"
fi

if [[ "${COMPACT_OUTPUTS}" == "1" ]]; then
  printf '\n[compact] Quality passed; compact finished session outputs\n'
  "${HAMER_PYTHON}" "${ROOT}/scripts/compact_postprocess_session.py" --session "${SESSION}"
else
  printf '\n[compact] skip: COMPACT_OUTPUTS=%s\n' "${COMPACT_OUTPUTS}"
fi
printf '\nDone\n'
printf '  session:        %s\n' "${SESSION}"
printf '  outputs:        %s\n' "${OUTPUT_DIR}"
printf '  rtabmap poses:  %s\n' "${RTABMAP_TRAJECTORY_JSONL}"
if [[ "${RENDER_STABLE_BBOX_VIDEO}" == "1" ]]; then
  printf '  stable bbox:    %s\n' "${OUTPUT_DIR}/videos/02_stable_bbox.mp4"
fi
if [[ "${RENDER_HAMER_SMOOTH_VIDEO}" == "1" ]]; then
  printf '  HaMeR smooth:   %s\n' "${OUTPUT_DIR}/videos/04_dual_visual_21kpts_2d_smooth.mp4"
fi
if [[ "${RENDER_FINAL_VIDEO}" == "1" ]]; then
  printf '  dual 3D video:  %s\n' "${OUTPUT_DIR}/videos/07_dual_trajectory_3d_world.mp4"
fi
printf '  trajectory:     %s\n' "${OUTPUT_DIR}/data/trajectory_wristroot_track_cameraoptical.jsonl"
printf '  review web:     %s\n' "${REVIEW_WEB_HTML}"
printf '  compact summary: %s\n' "${COMPACT_SUMMARY}"
if [[ -n "${CALIB_INPUT_FUSION_LEFT_FOR_FK:-}" ]]; then
  printf '  left calib fusion:  %s\n' "${CALIB_INPUT_FUSION_LEFT_FOR_FK}"
  printf '  right calib fusion: %s\n' "${CALIB_INPUT_FUSION_RIGHT_FOR_FK}"
fi
