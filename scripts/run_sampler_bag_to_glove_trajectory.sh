#!/usr/bin/env bash
set -euo pipefail

# Full sampler-bag pipeline:
# ROS2 bag -> RGBD/hand_frame extract -> RTAB-Map camera pose -> LocateAnything bbox
# -> stable bbox -> HaMeR wrist visual prior -> depth-root correction
# -> visual+glove fusion -> glove FK/calibration/root tracking/trajectory videos
# -> review web.

ROOT="${ROOT:-/home/lenovo/Ego-loong-postprocess}"
BAG_SESSION="${BAG_SESSION:-${ROOT}/datatsets/sampler_ros2_bags/rotation/data_20260628_123410}"
# Accept either the acquisition root (.../<session>) or its data directory
# (.../<session>/data).  Older datasets may also keep bag/calibration directly
# under the acquisition root.
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
CALIBRATION_DIR="${BAG_DATA_DIR}/calibration"
if [[ -z "${BAG_DIR:-}" ]]; then
  BAG_DIR="${BAG_DATA_DIR}/bag"
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
  CALIB_BAG_SESSION="$(find "${CALIBRATION_DIR}" -maxdepth 1 -type d -name 'calib_video_*' | sort | tail -n 1 || true)"
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
PROMPT="${PROMPT:-white glove with imu}"
PROMPT_TAG="${PROMPT_TAG:-white_glove_with_imu}"
LOCATE_DTYPE="${LOCATE_DTYPE:-fp32}"
LOCATE_DEVICE="${LOCATE_DEVICE:-cuda}"
HAMER_DEVICE="${HAMER_DEVICE:-cuda}"
HAMER_HANDEDNESS="${HAMER_HANDEDNESS:-all_left}"
VISUAL_SIDE="${VISUAL_SIDE:-hand_l}"
GLOVE_SIDE="${GLOVE_SIDE:-left}"
REQUESTED_FPS="${FPS:-}"
FPS=""
MAX_FRAMES="${MAX_FRAMES:-}"
OVERWRITE="${OVERWRITE:-0}"
SAVE_BBOX_FRAMES="${SAVE_BBOX_FRAMES:-0}"
VISUAL_2D_SMOOTH_ALPHA="${VISUAL_2D_SMOOTH_ALPHA:-0.35}"
VISUAL_2D_MAX_INTERP_GAP="${VISUAL_2D_MAX_INTERP_GAP:-3}"
TIME_FILTER_REFERENCE_FPS="${TIME_FILTER_REFERENCE_FPS:-30}"
USE_RTABMAP_POSE="${USE_RTABMAP_POSE:-1}"
RTABMAP_DB="${RTABMAP_DB:-${CALIBRATION_DIR}/rtabmap.db}"
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
WRIST_TRACK_ALPHA="${WRIST_TRACK_ALPHA:-0.25}"
WRIST_TRACK_ACCEPT_STEP_M="${WRIST_TRACK_ACCEPT_STEP_M:-0.035}"
WRIST_TRACK_PENDING_RADIUS_M="${WRIST_TRACK_PENDING_RADIUS_M:-0.035}"
WRIST_TRACK_CONFIRM_FRAMES="${WRIST_TRACK_CONFIRM_FRAMES:-8}"
WRIST_TRACK_MAX_STEP_M="${WRIST_TRACK_MAX_STEP_M:-0.007}"
USE_CAMERA_OPTICAL_FIX="${USE_CAMERA_OPTICAL_FIX:-1}"
CONSTRAINT_PREALIGN="${CONSTRAINT_PREALIGN:-1}"
APPLY_PALM_QUAT="${APPLY_PALM_QUAT:-1}"
CONSTRAINT_MIDDLE_IDX="${CONSTRAINT_MIDDLE_IDX:-9}"
CONSTRAINT_THUMB_IDX="${CONSTRAINT_THUMB_IDX:-2}"
FLIP_PALM_NORMAL="${FLIP_PALM_NORMAL:-0}"
WRIST_TRACK_SNAP_ROOT="${WRIST_TRACK_SNAP_ROOT:-0}"
WRIST_TRACK_DEPTH_ONLY="${WRIST_TRACK_DEPTH_ONLY:-1}"
REVIEW_HAND_DISPLAY_ROTATE_DEG="${REVIEW_HAND_DISPLAY_ROTATE_DEG:-45}"
COMPACT_OUTPUTS="${COMPACT_OUTPUTS:-0}"

RGBD_DIR="${SESSION}/preprocess"
LOCATE_DIR="${SESSION}/locateanything_${PROMPT_TAG}"
STABLE_BBOX_DIR="${SESSION}/locateanything_${PROMPT_TAG}_stable"
HAMER_DIR="${SESSION}/hamer_from_stable_locateanything_${PROMPT_TAG}_force_right"
DEPTH_DIR="${SESSION}/depth_correct_hamer_force_right"
FUSION_DIR="${SESSION}/fusion_input_force_right_depthroot"
VISUAL_SMOOTH_DIR="${SESSION}/visual_2d_smooth"
OUTPUT_DIR="${SESSION}/outputs"
CALIB_SESSION="${CALIB_SESSION:-${SESSION}/calibration_handeye}"
CALIB_RGBD_DIR="${CALIB_SESSION}/preprocess"
CALIB_LOCATE_DIR="${CALIB_SESSION}/locateanything_${PROMPT_TAG}"
CALIB_STABLE_BBOX_DIR="${CALIB_SESSION}/locateanything_${PROMPT_TAG}_stable"
CALIB_HAMER_DIR="${CALIB_SESSION}/hamer_from_stable_locateanything_${PROMPT_TAG}_force_right"
CALIB_DEPTH_DIR="${CALIB_SESSION}/depth_correct_hamer_force_right"
CALIB_FUSION_DIR="${CALIB_SESSION}/fusion_input_force_right_depthroot"

LOCATE_JSON="${LOCATE_DIR}/bboxes.json"
LOCATE_MP4="${LOCATE_DIR}/bboxes.mp4"
STABLE_BBOX_JSON="${STABLE_BBOX_DIR}/bboxes_stable.json"
STABLE_BBOX_MP4="${STABLE_BBOX_DIR}/bboxes_stable.mp4"
HAMER_JSON_NAME="hamer_${PROMPT_TAG}_stablebbox_force_right.json"
HAMER_DEPTH_JSON_NAME="hamer_${PROMPT_TAG}_stablebbox_force_right_depthroot.json"
HAMER_AGG_JSON="${HAMER_DIR}/hamer_${PROMPT_TAG}_stablebbox_force_right_aggregate.json"
HAMER_MP4="${HAMER_DIR}/hamer_21kpts_stablebbox_force_right.mp4"
DEPTH_SUMMARY="${DEPTH_DIR}/depthroot_summary.json"
FUSION_JSONL="${FUSION_DIR}/fusion_frames.jsonl"
FUSION_SUMMARY="${FUSION_DIR}/fusion_summary.json"
VISUAL_SMOOTH_MP4="${VISUAL_SMOOTH_DIR}/visual_21kpts_2d_smooth.mp4"
VISUAL_SMOOTH_JSONL="${VISUAL_SMOOTH_DIR}/visual_2d_smooth.jsonl"
VISUAL_SMOOTH_SUMMARY="${VISUAL_SMOOTH_DIR}/visual_2d_smooth_summary.json"
RTABMAP_TRAJECTORY_JSONL="${RTABMAP_POSE_DIR}/rtabmap_camera_pose_rgb30_interp.jsonl"
RTABMAP_TRAJECTORY_SUMMARY="${RTABMAP_POSE_DIR}/rtabmap_camera_pose_rgb30_interp_summary.json"
RTABMAP_APPLY_SUMMARY="${RTABMAP_POSE_DIR}/apply_rtabmap_pose_to_preprocess_summary.json"
REVIEW_WEB_HTML="${OUTPUT_DIR}/web/index.html"
COMPACT_SUMMARY="${OUTPUT_DIR}/compact_summary.json"
CALIB_LOCATE_JSON="${CALIB_LOCATE_DIR}/bboxes.json"
CALIB_LOCATE_MP4="${CALIB_LOCATE_DIR}/bboxes.mp4"
CALIB_STABLE_BBOX_JSON="${CALIB_STABLE_BBOX_DIR}/bboxes_stable.json"
CALIB_STABLE_BBOX_MP4="${CALIB_STABLE_BBOX_DIR}/bboxes_stable.mp4"
CALIB_HAMER_AGG_JSON="${CALIB_HAMER_DIR}/hamer_${PROMPT_TAG}_stablebbox_force_right_aggregate.json"
CALIB_HAMER_MP4="${CALIB_HAMER_DIR}/hamer_21kpts_stablebbox_force_right.mp4"
CALIB_DEPTH_SUMMARY="${CALIB_DEPTH_DIR}/depthroot_summary.json"
CALIB_FUSION_JSONL="${CALIB_FUSION_DIR}/fusion_frames.jsonl"
CALIB_FUSION_SUMMARY="${CALIB_FUSION_DIR}/fusion_summary.json"
TF_STATIC_JSONL="${TF_STATIC_JSONL:-${SESSION}/preprocess/tf_static.jsonl}"
CACHE_DIR="${CACHE_DIR:-${SESSION}/.pipeline_cache}"
CACHE_TOOL="${CACHE_TOOL:-${ROOT}/scripts/pipeline_stage_cache.py}"
CACHE_PYTHON="${CACHE_PYTHON:-${ROS_PYTHON}}"
QUALITY_REPORT="${OUTPUT_DIR}/quality_report.json"

mkdir -p "${SESSION}" "${LOCATE_DIR}" "${STABLE_BBOX_DIR}" "${HAMER_DIR}" "${DEPTH_DIR}" "${FUSION_DIR}" "${VISUAL_SMOOTH_DIR}" "${OUTPUT_DIR}" "${RTABMAP_POSE_DIR}" "${CACHE_DIR}"

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
printf '  BAG_DATA_DIR: %s\n' "${BAG_DATA_DIR}"
printf '  BAG_DIR:     %s\n' "${BAG_DIR}"
printf '  CALIBRATION: %s\n' "${CALIBRATION_DIR}"
printf '  SESSION:     %s\n' "${SESSION}"
printf '  PROMPT:      %s\n' "${PROMPT}"
printf '  OUTPUT_DIR:  %s\n' "${OUTPUT_DIR}"
printf '  RTABMAP_DB:  %s\n' "${RTABMAP_DB}"
printf '  RTABMAP:     %s\n' "${USE_RTABMAP_POSE}"

printf '\n[1/11] Extract ROS2 bag RGBD/hand_frame\n'
extract_cache_args=(
  --input "${BAG_DIR}"
  --input "${CALIBRATION_DIR}/handcal.txt"
  --input "${RESOLVE_DRIVER}"
  --code "${ROOT}/preprocess/ExtractRosbagSampler.py"
  --param "max_frames=${MAX_FRAMES}"
  --param "bag_dir=${BAG_DIR}"
  --param "resolve_driver=${RESOLVE_DRIVER}"
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
    --overwrite
  )
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
    --output "${RTABMAP_TRAJECTORY_JSONL}"
    --output "${RTABMAP_TRAJECTORY_SUMMARY}"
    --output "${RTABMAP_APPLY_SUMMARY}"
  )
  if stage_cache_hit rtabmap_pose "${rtabmap_cache_args[@]}"; then
    printf '  skip valid cache: %s\n' "$(stage_manifest rtabmap_pose)"
  else
    "${HAMER_PYTHON}" "${ROOT}/preprocess/BuildRtabmapCameraTrajectory.py" \
      --rtabmap_db "${RTABMAP_DB}" \
      --timestamps_jsonl "${RGBD_DIR}/timestamps.jsonl" \
      --out_dir "${RTABMAP_POSE_DIR}" \
      --fps "${FPS}" \
      --max_interp_gap_sec "${RTABMAP_MAX_INTERP_GAP_SEC}" \
      "${rtabmap_strict_args[@]}"
    "${HAMER_PYTHON}" "${ROOT}/preprocess/ApplyCameraTrajectoryToPreprocess.py" \
      --all_data_dir "${RGBD_DIR}/all_data" \
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
  --code "${ROOT}/preprocess/Timebase.py"
  --param "prompt=${PROMPT}"
  --param "model=${LOCATE_MODEL}"
  --param "device=${LOCATE_DEVICE}"
  --param "dtype=${LOCATE_DTYPE}"
  --param "fps=${FPS}"
  --param "max_frames=${MAX_FRAMES}"
  --param "save_bbox_frames=${SAVE_BBOX_FRAMES}"
  --output "${LOCATE_JSON}"
  --output "${LOCATE_MP4}"
)
if stage_cache_hit locate "${locate_cache_args[@]}"; then
  printf '  skip valid cache: %s\n' "$(stage_manifest locate)"
else
  locate_frame_args=(--no_save_frames)
  if [[ "${SAVE_BBOX_FRAMES}" == "1" ]]; then
    locate_frame_args=(--out_frames_dir "${LOCATE_DIR}/bbox_frames")
  fi
  "${LOCATE_PYTHON}" "${ROOT}/preprocess/VisualizeLocateAnythingBboxes.py" \
    --session_path "${SESSION}" \
    --prompt "${PROMPT}" \
    --model_path "${LOCATE_MODEL}" \
    --device "${LOCATE_DEVICE}" \
    --dtype "${LOCATE_DTYPE}" \
    --fps "${FPS}" \
    --out_json "${LOCATE_JSON}" \
    --out_video "${LOCATE_MP4}" \
    "${locate_frame_args[@]}" \
    "${max_frames_args[@]}"
  stage_cache_write locate "${locate_cache_args[@]}"
fi

printf '\n[4/11] Track/stabilize single target bbox\n'
track_bbox_cache_args=(
  --input "$(stage_manifest locate)"
  --code "${ROOT}/preprocess/TrackSingleHandBboxes.py"
  --code "${ROOT}/preprocess/Timebase.py"
  --param "fps=${FPS}"
  --param "reference_fps=${TIME_FILTER_REFERENCE_FPS}"
  --output "${STABLE_BBOX_JSON}"
  --output "${STABLE_BBOX_DIR}/tracking_summary.json"
  --output "${STABLE_BBOX_MP4}"
)
if stage_cache_hit track_bbox "${track_bbox_cache_args[@]}"; then
  printf '  skip valid cache: %s\n' "$(stage_manifest track_bbox)"
else
  "${HAMER_PYTHON}" "${ROOT}/preprocess/TrackSingleHandBboxes.py" \
    --session_path "${SESSION}" \
    --input_json "${LOCATE_JSON}" \
    --output_json "${STABLE_BBOX_JSON}" \
    --summary_json "${STABLE_BBOX_DIR}/tracking_summary.json" \
    --out_video "${STABLE_BBOX_MP4}" \
    --fps "${FPS}" \
    --reference_fps "${TIME_FILTER_REFERENCE_FPS}"
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
  --code "${ROOT}/preprocess/VisualizeHandKpts.py"
  --code "${ROOT}/preprocess/Timebase.py"
  --param "device=${HAMER_DEVICE}"
  --param "handedness=${HAMER_HANDEDNESS}"
  --param "max_frames=${MAX_FRAMES}"
  --param "fps=${FPS}"
  --output "${SESSION}/preprocess/all_data/00000/${HAMER_JSON_NAME}"
  --output "${HAMER_AGG_JSON}"
  --output "${HAMER_MP4}"
)
if stage_cache_hit hamer "${hamer_cache_args[@]}"; then
  printf '  skip valid cache: %s\n' "$(stage_manifest hamer)"
else
  "${HAMER_PYTHON}" "${ROOT}/preprocess/run_hamer_from_locate_bboxes.py" \
    --session_path "${SESSION}" \
    --bbox_json "${STABLE_BBOX_JSON}" \
    --fallback_bbox_json "" \
    --aggregate_json "${HAMER_AGG_JSON}" \
    --out_json_name "${HAMER_JSON_NAME}" \
    --out_video "${HAMER_MP4}" \
    --device "${HAMER_DEVICE}" \
    --max_boxes 1 \
    --handedness "${HAMER_HANDEDNESS}" \
    --fps "${FPS}" \
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
  --output "${SESSION}/preprocess/all_data/00000/${HAMER_DEPTH_JSON_NAME}"
  --output "${DEPTH_SUMMARY}"
)
if stage_cache_hit depth_root "${depth_cache_args[@]}"; then
  printf '  skip valid cache: %s\n' "$(stage_manifest depth_root)"
else
  "${HAMER_PYTHON}" "${ROOT}/preprocess/DepthCorrectHandKpts.py" \
    --session_path "${SESSION}" \
    --input_json_name "${HAMER_JSON_NAME}" \
    --output_json_name "${HAMER_DEPTH_JSON_NAME}" \
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

printf '\n[7/11] Build visual + /hand_frame fusion input\n'
fusion_cache_args=(
  --input "$(stage_manifest depth_root)"
  --input "${POSE_STAGE_MANIFEST}"
  --code "${ROOT}/preprocess/BuildHandFusionInput.py"
  --param "visual_side=${VISUAL_SIDE}"
  --param "glove_side=${GLOVE_SIDE}"
  --param "hand_sync_key=bag_time_ns"
  --output "${FUSION_JSONL}"
  --output "${FUSION_SUMMARY}"
)
if stage_cache_hit fusion "${fusion_cache_args[@]}"; then
  printf '  skip valid cache: %s\n' "$(stage_manifest fusion)"
else
  "${HAMER_PYTHON}" "${ROOT}/preprocess/BuildHandFusionInput.py" \
    --session_path "${SESSION}" \
    --rgbd_subdir preprocess \
    --visual_json_name "${HAMER_DEPTH_JSON_NAME}" \
    --output_jsonl "${FUSION_JSONL}" \
    --summary_json "${FUSION_SUMMARY}" \
    --visual_side "${VISUAL_SIDE}" \
    --glove_side "${GLOVE_SIDE}" \
    --hand_sync_key bag_time_ns
  stage_cache_write fusion "${fusion_cache_args[@]}"
fi


CALIB_INPUT_FUSION_FOR_FK=""
if [[ "${USE_CALIB_VIDEO}" == "1" ]]; then
  printf '\n[calib] Build dedicated hand-eye calibration fusion from calib_video bag\n'
  calib_cache_args=(
    --input "${CALIB_BAG_DIR}"
    --input "${CALIB_BAG_SESSION}/../handcal.txt"
    --input "${CALIB_BAG_SESSION}/../../handcal.txt"
    --input "${LOCATE_MODEL}/config.json"
    --code "${ROOT}/scripts/build_calib_video_fusion.sh"
    --code "${ROOT}/preprocess/ExtractRosbagSampler.py"
    --code "${ROOT}/preprocess/VisualizeLocateAnythingBboxes.py"
    --code "${ROOT}/preprocess/TrackSingleHandBboxes.py"
    --code "${ROOT}/preprocess/run_hamer_from_locate_bboxes.py"
    --code "${ROOT}/preprocess/VisualizeHandKpts.py"
    --code "${ROOT}/preprocess/DepthCorrectHandKpts.py"
    --code "${ROOT}/preprocess/BuildHandFusionInput.py"
    --code "${ROOT}/preprocess/Timebase.py"
    --param "prompt=${PROMPT}"
    --param "handedness=${HAMER_HANDEDNESS}"
    --param "visual_side=${VISUAL_SIDE}"
    --param "glove_side=${GLOVE_SIDE}"
    --param "fps=${FPS}"
    --param "max_frames=${MAX_FRAMES}"
    --param "depth_radius=${DEPTH_RADIUS}"
    --param "depth_method=${DEPTH_METHOD}"
    --param "robust_indices=${DEPTH_ROBUST_INDICES}"
    --param "palm_indices=${DEPTH_PALM_INDICES}"
    --param "min_candidates=${DEPTH_MIN_CANDIDATES}"
    --param "inlier_m=${DEPTH_ROBUST_INLIER_M}"
    --param "locate_device=${LOCATE_DEVICE}"
    --param "locate_dtype=${LOCATE_DTYPE}"
    --param "hamer_device=${HAMER_DEVICE}"
    --param "save_bbox_frames=${SAVE_BBOX_FRAMES}"
    --param "bbox_max_center_jump_px=${CALIB_BBOX_MAX_CENTER_JUMP_PX}"
    --param "bbox_lost_jump_px=${CALIB_BBOX_LOST_JUMP_PX}"
    --param "bbox_max_area_ratio=${CALIB_BBOX_MAX_AREA_RATIO}"
    --param "bbox_min_iou=${CALIB_BBOX_MIN_IOU}"
    --param "bbox_min_iou_center_px=${CALIB_BBOX_MIN_IOU_CENTER_PX}"
    --param "bbox_max_gap=${CALIB_BBOX_MAX_GAP}"
    --param "reference_fps=${TIME_FILTER_REFERENCE_FPS}"
    --output "${CALIB_FUSION_JSONL}"
    --output "${CALIB_FUSION_SUMMARY}"
  )
  if stage_cache_hit calibration_fusion "${calib_cache_args[@]}"; then
    printf '  skip valid cache: %s\n' "$(stage_manifest calibration_fusion)"
  else
    ROOT="${ROOT}" \
    CALIB_BAG_SESSION="${CALIB_BAG_SESSION}" \
    CALIB_BAG_DIR="${CALIB_BAG_DIR}" \
    CALIB_SESSION="${CALIB_SESSION}" \
    ROS_SETUP="${ROS_SETUP}" \
    HAND_MSG_SETUP="${HAND_MSG_SETUP}" \
    ROS_PYTHON="${ROS_PYTHON}" \
    LOCATE_PYTHON="${LOCATE_PYTHON}" \
    HAMER_PYTHON="${HAMER_PYTHON}" \
    LOCATE_MODEL="${LOCATE_MODEL}" \
    PROMPT="${PROMPT}" \
    PROMPT_TAG="${PROMPT_TAG}" \
    LOCATE_DTYPE="${LOCATE_DTYPE}" \
    LOCATE_DEVICE="${LOCATE_DEVICE}" \
    HAMER_DEVICE="${HAMER_DEVICE}" \
    HAMER_HANDEDNESS="${HAMER_HANDEDNESS}" \
    VISUAL_SIDE="${VISUAL_SIDE}" \
    GLOVE_SIDE="${GLOVE_SIDE}" \
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
    "${ROOT}/scripts/build_calib_video_fusion.sh"
    stage_cache_write calibration_fusion "${calib_cache_args[@]}"
  fi
  CALIB_INPUT_FUSION_FOR_FK="${CALIB_FUSION_JSONL}"
else
  printf '\n[calib] skip dedicated calibration video: USE_CALIB_VIDEO=%s\n' "${USE_CALIB_VIDEO}"
fi

printf '\n[8/11] Render visual 21 keypoints with temporal 2D smoothing\n'
visual_smooth_cache_args=(
  --input "$(stage_manifest fusion)"
  --code "${ROOT}/preprocess/VisualizeVisual2DSmooth.py"
  --code "${ROOT}/preprocess/Timebase.py"
  --param "fps=${FPS}"
  --param "alpha=${VISUAL_2D_SMOOTH_ALPHA}"
  --param "max_interp_gap=${VISUAL_2D_MAX_INTERP_GAP}"
  --param "reference_fps=${TIME_FILTER_REFERENCE_FPS}"
  --output "${VISUAL_SMOOTH_MP4}"
  --output "${VISUAL_SMOOTH_JSONL}"
  --output "${VISUAL_SMOOTH_SUMMARY}"
)
if stage_cache_hit visual_smooth "${visual_smooth_cache_args[@]}"; then
  printf '  skip valid cache: %s\n' "$(stage_manifest visual_smooth)"
else
  "${HAMER_PYTHON}" "${ROOT}/preprocess/VisualizeVisual2DSmooth.py" \
    --input_jsonl "${FUSION_JSONL}" \
    --out_path "${VISUAL_SMOOTH_MP4}" \
    --output_jsonl "${VISUAL_SMOOTH_JSONL}" \
    --summary_json "${VISUAL_SMOOTH_SUMMARY}" \
    --fps "${FPS}" \
    --alpha "${VISUAL_2D_SMOOTH_ALPHA}" \
    --max_interp_gap "${VISUAL_2D_MAX_INTERP_GAP}" \
    --reference_fps "${TIME_FILTER_REFERENCE_FPS}"
  stage_cache_write visual_smooth "${visual_smooth_cache_args[@]}"
fi

printf '\n[9/11] Glove FK + visual-bone calibration + wristroot tracking\n'
FK_DIR="${SESSION}/glove_fk21_visual_bones_smooth_solve045"
FINAL_TRAJECTORY="${FK_DIR}/trajectory_wristroot_track_cameraoptical.jsonl"
FINAL_TRAJECTORY_SUMMARY="${FK_DIR}/trajectory_wristroot_track_cameraoptical_summary.json"
FINAL_CALIBRATION="${FK_DIR}/glove_fk_to_camera_calib_smooth_solve045.json"
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
  --code "${ROOT}/preprocess/FixTrajectoryCameraOpticalWorld.py"
  --code "${ROOT}/preprocess/Timebase.py"
  --code "${ROOT}/preprocess/VisualizeGloveFkVsVisual.py"
  --code "${ROOT}/preprocess/VisualizeTrajectory3D.py"
  --param "fps=${FPS}"
  --param "glove_side=${GLOVE_SIDE}"
  --param "base_hand_config=${BASE_HAND_CONFIG}"
  --param "retarget_root=${RETARGET_ROOT}"
  --param "use_camera_optical_fix=${USE_CAMERA_OPTICAL_FIX}"
  --param "frame_start=${FRAME_START}"
  --param "frame_end=${FRAME_END}"
  --param "calib_frame_start=${CALIB_FRAME_START}"
  --param "calib_frame_end=${CALIB_FRAME_END}"
  --param "alpha_angle=${ALPHA_ANGLE}"
  --param "alpha_quat=${ALPHA_QUAT}"
  --param "wrist_track_alpha=${WRIST_TRACK_ALPHA}"
  --param "wrist_track_accept_step_m=${WRIST_TRACK_ACCEPT_STEP_M}"
  --param "wrist_track_pending_radius_m=${WRIST_TRACK_PENDING_RADIUS_M}"
  --param "wrist_track_confirm_frames=${WRIST_TRACK_CONFIRM_FRAMES}"
  --param "wrist_track_max_step_m=${WRIST_TRACK_MAX_STEP_M}"
  --param "apply_palm_quat=${APPLY_PALM_QUAT}"
  --param "constraint_prealign=${CONSTRAINT_PREALIGN}"
  --param "constraint_middle_idx=${CONSTRAINT_MIDDLE_IDX}"
  --param "constraint_thumb_idx=${CONSTRAINT_THUMB_IDX}"
  --param "flip_palm_normal=${FLIP_PALM_NORMAL}"
  --param "wrist_track_snap_root=${WRIST_TRACK_SNAP_ROOT}"
  --param "wrist_track_depth_only=${WRIST_TRACK_DEPTH_ONLY}"
  --param "reference_fps=${TIME_FILTER_REFERENCE_FPS}"
  --output "${FINAL_TRAJECTORY}"
  --output "${FINAL_TRAJECTORY_SUMMARY}"
  --output "${FINAL_CALIBRATION}"
)
if [[ -n "${CALIB_INPUT_FUSION_FOR_FK}" ]]; then
  glove_cache_args+=(--input "$(stage_manifest calibration_fusion)")
fi
if stage_cache_hit glove_fk_trajectory "${glove_cache_args[@]}"; then
  printf '  skip valid cache: %s\n' "$(stage_manifest glove_fk_trajectory)"
else
  SESSION="${SESSION}" \
  INPUT_FUSION="${FUSION_JSONL}" \
  CALIB_INPUT_FUSION="${CALIB_INPUT_FUSION_FOR_FK}" \
  PYTHON="${HAMER_PYTHON}" \
  FPS="${FPS}" \
  GLOVE_SIDE="${GLOVE_SIDE}" \
  BASE_HAND_CONFIG="${BASE_HAND_CONFIG}" \
  RETARGET_ROOT="${RETARGET_ROOT}" \
  FRAME_START="${FRAME_START}" \
  FRAME_END="${FRAME_END}" \
  CALIB_FRAME_START="${CALIB_FRAME_START}" \
  CALIB_FRAME_END="${CALIB_FRAME_END}" \
  ALPHA_ANGLE="${ALPHA_ANGLE}" \
  ALPHA_QUAT="${ALPHA_QUAT}" \
  WRIST_TRACK_ALPHA="${WRIST_TRACK_ALPHA}" \
  WRIST_TRACK_ACCEPT_STEP_M="${WRIST_TRACK_ACCEPT_STEP_M}" \
  WRIST_TRACK_PENDING_RADIUS_M="${WRIST_TRACK_PENDING_RADIUS_M}" \
  WRIST_TRACK_CONFIRM_FRAMES="${WRIST_TRACK_CONFIRM_FRAMES}" \
  WRIST_TRACK_MAX_STEP_M="${WRIST_TRACK_MAX_STEP_M}" \
  USE_CAMERA_OPTICAL_FIX="${USE_CAMERA_OPTICAL_FIX}" \
  CONSTRAINT_PREALIGN="${CONSTRAINT_PREALIGN}" \
  APPLY_PALM_QUAT="${APPLY_PALM_QUAT}" \
  CONSTRAINT_MIDDLE_IDX="${CONSTRAINT_MIDDLE_IDX}" \
  CONSTRAINT_THUMB_IDX="${CONSTRAINT_THUMB_IDX}" \
  FLIP_PALM_NORMAL="${FLIP_PALM_NORMAL}" \
  WRIST_TRACK_SNAP_ROOT="${WRIST_TRACK_SNAP_ROOT}" \
  WRIST_TRACK_DEPTH_ONLY="${WRIST_TRACK_DEPTH_ONLY}" \
  TF_STATIC_JSONL="${TF_STATIC_JSONL}" \
  TIME_FILTER_REFERENCE_FPS="${TIME_FILTER_REFERENCE_FPS}" \
  "${ROOT}/scripts/run_glove_fk_visual_bones_pipeline.sh"
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
  --output "${OUTPUT_DIR}/manifest.json"
  --output "${OUTPUT_DIR}/data/trajectory_wristroot_track_cameraoptical.jsonl"
)
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
  "${HAMER_PYTHON}" "${ROOT}/preprocess/CollectPipelineOutputs.py" \
    --session_path "${SESSION}" \
    --output_dir "${OUTPUT_DIR}" \
    --prompt_tag "${PROMPT_TAG}" \
    --out_tag visual_bones_smooth_solve045
  stage_cache_write collect_outputs "${collect_cache_args[@]}"
fi


printf '\n[11/11] Generate review web visualization\n'
web_cache_args=(
  --input "$(stage_manifest collect_outputs)"
  --input "${OUTPUT_DIR}/data/trajectory_wristroot_track_cameraoptical.jsonl"
  --code "${ROOT}/scripts/generate_review_web.py"
  --code "${ROOT}/preprocess/Timebase.py"
  --param "fps=${FPS}"
  --param "hand_display_rotate_deg=${REVIEW_HAND_DISPLAY_ROTATE_DEG}"
  --output "${REVIEW_WEB_HTML}"
  --output "${OUTPUT_DIR}/web/rgb_frames"
  --output "${OUTPUT_DIR}/web/traj_frames"
  --output "${OUTPUT_DIR}/web/tactile_frames"
)
if stage_cache_hit review_web "${web_cache_args[@]}"; then
  printf '  skip valid cache: %s\n' "$(stage_manifest review_web)"
else
  "${HAMER_PYTHON}" "${ROOT}/scripts/generate_review_web.py" \
    --session "${SESSION}" \
    --fps "${FPS}" \
    --hand_display_rotate_deg "${REVIEW_HAND_DISPLAY_ROTATE_DEG}"
  stage_cache_write review_web "${web_cache_args[@]}"
fi

if [[ "${COMPACT_OUTPUTS}" == "1" ]]; then
  printf '\n[quality] Validate outputs before compaction\n'
  "${HAMER_PYTHON}" "${ROOT}/scripts/validate_pipeline_quality.py" \
    --session "${SESSION}" \
    --report "${QUALITY_REPORT}" \
    --min_hand_match_ratio "${QUALITY_MIN_HAND_MATCH_RATIO:-0.95}" \
    --min_visual_ratio "${QUALITY_MIN_VISUAL_RATIO:-0.90}" \
    --min_depth_applied_ratio "${QUALITY_MIN_DEPTH_APPLIED_RATIO:-0.85}" \
    --max_calibration_median_m "${QUALITY_MAX_CALIBRATION_MEDIAN_M:-0.030}" \
    --max_calibration_p95_m "${QUALITY_MAX_CALIBRATION_P95_M:-0.060}" \
    --max_wrist_residual_p95_m "${QUALITY_MAX_WRIST_RESIDUAL_P95_M:-0.040}" \
    --min_rtabmap_coverage_ratio "${QUALITY_MIN_RTABMAP_COVERAGE_RATIO:-1.0}" \
    --max_rtabmap_interp_gap_sec "${QUALITY_MAX_RTABMAP_INTERP_GAP_SEC:-${RTABMAP_MAX_INTERP_GAP_SEC}}" \
    "${quality_rtabmap_args[@]}"
  printf '\n[compact] Quality passed; compact finished session outputs\n'
  "${HAMER_PYTHON}" "${ROOT}/scripts/compact_postprocess_session.py" --session "${SESSION}"
else
  printf '\n[compact] skip: COMPACT_OUTPUTS=%s\n' "${COMPACT_OUTPUTS}"
fi
printf '\nDone\n'
printf '  session:        %s\n' "${SESSION}"
printf '  outputs:        %s\n' "${OUTPUT_DIR}"
printf '  rtabmap poses:  %s\n' "${RTABMAP_TRAJECTORY_JSONL}"
printf '  2D smooth video:%s\n' "${OUTPUT_DIR}/videos/04_visual_21kpts_2d_smooth.mp4"
printf '  3D video:       %s\n' "${OUTPUT_DIR}/videos/07_trajectory_3d_world.mp4"
printf '  trajectory:     %s\n' "${OUTPUT_DIR}/data/trajectory_wristroot_track_cameraoptical.jsonl"
printf '  review web:     %s\n' "${REVIEW_WEB_HTML}"
printf '  compact summary: %s\n' "${COMPACT_SUMMARY}"
if [[ -n "${CALIB_INPUT_FUSION_FOR_FK:-}" ]]; then
  printf '  calib fusion:   %s\n' "${CALIB_INPUT_FUSION_FOR_FK}"
fi
