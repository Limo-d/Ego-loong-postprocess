#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lenovo/Ego-loong-postprocess}"
CALIB_BAG_SESSION="${CALIB_BAG_SESSION:?CALIB_BAG_SESSION is required}"
CALIB_BAG_DIR="${CALIB_BAG_DIR:-${CALIB_BAG_SESSION}/bag}"
CALIB_SESSION="${CALIB_SESSION:?CALIB_SESSION is required}"
HAND_CALIBRATION_FILE="${HAND_CALIBRATION_FILE:-}"
CAMERA_EXTRINSICS_FILE="${CAMERA_EXTRINSICS_FILE:-}"

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
HAMER_HANDEDNESS="${HAMER_HANDEDNESS:-all_left}"
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
TIME_FILTER_REFERENCE_FPS="${TIME_FILTER_REFERENCE_FPS:-30}"
RESOLVE_DRIVER="${RESOLVE_DRIVER:-/home/lenovo/Retarget/data/ros_ws/resolve_check/resolve_driver}"
RENDER_DEBUG_VIDEOS="${RENDER_DEBUG_VIDEOS:-0}"
PARALLEL_HANDS="${PARALLEL_HANDS:-1}"

RGBD_DIR="${CALIB_SESSION}/preprocess"
LOCATE_DIR="${CALIB_SESSION}/locateanything_${PROMPT_TAG}"
STABLE_BBOX_DIR="${CALIB_SESSION}/locateanything_${PROMPT_TAG}_stable"
HAMER_DIR="${CALIB_SESSION}/hamer_from_stable_locateanything_${PROMPT_TAG}_force_right"
DEPTH_DIR="${CALIB_SESSION}/depth_correct_hamer_force_right"
FUSION_LEFT_DIR="${CALIB_SESSION}/fusion_input_left_depthroot"
FUSION_RIGHT_DIR="${CALIB_SESSION}/fusion_input_right_depthroot"

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
HAMER_FRAME_DIR="${HAMER_DIR}/per_frame"
DEPTH_FRAME_DIR="${DEPTH_DIR}/per_frame"

mkdir -p "${CALIB_SESSION}" "${LOCATE_DIR}" "${STABLE_BBOX_DIR}" "${HAMER_DIR}" "${DEPTH_DIR}" "${FUSION_LEFT_DIR}" "${FUSION_RIGHT_DIR}"

has_output() {
  [[ -s "$1" && "${OVERWRITE}" != "1" ]]
}

max_frames_args=()
if [[ -n "${MAX_FRAMES}" ]]; then
  max_frames_args=(--max_frames "${MAX_FRAMES}")
fi

printf '\n[calib 1/6] Extract calibration ROS2 bag RGBD/hand_frame\n'
if [[ ! -d "${CALIB_BAG_DIR}" ]]; then
  printf '  ERROR: CALIB_BAG_DIR not found: %s\n' "${CALIB_BAG_DIR}" >&2
  exit 1
fi
if has_output "${RGBD_DIR}/timestamps.jsonl"; then
  printf '  skip existing: %s\n' "${RGBD_DIR}/timestamps.jsonl"
else
  set +u
  source "${ROS_SETUP}"
  source "${HAND_MSG_SETUP}"
  set -u
  extract_args=(
    --session_path "${CALIB_BAG_SESSION}"
    --bag_dir "${CALIB_BAG_DIR}"
    --output_dir "${RGBD_DIR}"
    --resolve_driver "${RESOLVE_DRIVER}"
    --image_write_workers "${EXTRACT_IMAGE_WRITE_WORKERS}"
  )
  if [[ -n "${HAND_CALIBRATION_FILE}" && -f "${HAND_CALIBRATION_FILE}" ]]; then
    extract_args+=(--handcal_path "${HAND_CALIBRATION_FILE}")
  fi
  if [[ -n "${CAMERA_EXTRINSICS_FILE}" && -f "${CAMERA_EXTRINSICS_FILE}" ]]; then
    extract_args+=(--camera_extrinsics "${CAMERA_EXTRINSICS_FILE}")
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    extract_args+=(--overwrite)
  fi
  if [[ -n "${MAX_FRAMES}" ]]; then
    extract_args+=(--max_frames "${MAX_FRAMES}")
  fi
  "${ROS_PYTHON}" "${ROOT}/preprocess/ExtractRosbagSampler.py" "${extract_args[@]}"
fi

FPS="$("${ROS_PYTHON}" -c 'import json,sys; value=float(json.load(open(sys.argv[1]))["fps"]); assert value > 0; print(f"{value:.9f}")' "${RGBD_DIR}/extract_summary.json")"
printf '  calibration RGB timebase: %.6f fps from rgb_stamp_ns\n' "${FPS}"
if [[ -n "${REQUESTED_FPS}" ]]; then
  printf '  note: ignoring requested FPS=%s; calibration RGB timebase is authoritative\n' "${REQUESTED_FPS}"
fi

printf '\n[calib 2/6] LocateAnything bbox detector\n'
if has_output "${LOCATE_JSON}"; then
  printf '  skip existing: %s\n' "${LOCATE_JSON}"
else
  locate_frame_args=(--no_save_frames)
  if [[ "${SAVE_BBOX_FRAMES}" == "1" ]]; then
    locate_frame_args=(--out_frames_dir "${LOCATE_DIR}/bbox_frames")
  fi
  if [[ "${RENDER_DEBUG_VIDEOS}" != "1" ]]; then
    locate_frame_args+=(--no_video)
  fi
  "${LOCATE_PYTHON}" "${ROOT}/preprocess/VisualizeLocateAnythingBboxes.py" \
    --session_path "${CALIB_SESSION}" \
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
fi

printf '\n[calib 3/6] Track/stabilize calibration bbox\n'
if has_output "${STABLE_BBOX_JSON}"; then
  printf '  skip existing: %s\n' "${STABLE_BBOX_JSON}"
else
  track_video_args=()
  if [[ "${RENDER_DEBUG_VIDEOS}" == "1" ]]; then
    track_video_args=(--out_video "${STABLE_BBOX_MP4}")
  fi
  "${HAMER_PYTHON}" "${ROOT}/preprocess/TrackDualHandBboxes.py" \
    --session_path "${CALIB_SESSION}" \
    --input_json "${LOCATE_JSON}" \
    --output_json "${STABLE_BBOX_JSON}" \
    --summary_json "${STABLE_BBOX_DIR}/tracking_summary.json" \
    "${track_video_args[@]}" \
    --fps "${FPS}" \
    --max_center_jump_px "${CALIB_BBOX_MAX_CENTER_JUMP_PX}" \
    --lost_jump_px "${CALIB_BBOX_LOST_JUMP_PX}" \
    --max_area_ratio "${CALIB_BBOX_MAX_AREA_RATIO}" \
    --min_iou "${CALIB_BBOX_MIN_IOU}" \
    --min_iou_center_px "${CALIB_BBOX_MIN_IOU_CENTER_PX}" \
    --max_gap "${CALIB_BBOX_MAX_GAP}" \
    --reference_fps "${TIME_FILTER_REFERENCE_FPS}" \
    --image_left_side "${IMAGE_LEFT_PHYSICAL_SIDE}"
fi

printf '\n[calib 4/6] HaMeR from calibration bbox\n'
if has_output "${HAMER_FRAME_DIR}/00000/${HAMER_JSON_NAME}"; then
  printf '  skip existing per-frame HaMeR json: %s\n' "${HAMER_JSON_NAME}"
else
  hamer_video_args=()
  if [[ "${RENDER_DEBUG_VIDEOS}" != "1" ]]; then
    hamer_video_args=(--no_video)
  fi
  "${HAMER_PYTHON}" "${ROOT}/preprocess/run_hamer_from_locate_bboxes.py" \
    --session_path "${CALIB_SESSION}" \
    --bbox_json "${STABLE_BBOX_JSON}" \
    --fallback_bbox_json "" \
    --aggregate_json "${HAMER_AGG_JSON}" \
    --per_frame_output_dir "${HAMER_FRAME_DIR}" \
    --out_json_name "${HAMER_JSON_NAME}" \
    --out_video "${HAMER_MP4}" \
    --device "${HAMER_DEVICE}" \
    --batch_size "${HAMER_BATCH_SIZE}" \
    --max_boxes 2 \
    --handedness "${HAMER_HANDEDNESS}" \
    --fps "${FPS}" \
    "${hamer_video_args[@]}" \
    "${max_frames_args[@]}"
fi

printf '\n[calib 5/6] Correct calibration HaMeR wrist root with aligned depth\n'
if has_output "${DEPTH_FRAME_DIR}/00000/${HAMER_DEPTH_JSON_NAME}"; then
  printf '  skip existing per-frame depth-root json: %s\n' "${HAMER_DEPTH_JSON_NAME}"
else
  "${HAMER_PYTHON}" "${ROOT}/preprocess/DepthCorrectHandKpts.py" \
    --session_path "${CALIB_SESSION}" \
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
fi

fusion_swap_args=()
if [[ "${HAND_FRAME_SWAP_LR}" == "1" ]]; then
  fusion_swap_args+=(--swap_hand_frame_lr)
fi
printf '\n[calib 6/6] Build left/right calibration visual + /hand_frame fusion inputs\n'
fusion_pids=()
fusion_sides=()
for side in left right; do
  if [[ "${side}" == "left" ]]; then
    visual_side=hand_l
    fusion_jsonl="${FUSION_LEFT_JSONL}"
    fusion_summary="${FUSION_LEFT_SUMMARY}"
  else
    visual_side=hand_r
    fusion_jsonl="${FUSION_RIGHT_JSONL}"
    fusion_summary="${FUSION_RIGHT_SUMMARY}"
  fi
  if has_output "${fusion_jsonl}"; then
    printf '  skip existing: %s\n' "${fusion_jsonl}"
    continue
  fi
  fusion_cmd=(
    "${HAMER_PYTHON}" "${ROOT}/preprocess/BuildHandFusionInput.py"
    --session_path "${CALIB_SESSION}"
    --rgbd_subdir preprocess
    --visual_json_name "${HAMER_DEPTH_JSON_NAME}"
    --visual_json_dir "${DEPTH_FRAME_DIR}"
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
    printf 'ERROR: calibration %s fusion failed\n' "${fusion_sides[$i]}" >&2
    fusion_failed=1
  fi
done
if [[ "${fusion_failed}" == "1" ]]; then
  exit 1
fi

printf '\n[calib] Done\n'
printf '  calibration session: %s\n' "${CALIB_SESSION}"
printf '  left calibration fusion:  %s\n' "${FUSION_LEFT_JSONL}"
printf '  right calibration fusion: %s\n' "${FUSION_RIGHT_JSONL}"
