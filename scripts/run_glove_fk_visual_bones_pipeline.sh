#!/usr/bin/env bash
set -euo pipefail

# Reproduce the current glove-FK + visual-bone calibration branch.
# This script starts from an existing fusion_frames.jsonl produced by BuildHandFusionInput.py.
# If CALIB_INPUT_FUSION is set, visual hand geometry and R_cam_glove are estimated
# from that dedicated calibration sequence; the final trajectory is still built
# from INPUT_FUSION.

ROOT="${ROOT:-/home/lenovo/Ego-loong-postprocess}"
PYTHON="${PYTHON:-/home/lenovo/miniconda3/envs/hamer/bin/python}"
SESSION="${SESSION:-${ROOT}/postprocess_data/data_20260628_123410}"
INPUT_FUSION="${INPUT_FUSION:-${SESSION}/fusion_input_force_right_depthroot/fusion_frames.jsonl}"
CALIB_INPUT_FUSION="${CALIB_INPUT_FUSION:-}"
BASE_HAND_CONFIG="${BASE_HAND_CONFIG:-/home/lenovo/Retarget/host/hand_config.json}"
RETARGET_ROOT="${RETARGET_ROOT:-/home/lenovo/Retarget/retarget}"
GLOVE_SIDE="${GLOVE_SIDE:-left}"
FRAME_START="${FRAME_START:-120}"
FRAME_END="${FRAME_END:-170}"
CALIB_FRAME_START="${CALIB_FRAME_START:-}"
CALIB_FRAME_END="${CALIB_FRAME_END:-}"
ALPHA_ANGLE="${ALPHA_ANGLE:-0.45}"
ALPHA_QUAT="${ALPHA_QUAT:-0.45}"
FPS="${FPS:-20}"
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
TIME_FILTER_REFERENCE_FPS="${TIME_FILTER_REFERENCE_FPS:-30}"

OUT_TAG="visual_bones_smooth_solve045"
if [[ -n "${CALIB_INPUT_FUSION}" ]]; then
  CONFIG_DIR="${SESSION}/hand_config_visual_bones_calib_video"
  VISUAL_CONFIG="${CONFIG_DIR}/hand_config_visual_bones_calib_video.json"
  VISUAL_CONFIG_SUMMARY="${CONFIG_DIR}/hand_config_visual_bones_calib_video_summary.json"
else
  CONFIG_DIR="${SESSION}/hand_config_visual_bones_${FRAME_START}_${FRAME_END}"
  VISUAL_CONFIG="${CONFIG_DIR}/hand_config_visual_bones_${FRAME_START}_${FRAME_END}.json"
  VISUAL_CONFIG_SUMMARY="${CONFIG_DIR}/hand_config_visual_bones_${FRAME_START}_${FRAME_END}_summary.json"
fi
SMOOTH_DIR="${SESSION}/fusion_input_force_right_depthroot_smooth_solve045"
FK_DIR="${SESSION}/glove_fk21_${OUT_TAG}"
CALIB_WORK_DIR="${SESSION}/handeye_calibration_${OUT_TAG}"

SMOOTH_FUSION="${SMOOTH_DIR}/fusion_frames_smooth_solve045.jsonl"
SMOOTH_SUMMARY="${SMOOTH_DIR}/smooth_solve045_summary.json"
FK_FRAMES="${FK_DIR}/glove_fk21_${OUT_TAG}_frames.jsonl"
FK_SUMMARY="${FK_DIR}/glove_fk21_${OUT_TAG}_summary.json"
CALIB_JSON="${FK_DIR}/glove_fk_to_camera_calib_smooth_solve045.json"
CALIBRATED_FRAMES="${FK_DIR}/glove_fk21_${OUT_TAG}_calibrated_frames.jsonl"
CALIBRATED_SUMMARY="${FK_DIR}/glove_fk21_${OUT_TAG}_calibrated_summary.json"
WRIST_TRACK_FRAMES="${FK_DIR}/glove_fk21_${OUT_TAG}_calibrated_wristroot_track_frames.jsonl"
WRIST_TRACK_SUMMARY="${FK_DIR}/wristroot_track_summary.json"
OVERLAY_MP4="${FK_DIR}/glove_fk_vs_hamer_${OUT_TAG}_overlay.mp4"
OVERLAY_SUMMARY="${FK_DIR}/glove_fk_vs_hamer_${OUT_TAG}_overlay_summary.json"
TRACK_OVERLAY_MP4="${FK_DIR}/glove_fk_vs_hamer_${OUT_TAG}_wristroot_track_overlay.mp4"
TRACK_OVERLAY_SUMMARY="${FK_DIR}/glove_fk_vs_hamer_${OUT_TAG}_wristroot_track_overlay_summary.json"
TRAJECTORY_JSONL="${FK_DIR}/trajectory_wristroot_track.jsonl"
TRAJECTORY_SUMMARY="${FK_DIR}/trajectory_wristroot_track_summary.json"
TRAJECTORY_CAMERAOPTICAL_JSONL="${FK_DIR}/trajectory_wristroot_track_cameraoptical.jsonl"
TRAJECTORY_CAMERAOPTICAL_SUMMARY="${FK_DIR}/trajectory_wristroot_track_cameraoptical_summary.json"
TRAJECTORY_3D_MP4="${FK_DIR}/trajectory_3d_world_wristroot_track_cameraoptical.mp4"
TRAJECTORY_3D_SUMMARY="${FK_DIR}/trajectory_3d_world_wristroot_track_cameraoptical_summary.json"
TF_STATIC_JSONL="${TF_STATIC_JSONL:-${SESSION}/preprocess/tf_static.jsonl}"

mkdir -p "${CONFIG_DIR}" "${SMOOTH_DIR}" "${FK_DIR}" "${CALIB_WORK_DIR}"

if [[ -n "${CALIB_INPUT_FUSION}" ]]; then
  HAND_CONFIG_FUSION="${CALIB_INPUT_FUSION}"
  HAND_CONFIG_FRAME_START="${CALIB_FRAME_START}"
  HAND_CONFIG_FRAME_END="${CALIB_FRAME_END}"
  CALIB_SOURCE_LABEL="dedicated calibration fusion"
else
  HAND_CONFIG_FUSION="${INPUT_FUSION}"
  HAND_CONFIG_FRAME_START="${FRAME_START}"
  HAND_CONFIG_FRAME_END="${FRAME_END}"
  CALIB_SOURCE_LABEL="main sequence window"
fi

ESTIMATE_ARGS=(
  --input_jsonl "${HAND_CONFIG_FUSION}"
  --base_config "${BASE_HAND_CONFIG}"
  --output_config "${VISUAL_CONFIG}"
  --summary_json "${VISUAL_CONFIG_SUMMARY}"
  --hand "${GLOVE_SIDE}"
)
if [[ -n "${HAND_CONFIG_FRAME_START}" ]]; then
  ESTIMATE_ARGS+=(--frame_start "${HAND_CONFIG_FRAME_START}")
fi
if [[ -n "${HAND_CONFIG_FRAME_END}" ]]; then
  ESTIMATE_ARGS+=(--frame_end "${HAND_CONFIG_FRAME_END}")
fi

printf '\n[1/13] Estimate visual hand bones from %s\n' "${CALIB_SOURCE_LABEL}"
printf '  input: %s\n' "${HAND_CONFIG_FUSION}"
"${PYTHON}" "${ROOT}/preprocess/EstimateHandConfigFromVisual.py" "${ESTIMATE_ARGS[@]}"

printf '\n[2/13] Smooth main glove solve_state (%s, alpha_angle=%s, alpha_quat=%s)\n' "${GLOVE_SIDE}" "${ALPHA_ANGLE}" "${ALPHA_QUAT}"
"${PYTHON}" "${ROOT}/preprocess/SmoothGloveSolveState.py" \
  --input_jsonl "${INPUT_FUSION}" \
  --output_jsonl "${SMOOTH_FUSION}" \
  --summary_json "${SMOOTH_SUMMARY}" \
  --glove_side "${GLOVE_SIDE}" \
  --alpha_angle "${ALPHA_ANGLE}" \
  --alpha_quat "${ALPHA_QUAT}" \
  --reference_fps "${TIME_FILTER_REFERENCE_FPS}"

printf '\n[3/13] Build main glove FK 21 points with visual bones (apply_palm_quat=%s)\n' "${APPLY_PALM_QUAT}"
BUILD_MAIN_FK_ARGS=(
  --input_jsonl "${SMOOTH_FUSION}"
  --output_jsonl "${FK_FRAMES}"
  --summary_json "${FK_SUMMARY}"
  --hand_config "${VISUAL_CONFIG}"
  --retarget_root "${RETARGET_ROOT}"
  --glove_side "${GLOVE_SIDE}"
  --alignment visual_wrist_rotation
  --no_mirror_to_right
)
if [[ "${APPLY_PALM_QUAT}" == "1" ]]; then
  BUILD_MAIN_FK_ARGS+=(--apply_palm_quat)
fi
"${PYTHON}" "${ROOT}/preprocess/BuildGloveFk21FromFusion.py" "${BUILD_MAIN_FK_ARGS[@]}"

CALIB_FK_FOR_ROTATION="${FK_FRAMES}"
CALIB_ROT_FRAME_START="${FRAME_START}"
CALIB_ROT_FRAME_END="${FRAME_END}"
if [[ -n "${CALIB_INPUT_FUSION}" ]]; then
  CALIB_SMOOTH_FUSION="${CALIB_WORK_DIR}/fusion_frames_smooth_solve045.jsonl"
  CALIB_SMOOTH_SUMMARY="${CALIB_WORK_DIR}/smooth_solve045_summary.json"
  CALIB_FK_FOR_ROTATION="${CALIB_WORK_DIR}/glove_fk21_${OUT_TAG}_frames.jsonl"
  CALIB_FK_SUMMARY="${CALIB_WORK_DIR}/glove_fk21_${OUT_TAG}_summary.json"
  CALIB_ROT_FRAME_START="${CALIB_FRAME_START}"
  CALIB_ROT_FRAME_END="${CALIB_FRAME_END}"

  printf '\n[4/13] Build dedicated hand-eye calibration FK from calib fusion\n'
  "${PYTHON}" "${ROOT}/preprocess/SmoothGloveSolveState.py" \
    --input_jsonl "${CALIB_INPUT_FUSION}" \
    --output_jsonl "${CALIB_SMOOTH_FUSION}" \
    --summary_json "${CALIB_SMOOTH_SUMMARY}" \
    --glove_side "${GLOVE_SIDE}" \
    --alpha_angle "${ALPHA_ANGLE}" \
    --alpha_quat "${ALPHA_QUAT}" \
    --reference_fps "${TIME_FILTER_REFERENCE_FPS}"
  BUILD_CALIB_FK_ARGS=(
    --input_jsonl "${CALIB_SMOOTH_FUSION}"
    --output_jsonl "${CALIB_FK_FOR_ROTATION}"
    --summary_json "${CALIB_FK_SUMMARY}"
    --hand_config "${VISUAL_CONFIG}"
    --retarget_root "${RETARGET_ROOT}"
    --glove_side "${GLOVE_SIDE}"
    --alignment visual_wrist_rotation
  --no_mirror_to_right
  )
  if [[ "${APPLY_PALM_QUAT}" == "1" ]]; then
    BUILD_CALIB_FK_ARGS+=(--apply_palm_quat)
  fi
  "${PYTHON}" "${ROOT}/preprocess/BuildGloveFk21FromFusion.py" "${BUILD_CALIB_FK_ARGS[@]}"
else
  printf '\n[4/13] Use main FK calibration window %s-%s\n' "${FRAME_START}" "${FRAME_END}"
fi

CALIB_ARGS=(
  --input_jsonl "${CALIB_FK_FOR_ROTATION}"
  --output_json "${CALIB_JSON}"
)
if [[ -n "${CALIB_ROT_FRAME_START}" ]]; then
  CALIB_ARGS+=(--frame_start "${CALIB_ROT_FRAME_START}")
fi
if [[ -n "${CALIB_ROT_FRAME_END}" ]]; then
  CALIB_ARGS+=(--frame_end "${CALIB_ROT_FRAME_END}")
fi
if [[ "${CONSTRAINT_PREALIGN}" == "1" ]]; then
  CALIB_ARGS+=(
    --constraint_prealign
    --middle_idx "${CONSTRAINT_MIDDLE_IDX}"
    --thumb_idx "${CONSTRAINT_THUMB_IDX}"
  )
fi

printf '\n[5/13] Calibrate fixed R_cam_glove from %s (constraint_prealign=%s, middle=%s, thumb=%s)\n' "${CALIB_SOURCE_LABEL}" "${CONSTRAINT_PREALIGN}" "${CONSTRAINT_MIDDLE_IDX}" "${CONSTRAINT_THUMB_IDX}"
printf '  input: %s\n' "${CALIB_FK_FOR_ROTATION}"
"${PYTHON}" "${ROOT}/preprocess/CalibrateGloveFkToCamera.py" "${CALIB_ARGS[@]}"

printf '\n[6/13] Apply calibration to all main frames (flip_palm_normal=%s)\n' "${FLIP_PALM_NORMAL}"
APPLY_ARGS=(
  --input_jsonl "${FK_FRAMES}"
  --calib_json "${CALIB_JSON}"
  --output_jsonl "${CALIBRATED_FRAMES}"
  --summary_json "${CALIBRATED_SUMMARY}"
)
if [[ "${FLIP_PALM_NORMAL}" == "1" ]]; then
  APPLY_ARGS+=(--flip_palm_normal)
fi
"${PYTHON}" "${ROOT}/preprocess/ApplyGloveFkCalibration.py" "${APPLY_ARGS[@]}"

printf '\n[7/13] Track wrist/root depth mode with hysteresis\n'
TRACK_ARGS=(
  --input_jsonl "${CALIBRATED_FRAMES}"
  --output_jsonl "${WRIST_TRACK_FRAMES}"
  --summary_json "${WRIST_TRACK_SUMMARY}"
  --alpha "${WRIST_TRACK_ALPHA}"
  --accept_step_m "${WRIST_TRACK_ACCEPT_STEP_M}"
  --pending_radius_m "${WRIST_TRACK_PENDING_RADIUS_M}"
  --confirm_frames "${WRIST_TRACK_CONFIRM_FRAMES}"
  --max_step_m "${WRIST_TRACK_MAX_STEP_M}"
  --reference_fps "${TIME_FILTER_REFERENCE_FPS}"
)
if [[ "${WRIST_TRACK_SNAP_ROOT}" == "1" ]]; then
  TRACK_ARGS+=(--snap_to_input_root)
elif [[ "${WRIST_TRACK_DEPTH_ONLY}" == "1" ]]; then
  TRACK_ARGS+=(--filter_root_depth_only)
fi
"${PYTHON}" "${ROOT}/preprocess/TrackGloveWristRoot.py" "${TRACK_ARGS[@]}"

printf '\n[8/13] Render raw calibrated HaMeR-vs-glove overlay video\n'
"${PYTHON}" "${ROOT}/preprocess/VisualizeGloveFkVsVisual.py" \
  --input_jsonl "${CALIBRATED_FRAMES}" \
  --out_path "${OVERLAY_MP4}" \
  --summary_json "${OVERLAY_SUMMARY}" \
  --fps "${FPS}"

printf '\n[9/13] Render wristroot_track HaMeR-vs-glove overlay video\n'
"${PYTHON}" "${ROOT}/preprocess/VisualizeGloveFkVsVisual.py" \
  --input_jsonl "${WRIST_TRACK_FRAMES}" \
  --out_path "${TRACK_OVERLAY_MP4}" \
  --summary_json "${TRACK_OVERLAY_SUMMARY}" \
  --fps "${FPS}"

printf '\n[10/13] Build unified wristroot_track trajectory\n'
"${PYTHON}" "${ROOT}/preprocess/BuildUnifiedTrajectory.py" \
  --calibrated_fk_jsonl "${WRIST_TRACK_FRAMES}" \
  --smoothed_fusion_jsonl "${SMOOTH_FUSION}" \
  --output_jsonl "${TRAJECTORY_JSONL}" \
  --summary_json "${TRAJECTORY_SUMMARY}" \
  --glove_side "${GLOVE_SIDE}" \
  --calib_json "${CALIB_JSON}" \
  --hand_config_json "${VISUAL_CONFIG}" \
  --hand_config_summary_json "${VISUAL_CONFIG_SUMMARY}" \
  --smoothing_summary_json "${SMOOTH_SUMMARY}"

printf '\n[11/13] Fix trajectory world pose with base_link -> RGB optical TF\n'
if [[ "${USE_CAMERA_OPTICAL_FIX}" == "1" ]]; then
  "${PYTHON}" "${ROOT}/preprocess/FixTrajectoryCameraOpticalWorld.py" \
    --input_jsonl "${TRAJECTORY_JSONL}" \
    --output_jsonl "${TRAJECTORY_CAMERAOPTICAL_JSONL}" \
    --summary_json "${TRAJECTORY_CAMERAOPTICAL_SUMMARY}" \
    --tf_static_jsonl "${TF_STATIC_JSONL}" \
    --base_frame base_link \
    --camera_frame oak_rgb_optical_frame \
    --allow_default_oak_rgb_optical
else
  cp "${TRAJECTORY_JSONL}" "${TRAJECTORY_CAMERAOPTICAL_JSONL}"
  "${PYTHON}" -c 'import json, sys; from pathlib import Path; Path(sys.argv[2]).write_text(json.dumps({"method":"skipped_camera_optical_fix","reason":"camera.c2w already comes from RTAB-Map camera pose","input_jsonl":sys.argv[1],"output_jsonl":sys.argv[3]}, indent=4) + "\n")' "${TRAJECTORY_JSONL}" "${TRAJECTORY_CAMERAOPTICAL_SUMMARY}" "${TRAJECTORY_CAMERAOPTICAL_JSONL}"
  printf '  skip camera optical fix: USE_CAMERA_OPTICAL_FIX=%s\n' "${USE_CAMERA_OPTICAL_FIX}"
fi

printf '\n[12/13] Render wristroot_track camera-optical world trajectory video\n'
"${PYTHON}" "${ROOT}/preprocess/VisualizeTrajectory3D.py" \
  --trajectory_jsonl "${TRAJECTORY_CAMERAOPTICAL_JSONL}" \
  --out_path "${TRAJECTORY_3D_MP4}" \
  --summary_json "${TRAJECTORY_3D_SUMMARY}" \
  --fps "${FPS}"

printf '\n[13/13] Done. Main outputs:\n'
printf '  visual config:    %s\n' "${VISUAL_CONFIG}"
printf '  calibration json: %s\n' "${CALIB_JSON}"
printf '  calibrated frames:%s\n' "${CALIBRATED_FRAMES}"
printf '  wrist track frames:%s\n' "${WRIST_TRACK_FRAMES}"
printf '  raw overlay video: %s\n' "${OVERLAY_MP4}"
printf '  track overlay video:%s\n' "${TRACK_OVERLAY_MP4}"
printf '  trajectory jsonl:  %s\n' "${TRAJECTORY_JSONL}"
printf '  cameraopt jsonl:   %s\n' "${TRAJECTORY_CAMERAOPTICAL_JSONL}"
printf '  3D trajectory mp4: %s\n' "${TRAJECTORY_3D_MP4}"
