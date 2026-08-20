#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lenovo/Ego-loong-postprocess}"
BATCH_ROOT="${BATCH_ROOT:-${1:-}}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

if [[ -z "${BATCH_ROOT}" ]]; then
  printf 'Usage: BATCH_ROOT=/path/to/batch %s\n' "$0" >&2
  exit 2
fi
BATCH_ROOT="${BATCH_ROOT%/}"
CALIBRATION_DIR="${CALIBRATION_DIR:-${BATCH_ROOT}/calibrations}"
PIPELINE="${ROOT}/scripts/run_sampler_bag_to_glove_trajectory.sh"
BATCH_NAME="$(basename "${BATCH_ROOT}")"
BATCH_CALIB_SESSION="${BATCH_CALIB_SESSION:-${ROOT}/postprocess_data/_batch_calibration/${BATCH_NAME}}"
BATCH_CALIB_CACHE_DIR="${BATCH_CALIB_CACHE_DIR:-${BATCH_CALIB_SESSION}/.pipeline_cache}"

if [[ ! -d "${CALIBRATION_DIR}" ]]; then
  printf 'ERROR: shared calibration directory not found: %s\n' "${CALIBRATION_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CALIBRATION_DIR}/calibration_manifest.json" ]]; then
  printf 'WARNING: calibration manifest not found: %s\n' "${CALIBRATION_DIR}/calibration_manifest.json" >&2
fi
if [[ -n "${SESSION:-}" || -n "${SESSION_NAME:-}" ]]; then
  printf 'ERROR: SESSION and SESSION_NAME must be unset for batch processing\n' >&2
  exit 2
fi

sessions=()
while IFS= read -r -d '' candidate; do
  if [[ -d "${candidate}/data/bag" ]]; then
    sessions+=("${candidate}")
  fi
done < <(find "${BATCH_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

if [[ "${#sessions[@]}" -eq 0 ]]; then
  printf 'ERROR: no session containing data/bag found under: %s\n' "${BATCH_ROOT}" >&2
  exit 1
fi

printf 'Batch: %s\nShared calibration: %s\nSessions: %d\n' \
  "${BATCH_ROOT}" "${CALIBRATION_DIR}" "${#sessions[@]}"
printf 'Shared calibration output: %s\n' "${BATCH_CALIB_SESSION}"

failures=()
next_calib_overwrite="${CALIB_OVERWRITE:-${OVERWRITE:-0}}"
for bag_session in "${sessions[@]}"; do
  printf '\n===== session: %s =====\n' "${bag_session}"
  if ! BATCH_ROOT="${BATCH_ROOT}" \
    CALIBRATION_DIR="${CALIBRATION_DIR}" \
    CALIB_SESSION="${BATCH_CALIB_SESSION}" \
    CALIB_CACHE_DIR="${BATCH_CALIB_CACHE_DIR}" \
    CALIB_OVERWRITE="${next_calib_overwrite}" \
    BAG_SESSION="${bag_session}" \
    "${PIPELINE}"; then
    failures+=("${bag_session}")
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit 1
    fi
  fi
  next_calib_overwrite=0
done

if [[ "${#failures[@]}" -gt 0 ]]; then
  printf '\nFailed sessions:\n' >&2
  printf '  %s\n' "${failures[@]}" >&2
  exit 1
fi
