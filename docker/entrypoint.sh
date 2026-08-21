#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/opt/ego-loong-postprocess}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
HAND_MSG_SETUP="${HAND_MSG_SETUP:-${ROOT}/hand_msg_ws/install/setup.bash}"

set +u
source "${ROS_SETUP}"
source "${HAND_MSG_SETUP}"
set -u

runtime_home="${HOME:-/tmp/ego-loong-home}"
if [[ ! -d "${runtime_home}" || ! -w "${runtime_home}" ]]; then
  runtime_home="/tmp/ego-loong-home-$(id -u)"
  mkdir -p "${runtime_home}"
  export HOME="${runtime_home}"
fi

for cache_dir in \
  "${XDG_CACHE_HOME:-/tmp/ego-loong-cache/xdg}" \
  "${HF_HOME:-/tmp/ego-loong-cache/huggingface}" \
  "${MPLCONFIGDIR:-/tmp/ego-loong-cache/matplotlib}" \
  "${NUMBA_CACHE_DIR:-/tmp/ego-loong-cache/numba}"; do
  if ! mkdir -p "${cache_dir}" 2>/dev/null; then
    fallback_cache="/tmp/ego-loong-cache-$(id -u)/$(basename "${cache_dir}")"
    mkdir -p "${fallback_cache}"
    case "${cache_dir}" in
      "${XDG_CACHE_HOME:-}") export XDG_CACHE_HOME="${fallback_cache}" ;;
      "${HF_HOME:-}") export HF_HOME="${fallback_cache}" ;;
      "${MPLCONFIGDIR:-}") export MPLCONFIGDIR="${fallback_cache}" ;;
      "${NUMBA_CACHE_DIR:-}") export NUMBA_CACHE_DIR="${fallback_cache}" ;;
    esac
  fi
done

case "${1:-}" in
  pipeline)
    shift
    exec "${ROOT}/scripts/run_sampler_bag_to_glove_trajectory.sh" "$@"
    ;;
  batch)
    shift
    exec "${ROOT}/scripts/run_sampler_batch_to_glove_trajectory.sh" "$@"
    ;;
  smoke)
    shift
    exec "${ROOT}/docker/smoke-test.sh" "$@"
    ;;
  *)
    exec "$@"
    ;;
esac

