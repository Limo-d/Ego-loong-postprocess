#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${ROOT}/.venv-mujoco"
MENAGERIE="${ROOT}/third_party/mujoco_menagerie"

if [[ ! -x "${VENV}/bin/python" ]]; then
  /usr/bin/python3 -m venv "${VENV}"
fi
"${VENV}/bin/pip" install mujoco imageio imageio-ffmpeg mink==1.3.0

if [[ ! -d "${MENAGERIE}/.git" ]]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git "${MENAGERIE}"
fi
git -C "${MENAGERIE}" sparse-checkout set universal_robots_ur5e

echo "MuJoCo environment: ${VENV}"
echo "UR5e model: ${MENAGERIE}/universal_robots_ur5e/ur5e.xml"
