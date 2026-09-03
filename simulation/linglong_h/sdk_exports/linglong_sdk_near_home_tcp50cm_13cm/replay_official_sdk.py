#!/usr/bin/env python3
"""Load or execute the exported trajectory through LingLong's official SDK runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SDK_ROOT = Path("/home/user/sdk")
DEFAULT_TASK = "ego_sdk_near_home_joint"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Connect to the robot and execute; omitted means load-only dry-run.",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--keep-final-pose",
        action="store_true",
        help="Do not call reset_to_init after playback.",
    )
    args = parser.parse_args()

    if not SDK_ROOT.is_dir():
        parser.error(f"official SDK directory does not exist: {SDK_ROOT}")
    sys.path.insert(0, str(SDK_ROOT))

    from sdk_test import test_trajectory_playback_robot_joint as official

    official.PlaybackSettings.TASK_NAME = args.task
    official.PlaybackSettings.DRY_RUN_ONLY_LOAD = not args.execute
    official.PlaybackSettings.RUN_ROBOT_ENABLE_UP = args.execute
    official.PlaybackSettings.RUN_ROBOT_AUTONOMOUS_MODE = args.execute
    official.PlaybackSettings.RUN_RESET_TO_INIT_BEFORE_PLAYBACK = True
    official.PlaybackSettings.RUN_RESET_TO_INIT_AFTER_PLAYBACK = (
        not args.keep_final_pose
    )

    if args.execute:
        print(
            "[ego-replay] EXECUTE enabled: the robot will enable, enter autonomous "
            "mode, reset to the SDK init pose, interpolate for 5 s to trajectory "
            "frame 0, and then replay.",
            flush=True,
        )
    else:
        print("[ego-replay] DRY RUN: loading CSV/YAML only; robot will not move.")
    return int(official.main())


if __name__ == "__main__":
    raise SystemExit(main())
