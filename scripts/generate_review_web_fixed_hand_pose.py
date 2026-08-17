#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a review web page with a fixed-wrist hand-pose panel.

Compared with generate_review_web.py, this script removes the trajectory/head
rendering from the lower-left panel. It keeps both wrists fixed at the bottom
and draws 21-keypoint hand skeletons with the existing finger colors.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import generate_review_web as base


def build_fixed_hand_web_data(
    rows: List[Dict[str, Any]],
    all_points: List[np.ndarray],
    hand_display_scale: float = 3.0,
) -> Dict[str, Any]:
    frames: List[Dict[str, Any]] = []
    rel_sets: List[np.ndarray] = []
    for row in rows:
        item: Dict[str, Any] = {}
        hand = row.get("hand")
        if hand is not None:
            hand_arr = np.asarray(hand, dtype=np.float64).copy()
            if hand_arr.shape[0] >= 21 and np.isfinite(hand_arr[:21]).all():
                root = hand_arr[0].copy()
                rel = (hand_arr[:21] - root[None, :]) * float(hand_display_scale)
                rel[0] = 0.0
                item["hand"] = np.round(rel, 5).tolist()
                item["hand_l"] = np.round(rel, 5).tolist()
                rel_sets.append(rel)
        frames.append(item)

    if rel_sets:
        right_template = np.round(rel_sets[0], 5).tolist()
        for item in frames:
            item["hand_r"] = right_template
        cloud = np.concatenate(rel_sets, axis=0)
        radius = float(np.nanpercentile(np.linalg.norm(cloud, axis=1), 98.0)) * 1.08
    else:
        zero_right = np.zeros((21, 3), dtype=np.float64)
        for item in frames:
            item["hand_r"] = np.round(zero_right, 5).tolist()
        radius = 1.0
    if not np.isfinite(radius) or radius < 1e-6:
        radius = 1.0

    return {
        "frames": frames,
        "center": [0.0, 0.0, 0.0],
        "radius": round(radius, 5),
        "bones": base.BONES,
    }


HAND_POSE_DRAW_JS = r"""function poseProject(pt,originX,originY,mirrorX=false){const root=projectTraj([0,0,0]),q=projectTraj(pt);let dx=q[0]-root[0],dy=q[1]-root[1];if(mirrorX)dx=-dx;return [originX+dx,originY+dy,q[2]]}
function poseLine(a,b,color,originX,originY,mirrorX=false,alpha=1){const p=poseProject(a,originX,originY,mirrorX),q=poseProject(b,originX,originY,mirrorX);trajCtx.globalAlpha=alpha;trajCtx.strokeStyle=color;trajCtx.lineWidth=3;trajCtx.lineCap='round';trajCtx.beginPath();trajCtx.moveTo(p[0],p[1]);trajCtx.lineTo(q[0],q[1]);trajCtx.stroke();trajCtx.globalAlpha=1;}
function poseDot(p,r,color,originX,originY,mirrorX=false,alpha=1){const q=poseProject(p,originX,originY,mirrorX);trajCtx.globalAlpha=alpha;trajCtx.fillStyle=color;trajCtx.beginPath();trajCtx.arc(q[0],q[1],r,0,Math.PI*2);trajCtx.fill();trajCtx.strokeStyle='#1f2937';trajCtx.lineWidth=1;trajCtx.stroke();trajCtx.globalAlpha=1;}
function isZeroHand(hand){return !hand||!hand.some((p,i)=>i>0&&p&&Math.abs(p[0])+Math.abs(p[1])+Math.abs(p[2])>1e-8)}
function drawPoseHand(hand,originX,originY,mirrorX=false,alpha=1){if(!hand)return;for(const [a,b] of TRAJ.bones){const ci=Math.max(0,Math.min(4,Math.floor((b-1)/4)));poseLine(hand[a],hand[b],FINGER_RGB[ci],originX,originY,mirrorX,alpha);}for(let i=0;i<hand.length;i++){const ci=Math.max(0,Math.min(4,Math.floor((i-1)/4)));poseDot(hand[i],i===0?5:4,i===0?'#f8fafc':FINGER_RGB[ci],originX,originY,mirrorX,alpha);}}
function drawTrajectory(){const w=trajCanvas.width,h=trajCanvas.height;trajCtx.setTransform(1,0,0,1,0,0);drawTrajGrid(w,h);const frames=TRAJ.frames||[],f=frames[Math.max(0,Math.min(frames.length-1,frame))]||{},left=f.hand_l||f.hand,right=f.hand_r;const wristY=h*.90;drawPoseHand(left,w*.25,wristY,true,1);drawPoseHand(right,w*.75,wristY,false,1);}"""


def patch_html(html_path: Path) -> None:
    html = html_path.read_text(encoding="utf-8")
    html = html.replace('<div class="badge">Trajectory</div>', '<div class="badge">Hand Pose</div>')
    html = html.replace("const DEFAULT_TRAJ_VIEW={yaw:-2.3517,pitch:1.2078,roll:2.3040,zoom:1.0};",
                        "const DEFAULT_TRAJ_VIEW={yaw:-2.3517,pitch:1.2078,roll:2.3040,zoom:1.0};")
    layout_css = """
.main{grid-template-columns:760px 396px;gap:28px;height:554px;align-items:stretch}
.stage.rgbStage{display:grid;grid-template-rows:116px 428px;gap:10px;width:760px;height:554px;min-height:0}
.rgbStage>.videoPanel{height:428px;aspect-ratio:16/9;min-height:0}
.rgbStage>.videoPanel canvas{width:100%;height:100%;display:block;background:#06101d}
.infoPanel{width:760px;height:116px;padding:12px;display:block;min-height:0}
.infoPanel .section-title{margin-bottom:8px;font-size:16px}
.infoPanel .grid3{grid-template-columns:repeat(3,1fr);gap:12px}
.infoPanel .metric{height:64px;padding:10px;background:#f8fbfe}
.visualStack{display:grid;grid-template-rows:272px 272px;gap:10px;width:396px;height:554px;min-height:0}
.visualStack .videoPanel{height:100%;min-height:0;background:#e4eef6}
.visualStack .videoPanel canvas{width:100%;height:100%;display:block;background:#e4eef6}
.visualStack .trajPanel canvas{cursor:grab}
.visualStack .trajPanel.dragging canvas{cursor:grabbing}
.compatHidden{position:absolute;left:-10000px;top:-10000px;width:1px;height:1px;overflow:hidden}
"""
    html = html.replace("</style>", layout_css + "\n</style>", 1)

    rgb_panel = '<div class="panel videoPanel"><div class="badge">RGB</div><canvas id="rgbCanvas"></canvas></div>'
    hand_panel = '<div class="panel videoPanel trajPanel"><div class="badge">Hand Pose</div><canvas id="trajCanvas"></canvas></div>'
    tactile_panel = '<div class="panel videoPanel tactilePanel"><div class="badge">Tactile</div><canvas id="tactileCanvas"></canvas></div>'
    side_match = re.search(
        r'(?P<side><aside class="panel side"><section>(?P<info>.*?)</section><section class="chartBox">.*?</section><section class="current">.*?</section></aside>)',
        html,
        flags=re.S,
    )
    if not side_match:
        raise RuntimeError(f"Could not find side panel in {html_path}")
    old_main = f'<main class="main"><section class="stage">{rgb_panel}<div class="lowerStage">{hand_panel}{tactile_panel}</div></section>\n{side_match.group("side")}</main>'
    new_main = (
        f'<main class="main"><section class="stage rgbStage"><aside class="panel side infoPanel"><section>{side_match.group("info")}</section></aside>{rgb_panel}</section>'
        f'<section class="visualStack">{hand_panel}{tactile_panel}</section>'
        '<div class="compatHidden"><canvas id="chartCanvas"></canvas><span id="xNow"></span><span id="yNow"></span><span id="zNow"></span></div></main>'
    )
    if old_main not in html:
        raise RuntimeError(f"Could not replace main layout in {html_path}")
    html = html.replace(old_main, new_main, 1)

    html = html.replace(
        " function drawFrames(){preload(frame);drawImageFit(rgbCtx,rgbCanvas,loadImage(rgbFrames[frame]));drawTrajectory();drawImageFit(tactileCtx,tactileCanvas,loadImage(tactileFrames[frame]),TACTILE_CANVAS_BG);}",
        " function drawFrames(){preload(frame);drawImageFit(rgbCtx,rgbCanvas,loadImage(rgbFrames[frame]));drawTrajectory();drawImageFit(tactileCtx,tactileCanvas,loadImage(tactileFrames[frame]),TACTILE_CANVAS_BG);}",
        1,
    )

    html, n = re.subn(
        r"function drawTrajectory\(\)\{.*?\}\ntrajCanvas\.addEventListener\('pointerdown'",
        HAND_POSE_DRAW_JS + "\ntrajCanvas.addEventListener('pointerdown'",
        html,
        count=1,
        flags=re.S,
    )
    if n != 1:
        raise RuntimeError(f"Could not replace drawTrajectory() in {html_path}")
    html_path.write_text(html, encoding="utf-8")


_orig_write_html = base.write_html


def write_html_and_patch(*args: Any, **kwargs: Any) -> Path:
    out = _orig_write_html(*args, **kwargs)
    patch_html(out)
    return out


base.build_trajectory_web_data = build_fixed_hand_web_data
base.write_html = write_html_and_patch


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
