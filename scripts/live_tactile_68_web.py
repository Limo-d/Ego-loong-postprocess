#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live 68-channel tactile glove web viewer.

Reads ASCII TOUCH lines from the USB serial glove and serves a local web page
that overlays a 68-node heatmap on the PALMSCOPE x-ray hand artwork.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import select
import signal
import statistics
import threading
import time
import termios
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse


TOUCH_RE = re.compile(
    r"TOUCH\s+seq=(\d+)\s+"
    r"thumb=([\d,]+)\s+"
    r"index=([\d,]+)\s+"
    r"middle=([\d,]+)\s+"
    r"ring=([\d,]+)\s+"
    r"little=([\d,]+)\s+"
    r"palm=([\d,]+)"
)

SENSOR_NAMES = (
    [f"A{i}" for i in range(4)]
    + [f"B{i}" for i in range(4)]
    + [f"C{i}" for i in range(4)]
    + [f"D{i}" for i in range(4)]
    + [f"E{i}" for i in range(4)]
    + [f"F{i}" for i in range(48)]
)

# Normalized coordinates against ourhost/assets/hand_live.png, cropped from hand.png.
# Fingers are 2x2: first phalanx row = 0,1; second phalanx row = 2,3.
FINGER_POINTS = {
    "A": [(0.876, 0.518), (0.909, 0.502), (0.815, 0.607), (0.850, 0.586)],
    "B": [(0.614, 0.183), (0.643, 0.183), (0.599, 0.344), (0.632, 0.344)],
    "C": [(0.453, 0.115), (0.482, 0.115), (0.453, 0.288), (0.482, 0.288)],
    "D": [(0.294, 0.176), (0.323, 0.176), (0.309, 0.344), (0.342, 0.344)],
    "E": [(0.155, 0.318), (0.184, 0.318), (0.198, 0.483), (0.231, 0.483)],
}
PALM_ROWS = [
    (0.663, 0.247, 0.637),
    (0.702, 0.238, 0.663),
    (0.740, 0.231, 0.692),
    (0.778, 0.243, 0.677),
    (0.817, 0.265, 0.625),
    (0.855, 0.289, 0.570),
]


def build_points() -> List[List[float]]:
    points: List[List[float]] = []
    for key in "ABCDE":
        points.extend([[x, y] for x, y in FINGER_POINTS[key]])
    for y, x0, x1 in PALM_ROWS:
        for col in range(8):
            x = x1 - (x1 - x0) * col / 7.0
            points.append([x, y])
    if len(points) != 68:
        raise RuntimeError(f"Expected 68 points, got {len(points)}")
    return points


POINTS = build_points()


def parse_touch_line(line: str) -> Optional[tuple[int, List[int]]]:
    match = TOUCH_RE.search(line)
    if not match:
        return None
    seq = int(match.group(1))
    values: List[int] = []
    for group in match.groups()[1:]:
        values.extend(int(x) for x in group.split(",") if x)
    if len(values) != 68:
        return None
    return seq, values


class SerialTouchReader(threading.Thread):
    def __init__(
        self,
        port: str,
        baudrate: int,
        baseline_frames: int,
        noise_gate: float,
        ema_rise: float,
        ema_fall: float,
    ) -> None:
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.baseline_frames = baseline_frames
        self.noise_gate = noise_gate
        self.ema_rise = ema_rise
        self.ema_fall = ema_fall
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.frame_id = 0
        self.latest: Dict[str, Any] = {
            "connected": False,
            "status": "starting",
            "seq": None,
            "frame_id": 0,
            "raw": [0] * 68,
            "delta": [0.0] * 68,
            "display": [0.0] * 68,
            "baseline_ready": False,
            "baseline_count": 0,
            "hz": 0.0,
            "peak": 0.0,
            "peak_name": "--",
            "total": 0.0,
            "display_range": 160.0,
            "range_mode": "160",
        }
        self._baseline_buf: List[List[int]] = []
        self._baseline: Optional[List[float]] = None
        self._filtered = [0.0] * 68
        self._display_range = 160.0
        self._manual_display_range: Optional[float] = 160.0
        self._frame_times: List[float] = []

    def reset_baseline(self) -> None:
        with self.lock:
            self._baseline_buf = []
            self._baseline = None
            self._filtered = [0.0] * 68
            self._display_range = self._manual_display_range or 8.0
            self.latest["baseline_ready"] = False
            self.latest["baseline_count"] = 0
            self.latest["status"] = "baseline reset"
            self.latest["display_range"] = round(self._display_range, 3)
            self.latest["range_mode"] = "auto" if self._manual_display_range is None else str(int(self._manual_display_range))

    def set_display_range(self, value: Optional[float]) -> None:
        with self.lock:
            self._manual_display_range = None if value is None else max(1.0, float(value))
            if self._manual_display_range is not None:
                self._display_range = self._manual_display_range
            self.latest["range_mode"] = "auto" if self._manual_display_range is None else str(int(self._manual_display_range))
            self.latest["display_range"] = round(self._display_range, 3)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.latest)

    def _set_status(self, status: str, connected: bool = False) -> None:
        with self.lock:
            self.latest["status"] = status
            self.latest["connected"] = connected

    def _speed_const(self) -> int:
        name = f"B{self.baudrate}"
        speed = getattr(termios, name, None)
        if speed is None:
            raise RuntimeError(f"Unsupported baudrate by termios: {self.baudrate}")
        return int(speed)

    def _open_serial(self) -> int:
        fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attrs = termios.tcgetattr(fd)
        speed = self._speed_const()
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attrs[3] = 0
        attrs[4] = speed
        attrs[5] = speed
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 1
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        termios.tcflush(fd, termios.TCIFLUSH)
        return fd

    def _update_hz(self, now: float) -> float:
        self._frame_times.append(now)
        self._frame_times = [t for t in self._frame_times if now - t <= 2.0]
        if len(self._frame_times) < 2:
            return 0.0
        span = max(1e-6, self._frame_times[-1] - self._frame_times[0])
        return (len(self._frame_times) - 1) / span

    def _apply_frame(self, seq: int, raw: List[int]) -> None:
        now = time.time()
        if self._baseline is None:
            self._baseline_buf.append(raw)
            if len(self._baseline_buf) >= self.baseline_frames:
                cols = zip(*self._baseline_buf)
                self._baseline = [float(statistics.median(col)) for col in cols]
            display = [0.0] * 68
            delta = [0.0] * 68
        else:
            delta = []
            for i, value in enumerate(raw):
                d = abs(float(value) - self._baseline[i])
                if d <= self.noise_gate:
                    d = 0.0
                    self._baseline[i] = self._baseline[i] * 0.999 + float(value) * 0.001
                alpha = self.ema_rise if d >= self._filtered[i] else self.ema_fall
                self._filtered[i] = self._filtered[i] * (1.0 - alpha) + d * alpha
                if self._filtered[i] < max(0.25, self.noise_gate * 0.35):
                    self._filtered[i] = 0.0
                delta.append(self._filtered[i])
            delta_max = max(delta) if delta else 0.0
            if self._manual_display_range is not None:
                self._display_range = self._manual_display_range
            else:
                target = max(4.0, delta_max * 1.20)
                if target > self._display_range:
                    self._display_range = target
                else:
                    self._display_range = self._display_range * 0.96 + target * 0.04
                self._display_range = max(8.0, min(4096.0, self._display_range))
            display = [max(0.0, min(100.0, d / self._display_range * 100.0)) for d in delta]

        peak = max(delta) if delta else 0.0
        peak_idx = delta.index(peak) if delta and peak > 0 else -1
        hz = self._update_hz(now)
        with self.lock:
            self.frame_id += 1
            self.latest = {
                "connected": True,
                "status": "streaming",
                "seq": seq,
                "frame_id": self.frame_id,
                "raw": raw,
                "delta": [round(v, 3) for v in delta],
                "display": [round(v, 3) for v in display],
                "baseline_ready": self._baseline is not None,
                "baseline_count": min(len(self._baseline_buf), self.baseline_frames),
                "hz": round(hz, 1),
                "peak": round(peak, 3),
                "peak_name": SENSOR_NAMES[peak_idx] if peak_idx >= 0 else "--",
                "total": round(sum(delta), 3),
                "display_range": round(self._display_range, 3),
                "range_mode": "auto" if self._manual_display_range is None else str(int(self._manual_display_range)),
            }

    def run(self) -> None:
        fd: Optional[int] = None
        try:
            fd = self._open_serial()
            self._set_status("serial open", connected=True)
            buf = ""
            while not self.stop_event.is_set():
                ready, _, _ = select.select([fd], [], [], 0.2)
                if not ready:
                    continue
                chunk = os.read(fd, 8192)
                if not chunk:
                    continue
                buf += chunk.decode("utf-8", errors="ignore")
                lines = re.split(r"\r?\n", buf)
                buf = lines.pop() if lines else ""
                for line in lines:
                    parsed = parse_touch_line(line)
                    if parsed:
                        seq, raw = parsed
                        self._apply_frame(seq, raw)
        except Exception as exc:
            self._set_status(f"serial error: {exc}", connected=False)
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass


def make_html(baseline_frames: int = 24) -> str:
    points_json = json.dumps(POINTS, separators=(",", ":"))
    names_json = json.dumps(SENSOR_NAMES, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>68 点触觉热力图</title>
<style>
:root{{--bg:#071018;--panel:#0b1720;--line:#1d4050;--cyan:#42eadc;--muted:#65828d;--orange:#ffb547;--red:#ff5d57}}
*{{box-sizing:border-box}}body{{margin:0;height:100vh;overflow:hidden;background:radial-gradient(circle at 50% 45%,#11313a 0,#071018 58%,#04080c 100%);color:#d7f8f4;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
body:before{{content:"";position:fixed;inset:0;pointer-events:none;background:linear-gradient(#0e253033 1px,transparent 1px),linear-gradient(90deg,#0e253033 1px,transparent 1px);background-size:36px 36px}}
.shell{{height:100vh;display:grid;grid-template-rows:56px 1fr 42px;padding:10px 18px 14px;gap:10px}}
.top{{border:1px solid var(--line);background:rgba(6,16,24,.72);display:flex;align-items:center;justify-content:space-between;padding:0 14px;box-shadow:0 0 26px #1ad7ff18}}
.brand{{display:flex;align-items:center;gap:12px;font:700 12px "Space Mono",monospace;letter-spacing:.8px}}.dot{{width:6px;height:6px;border-radius:50%;background:var(--cyan);box-shadow:0 0 10px var(--cyan)}}.brand small{{color:var(--muted);font-weight:500;margin-left:12px}}
.actions{{display:flex;align-items:center;gap:10px}}button,select{{height:34px;border:1px solid #285c68;background:#10252d;color:#bff9f3;padding:0 12px;cursor:pointer;font-weight:700}}button.active{{border-color:#45e6d2;color:#45e6d2;background:#123139}}label{{display:flex;align-items:center;gap:6px;color:#78939d;font-size:12px}}input{{accent-color:#45e6d2}}
.main{{min-height:0;display:grid;grid-template-columns:minmax(520px,1fr) 330px;gap:14px}}
.visual{{position:relative;min-height:0;border:1px solid var(--line);background:rgba(5,14,20,.64);overflow:hidden}}
.visual:before,.visual:after{{content:"";position:absolute;left:50%;top:54%;transform:translate(-50%,-50%);width:54%;aspect-ratio:1;border:1px solid #2a4c5744;border-radius:50%;pointer-events:none}}.visual:after{{width:72%;border-style:dashed}}
.hand{{position:absolute;left:50%;top:50%;height:94%;aspect-ratio:920/1040;transform:translate(-50%,-50%);filter:drop-shadow(0 0 20px #159abc55)}}
.hand img,.hand canvas{{position:absolute;inset:0;width:100%;height:100%;display:block}}.hand img{{opacity:.88}}.hand canvas{{z-index:2}}
.scan{{position:absolute;left:18%;right:18%;height:1px;top:20%;background:linear-gradient(90deg,transparent,var(--cyan),transparent);box-shadow:0 0 12px var(--cyan);animation:scan 4.8s ease-in-out infinite}}@keyframes scan{{50%{{top:78%}}}}
.reticle{{position:absolute;width:24px;height:24px;border-left:1px solid var(--cyan);border-top:1px solid var(--cyan);opacity:.55}}.r1{{left:12%;top:22%}}.r2{{right:12%;bottom:14%;transform:rotate(180deg)}}
.side{{display:flex;flex-direction:column;gap:12px;min-height:0}}.card{{border:1px solid var(--line);background:rgba(7,18,26,.82);padding:14px;box-shadow:0 0 20px #0005}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}.metric small,.section small{{display:block;color:#62818d;font:8px "Space Mono",monospace;letter-spacing:1px}}.metric strong{{display:block;margin-top:8px;font:700 24px "Space Mono",monospace;color:#eaffff}}.metric em{{font-style:normal;color:#6d8791;font-size:10px;margin-left:4px}}
.section{{min-height:0}}.kv{{display:grid;grid-template-columns:92px 1fr;gap:7px;margin-top:10px;font:12px "Space Mono",monospace;color:#9cc3c9}}.kv b{{font-weight:500;color:#e2fffb}}.bar{{height:5px;background:#10242c;margin-top:10px;overflow:hidden}}.bar i{{display:block;height:100%;width:0;background:linear-gradient(90deg,#2067ff,#2ee7d0,#f6dd3d,#ff624f);box-shadow:0 0 12px #45e6d2}}
#chart{{width:100%;height:160px;display:block;margin-top:8px}}.legend{{height:42px;border:1px solid var(--line);background:rgba(6,16,24,.7);display:flex;align-items:center;gap:12px;padding:0 14px;color:#66818b;font:9px "Space Mono",monospace}}.grad{{height:4px;flex:1;background:linear-gradient(90deg,#123f8f,#2ee7d0,#f6dd3d,#ff624f)}}
.status{{color:#75f7a7}}.warn{{color:#ffcf66}}.err{{color:#ff7777}}
</style>
</head>
<body>
<div class="shell">
  <header class="top">
    <div class="brand"><span class="dot"></span><span>LIVE</span><small>68 SENSOR NODES · A0-F47 · USB TOUCH</small></div>
    <div class="actions">
      <button id="heatBtn" class="active">热力</button><button id="nodeBtn">点阵</button>
      <label>量程<select id="rangeSelect"><option value="auto">自动 · 高灵敏</option><option value="4096">4096 · 满量程</option><option value="2048">2048</option><option value="1024">1024</option><option value="512">512</option><option value="256">256</option><option value="160" selected>160</option><option value="128">128</option><option value="64">64</option><option value="32">32</option><option value="16">16</option><option value="8">8 · 最高灵敏</option></select></label>
      <label><input id="labelToggle" type="checkbox">标签</label>
      <button id="resetBtn">重置零点</button>
    </div>
  </header>
  <main class="main">
    <section class="visual">
      <div class="scan"></div><div class="reticle r1"></div><div class="reticle r2"></div>
      <div class="hand"><img src="/assets/hand.png" alt=""><canvas id="sensorCanvas"></canvas></div>
    </section>
    <aside class="side">
      <section class="grid">
        <article class="metric card"><small>PEAK NODE</small><strong id="peak">--</strong></article>
        <article class="metric card"><small>FRAME RATE</small><strong><span id="hz">0.0</span><em>Hz</em></strong></article>
        <article class="metric card"><small>TOTAL ΔAD</small><strong id="total">0</strong></article>
        <article class="metric card"><small>SEQ</small><strong id="seq">--</strong></article>
      </section>
      <section class="section card">
        <small>STREAM</small>
        <div class="kv">
          <span>状态</span><b id="status">connecting</b>
          <span>零点</span><b id="baseline">--</b>
          <span>量程</span><b id="range">--</b>
          <span>峰值</span><b id="peakValue">--</b>
        </div>
        <div class="bar"><i id="peakBar"></i></div>
      </section>
      <section class="section card">
        <small>PRESSURE CURVE</small>
        <canvas id="chart"></canvas>
      </section>
    </aside>
  </main>
  <footer class="legend"><span>ΔAD 0</span><div class="grad"></div><span id="legendMax">AUTO</span></footer>
</div>
<script>
const POINTS={points_json};
const NAMES={names_json};
const canvas=document.getElementById('sensorCanvas'),ctx=canvas.getContext('2d');
const chart=document.getElementById('chart'),cctx=chart.getContext('2d');
let mode='heat', showLabels=false, latest=null, history=Array(160).fill(0);
const $=id=>document.getElementById(id);
function fitCanvas(c){{const r=c.getBoundingClientRect(),d=Math.min(window.devicePixelRatio||1,2);const w=Math.max(1,Math.round(r.width*d)),h=Math.max(1,Math.round(r.height*d));if(c.width!==w||c.height!==h){{c.width=w;c.height=h;c.getContext('2d').setTransform(d,0,0,d,0,0)}}return r}}
const STOPS=[[0,35,108,170],[.45,44,210,220],[.7,65,230,205],[.86,255,178,64],[1,255,76,76]];
function ramp(v){{let hi=1;while(hi<STOPS.length-1&&v>STOPS[hi][0])hi++;const a=STOPS[hi-1],b=STOPS[hi],t=(v-a[0])/(b[0]-a[0]);return [a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t,a[3]+(b[3]-a[3])*t]}}
function color(v,a=1){{const c=ramp(Math.max(0,Math.min(1,v)));return `rgba(${{c[0]}},${{c[1]}},${{c[2]}},${{a}})`}}
function visual(v){{return v<=0?0:Math.pow(Math.min(1,v/100),.62)}}
function drawSensors(){{const r=fitCanvas(canvas),w=r.width,h=r.height;ctx.clearRect(0,0,w,h);if(!latest)return;const vals=latest.display||[];
  POINTS.forEach((p,i)=>{{const x=p[0]*w,y=p[1]*h,v=visual(vals[i]||0);if(mode==='heat'){{const radius=9+v*24,g=ctx.createRadialGradient(x,y,0,x,y,radius);g.addColorStop(0,color(v,.92));g.addColorStop(.34,color(v,.42));g.addColorStop(1,color(v,0));ctx.fillStyle=g;ctx.beginPath();ctx.arc(x,y,radius,0,Math.PI*2);ctx.fill();ctx.fillStyle=color(Math.min(1,v+.16),.95);ctx.beginPath();ctx.arc(x,y,2.2+v*1.8,0,Math.PI*2);ctx.fill();}}else{{ctx.shadowBlur=4+v*14;ctx.shadowColor=color(v,.9);ctx.fillStyle=color(v,.16+v*.7);ctx.beginPath();ctx.arc(x,y,4.4,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;ctx.strokeStyle=color(Math.min(1,v+.2),.6+v*.3);ctx.stroke();ctx.fillStyle=color(Math.min(1,v+.25),.9);ctx.beginPath();ctx.arc(x,y,1.3+v*1.6,0,Math.PI*2);ctx.fill();}}
    if(showLabels){{ctx.font='9px Space Mono, monospace';ctx.fillStyle='rgba(208,255,250,.82)';ctx.fillText(NAMES[i],x+5,y-5);}}
  }});
}}
function drawChart(){{const r=fitCanvas(chart),w=r.width,h=r.height;cctx.clearRect(0,0,w,h);cctx.strokeStyle='#17313b';cctx.lineWidth=1;for(let i=1;i<4;i++){{cctx.beginPath();cctx.moveTo(0,i*h/4);cctx.lineTo(w,i*h/4);cctx.stroke();}}const max=Math.max(8,...history),pts=history.map((v,i)=>[i*w/(history.length-1),h-(v/max)*(h-8)-4]);const grad=cctx.createLinearGradient(0,0,0,h);grad.addColorStop(0,'rgba(69,230,210,.28)');grad.addColorStop(1,'rgba(69,230,210,0)');cctx.beginPath();pts.forEach((p,i)=>i?cctx.lineTo(p[0],p[1]):cctx.moveTo(p[0],p[1]));cctx.lineTo(w,h);cctx.lineTo(0,h);cctx.closePath();cctx.fillStyle=grad;cctx.fill();cctx.beginPath();pts.forEach((p,i)=>i?cctx.lineTo(p[0],p[1]):cctx.moveTo(p[0],p[1]));cctx.strokeStyle='#45e6d2';cctx.lineWidth=1.6;cctx.shadowBlur=7;cctx.shadowColor='#45e6d2';cctx.stroke();cctx.shadowBlur=0;}}
function updateUi(d){{latest=d;$('seq').textContent=d.seq??'--';$('hz').textContent=(d.hz||0).toFixed(1);$('total').textContent=Math.round(d.total||0);$('peak').textContent=d.peak_name||'--';$('peakValue').textContent=(d.peak||0).toFixed(2)+' ΔAD';const rangeMode=d.range_mode||'auto',rangeText=(rangeMode==='auto'?'AUTO ':'')+(d.display_range||0).toFixed(1)+' ΔAD';$('range').textContent=rangeText;$('legendMax').textContent=(rangeMode==='auto'?'AUTO ':'')+'ΔAD '+Math.round(d.display_range||0);$('peakBar').style.width=Math.max(0,Math.min(100,(d.peak||0)/(d.display_range||8)*100))+'%';$('status').textContent=d.status||'--';$('status').className=d.connected?'status':'err';$('baseline').textContent=d.baseline_ready?'ready':`${{d.baseline_count||0}}/{baseline_frames}`;if($('rangeSelect')&&$('rangeSelect').value!==rangeMode)$('rangeSelect').value=rangeMode;history.push(d.peak||0);history.shift();drawSensors();drawChart();}}
const es=new EventSource('/events');es.onmessage=e=>updateUi(JSON.parse(e.data));es.onerror=()=>{{$('status').textContent='event stream disconnected';$('status').className='err';}};
$('heatBtn').onclick=()=>{{mode='heat';$('heatBtn').classList.add('active');$('nodeBtn').classList.remove('active');drawSensors();}};
$('nodeBtn').onclick=()=>{{mode='nodes';$('nodeBtn').classList.add('active');$('heatBtn').classList.remove('active');drawSensors();}};
$('labelToggle').onchange=e=>{{showLabels=e.target.checked;drawSensors();}};
$('resetBtn').onclick=async()=>{{await fetch('/reset_baseline',{{method:'POST'}});}};
$('rangeSelect').onchange=async e=>{{await fetch('/set_range?value='+encodeURIComponent(e.target.value),{{method:'POST'}});}};
window.addEventListener('resize',()=>{{drawSensors();drawChart();}});
drawSensors();drawChart();
</script>
</body>
</html>"""


class TactileRequestHandler(BaseHTTPRequestHandler):
    reader: SerialTouchReader
    hand_asset: Path
    html: str

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_bytes(self.html.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/assets/hand.png":
            self._send_bytes(self.hand_asset.read_bytes(), "image/png")
        elif path == "/status":
            data = json.dumps(self.reader.snapshot(), ensure_ascii=False).encode("utf-8")
            self._send_bytes(data, "application/json; charset=utf-8")
        elif path == "/events":
            self._serve_events()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/reset_baseline":
            self.reader.reset_baseline()
            self._send_bytes(b'{"ok":true}', "application/json")
        elif path == "/set_range":
            value = parse_qs(parsed.query).get("value", ["auto"])[0]
            try:
                self.reader.set_display_range(None if value == "auto" else float(value))
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "invalid range")
                return
            self._send_bytes(b'{"ok":true}', "application/json")
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _serve_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        last_frame_id = -1
        try:
            while True:
                snap = self.reader.snapshot()
                frame_id = int(snap.get("frame_id") or 0)
                if frame_id != last_frame_id:
                    payload = json.dumps(snap, ensure_ascii=False, separators=(",", ":"))
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    last_frame_id = frame_id
                time.sleep(1.0 / 30.0)
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a live 68-channel tactile heatmap web page.")
    parser.add_argument("--port", default="/dev/ttyACM0", help="USB serial device")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--web_port", type=int, default=8790)
    parser.add_argument("--hand_asset", default="/home/lenovo/Downloads/ourhost/assets/hand_live.png")
    parser.add_argument("--baseline_frames", type=int, default=24)
    parser.add_argument("--noise_gate", type=float, default=1.5)
    parser.add_argument("--ema_rise", type=float, default=0.45)
    parser.add_argument("--ema_fall", type=float, default=0.22)
    args = parser.parse_args()

    hand_asset = Path(args.hand_asset).expanduser().resolve()
    if not hand_asset.exists():
        raise SystemExit(f"hand asset not found: {hand_asset}")

    reader = SerialTouchReader(
        port=args.port,
        baudrate=args.baudrate,
        baseline_frames=max(1, args.baseline_frames),
        noise_gate=args.noise_gate,
        ema_rise=args.ema_rise,
        ema_fall=args.ema_fall,
    )
    reader.start()

    TactileRequestHandler.reader = reader
    TactileRequestHandler.hand_asset = hand_asset
    TactileRequestHandler.html = make_html(max(1, args.baseline_frames))
    server = ThreadingHTTPServer((args.host, args.web_port), TactileRequestHandler)

    def stop(_signum: int, _frame: Any) -> None:
        reader.stop_event.set()
        server.shutdown()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(json.dumps({"url": f"http://{args.host}:{args.web_port}", "serial": args.port, "baudrate": args.baudrate}, ensure_ascii=False))
    try:
        server.serve_forever()
    finally:
        reader.stop_event.set()


if __name__ == "__main__":
    main()
