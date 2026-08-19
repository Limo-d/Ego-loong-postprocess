# -*- coding: utf-8 -*-
"""Visualize LocateAnything text-prompt bboxes on extracted RGB frames."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
for _proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    _proxy_val = os.environ.get(_proxy_key)
    if _proxy_val and _proxy_val.startswith("socks://"):
        os.environ[_proxy_key] = "socks5://" + _proxy_val[len("socks://"):]

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from preprocess.Timebase import repeat_counts
except ModuleNotFoundError:
    from Timebase import repeat_counts
from tqdm import tqdm


def _frame_dirs(session_path: str, max_frames: Optional[int]) -> List[str]:
    all_data_dir = os.path.join(session_path, "preprocess", "all_data")
    if not os.path.isdir(all_data_dir):
        raise FileNotFoundError(f"Missing frame directory: {all_data_dir}")
    dirs = [
        os.path.join(all_data_dir, name)
        for name in sorted(os.listdir(all_data_dir))
        if name.isdigit() and os.path.isdir(os.path.join(all_data_dir, name))
    ]
    if max_frames is not None:
        dirs = dirs[:max_frames]
    return dirs


def _read_fps(frame_dir: str, default: float) -> float:
    meta_path = os.path.join(frame_dir, "aria_cam_rgb.json")
    if not os.path.isfile(meta_path):
        return default
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return float(json.load(f).get("fps", default))
    except Exception:
        return default


def _dtype_from_name(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _draw_label(img: np.ndarray, text: str, x: int, y: int, color: tuple) -> None:
    y = max(18, y)
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _draw_detection(img: np.ndarray, det: Dict) -> Dict:
    bbox = np.asarray(det["bbox"], dtype=np.float32)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = int(np.clip(round(x1), 0, w - 1))
    y1 = int(np.clip(round(y1), 0, h - 1))
    x2 = int(np.clip(round(x2), 0, w - 1))
    y2 = int(np.clip(round(y2), 0, h - 1))

    label = str(det.get("label", "object"))
    score = float(det.get("score", 1.0))
    color = (80, 220, 80)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    _draw_label(img, f"{label} {score:.2f}", x1, y1 - 8, color)

    return {
        "bbox": [float(v) for v in bbox.tolist()],
        "label": label,
        "score": score,
        "raw": det.get("raw", ""),
    }


def _parse_locateanything_boxes(answer: str, image_w: int, image_h: int, default_label: str) -> List[Dict]:
    detections = []
    pattern = re.compile(
        r"(?:<ref>(?P<label>.*?)</ref>\s*)?"
        r"<box><(?P<x1>\d+)><(?P<y1>\d+)><(?P<x2>\d+)><(?P<y2>\d+)></box>"
    )

    for match in pattern.finditer(answer):
        label = (match.group("label") or default_label).strip() or default_label
        raw_vals = [int(match.group(k)) for k in ("x1", "y1", "x2", "y2")]
        x1, y1, x2, y2 = [float(np.clip(v, 0, 1000)) for v in raw_vals]
        x1, x2 = sorted([x1 / 1000.0 * image_w, x2 / 1000.0 * image_w])
        y1, y2 = sorted([y1 / 1000.0 * image_h, y2 / 1000.0 * image_h])
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            continue
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "label": label,
            "score": 1.0,
            "raw": match.group(0),
        })

    return detections


class LocateAnythingDetector:
    def __init__(
        self,
        model_path: str,
        device: str,
        dtype: str,
        attn_implementation: str,
        generation_mode: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

        self.device = device
        self.torch_dtype = _dtype_from_name(dtype)
        self.generation_mode = generation_mode
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config._attn_implementation = attn_implementation
        config.text_config._attn_implementation = attn_implementation
        config.vision_config._attn_implementation = (
            "flash_attention_2" if attn_implementation == "magi" else attn_implementation
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            config=config,
            dtype=self.torch_dtype,
            trust_remote_code=True,
        ).to(device).eval()

    def detect(self, image: Image.Image, prompt: str) -> Dict:
        question = f"Locate all the instances that match the following description: {prompt}."
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]

        text = self.processor.py_apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            return_tensors="pt",
        ).to(self.device)

        kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "tokenizer": self.tokenizer,
            "max_new_tokens": self.max_new_tokens,
            "use_cache": True,
            "generation_mode": self.generation_mode,
            "temperature": self.temperature,
            "do_sample": self.temperature > 0,
            "top_p": self.top_p,
            "repetition_penalty": 1.1,
            "verbose": False,
        }
        if "pixel_values" in inputs:
            kwargs["pixel_values"] = inputs["pixel_values"].to(self.torch_dtype)
        if "image_grid_hws" in inputs:
            kwargs["image_grid_hws"] = inputs["image_grid_hws"]

        with torch.no_grad():
            response = self.model.generate(**kwargs)

        answer = response[0] if isinstance(response, (tuple, list)) else response
        if not isinstance(answer, str):
            answer = str(answer)

        w, h = image.size
        detections = _parse_locateanything_boxes(answer, w, h, default_label=prompt)
        return {"answer": answer, "detections": detections}

    def detect_batch(self, images: List[Image.Image], prompt: str) -> List[Dict]:
        return [self.detect(image, prompt) for image in images]


class LocateAnythingBatchDetector:
    """NVIDIA's batched hybrid runtime with the legacy per-frame output contract."""

    def __init__(
        self,
        model_path: str,
        attn_implementation: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        runtime_root = Path(__file__).resolve().parents[1] / "third_party" / "nvidia_locateanything_batch"
        if not runtime_root.is_dir():
            raise FileNotFoundError(f"Missing LocateAnything batch runtime: {runtime_root}")
        sys.path.insert(0, str(runtime_root))
        os.environ["LA_FLASH_MODEL"] = str(Path(model_path).expanduser().resolve())
        os.environ["LA_FLASH_ATTN"] = attn_implementation
        os.environ["LA_FLASH_VISION_ATTN"] = "sdpa"
        os.environ["LA_FLASH_HYBRID_SCHEDULER"] = "pipeline"

        from batch_utils import generate_batch_hybrid, load

        self.generate_batch_hybrid = generate_batch_hybrid
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        load()

    def detect_batch(self, images: List[Image.Image], prompt: str) -> List[Dict]:
        if not images:
            return []
        answers = self.generate_batch_hybrid(
            [(image, prompt) for image in images],
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=1.1,
            max_new_tokens=self.max_new_tokens,
            scheduler="pipeline",
        )
        output = []
        for image, answer in zip(images, answers):
            answer = answer if isinstance(answer, str) else str(answer)
            w, h = image.size
            output.append({
                "answer": answer,
                "detections": _parse_locateanything_boxes(answer, w, h, default_label=prompt),
            })
        return output


def visualize_locateanything_bboxes(
    session_path: str,
    prompt: str,
    model_path: str,
    device: str,
    dtype: str,
    attn_implementation: str,
    generation_mode: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    out_video: str,
    out_json: str,
    out_frames_dir: str,
    fps: float,
    max_frames: Optional[int],
    save_frames: bool,
    batch_size: int,
) -> Dict[str, int]:
    frame_dirs = _frame_dirs(session_path, max_frames)
    if not frame_dirs:
        raise FileNotFoundError(f"No numeric frame directories under: {session_path}/preprocess/all_data")

    if fps <= 0:
        fps = _read_fps(frame_dirs[0], 30.0)

    timestamp_path = Path(session_path) / "preprocess" / "timestamps.jsonl"
    timestamp_by_frame = {}
    if timestamp_path.is_file():
        timestamp_rows = [json.loads(line) for line in timestamp_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        timestamp_by_frame = {str(row.get("frame")): row.get("rgb_stamp_ns") for row in timestamp_rows}
    timeline_rows = [
        {"rgb_stamp_ns": timestamp_by_frame.get(os.path.basename(frame_dir))}
        for frame_dir in frame_dirs
    ]
    repeats = repeat_counts(timeline_rows, output_fps=fps)

    os.makedirs(os.path.dirname(out_video), exist_ok=True)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    if save_frames:
        os.makedirs(out_frames_dir, exist_ok=True)

    batch_size = max(1, int(batch_size))
    use_batch_runtime = (
        batch_size > 1
        and device.startswith("cuda")
        and dtype == "bf16"
        and generation_mode == "hybrid"
        and attn_implementation in {"sdpa", "magi"}
    )
    if use_batch_runtime:
        detector = LocateAnythingBatchDetector(
            model_path=model_path,
            attn_implementation=attn_implementation,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        batch_size = 1
        detector = LocateAnythingDetector(
            model_path=model_path,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            generation_mode=generation_mode,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    writer = None
    result = {
        "session_path": session_path,
        "detector": "locateanything",
        "model_path": model_path,
        "prompt": prompt,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "batch_size": batch_size,
        "batch_runtime": use_batch_runtime,
        "generation_mode": generation_mode,
        "frames": [],
    }
    stats = {"frames": 0, "detections": 0, "written": 0}

    progress = tqdm(total=len(frame_dirs), desc=f"LocateAnything bbox batch={batch_size}")
    for chunk_start in range(0, len(frame_dirs), batch_size):
        records = []
        images = []
        for frame_i, frame_dir in enumerate(frame_dirs[chunk_start:chunk_start + batch_size], start=chunk_start):
            frame_name = os.path.basename(frame_dir)
            rgb_stamp_ns = timestamp_by_frame.get(frame_name)
            rgb_path = os.path.join(frame_dir, "rgb.png")
            img_bgr = cv2.imread(rgb_path)
            image = None if img_bgr is None else Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            records.append((frame_i, frame_name, rgb_stamp_ns, rgb_path, img_bgr, image))
            if image is not None:
                images.append(image)

        predictions = iter(detector.detect_batch(images, prompt=prompt))
        for frame_i, frame_name, rgb_stamp_ns, rgb_path, img_bgr, image in records:
            if img_bgr is None or image is None:
                result["frames"].append({"frame": frame_name, "rgb_stamp_ns": rgb_stamp_ns, "rgb_path": rgb_path, "answer": "", "detections": []})
                progress.update(1)
                continue

            pred = next(predictions)
            vis = img_bgr.copy()
            drawn = []
            for det in pred["detections"]:
                drawn.append(_draw_detection(vis, det))
                stats["detections"] += 1

            _draw_label(vis, f"{frame_name}  dets={len(drawn)}  prompt={prompt}", 10, 24, (255, 255, 255))

            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {out_video}")

            for _ in range(repeats[frame_i]):
                writer.write(vis)
                stats["written"] += 1
            stats["frames"] += 1

            if save_frames:
                cv2.imwrite(os.path.join(out_frames_dir, f"{frame_name}.png"), vis)

            result["frames"].append({
                "frame": frame_name,
                "rgb_stamp_ns": rgb_stamp_ns,
                "rgb_path": rgb_path,
                "answer": pred["answer"],
                "detections": drawn,
            })
            progress.update(1)
    progress.close()

    if writer is not None:
        writer.release()

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize LocateAnything text-prompt bboxes on rgb.png frames.")
    parser.add_argument("--session_path", required=True)
    parser.add_argument("--prompt", default="white glove")
    parser.add_argument("--model_path", default="nvidia/LocateAnything-3B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn_implementation", choices=["sdpa", "flash_attention_2", "magi"], default="sdpa")
    parser.add_argument("--batch_size", type=int, default=8, help="Batched hybrid inference size; use 1 for the legacy per-frame path.")
    parser.add_argument("--generation_mode", default="hybrid")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--fps", type=float, default=0.0, help="0 means read fps from aria_cam_rgb.json or use 30.")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--out_video", default=None)
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--out_frames_dir", default=None)
    parser.add_argument("--no_save_frames", action="store_true")
    args = parser.parse_args()

    session_path = os.path.abspath(args.session_path)
    vis_dir = os.path.join(session_path, "preprocess", "vis")
    safe_prompt = "".join(c if c.isalnum() else "_" for c in args.prompt).strip("_").lower()
    out_video = args.out_video or os.path.join(vis_dir, f"locateanything_{safe_prompt}_bboxes.mp4")
    out_json = args.out_json or os.path.join(session_path, "preprocess", f"locateanything_{safe_prompt}_bboxes.json")
    out_frames_dir = args.out_frames_dir or os.path.join(vis_dir, f"locateanything_{safe_prompt}_bbox_frames")

    stats = visualize_locateanything_bboxes(
        session_path=session_path,
        prompt=args.prompt,
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        generation_mode=args.generation_mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        out_video=out_video,
        out_json=out_json,
        out_frames_dir=out_frames_dir,
        fps=args.fps,
        max_frames=args.max_frames,
        save_frames=not args.no_save_frames,
        batch_size=args.batch_size,
    )
    print(f"[VisualizeLocateAnythingBboxes] Saved video: {out_video}")
    print(f"[VisualizeLocateAnythingBboxes] Saved json: {out_json}")
    if not args.no_save_frames:
        print(f"[VisualizeLocateAnythingBboxes] Saved frames: {out_frames_dir}")
    print(f"[VisualizeLocateAnythingBboxes] Stats: {stats}")


if __name__ == "__main__":
    main()
