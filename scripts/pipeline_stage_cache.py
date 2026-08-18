#!/usr/bin/env python3
"""Content-addressed stage cache manifests for the postprocess pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


SCHEMA_VERSION = 2
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_path(raw_path: str, previous: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        return {"path": str(path), "kind": "missing"}
    if path.is_file():
        # Downstream stages consume the stable meaning of an upstream stage
        # manifest, not volatile metadata such as completed_at or JSON spacing.
        payload = None
        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, OSError):
                payload = None
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("signature"), str)
            and isinstance(payload.get("stage"), str)
            and isinstance(payload.get("outputs"), list)
        ):
            dependency = {
                "stage": payload["stage"],
                "signature": payload["signature"],
                "status": payload.get("status"),
                "outputs": payload["outputs"],
            }
            encoded = json.dumps(dependency, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            return {
                "path": str(path),
                "kind": "stage_manifest",
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        stat = path.stat()
        size = int(stat.st_size)
        mtime_ns = int(stat.st_mtime_ns)
        ctime_ns = int(stat.st_ctime_ns)
        if (
            previous
            and previous.get("path") == str(path)
            and previous.get("kind") == "file"
            and previous.get("size") == size
            and previous.get("mtime_ns") == mtime_ns
            and previous.get("ctime_ns") == ctime_ns
            and isinstance(previous.get("sha256"), str)
        ):
            file_hash = previous["sha256"]
        else:
            file_hash = sha256_file(path)
        return {
            "path": str(path),
            "kind": "file",
            "size": size,
            "mtime_ns": mtime_ns,
            "ctime_ns": ctime_ns,
            "sha256": file_hash,
        }
    if path.is_dir():
        entries: List[Dict[str, Any]] = []
        digest = hashlib.sha256()
        previous_entries = {
            entry["relative_path"]: entry
            for entry in (previous or {}).get("_cache_entries", [])
            if isinstance(entry, dict) and isinstance(entry.get("relative_path"), str)
        }
        for item in sorted(p for p in path.rglob("*") if p.is_file()):
            rel = item.relative_to(path).as_posix()
            stat = item.stat()
            size = int(stat.st_size)
            mtime_ns = int(stat.st_mtime_ns)
            ctime_ns = int(stat.st_ctime_ns)
            old = previous_entries.get(rel)
            if (
                old
                and old.get("size") == size
                and old.get("mtime_ns") == mtime_ns
                and old.get("ctime_ns") == ctime_ns
                and isinstance(old.get("sha256"), str)
            ):
                file_hash = old["sha256"]
            else:
                file_hash = sha256_file(item)
            entry = {
                "relative_path": rel,
                "size": size,
                "mtime_ns": mtime_ns,
                "ctime_ns": ctime_ns,
                "sha256": file_hash,
            }
            entries.append(entry)
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(b"\0")
            digest.update(file_hash.encode("ascii"))
            digest.update(b"\n")
        return {
            "path": str(path),
            "kind": "directory",
            "file_count": len(entries),
            "sha256": digest.hexdigest(),
            "_cache_entries": entries,
        }
    return {"path": str(path), "kind": "other"}


def parse_params(values: Iterable[str]) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected --param KEY=VALUE, got {value!r}")
        key, raw = value.split("=", 1)
        if not key:
            raise ValueError(f"Empty parameter name in {value!r}")
        params[key] = raw
    return dict(sorted(params.items()))


def previous_by_path(manifest: Mapping[str, Any] | None, key: str) -> Dict[str, Mapping[str, Any]]:
    if not manifest:
        return {}
    values = manifest.get(key, [])
    if not isinstance(values, list):
        return {}
    return {
        value["path"]: value
        for value in values
        if isinstance(value, dict) and isinstance(value.get("path"), str)
    }


def build_identity(args: argparse.Namespace, previous: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    previous_inputs = previous_by_path(previous, "inputs")
    previous_code = previous_by_path(previous, "code")
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": args.stage,
        "inputs": [
            fingerprint_path(path, previous_inputs.get(str(Path(path).expanduser().resolve())))
            for path in args.input
        ],
        "parameters": parse_params(args.param),
        "code": [
            fingerprint_path(path, previous_code.get(str(Path(path).expanduser().resolve())))
            for path in args.code
        ],
    }


def identity_signature(identity: Dict[str, Any]) -> str:
    def stable(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: stable(item) for key, item in value.items() if not key.startswith("_cache_")}
        if isinstance(value, list):
            return [stable(item) for item in value]
        return value

    encoded = json.dumps(stable(identity), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def output_state(raw_path: str) -> Dict[str, Any]:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        return {"path": str(path), "kind": "missing", "complete": False}
    if path.is_file():
        size = int(path.stat().st_size)
        return {"path": str(path), "kind": "file", "size": size, "complete": size > 0}
    if path.is_dir():
        files = [item for item in path.rglob("*") if item.is_file() and item.stat().st_size > 0]
        return {"path": str(path), "kind": "directory", "nonempty_files": len(files), "complete": bool(files)}
    return {"path": str(path), "kind": "other", "complete": False}


def outputs_complete(paths: Iterable[str]) -> tuple[bool, List[Dict[str, Any]]]:
    states = [output_state(path) for path in paths]
    return bool(states) and all(bool(state["complete"]) for state in states), states


def outputs_match_manifest(states: List[Dict[str, Any]], recorded: Any) -> bool:
    if not isinstance(recorded, list):
        return False
    previous = {
        state["path"]: state
        for state in recorded
        if isinstance(state, dict) and isinstance(state.get("path"), str)
    }
    for state in states:
        old = previous.get(state["path"])
        if old is None or old.get("kind") != state.get("kind"):
            return False
        if state["kind"] == "file" and old.get("size") != state.get("size"):
            return False
        if state["kind"] == "directory" and old.get("nonempty_files") != state.get("nonempty_files"):
            return False
    return len(previous) == len(states)


def load_manifest(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def check(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    if manifest is None:
        print(f"[cache] MISS {args.stage}: manifest missing or invalid")
        return 1
    identity = build_identity(args, manifest)
    signature = identity_signature(identity)
    if manifest.get("status") != "complete":
        print(f"[cache] MISS {args.stage}: previous stage not complete")
        return 1
    if manifest.get("signature") != signature:
        print(f"[cache] MISS {args.stage}: input/parameter/code fingerprint changed")
        return 1
    complete, states = outputs_complete(args.output)
    if not complete:
        missing = [state["path"] for state in states if not state["complete"]]
        print(f"[cache] MISS {args.stage}: outputs incomplete: {missing}")
        return 1
    if not outputs_match_manifest(states, manifest.get("outputs")):
        print(f"[cache] MISS {args.stage}: output size or file count changed")
        return 1
    # Persist reusable per-file directory hash metadata without changing the
    # stage signature or completion time. Downstream semantic fingerprints
    # intentionally ignore these private cache fields.
    if manifest.get("inputs") != identity["inputs"] or manifest.get("code") != identity["code"]:
        refreshed = dict(manifest)
        refreshed["inputs"] = identity["inputs"]
        refreshed["code"] = identity["code"]
        temp_path = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
        temp_path.write_text(json.dumps(refreshed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(manifest_path)
    print(f"[cache] HIT  {args.stage}: {manifest_path}")
    return 0


def write(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    complete, states = outputs_complete(args.output)
    if not complete:
        missing = [state["path"] for state in states if not state["complete"]]
        raise RuntimeError(f"Cannot mark stage {args.stage!r} complete; outputs incomplete: {missing}")
    identity = build_identity(args, load_manifest(manifest_path))
    payload = {
        **identity,
        "signature": identity_signature(identity),
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "outputs": states,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(manifest_path)
    print(f"[cache] WRITE {args.stage}: {manifest_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "write"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--stage", required=True)
        sub.add_argument("--manifest", required=True)
        sub.add_argument("--input", action="append", default=[])
        sub.add_argument("--code", action="append", default=[])
        sub.add_argument("--param", action="append", default=[])
        sub.add_argument("--output", action="append", default=[], required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(check(args) if args.command == "check" else write(args))


if __name__ == "__main__":
    main()
