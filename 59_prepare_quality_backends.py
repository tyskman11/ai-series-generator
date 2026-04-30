#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from pipeline_common import (
    CONFIG_PATH,
    coalesce_text,
    headline,
    info,
    load_config,
    ok,
    read_json,
    resolve_project_path,
    warn,
    write_json,
)

DOWNLOAD_METADATA_FILE = ".quality_backend_asset.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and update project-local quality backend tools and models with revision tracking."
    )
    parser.add_argument("--print-plan", action="store_true", help="Only print the resolved asset plan without downloading.")
    parser.add_argument("--skip-downloads", action="store_true", help="Only validate the current local asset state.")
    parser.add_argument("--force", action="store_true", help="Force a refresh even when the local revision already matches.")
    return parser.parse_args()


def ensure_runtime_package(module_name: str, package_name: str) -> None:
    try:
        __import__(module_name)
        return
    except Exception:
        pass
    subprocess.run(
        ["python", "-m", "pip", "install", package_name],
        check=True,
    )


def default_quality_backend_asset_targets(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    clone_cfg = cfg.get("cloning", {}) if isinstance(cfg.get("cloning"), dict) else {}
    targets: list[dict[str, Any]] = [
        {
            "name": "comfyui",
            "kind": "git",
            "repo_url": "https://github.com/comfyanonymous/ComfyUI.git",
            "ref": "master",
            "target_dir": "tools/quality_backends/comfyui",
            "required_files": ["main.py", "nodes.py", "server.py", "requirements.txt"],
        }
    ]
    model_specs = [
        ("image_base_model", "image", ["model_index.json", "*.safetensors"]),
        ("video_base_model", "video", ["*.safetensors"]),
        ("voice_base_model", "voice", []),
    ]
    for config_key, group_name, required_patterns in model_specs:
        model_id = coalesce_text(foundation_cfg.get(config_key, ""))
        if not model_id:
            continue
        targets.append(
            {
                "name": config_key,
                "kind": "huggingface",
                "repo_id": model_id,
                "target_dir": f"tools/quality_models/{group_name}/{model_id.replace('/', '__')}",
                "required_patterns": required_patterns,
            }
        )
    xtts_model_name = coalesce_text(clone_cfg.get("xtts_model_name", ""))
    if xtts_model_name:
        targets.append(
            {
                "name": "xtts_model_name_record",
                "kind": "metadata",
                "repo_id": xtts_model_name,
                "target_dir": "tools/quality_models/voice/xtts_runtime",
                "required_patterns": [],
            }
        )
    lipsync_model = coalesce_text(foundation_cfg.get("lipsync_model", ""))
    if lipsync_model:
        targets.append(
            {
                "name": "lipsync_model_name_record",
                "kind": "metadata",
                "repo_id": lipsync_model,
                "target_dir": "tools/quality_models/lipsync/runtime",
                "required_patterns": [],
            }
        )
    return targets


def quality_backend_asset_targets(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    assets_cfg = cfg.get("quality_backend_assets", {}) if isinstance(cfg.get("quality_backend_assets"), dict) else {}
    configured = assets_cfg.get("targets", [])
    if isinstance(configured, list) and configured:
        return [dict(item) for item in configured if isinstance(item, dict)]
    return default_quality_backend_asset_targets(cfg)


def asset_target_dir(target: dict[str, Any]) -> Path:
    raw = coalesce_text(target.get("target_dir", ""))
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else resolve_project_path(raw)


def asset_metadata_path(target: dict[str, Any]) -> Path:
    return asset_target_dir(target) / DOWNLOAD_METADATA_FILE


def read_asset_metadata(target: dict[str, Any]) -> dict[str, Any]:
    path = asset_metadata_path(target)
    return read_json(path, {}) if path.exists() else {}


def write_asset_metadata(target: dict[str, Any], payload: dict[str, Any]) -> None:
    metadata_path = asset_metadata_path(target)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(metadata_path, payload)


def target_required_files_ready(target: dict[str, Any]) -> bool:
    target_dir = asset_target_dir(target)
    required_files = target.get("required_files", [])
    if isinstance(required_files, list) and required_files:
        return all((target_dir / str(value)).exists() for value in required_files if str(value).strip())
    required_patterns = target.get("required_patterns", [])
    if isinstance(required_patterns, list) and required_patterns:
        for pattern in required_patterns:
            text = str(pattern or "").strip()
            if not text:
                continue
            if any(path.is_file() for path in target_dir.glob(text)):
                return True
        return False
    return any(path.name != DOWNLOAD_METADATA_FILE for path in target_dir.iterdir()) if target_dir.exists() else False


def asset_target_ready(target: dict[str, Any]) -> bool:
    target_dir = asset_target_dir(target)
    if not target_dir.exists():
        return False
    if not any(path.name != DOWNLOAD_METADATA_FILE for path in target_dir.iterdir()):
        return False
    return target_required_files_ready(target)


def incomplete_download_files(target: dict[str, Any]) -> list[Path]:
    target_dir = asset_target_dir(target)
    cache_dir = target_dir / ".cache" / "huggingface" / "download"
    if not cache_dir.exists():
        return []
    return sorted(path for path in cache_dir.rglob("*.incomplete") if path.is_file())


def cleanup_incomplete_download_files(target: dict[str, Any]) -> int:
    removed = 0
    for path in incomplete_download_files(target):
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def infer_local_hf_revision(target: dict[str, Any]) -> str:
    target_dir = asset_target_dir(target)
    cache_dir = target_dir / ".cache" / "huggingface" / "download"
    if not cache_dir.exists():
        return ""
    for metadata_file in sorted(cache_dir.rglob("*.metadata")):
        try:
            payload = metadata_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in payload.splitlines():
            if line.lower().startswith("commit_hash"):
                _, _, value = line.partition(":")
                return value.strip()
    return ""


def infer_local_git_revision(target: dict[str, Any]) -> str:
    target_dir = asset_target_dir(target)
    git_dir = target_dir / ".git"
    if not git_dir.exists():
        return ""
    completed = subprocess.run(
        ["git", "-C", str(target_dir), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def local_asset_state(target: dict[str, Any]) -> dict[str, Any]:
    metadata = read_asset_metadata(target)
    local_revision = coalesce_text(metadata.get("revision", ""))
    inferred_revision = ""
    if target.get("kind") == "huggingface" and not local_revision and asset_target_ready(target):
        inferred_revision = infer_local_hf_revision(target)
    elif target.get("kind") == "git" and not local_revision and asset_target_ready(target):
        inferred_revision = infer_local_git_revision(target)
    if inferred_revision and not local_revision:
        local_revision = inferred_revision
    return {
        "local_revision": local_revision,
        "inferred_revision": inferred_revision,
        "has_incomplete_files": bool(incomplete_download_files(target)),
        "ready": asset_target_ready(target),
    }


def fetch_remote_git_revision(target: dict[str, Any]) -> str:
    ref = coalesce_text(target.get("ref", "master")) or "master"
    repo_url = coalesce_text(target.get("repo_url", ""))
    completed = subprocess.run(
        ["git", "ls-remote", repo_url, ref],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout.strip() or f"Could not inspect {repo_url}")
    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    return line.split()[0] if line else ""


def fetch_remote_hf_revision(api: Any, target: dict[str, Any], token: str) -> str:
    info_payload = api.model_info(repo_id=target["repo_id"], token=token or None)
    return coalesce_text(getattr(info_payload, "sha", ""))


def resolve_target_action(target: dict[str, Any], remote_revision: str) -> str:
    if target.get("kind") == "metadata":
        return "record"
    if not asset_target_ready(target):
        return "download"
    state = local_asset_state(target)
    if state["has_incomplete_files"] and (not remote_revision or state["local_revision"] == remote_revision):
        return "cleanup"
    if state["has_incomplete_files"]:
        return "repair"
    if not remote_revision:
        return "current"
    if state["local_revision"] != remote_revision:
        return "update"
    return "current"


def ensure_git_target(target: dict[str, Any], remote_revision: str, action: str) -> dict[str, Any]:
    target_dir = asset_target_dir(target)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if action == "download":
        subprocess.run(["git", "clone", coalesce_text(target["repo_url"]), str(target_dir)], check=True)
    else:
        subprocess.run(["git", "-C", str(target_dir), "fetch", "origin"], check=True)
    ref = coalesce_text(target.get("ref", "master")) or "master"
    subprocess.run(["git", "-C", str(target_dir), "checkout", remote_revision or f"origin/{ref}"], check=True)
    final_revision = infer_local_git_revision(target)
    write_asset_metadata(
        target,
        {
            "kind": target["kind"],
            "source": target["repo_url"],
            "revision": final_revision,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_dir": str(target_dir),
        },
    )
    return {
        **target,
        "revision": final_revision,
        "target_dir": str(target_dir),
        "downloaded": action == "download",
        "updated": action in {"update", "repair"},
    }


def ensure_hf_target(target: dict[str, Any], remote_revision: str, token: str, action: str) -> dict[str, Any]:
    ensure_runtime_package("huggingface_hub", "huggingface_hub")
    from huggingface_hub import snapshot_download

    target_dir = asset_target_dir(target)
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=target["repo_id"],
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        token=token or None,
        resume_download=True,
        revision=remote_revision or None,
    )
    removed_incomplete = cleanup_incomplete_download_files(target)
    final_revision = remote_revision or infer_local_hf_revision(target)
    write_asset_metadata(
        target,
        {
            "kind": target["kind"],
            "source": target["repo_id"],
            "revision": final_revision,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "snapshot_path": str(snapshot_path),
            "repaired_incomplete_files": removed_incomplete,
        },
    )
    return {
        **target,
        "revision": final_revision,
        "target_dir": str(target_dir),
        "snapshot_path": str(snapshot_path),
        "downloaded": action == "download",
        "updated": action in {"update", "repair"},
        "repaired_incomplete_files": removed_incomplete,
    }


def ensure_metadata_target(target: dict[str, Any]) -> dict[str, Any]:
    target_dir = asset_target_dir(target)
    target_dir.mkdir(parents=True, exist_ok=True)
    write_asset_metadata(
        target,
        {
            "kind": target["kind"],
            "source": coalesce_text(target.get("repo_id", "")),
            "revision": "",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_dir": str(target_dir),
        },
    )
    return {
        **target,
        "revision": "",
        "target_dir": str(target_dir),
        "downloaded": False,
        "updated": False,
        "recorded": True,
    }


def prepare_quality_backend_assets(cfg: dict[str, Any], *, force: bool, skip_downloads: bool) -> list[dict[str, Any]]:
    assets_cfg = cfg.get("quality_backend_assets", {}) if isinstance(cfg.get("quality_backend_assets"), dict) else {}
    token_env = coalesce_text(assets_cfg.get("huggingface_token_env", "HF_TOKEN")) or "HF_TOKEN"
    token = coalesce_text(__import__("os").environ.get(token_env, ""))
    api = None
    results: list[dict[str, Any]] = []

    for target in quality_backend_asset_targets(cfg):
        kind = coalesce_text(target.get("kind", ""))
        state = local_asset_state(target)
        remote_revision = ""
        if kind == "git":
            remote_revision = fetch_remote_git_revision(target)
        elif kind == "huggingface":
            ensure_runtime_package("huggingface_hub", "huggingface_hub")
            if api is None:
                from huggingface_hub import HfApi

                api = HfApi()
            remote_revision = fetch_remote_hf_revision(api, target, token)

        action = "update" if force and kind in {"git", "huggingface"} else resolve_target_action(target, remote_revision)
        target_dir = asset_target_dir(target)

        if skip_downloads or action == "current":
            if state.get("inferred_revision") and not read_asset_metadata(target):
                write_asset_metadata(
                    target,
                    {
                        "kind": kind,
                        "source": coalesce_text(target.get("repo_url", target.get("repo_id", ""))),
                        "revision": state["local_revision"],
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "target_dir": str(target_dir),
                    },
                )
            results.append(
                {
                    **target,
                    "target_dir": str(target_dir),
                    "downloaded": False,
                    "updated": False,
                    "revision": remote_revision or coalesce_text(state.get("local_revision", "")),
                    "ready": bool(state.get("ready", False)),
                    "action": "current" if not skip_downloads else "validated",
                }
            )
            continue

        if action == "cleanup":
            removed = cleanup_incomplete_download_files(target)
            results.append(
                {
                    **target,
                    "target_dir": str(target_dir),
                    "downloaded": False,
                    "updated": False,
                    "revision": remote_revision or coalesce_text(state.get("local_revision", "")),
                    "cleaned": True,
                    "cleaned_incomplete_files": removed,
                    "ready": asset_target_ready(target),
                    "action": action,
                }
            )
            continue

        if kind == "metadata":
            info(f"Erfasse Backend-Metadaten: {target['name']}")
            result = ensure_metadata_target(target)
        elif kind == "git":
            info(f"{'Aktualisiere' if action in {'update', 'repair'} else 'Lade'} Backend-Tool: {target['name']}")
            result = ensure_git_target(target, remote_revision, action)
        elif kind == "huggingface":
            info(f"{'Aktualisiere' if action in {'update', 'repair'} else 'Lade'} Backend-Modell: {target['repo_id']}")
            result = ensure_hf_target(target, remote_revision, token, action)
        else:
            warn(f"Unknown quality backend asset kind skipped: {kind}")
            continue
        result["action"] = action
        result["ready"] = asset_target_ready(target)
        results.append(result)
    return results


def write_summary(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    assets_cfg = cfg.get("quality_backend_assets", {}) if isinstance(cfg.get("quality_backend_assets"), dict) else {}
    summary_relative = coalesce_text(assets_cfg.get("summary_path", "tools/quality_backends/quality_backend_assets.json"))
    summary_path = resolve_project_path(summary_relative)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(CONFIG_PATH),
        "assets": rows,
    }
    write_json(summary_path, payload)
    return summary_path


def main() -> None:
    args = parse_args()
    headline("Prepare Quality Backends")
    cfg = load_config()
    rows = prepare_quality_backend_assets(cfg, force=args.force, skip_downloads=args.skip_downloads or args.print_plan)
    summary_path = write_summary(cfg, rows)
    for row in rows:
        source = coalesce_text(row.get("repo_id", row.get("repo_url", row.get("name", ""))))
        state = "ready" if row.get("ready") else "incomplete"
        action = coalesce_text(row.get("action", "current")) or "current"
        info(f"{coalesce_text(row.get('kind', 'asset'))}: {source} -> {row.get('target_dir', '')} ({action}, {state})")
    ok(f"Quality backend asset summary written: {summary_path}")


if __name__ == "__main__":
    main()
