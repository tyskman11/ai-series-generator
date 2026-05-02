#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from support_scripts.pipeline_common import (
    CONFIG_PATH,
    CONFIG_TEMPLATE_PATH,
    coalesce_text,
    current_os,
    headline,
    info,
    load_config,
    ok,
    pip_install_command,
    read_json,
    resolve_project_path,
    rerun_in_runtime,
    runtime_python,
    warn,
    write_json,
)

DOWNLOAD_METADATA_FILE = ".quality_backend_asset.json"
GITHUB_API_BASE = "https://api.github.com"


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
        pip_install_command(runtime_python(), package_name),
        check=True,
    )


def git_command_available() -> bool:
    return shutil.which("git") is not None


def parse_github_repo(target: dict[str, Any]) -> tuple[str, str] | tuple[None, None]:
    repo_url = coalesce_text(target.get("repo_url", ""))
    if not repo_url:
        return None, None
    parsed = urlparse(repo_url)
    if parsed.netloc.lower() != "github.com":
        return None, None
    parts = [segment for segment in parsed.path.strip("/").split("/") if segment]
    if len(parts) < 2:
        return None, None
    owner = parts[0]
    repo = parts[1][:-4] if parts[1].endswith(".git") else parts[1]
    if not owner or not repo:
        return None, None
    return owner, repo


def github_request_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "ai-series-training"})
    with urlopen(request, timeout=60) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    return data if isinstance(data, dict) else {}


def github_archive_url(owner: str, repo: str, ref: str) -> str:
    return f"https://github.com/{owner}/{repo}/archive/refs/heads/{ref}.zip"


def github_commit_api_url(owner: str, repo: str, ref: str) -> str:
    return f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{ref}"


def is_windows_unc_path(path: Path) -> bool:
    return current_os() == "windows" and str(path).startswith("\\\\")


def project_git_config_path() -> Path:
    config_path = resolve_project_path("runtime/git/project_git_config.ini")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text("[safe]\n", encoding="utf-8")
    return config_path


def git_subprocess_env() -> dict[str, str]:
    return dict(os.environ)


def run_git_command(
    args: list[str],
    *,
    target_dir: Path | None = None,
    check: bool,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = ["git", "-c", "safe.directory=*"]
    if target_dir is not None:
        command.extend(["-C", str(target_dir)])
    command.extend(args)
    kwargs: dict[str, Any] = {
        "check": check,
        "env": git_subprocess_env(),
        "text": True,
    }
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
    return subprocess.run(command, **kwargs)


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
    if not git_command_available():
        return ""
    completed = run_git_command(["rev-parse", "HEAD"], target_dir=target_dir, check=False, capture_output=True)
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
    if git_command_available():
        completed = run_git_command(["ls-remote", repo_url, ref], check=False, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout.strip() or f"Could not inspect {repo_url}")
        for line in completed.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 2 and len(parts[0]) >= 7 and all(char in "0123456789abcdefABCDEF" for char in parts[0]):
                return parts[0]
        raise RuntimeError(completed.stdout.strip() or f"Could not determine the remote revision for {repo_url}")

    owner, repo = parse_github_repo(target)
    if owner and repo:
        payload = github_request_json(github_commit_api_url(owner, repo, ref))
        return coalesce_text(payload.get("sha", ""))
    return ""


def ensure_git_safe_directory(target_dir: Path) -> None:
    return


def project_archive_staging_dir(target: dict[str, Any]) -> Path:
    name = coalesce_text(target.get("name", target.get("repo_id", "asset"))) or "asset"
    safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
    return resolve_project_path(f"runtime/backend_asset_staging/{safe_name}")


def download_and_extract_github_archive(target: dict[str, Any]) -> None:
    owner, repo = parse_github_repo(target)
    if not owner or not repo:
        raise RuntimeError(f"Git is not available and no GitHub ZIP fallback exists for {coalesce_text(target.get('repo_url', ''))}")
    ref = coalesce_text(target.get("ref", "master")) or "master"
    archive_url = github_archive_url(owner, repo, ref)
    target_dir = asset_target_dir(target)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root = project_archive_staging_dir(target)
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    zip_path = staging_root / f"{repo}-{ref}.zip"
    request = Request(archive_url, headers={"User-Agent": "ai-series-training"})
    with urlopen(request, timeout=180) as response, zip_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    shutil.unpack_archive(str(zip_path), str(staging_root))
    extracted_candidates = [path for path in staging_root.iterdir() if path.is_dir()]
    extracted_root = next((path for path in extracted_candidates if path.name != "__MACOSX"), None)
    if extracted_root is None:
        raise RuntimeError(f"Could not extract GitHub archive for {owner}/{repo}@{ref}")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.move(str(extracted_root), str(target_dir))
    shutil.rmtree(staging_root, ignore_errors=True)


def refresh_git_target_via_archive(target: dict[str, Any], remote_revision: str) -> str:
    warn(f"Git checkout for {target.get('name', 'git target')} is invalid or incomplete. Rebuilding it from a project-local GitHub ZIP download.")
    download_and_extract_github_archive(target)
    return remote_revision or coalesce_text(read_asset_metadata(target).get("revision", ""))


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
    if git_command_available():
        try:
            if action == "download":
                run_git_command(["clone", coalesce_text(target["repo_url"]), str(target_dir)], check=True)
            else:
                run_git_command(["fetch", "origin"], target_dir=target_dir, check=True)
            ref = coalesce_text(target.get("ref", "master")) or "master"
            run_git_command(["checkout", remote_revision or f"origin/{ref}"], target_dir=target_dir, check=True)
            final_revision = infer_local_git_revision(target)
        except subprocess.CalledProcessError:
            final_revision = refresh_git_target_via_archive(target, remote_revision)
    else:
        download_and_extract_github_archive(target)
        final_revision = remote_revision or coalesce_text(read_asset_metadata(target).get("revision", ""))
    write_asset_metadata(
        target,
        {
            "kind": target["kind"],
            "source": target["repo_url"],
            "revision": final_revision,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_dir": str(target_dir),
            "transport": "git" if git_command_available() else "github-zip",
        },
    )
    return {
        **target,
        "revision": final_revision,
        "target_dir": str(target_dir),
        "downloaded": action == "download",
        "updated": action in {"update", "repair"},
    }


@contextlib.contextmanager
def hf_download_environment() -> Any:
    previous_disable_xet = os.environ.get("HF_HUB_DISABLE_XET")
    previous_hf_xet = os.environ.get("HF_XET_HIGH_PERFORMANCE")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
    try:
        yield
    finally:
        if previous_disable_xet is None:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
        else:
            os.environ["HF_HUB_DISABLE_XET"] = previous_disable_xet
        if previous_hf_xet is None:
            os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
        else:
            os.environ["HF_XET_HIGH_PERFORMANCE"] = previous_hf_xet


def load_hf_snapshot_download() -> Any:
    with hf_download_environment():
        ensure_runtime_package("huggingface_hub", "huggingface_hub")
        for module_name in [name for name in list(sys.modules) if name == "huggingface_hub" or name.startswith("huggingface_hub.")]:
            sys.modules.pop(module_name, None)
        from huggingface_hub import snapshot_download

    return snapshot_download


def ensure_hf_target(target: dict[str, Any], remote_revision: str, token: str, action: str) -> dict[str, Any]:
    snapshot_download = load_hf_snapshot_download()

    target_dir = asset_target_dir(target)
    target_dir.mkdir(parents=True, exist_ok=True)
    download_dir = target_dir
    staged_from_local_temp = False
    local_stage_root: tempfile.TemporaryDirectory[str] | None = None
    if is_windows_unc_path(target_dir):
        local_stage_root = tempfile.TemporaryDirectory(prefix="hf-quality-backend-")
        download_dir = Path(local_stage_root.name) / target_dir.name
        download_dir.mkdir(parents=True, exist_ok=True)
        staged_from_local_temp = True
    with hf_download_environment():
        snapshot_path = snapshot_download(
            repo_id=target["repo_id"],
            local_dir=str(download_dir),
            token=token or None,
            revision=remote_revision or None,
            max_workers=1,
        )
    if staged_from_local_temp:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(download_dir, target_dir)
        snapshot_path = str(target_dir)
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
            "transport": "huggingface-local-stage" if staged_from_local_temp else "huggingface-direct",
        },
    )
    if local_stage_root is not None:
        local_stage_root.cleanup()
    return {
        **target,
        "revision": final_revision,
        "target_dir": str(target_dir),
        "snapshot_path": str(snapshot_path),
        "downloaded": action == "download",
        "updated": action in {"update", "repair"},
        "repaired_incomplete_files": removed_incomplete,
        "transport": "huggingface-local-stage" if staged_from_local_temp else "huggingface-direct",
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
        "template_config_path": str(CONFIG_TEMPLATE_PATH),
        "assets": rows,
    }
    write_json(summary_path, payload)
    return summary_path


def main() -> None:
    rerun_in_runtime(__file__)
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

