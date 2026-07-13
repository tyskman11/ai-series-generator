#!/usr/bin/env python3
"""Generate and quality-check standalone season intro/outro assets."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent / "ai_series_project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from support_scripts.pipeline_common import (
    LiveProgressReporter,
    add_shared_worker_arguments,
    detect_tool,
    distributed_item_lease,
    distributed_step_runtime_root,
    ensure_quality_first_ready,
    error,
    headline,
    info,
    load_config,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    read_json,
    rerun_in_runtime,
    resolve_project_path,
    shared_worker_id_for_args,
    shared_workers_enabled_for_args,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate standalone, quality-gated season intro/outro assets without rendering an episode."
    )
    parser.add_argument("--season", help="Season ID, for example season_01. Uses the selected episode/default season otherwise.")
    parser.add_argument("--episode-id", help="Use this generated episode's shotlist as the season-style/reference source.")
    parser.add_argument("--kind", choices=("intro", "outro", "both"), default="both", help="Season asset to create.")
    parser.add_argument("--force", action="store_true", help="Explicitly replace an existing locked canonical season asset.")
    parser.add_argument(
        "--max-quality-cycles",
        type=int,
        help="Maximum regeneration attempts per season asset. Zero means retry until the standalone gate passes or a blocker is found.",
    )
    add_shared_worker_arguments(parser)
    return parser.parse_args()


def load_render_module() -> Any:
    target = Path(__file__).resolve().parent / "17_render_episode.py"
    spec = importlib.util.spec_from_file_location("render_episode_season_assets", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load shared render helpers from {target}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_shotlist(cfg: dict[str, Any], episode_id: str = "") -> tuple[Path, dict[str, Any]]:
    shotlist_root = resolve_project_path("generation/shotlists")
    candidate = shotlist_root / f"{episode_id}.json" if episode_id else None
    if candidate is None:
        rows = sorted((path for path in shotlist_root.glob("*.json") if path.is_file()), key=lambda path: path.stat().st_mtime, reverse=True)
        candidate = rows[0] if rows else None
    if candidate is None or not candidate.is_file():
        raise RuntimeError(
            "No episode shotlist is available for season-asset generation. Generate at least one episode with "
            "14_generate_episode.py first, then run this step again."
        )
    payload = read_json(candidate, {})
    if not isinstance(payload, dict):
        raise RuntimeError(f"Season-asset source shotlist is invalid: {candidate}")
    return candidate, payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def season_asset_quality_gate(asset: dict[str, Any], asset_kind: str) -> dict[str, Any]:
    canonical = Path(str(asset.get("canonical_video", "") or ""))
    manifest = Path(str(asset.get("manifest", "") or ""))
    expected_hash = str(asset.get("canonical_sha256", "") or "")
    actual_hash = file_sha256(canonical) if canonical.is_file() else ""
    blockers: list[str] = []
    if not canonical.is_file() or canonical.stat().st_size <= 0:
        blockers.append("canonical video is missing")
    if not manifest.is_file():
        blockers.append("asset manifest is missing")
    if not expected_hash or actual_hash != expected_hash:
        blockers.append("canonical video hash is missing or does not match the manifest")
    if bool(asset.get("fallback_used", False)):
        blockers.append("a fallback backend was used")
    if str(asset.get("source_origin", "") or "") == "backend_generated":
        statuses = asset.get("backend_runner_statuses", {}) if isinstance(asset.get("backend_runner_statuses", {}), dict) else {}
        manifests = asset.get("backend_manifests", {}) if isinstance(asset.get("backend_manifests", {}), dict) else {}
        for runner in ("finished_episode_image_runner", "finished_episode_video_runner"):
            if str(statuses.get(runner, "") or "") not in {"completed", "existing_outputs"}:
                blockers.append(f"{runner} did not complete")
            runner_manifest = Path(str(manifests.get(runner, "") or ""))
            if not runner_manifest.is_file():
                blockers.append(f"{runner} manifest is missing")
    return {
        "asset_kind": asset_kind,
        "season_id": str(asset.get("season_id", "") or ""),
        "status": "PASS" if not blockers else "FAIL",
        "passed": not blockers,
        "canonical_video": str(canonical),
        "canonical_sha256": expected_hash,
        "actual_sha256": actual_hash,
        "source_origin": str(asset.get("source_origin", "") or ""),
        "backend_generated": str(asset.get("source_origin", "") or "") == "backend_generated",
        "blockers": blockers,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def write_quality_report(asset: dict[str, Any], report: dict[str, Any]) -> tuple[Path, Path]:
    manifest_path = Path(str(asset.get("manifest", "") or ""))
    report_root = manifest_path.parent if manifest_path.parent.exists() else resolve_project_path("generation/season_assets")
    json_path = report_root / f"{report['asset_kind']}_quality_gate.json"
    markdown_path = report_root / f"{report['asset_kind']}_quality_gate.md"
    write_json(json_path, report)
    lines = [
        f"# Season {str(report['asset_kind']).title()} Quality Gate",
        "",
        f"- Season: {report.get('season_id') or '-'}",
        f"- Status: {report.get('status')}",
        f"- Canonical video: {report.get('canonical_video') or '-'}",
        f"- Source: {report.get('source_origin') or '-'}",
        "",
        "## Blockers",
        "",
    ]
    blockers = report.get("blockers", []) if isinstance(report.get("blockers", []), list) else []
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- None")
    write_text(markdown_path, "\n".join(lines).rstrip() + "\n")
    return json_path, markdown_path


def remove_locked_asset(render_module: Any, cfg: dict[str, Any], season_id: str, asset_kind: str) -> None:
    root = resolve_project_path(f"generation/season_assets/{season_id}/{asset_kind}")
    for path in (root / f"{asset_kind}.mp4", root / f"{asset_kind}_manifest.json"):
        if path.exists():
            path.unlink()
    generated_root = root / "generated"
    if generated_root.exists():
        import shutil

        shutil.rmtree(generated_root)
    info(f"Removed previous {season_id} {asset_kind} outputs for regeneration.")


def season_asset_retry_policy(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[bool, int]:
    release_cfg = cfg.get("release_mode", {}) if isinstance(cfg.get("release_mode", {}), dict) else {}
    retry_until_pass = bool(release_cfg.get("retry_until_pass", True))
    try:
        configured_limit = int(release_cfg.get("max_auto_retry_cycles", 0) or 0)
    except (TypeError, ValueError):
        configured_limit = 0
    if args.max_quality_cycles is not None:
        configured_limit = max(0, int(args.max_quality_cycles))
    return retry_until_pass, configured_limit


def season_asset_retry_blocker(asset: dict[str, Any], report: dict[str, Any]) -> str:
    blockers = {str(item or "").strip() for item in report.get("blockers", []) if str(item or "").strip()}
    if "a fallback backend was used" in blockers:
        return "a fallback backend was used; fix the backend configuration instead of retrying"
    source_origin = str(asset.get("source_origin", "") or "").strip()
    if source_origin and source_origin != "backend_generated":
        return "the configured source asset is invalid; replace the source file before retrying"
    statuses = asset.get("backend_runner_statuses", {}) if isinstance(asset.get("backend_runner_statuses", {}), dict) else {}
    unavailable_statuses = {"blocked", "disabled", "missing", "not_configured", "unavailable", "unsupported"}
    for runner_name, raw_status in statuses.items():
        status = str(raw_status or "").strip().lower()
        if status in unavailable_statuses:
            return f"{runner_name} is {status}; configure a real backend before retrying"
    return ""


def render_season_asset_until_quality_gate(
    render_module: Any,
    cfg: dict[str, Any],
    shotlist: dict[str, Any],
    ffmpeg: str,
    *,
    season_id: str,
    asset_kind: str,
    render_cfg: dict[str, Any],
    reporter: LiveProgressReporter,
    overall_base: float,
    force: bool,
    retry_until_pass: bool,
    max_quality_cycles: int,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    """Regenerate one standalone asset until it passes, or report a real blocker."""
    attempt = 0
    previous_fingerprint = ""
    repeated_fingerprint_count = 0
    while True:
        attempt += 1
        if force or attempt > 1:
            remove_locked_asset(render_module, cfg, season_id, asset_kind)
        reporter.update(
            overall_base,
            current_label=f"{season_id} {asset_kind}",
            extra_label=f"Generating real backend asset (quality attempt {attempt})",
            force=True,
        )
        asset_cfg = copy.deepcopy(cfg)
        section = asset_cfg.setdefault(f"season_{asset_kind}", {})
        section["enabled"] = True
        section["auto_generate_if_missing"] = True
        section["require_in_finished_episode_mode"] = True
        if season_id:
            section["default_season_id"] = season_id
        asset = render_module.materialize_season_asset(
            asset_cfg,
            shotlist,
            ffmpeg,
            asset_kind=asset_kind,
            fps=max(12, int(render_cfg.get("fps", 30) or 30)),
            width=max(512, int(render_cfg.get("width", 1280) or 1280)),
            height=max(512, int(render_cfg.get("height", 720) or 720)),
            reporter=reporter,
            overall_base=overall_base,
            overall_span=0.95,
        )
        report = season_asset_quality_gate(asset, asset_kind)
        report["attempt"] = attempt
        report["retry_until_pass"] = retry_until_pass
        report["max_quality_cycles"] = max_quality_cycles
        report_json, report_md = write_quality_report(asset, report)
        asset["quality_gate_json"] = str(report_json)
        asset["quality_gate_markdown"] = str(report_md)
        if report["passed"]:
            return asset, report, report_json, report_md

        blocker = season_asset_retry_blocker(asset, report)
        fingerprint = "|".join(sorted(str(item) for item in report.get("blockers", []) or []))
        repeated_fingerprint_count = repeated_fingerprint_count + 1 if fingerprint and fingerprint == previous_fingerprint else 0
        previous_fingerprint = fingerprint
        if blocker:
            raise RuntimeError(f"{season_id} {asset_kind} quality retry is blocked: {blocker}")
        if repeated_fingerprint_count >= 2:
            raise RuntimeError(
                f"{season_id} {asset_kind} quality retry is blocked: the same gate blockers repeated "
                f"for {repeated_fingerprint_count + 1} attempts without a changed backend result: {fingerprint or 'unknown'}"
            )
        if not retry_until_pass:
            raise RuntimeError(
                f"{season_id} {asset_kind} did not pass its standalone quality gate and retry_until_pass is disabled: "
                + "; ".join(report["blockers"])
            )
        if max_quality_cycles and attempt >= max_quality_cycles:
            raise RuntimeError(
                f"{season_id} {asset_kind} did not pass its standalone quality gate after {attempt} attempt(s): "
                + "; ".join(report["blockers"])
            )
        info(
            f"{season_id} {asset_kind} standalone quality gate failed on attempt {attempt}; "
            "removing the generated output and retrying."
        )


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Generate Season Intro And Outro")
    cfg = load_config()
    ensure_quality_first_ready(cfg, context_label="25_generate_season_assets.py")
    worker_id = shared_worker_id_for_args(args)
    shared_workers = shared_workers_enabled_for_args(cfg, args)
    source_path, shotlist = find_shotlist(cfg, str(args.episode_id or ""))
    season_id = str(args.season or "").strip() or str(shotlist.get("season_id", "") or "").strip()
    if season_id:
        shotlist = {**shotlist, "season_id": season_id}
    render_module = load_render_module()
    ffmpeg = detect_tool(PROJECT_DIR / "tools" / "ffmpeg" / "bin", "ffmpeg")
    render_cfg = cfg.get("render", {}) if isinstance(cfg.get("render", {}), dict) else {}
    asset_kinds = ["intro", "outro"] if args.kind == "both" else [args.kind]
    retry_until_pass, max_quality_cycles = season_asset_retry_policy(cfg, args)
    actual_season_id = season_id or render_module.season_id_for_episode(cfg, shotlist, asset_kinds[0])
    mark_step_started("25_generate_season_assets", actual_season_id, {"season_id": actual_season_id, "source_shotlist": str(source_path)})
    lease_manager = distributed_item_lease(
        root=distributed_step_runtime_root("25_generate_season_assets", "seasons"),
        lease_name=actual_season_id,
        cfg=cfg,
        worker_id=worker_id,
        enabled=shared_workers,
        meta={"step": "25_generate_season_assets", "season_id": actual_season_id, "worker_id": worker_id},
    )
    acquired = lease_manager.__enter__()
    if not acquired:
        info(f"Season assets for {actual_season_id} are already being generated on another worker.")
        lease_manager.__exit__(None, None, None)
        return
    reporter = LiveProgressReporter(
        script_name="25_generate_season_assets.py",
        total=len(asset_kinds),
        phase_label="Generate Season Assets",
        parent_label=actual_season_id,
    )
    results: dict[str, dict[str, Any]] = {}
    try:
        for index, asset_kind in enumerate(asset_kinds, start=1):
            reporter.update(index - 1, current_label=f"{actual_season_id} {asset_kind}", extra_label="Preparing real backend generation", force=True)
            asset, report, _report_json, _report_md = render_season_asset_until_quality_gate(
                render_module,
                cfg,
                shotlist,
                ffmpeg,
                season_id=actual_season_id,
                asset_kind=asset_kind,
                render_cfg=render_cfg,
                reporter=reporter,
                overall_base=float(index - 1),
                force=bool(args.force),
                retry_until_pass=retry_until_pass,
                max_quality_cycles=max_quality_cycles,
            )
            results[asset_kind] = asset
            reporter.update(index, current_label=f"{actual_season_id} {asset_kind}", extra_label="Standalone quality gate passed", force=True)
            ok(f"Season {asset_kind} ready: {asset.get('canonical_video')}")
        mark_step_completed("25_generate_season_assets", actual_season_id, {"season_id": actual_season_id, "assets": results})
        reporter.finish(current_label=actual_season_id, extra_label="Intro/outro generation and quality checks completed")
    except Exception as exc:
        mark_step_failed("25_generate_season_assets", str(exc), actual_season_id, {"season_id": actual_season_id, "assets": results})
        raise
    finally:
        lease_manager.__exit__(None, None, None)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
