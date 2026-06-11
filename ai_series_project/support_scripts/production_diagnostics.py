from __future__ import annotations

import shutil
import socket
import wave
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from support_scripts.pipeline_common import (
    HOST_RUNTIME_ROOT,
    PROJECT_ROOT,
    canonical_person_name,
    detect_gpu_availability,
    display_person_name,
    external_backend_runner_prerequisite_gaps,
    quality_first_requirements_report,
    read_json,
    resolve_project_path,
    runtime_environment_tag,
    runtime_python,
    write_json,
)


REFERENCE_AUDIO_PATTERNS = ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.ogg")


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def clean_text(value: object) -> str:
    return str(value or "").strip()


def audio_duration_seconds(path: Path) -> float:
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".wav":
        return 0.0
    try:
        with wave.open(str(path), "rb") as handle:
            rate = int(handle.getframerate() or 0)
            return round(handle.getnframes() / rate, 3) if rate > 0 else 0.0
    except (OSError, EOFError, wave.Error):
        return 0.0


def existing_project_file(value: object) -> Path | None:
    text = clean_text(value)
    if not text:
        return None
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = resolve_project_path(text)
    return candidate if candidate.exists() and candidate.is_file() else None


def named_map_rows(payload: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    clusters = payload.get("clusters", {}) if isinstance(payload.get("clusters", {}), dict) else {}
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for cluster_id, cluster in clusters.items():
        if not isinstance(cluster, dict):
            continue
        name = canonical_person_name(clean_text(cluster.get("name", "")))
        if not name or name.lower().startswith(("unknown", "speaker_", "face_")):
            continue
        if bool(cluster.get("ignored", False)):
            continue
        rows.append((clean_text(cluster_id), name, cluster))
    return rows


def voice_model_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    root = resolve_project_path(clean_text(paths.get("voice_models", "characters/voice_models")))
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("*.json")):
        payload = read_json(path, {})
        if not isinstance(payload, dict):
            continue
        name = canonical_person_name(
            clean_text(payload.get("character", ""))
            or clean_text(payload.get("speaker_name", ""))
            or clean_text(payload.get("name", ""))
            or path.stem.replace("_voice_model", "").replace("_", " ")
        )
        if name:
            rows.append({"name": name, "path": str(path), "payload": payload})
    return rows


def reference_values(payload: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("reference_audio", "audio_path"):
        if clean_text(payload.get(key, "")):
            values.append(clean_text(payload.get(key, "")))
    for key in ("sample_paths", "reference_audio_candidates"):
        rows = payload.get(key, []) if isinstance(payload.get(key, []), list) else []
        values.extend(clean_text(row) for row in rows if clean_text(row))
    for row in payload.get("reference_segments", []) if isinstance(payload.get("reference_segments", []), list) else []:
        if not isinstance(row, dict) or row.get("voice_reference_eligible") is False:
            continue
        if clean_text(row.get("audio_path", "")):
            values.append(clean_text(row.get("audio_path", "")))
    return values


def voice_sample_files(cfg: dict[str, Any], name: str) -> list[Path]:
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    root = resolve_project_path(clean_text(paths.get("voice_samples", "characters/voice_samples")))
    if not root.exists():
        return []
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in display_person_name(name, name)).strip("_")
    files: list[Path] = []
    candidate_dirs = [root / normalized, root / name, root / name.replace(" ", "_")]
    for directory in candidate_dirs:
        if not directory.exists() or not directory.is_dir():
            continue
        for pattern in REFERENCE_AUDIO_PATTERNS:
            files.extend(path for path in directory.glob(pattern) if path.is_file() and path.stat().st_size > 0)
    return sorted(dict.fromkeys(files))


def reference_audio_for_character(cfg: dict[str, Any], model_rows: list[dict[str, Any]], name: str) -> list[Path]:
    candidates: list[Path] = voice_sample_files(cfg, name)
    for row in model_rows:
        if canonical_person_name(row.get("name", "")) != name:
            continue
        payload = row.get("payload", {}) if isinstance(row.get("payload", {}), dict) else {}
        for value in reference_values(payload):
            path = existing_project_file(value)
            if path is not None:
                candidates.append(path)
    return sorted(dict.fromkeys(path.resolve(strict=False) for path in candidates))


def character_reference_score(face_count: int, voice_cluster_count: int, reference_count: int, voice_seconds: float) -> float:
    score = 0.0
    score += 0.28 if face_count >= 2 else 0.14 if face_count else 0.0
    score += 0.18 if voice_cluster_count else 0.0
    score += min(0.28, reference_count * 0.07)
    score += min(0.26, voice_seconds / 90.0)
    return round(min(1.0, score), 4)


def build_reference_quality_dashboard(
    cfg: dict[str, Any],
    *,
    character_map: dict[str, Any] | None = None,
    voice_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    character_map = character_map if isinstance(character_map, dict) else read_json(
        resolve_project_path(clean_text(paths.get("character_map", "characters/maps/character_map.json"))),
        {"clusters": {}},
    )
    voice_map = voice_map if isinstance(voice_map, dict) else read_json(
        resolve_project_path(clean_text(paths.get("voice_map", "characters/maps/voice_map.json"))),
        {"clusters": {}},
    )
    face_rows = named_map_rows(character_map if isinstance(character_map, dict) else {})
    speaker_rows = named_map_rows(voice_map if isinstance(voice_map, dict) else {})
    model_rows = voice_model_rows(cfg)
    face_counts = Counter(name for _cluster_id, name, _cluster in face_rows)
    speaker_counts = Counter(name for _cluster_id, name, _cluster in speaker_rows)
    known_names = sorted(set(face_counts) | set(speaker_counts) | {row["name"] for row in model_rows})
    reference_owners: defaultdict[str, set[str]] = defaultdict(set)
    rows: list[dict[str, Any]] = []
    for name in known_names:
        references = reference_audio_for_character(cfg, model_rows, name)
        for path in references:
            reference_owners[str(path)].add(name)
        duration_seconds = round(sum(audio_duration_seconds(path) for path in references), 3)
        score = character_reference_score(face_counts[name], speaker_counts[name], len(references), duration_seconds)
        issues: list[str] = []
        actions: list[str] = []
        if face_counts[name] == 0:
            issues.append("no reviewed face identity")
            actions.append("review or add canonical face anchors")
        if speaker_counts[name] == 0:
            issues.append("no named speaker cluster")
            actions.append("link a clean speech cluster to this character")
        if not references:
            issues.append("no usable voice-reference file")
            actions.append("collect clean rights-safe speech references")
        elif duration_seconds and duration_seconds < 12.0:
            issues.append("voice-reference duration is short")
            actions.append("collect more clean speech for clone stability")
        rows.append(
            {
                "name": name,
                "face_identity_count": int(face_counts[name]),
                "speaker_cluster_count": int(speaker_counts[name]),
                "voice_reference_count": len(references),
                "voice_reference_seconds": duration_seconds,
                "voice_reference_paths": [str(path) for path in references[:12]],
                "score": score,
                "status": "ready" if score >= 0.72 and not issues else "needs_review",
                "issues": issues,
                "recommended_actions": list(dict.fromkeys(actions)),
            }
        )
    contaminated = {
        path: sorted(owners)
        for path, owners in reference_owners.items()
        if len(owners) > 1
    }
    for row in rows:
        shared = [path for path in row["voice_reference_paths"] if path in contaminated]
        if shared:
            row["issues"].append("voice reference is shared across multiple character labels")
            row["recommended_actions"].append("audit cross-speaker reference contamination")
            row["shared_reference_paths"] = shared
            row["status"] = "needs_review"
            row["score"] = round(max(0.0, float(row["score"]) - 0.18), 4)
    scored_rows = [float(row.get("score", 0.0) or 0.0) for row in rows]
    return {
        "created_at": utc_now_iso(),
        "character_count": len(rows),
        "ready_character_count": len([row for row in rows if row.get("status") == "ready"]),
        "average_score": round(sum(scored_rows) / max(1, len(scored_rows)), 4),
        "shared_reference_paths": contaminated,
        "characters": rows,
        "rights_review": {
            "required": True,
            "note": "Use face, voice, and lip-sync references only with rights and consent.",
        },
    }


def dashboard_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Reference Quality Dashboard",
        "",
        f"- Characters: {payload.get('character_count', 0)}",
        f"- Ready characters: {payload.get('ready_character_count', 0)}",
        f"- Average reference score: {int(round(float(payload.get('average_score', 0.0) or 0.0) * 100))}%",
        "",
        "## Character Audit",
    ]
    for row in payload.get("characters", []) if isinstance(payload.get("characters", []), list) else []:
        if not isinstance(row, dict):
            continue
        issues = "; ".join(str(item) for item in row.get("issues", []) if clean_text(item)) or "none"
        lines.append(
            f"- {row.get('name', 'character')}: {row.get('status', 'unknown')} | "
            f"faces={row.get('face_identity_count', 0)} | speakers={row.get('speaker_cluster_count', 0)} | "
            f"voice_refs={row.get('voice_reference_count', 0)} | issues={issues}"
        )
    if not payload.get("characters"):
        lines.append("- No named characters have reference evidence yet.")
    lines.extend(["", "## Rights", f"- {payload.get('rights_review', {}).get('note', '')}"])
    return "\n".join(lines).rstrip() + "\n"


def backend_row(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    external = cfg.get("external_backends", {}) if isinstance(cfg.get("external_backends", {}), dict) else {}
    payload = external.get(name, {}) if isinstance(external.get(name, {}), dict) else {}
    return {
        "name": name,
        "enabled": bool(payload.get("enabled", False)),
        "command_template": payload.get("command_template", []) if isinstance(payload.get("command_template", []), list) else [],
        "prerequisite_gaps": external_backend_runner_prerequisite_gaps(cfg, name),
    }


def asset_summary_status(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    path = resolve_project_path(clean_text(paths.get("quality_backend_asset_summary", "tools/quality_backends/quality_backend_assets.json")))
    payload = read_json(path, {}) if path.exists() else {}
    return {
        "path": str(path),
        "exists": path.exists(),
        "target_count": len(payload.get("targets", []) if isinstance(payload.get("targets", []), list) else []),
        "download_summary": payload.get("summary", {}) if isinstance(payload.get("summary", {}), dict) else {},
    }


def build_backend_readiness_report(cfg: dict[str, Any]) -> dict[str, Any]:
    quality_first = quality_first_requirements_report(cfg)
    package_status = read_json(HOST_RUNTIME_ROOT / "package_status.json", {})
    gpu = detect_gpu_availability()
    disk = shutil.disk_usage(PROJECT_ROOT)
    runners = [backend_row(cfg, name) for name in quality_first.get("required_runners", [])]
    missing = list(quality_first.get("missing", []) if isinstance(quality_first.get("missing", []), list) else [])
    if disk.free < 12 * 1024**3:
        missing.append("project volume has less than 12 GiB free for model/render outputs")
    return {
        "created_at": utc_now_iso(),
        "ready": not missing,
        "quality_first": quality_first,
        "missing": missing,
        "warnings": quality_first.get("warnings", []),
        "runtime": {
            "python": str(runtime_python()),
            "runtime_tag": runtime_environment_tag(),
            "package_status": package_status,
            "gpu": gpu,
            "disk_free_gib": round(disk.free / 1024**3, 2),
        },
        "backend_assets": asset_summary_status(cfg),
        "runners": runners,
        "rights_review_checklist": [
            "source episodes may be used for this project",
            "face/voice/lip-sync references are rights-safe and consented where required",
            "public exports keep synthetic-media disclosure metadata",
        ],
    }


def backend_readiness_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Backend Readiness Report",
        "",
        f"- Ready: {'yes' if payload.get('ready') else 'no'}",
        f"- Runtime: {payload.get('runtime', {}).get('runtime_tag', '-')}",
        f"- Python: {payload.get('runtime', {}).get('python', '-')}",
        f"- Free project disk: {payload.get('runtime', {}).get('disk_free_gib', 0)} GiB",
        "",
        "## Blockers",
    ]
    lines.extend([f"- {item}" for item in payload.get("missing", []) if clean_text(item)] or ["- none"])
    lines.extend(["", "## Runners"])
    for runner in payload.get("runners", []) if isinstance(payload.get("runners", []), list) else []:
        gaps = "; ".join(str(item) for item in runner.get("prerequisite_gaps", []) if clean_text(item)) or "none"
        lines.append(f"- {runner.get('name', 'runner')}: enabled={bool(runner.get('enabled'))} | gaps={gaps}")
    lines.extend(["", "## Rights Review Checklist"])
    lines.extend([f"- {item}" for item in payload.get("rights_review_checklist", []) if clean_text(item)])
    return "\n".join(lines).rstrip() + "\n"


def build_worker_capability_snapshot(cfg: dict[str, Any], backend_report: dict[str, Any] | None = None) -> dict[str, Any]:
    backend_report = backend_report if isinstance(backend_report, dict) else build_backend_readiness_report(cfg)
    gpu = backend_report.get("runtime", {}).get("gpu", {}) if isinstance(backend_report.get("runtime", {}), dict) else {}
    package_status = backend_report.get("runtime", {}).get("package_status", {}) if isinstance(backend_report.get("runtime", {}), dict) else {}
    ready_runners = [
        row.get("name", "")
        for row in backend_report.get("runners", []) if isinstance(backend_report.get("runners", []), list)
        if isinstance(row, dict) and bool(row.get("enabled", False)) and not row.get("prerequisite_gaps")
    ]
    return {
        "created_at": utc_now_iso(),
        "hostname": socket.gethostname(),
        "runtime_tag": runtime_environment_tag(),
        "python": str(runtime_python()),
        "has_gpu": bool(gpu.get("available", False)),
        "gpu_memory_mb": int(gpu.get("memory_total_mb", 0) or 0),
        "gpu_devices": gpu.get("devices", []) if isinstance(gpu.get("devices", []), list) else [],
        "package_capabilities": {
            "torch": bool(package_status.get("torch", False)),
            "quality_generation": bool(package_status.get("quality_generation", False)),
            "voice_cloning": bool(package_status.get("voice_cloning", False)),
            "speaker_embeddings": bool(package_status.get("speaker_embeddings", False)),
        },
        "ready_backend_runners": ready_runners,
        "routing_profiles": {
            "story_dialogue": {"gpu_required": False, "min_memory_mb": 0},
            "shot_image": {"gpu_required": True, "min_memory_mb": 8192},
            "shot_video": {"gpu_required": True, "min_memory_mb": 12288},
            "voice_clone": {"gpu_required": bool(gpu.get("available", False)), "min_memory_mb": 4096},
            "lipsync": {"gpu_required": True, "min_memory_mb": 4096},
            "audio_mix": {"gpu_required": False, "min_memory_mb": 0},
        },
    }


def write_production_diagnostics(cfg: dict[str, Any]) -> dict[str, str]:
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    root = resolve_project_path(clean_text(paths.get("quality_reports", "generation/quality_reports"))) / "readiness"
    root.mkdir(parents=True, exist_ok=True)
    references = build_reference_quality_dashboard(cfg)
    backend = build_backend_readiness_report(cfg)
    worker = build_worker_capability_snapshot(cfg, backend)
    outputs = {
        "reference_quality_json": str(root / "reference_quality_dashboard.json"),
        "reference_quality_markdown": str(root / "reference_quality_dashboard.md"),
        "backend_readiness_json": str(root / "backend_readiness_report.json"),
        "backend_readiness_markdown": str(root / "backend_readiness_report.md"),
        "worker_capabilities_json": str(root / "worker_capabilities.json"),
    }
    write_json(Path(outputs["reference_quality_json"]), references)
    Path(outputs["reference_quality_markdown"]).write_text(dashboard_markdown(references), encoding="utf-8")
    write_json(Path(outputs["backend_readiness_json"]), backend)
    Path(outputs["backend_readiness_markdown"]).write_text(backend_readiness_markdown(backend), encoding="utf-8")
    write_json(Path(outputs["worker_capabilities_json"]), worker)
    write_json(root / "latest_production_diagnostics.json", outputs)
    return outputs
