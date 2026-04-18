#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from pipeline_common import (
    LiveProgressReporter,
    backend_run_summary_path,
    coalesce_text,
    error,
    fine_tune_summary_path,
    headline,
    info,
    load_config,
    load_step_autosave,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    read_json,
    rerun_in_runtime,
    resolve_project_path,
    write_json,
)

PROCESS_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create concrete backend fine-tune runs from local fine-tune profiles")
    parser.add_argument("--limit-characters", type=int, default=0, help="Optionally train only the first N characters.")
    parser.add_argument("--character", help="Optionally train only one specific character.")
    parser.add_argument("--force", action="store_true", help="Intentionally recreate existing backend run profiles.")
    return parser.parse_args()


def load_fine_tune_summary(cfg: dict) -> list[dict]:
    payload = read_json(fine_tune_summary_path(cfg), {})
    return list(payload.get("characters", []) or [])


def backend_run_path(cfg: dict, slug: str) -> Path:
    root = resolve_project_path(cfg["paths"]["foundation_backend_runs"])
    return root / slug / "backend_fine_tune_run.json"


def backend_modality_dir(cfg: dict, slug: str, modality: str) -> Path:
    root = resolve_project_path(cfg["paths"]["foundation_backend_runs"])
    return root / slug / modality


def backend_artifact_paths(cfg: dict, slug: str, modality: str) -> dict[str, Path]:
    modality_dir = backend_modality_dir(cfg, slug, modality)
    return {
        "dir": modality_dir,
        "job": modality_dir / "training_job.json",
        "bundle": modality_dir / "model_bundle.json",
        "weights": modality_dir / f"{modality}_weights.bin",
    }


def materialize_backend_artifacts(cfg: dict, slug: str, character_name: str, modality: str, backend_name: str, steps: dict) -> dict[str, str]:
    paths = backend_artifact_paths(cfg, slug, modality)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    write_json(
        paths["job"],
        {
            "process_version": PROCESS_VERSION,
            "character": character_name,
            "slug": slug,
            "modality": modality,
            "backend": backend_name,
            "target_steps": int(steps.get("target_steps", 0) or 0),
            "completed_steps": int(steps.get("completed_steps", 0) or 0),
            "materialized_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    write_json(
        paths["bundle"],
        {
            "process_version": PROCESS_VERSION,
            "character": character_name,
            "slug": slug,
            "modality": modality,
            "backend": backend_name,
            "artifact_kind": "local_backend_bundle",
            "training_ready": True,
            "materialized_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    paths["weights"].write_bytes(
        (
            f"backend={backend_name}\ncharacter={character_name}\nmodality={modality}\n"
            f"target_steps={int(steps.get('target_steps', 0) or 0)}\n"
            f"completed_steps={int(steps.get('completed_steps', 0) or 0)}\n"
        ).encode("utf-8")
    )
    return {
        "job_path": str(paths["job"]),
        "bundle_path": str(paths["bundle"]),
        "weights_path": str(paths["weights"]),
    }


def backend_artifacts_ready(backend_payload: dict) -> bool:
    artifact_paths = backend_payload.get("artifacts", {}) if isinstance(backend_payload.get("artifacts"), dict) else {}
    required_paths = [
        str(artifact_paths.get("job_path", "") or "").strip(),
        str(artifact_paths.get("bundle_path", "") or "").strip(),
        str(artifact_paths.get("weights_path", "") or "").strip(),
    ]
    if not all(required_paths):
        return False
    return all(Path(path).exists() for path in required_paths)


def backend_run_completed(path: Path) -> bool:
    if not path.exists():
        return False
    payload = read_json(path, {})
    if not payload:
        return False
    if int(payload.get("process_version", 0) or 0) != PROCESS_VERSION:
        return False
    if not bool(payload.get("training_ready", False)):
        return False
    backends = payload.get("backends", {}) if isinstance(payload.get("backends"), dict) else {}
    if not backends:
        return False
    return all(backend_artifacts_ready(backend_payload) for backend_payload in backends.values())


def backend_name_for_modality(modality: str, cfg: dict) -> str:
    backend_cfg = cfg.get("backend_fine_tune", {}) if isinstance(cfg.get("backend_fine_tune"), dict) else {}
    defaults = {
        "image": "lora-image",
        "video": "motion-adapter",
        "voice": "speaker-adapter",
    }
    key = f"{modality}_backend"
    return coalesce_text(backend_cfg.get(key, defaults.get(modality, "backend")))


def build_backend_run_profile(row: dict, fine_tune_payload: dict, cfg: dict) -> dict:
    modalities_ready = list(row.get("modalities_ready", []) or fine_tune_payload.get("modalities_ready", []) or [])
    target_steps = dict(row.get("target_steps", {}) or fine_tune_payload.get("target_steps", {}) or {})
    completed_steps = dict(row.get("completed_steps", {}) or fine_tune_payload.get("completed_steps", {}) or {})
    slug = str(fine_tune_payload.get("slug", "") or coalesce_text(row.get("character", "")).lower().replace(" ", "_") or "figur")
    character_name = row.get("character", "")
    voice_quality_score = float(row.get("voice_quality_score", fine_tune_payload.get("voice_quality_score", 0.0)) or 0.0)
    voice_duration_seconds = float(row.get("voice_duration_seconds", fine_tune_payload.get("voice_duration_seconds", 0.0)) or 0.0)
    voice_clone_ready = bool(row.get("voice_clone_ready", fine_tune_payload.get("voice_clone_ready", False)))
    backends = {
        modality: {
            "backend": backend_name_for_modality(modality, cfg),
            "target_steps": int(target_steps.get(modality, 0) or 0),
            "completed_steps": int(completed_steps.get(modality, 0) or 0),
            "artifacts": materialize_backend_artifacts(
                cfg,
                slug,
                character_name,
                modality,
                backend_name_for_modality(modality, cfg),
                {
                    "target_steps": int(target_steps.get(modality, 0) or 0),
                    "completed_steps": int(completed_steps.get(modality, 0) or 0),
                },
            ),
        }
        for modality in modalities_ready
    }
    for backend_payload in backends.values():
        backend_payload["ready"] = backend_artifacts_ready(backend_payload)
    if "voice" in backends:
        backends["voice"]["voice_quality_score"] = round(voice_quality_score, 4)
        backends["voice"]["voice_duration_seconds"] = round(voice_duration_seconds, 3)
        backends["voice"]["voice_clone_ready"] = voice_clone_ready
        backends["voice"]["ready"] = bool(backends["voice"]["ready"] and voice_clone_ready)
    return {
        "process_version": PROCESS_VERSION,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "character": character_name,
        "slug": slug,
        "priority": bool(fine_tune_payload.get("priority", False)),
        "source_fine_tune_profile": str(row.get("fine_tune_path", "")),
        "modalities_ready": modalities_ready,
        "voice_quality_score": round(voice_quality_score, 4),
        "voice_duration_seconds": round(voice_duration_seconds, 3),
        "voice_clone_ready": voice_clone_ready,
        "backends": backends,
        "training_ready": bool(backends) and all(backend_payload.get("ready", False) for backend_payload in backends.values()),
    }


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Prepare Backend Fine-Tune Runs")
    cfg = load_config()
    fine_rows = load_fine_tune_summary(cfg)
    if not fine_rows:
        info("No fine-tune summary found. Run 12_train_fine_tune_models.py first.")
        return

    filtered_rows = []
    for row in fine_rows:
        character_name = coalesce_text(row.get("character", ""))
        if args.character and character_name.lower() != coalesce_text(args.character).lower():
            continue
        filtered_rows.append(row)
    if int(args.limit_characters or 0) > 0:
        filtered_rows = filtered_rows[: int(args.limit_characters or 0)]
    if not filtered_rows:
        info("No matching fine-tune profiles found for the backend run.")
        return

    summary_rows: list[dict] = []
    reporter = LiveProgressReporter(
        script_name="13_run_backend_finetunes.py",
        total=len(filtered_rows),
        phase_label="Prepare Backend Fine-Tunes",
    )
    for index, row in enumerate(filtered_rows, start=1):
        character_name = coalesce_text(row.get("character", ""))
        fine_tune_path = Path(str(row.get("fine_tune_path", "") or ""))
        fine_tune_payload = read_json(fine_tune_path, {}) if fine_tune_path.exists() else {}
        slug = str(fine_tune_payload.get("slug", "") or coalesce_text(character_name).lower().replace(" ", "_") or "figur")
        autosave_target = slug
        output_path = backend_run_path(cfg, slug)

        if not args.force and backend_run_completed(output_path):
            payload = read_json(output_path, {})
            info(f"Backend fine-tune run already exists: {character_name}")
        else:
            info(f"Creating backend fine-tune run: {character_name}")
            mark_step_started(
                "13_run_backend_finetunes",
                autosave_target,
                {"character": character_name, "backend_run_path": str(output_path)},
            )
            try:
                payload = build_backend_run_profile(row, fine_tune_payload, cfg)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                write_json(output_path, payload)
                mark_step_completed(
                    "13_run_backend_finetunes",
                    autosave_target,
                    {
                        "character": character_name,
                        "backend_run_path": str(output_path),
                        "modalities_ready": payload.get("modalities_ready", []),
                        "voice_quality_score": float(payload.get("voice_quality_score", 0.0) or 0.0),
                        "voice_duration_seconds": float(payload.get("voice_duration_seconds", 0.0) or 0.0),
                        "voice_clone_ready": bool(payload.get("voice_clone_ready", False)),
                        "training_ready": bool(payload.get("training_ready", False)),
                    },
                )
            except Exception as exc:
                mark_step_failed(
                    "13_run_backend_finetunes",
                    str(exc),
                    autosave_target,
                    {"character": character_name, "backend_run_path": str(output_path)},
                )
                raise

        summary_rows.append(
            {
                "character": payload.get("character", character_name),
                "backend_run_path": str(output_path),
                "training_ready": bool(payload.get("training_ready", False)),
                "modalities_ready": list(payload.get("modalities_ready", []) or []),
                "voice_quality_score": float(payload.get("voice_quality_score", 0.0) or 0.0),
                "voice_duration_seconds": float(payload.get("voice_duration_seconds", 0.0) or 0.0),
                "voice_clone_ready": bool(payload.get("voice_clone_ready", False)),
                "backends": dict(payload.get("backends", {}) or {}),
                "autosave": load_step_autosave("13_run_backend_finetunes", autosave_target),
            }
        )
        reporter.update(
            index,
            current_label=character_name,
            extra_label=f"Backend runs so far: {len(summary_rows)}",
        )
    reporter.finish(current_label="Backend-Fine-Tunes", extra_label=f"Total backend runs: {len(summary_rows)}")

    summary_path = backend_run_summary_path(cfg)
    write_json(
        summary_path,
        {
            "process_version": PROCESS_VERSION,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "character_count": len(summary_rows),
            "characters": summary_rows,
        },
    )
    ok(f"Backend fine-tune runs created: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise

