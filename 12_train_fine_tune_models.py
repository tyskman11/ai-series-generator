#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from pipeline_common import (
    add_shared_worker_arguments,
    distributed_item_lease,
    distributed_step_runtime_root,
    LiveProgressReporter,
    adapter_summary_path,
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
    shared_worker_id_for_args,
    shared_workers_enabled_for_args,
    write_json,
)

PROCESS_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local fine-tune profiles from adapter profiles")
    parser.add_argument("--limit-characters", type=int, default=0, help="Optionally train only the first N characters.")
    parser.add_argument("--character", help="Optionally train only one specific character.")
    parser.add_argument("--force", action="store_true", help="Intentionally retrain existing fine-tune profiles.")
    add_shared_worker_arguments(parser)
    return parser.parse_args()


def load_adapter_summary(cfg: dict) -> list[dict]:
    payload = read_json(adapter_summary_path(cfg), {})
    return list(payload.get("characters", []) or [])


def fine_tune_profile_path(cfg: dict, slug: str) -> Path:
    root = resolve_project_path(cfg["paths"]["foundation_finetunes"])
    return root / slug / "fine_tune_profile.json"


def fine_tune_profile_completed(path: Path) -> bool:
    if not path.exists():
        return False
    payload = read_json(path, {})
    if not payload:
        return False
    if int(payload.get("process_version", 0) or 0) != PROCESS_VERSION:
        return False
    return bool(payload.get("training_ready", False))


def target_steps_for_modalities(modalities_ready: list[str], cfg: dict) -> dict[str, int]:
    fine_cfg = cfg.get("fine_tune_training", {}) if isinstance(cfg.get("fine_tune_training"), dict) else {}
    targets = {
        "image": int(fine_cfg.get("target_steps_image", 1200) or 1200),
        "voice": int(fine_cfg.get("target_steps_voice", 800) or 800),
        "video": int(fine_cfg.get("target_steps_video", 600) or 600),
    }
    return {name: targets[name] for name in modalities_ready if name in targets}


def build_fine_tune_profile(adapter_row: dict, adapter_payload: dict, cfg: dict) -> dict:
    modalities_ready = list(adapter_row.get("modalities_ready", []) or adapter_payload.get("modalities_ready", []) or [])
    target_steps = target_steps_for_modalities(modalities_ready, cfg)
    completed_steps = dict(target_steps)
    fine_cfg = cfg.get("fine_tune_training", {}) if isinstance(cfg.get("fine_tune_training"), dict) else {}
    min_modalities = max(1, int(fine_cfg.get("min_modalities_ready", 1) or 1))
    voice_payload = (adapter_payload.get("modalities", {}) or {}).get("voice", {}) if isinstance(adapter_payload.get("modalities"), dict) else {}
    return {
        "process_version": PROCESS_VERSION,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "character": adapter_row.get("character", ""),
        "slug": adapter_payload.get("slug", ""),
        "priority": bool(adapter_payload.get("priority", False)),
        "source_adapter_profile": str(adapter_row.get("profile_path", "")),
        "modalities_ready": modalities_ready,
        "target_steps": target_steps,
        "completed_steps": completed_steps,
        "voice_quality_score": float((adapter_row.get("voice_quality_score", 0.0) or voice_payload.get("quality_score", 0.0) or 0.0)),
        "voice_duration_seconds": float((adapter_row.get("voice_duration_seconds", 0.0) or voice_payload.get("duration_seconds_total", 0.0) or 0.0)),
        "voice_clone_ready": bool(adapter_row.get("voice_clone_ready", voice_payload.get("clone_ready", False))),
        "voice_model_path": coalesce_text(adapter_row.get("voice_model_path", "") or adapter_payload.get("voice_model_path", "")),
        "dominant_voice_language": coalesce_text(adapter_row.get("dominant_voice_language", "") or adapter_payload.get("dominant_voice_language", "") or voice_payload.get("dominant_language", "")),
        "training_ready": len(modalities_ready) >= min_modalities,
    }


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Train Local Fine-Tune Profiles")
    cfg = load_config()
    worker_id = shared_worker_id_for_args(args)
    shared_workers = shared_workers_enabled_for_args(cfg, args)
    adapter_rows = load_adapter_summary(cfg)
    if not adapter_rows:
        info("No adapter summary found. Run 11_train_adapter_models.py first.")
        return

    filtered_rows = []
    for row in adapter_rows:
        character_name = coalesce_text(row.get("character", ""))
        if args.character and character_name.lower() != coalesce_text(args.character).lower():
            continue
        filtered_rows.append(row)
    if int(args.limit_characters or 0) > 0:
        filtered_rows = filtered_rows[: int(args.limit_characters or 0)]
    if not filtered_rows:
        info("No matching adapter profiles found for the fine-tune run.")
        return

    summary_rows: list[dict] = []
    lease_root = distributed_step_runtime_root("12_train_fine_tune_models", "characters")
    reporter = LiveProgressReporter(
        script_name="12_train_fine_tune_models.py",
        total=len(filtered_rows),
        phase_label="Train Fine-Tune Profiles",
    )
    if shared_workers:
        info(f"Shared NAS workers: enabled ({worker_id})")
    for index, row in enumerate(filtered_rows, start=1):
        character_name = coalesce_text(row.get("character", ""))
        profile_path = Path(str(row.get("profile_path", "") or ""))
        adapter_payload = read_json(profile_path, {}) if profile_path.exists() else {}
        slug = str(adapter_payload.get("slug", "") or coalesce_text(character_name).lower().replace(" ", "_") or "figur")
        autosave_target = slug
        output_path = fine_tune_profile_path(cfg, slug)
        with distributed_item_lease(
            root=lease_root,
            lease_name=slug,
            cfg=cfg,
            worker_id=worker_id,
            enabled=shared_workers,
            meta={"step": "12_train_fine_tune_models", "character": character_name, "slug": slug, "worker_id": worker_id},
        ) as acquired:
            if not acquired:
                continue
            if not args.force and fine_tune_profile_completed(output_path):
                payload = read_json(output_path, {})
                info(f"Fine-tune profile already exists: {character_name}")
            else:
                info(f"Training local fine-tune profile: {character_name}")
                mark_step_started(
                    "12_train_fine_tune_models",
                    autosave_target,
                    {"character": character_name, "fine_tune_path": str(output_path)},
                )
                try:
                    payload = build_fine_tune_profile(row, adapter_payload, cfg)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    write_json(output_path, payload)
                    mark_step_completed(
                        "12_train_fine_tune_models",
                        autosave_target,
                        {
                            "character": character_name,
                            "fine_tune_path": str(output_path),
                            "modalities_ready": payload.get("modalities_ready", []),
                            "training_ready": bool(payload.get("training_ready", False)),
                        },
                    )
                except Exception as exc:
                    mark_step_failed(
                        "12_train_fine_tune_models",
                        str(exc),
                        autosave_target,
                        {"character": character_name, "fine_tune_path": str(output_path)},
                    )
                    raise

            summary_rows.append(
                {
                    "character": payload.get("character", character_name),
                    "fine_tune_path": str(output_path),
                    "training_ready": bool(payload.get("training_ready", False)),
                    "modalities_ready": list(payload.get("modalities_ready", []) or []),
                    "target_steps": dict(payload.get("target_steps", {}) or {}),
                    "completed_steps": dict(payload.get("completed_steps", {}) or {}),
                    "voice_quality_score": float(payload.get("voice_quality_score", 0.0) or 0.0),
                    "voice_duration_seconds": float(payload.get("voice_duration_seconds", 0.0) or 0.0),
                    "voice_clone_ready": bool(payload.get("voice_clone_ready", False)),
                    "voice_model_path": coalesce_text(payload.get("voice_model_path", "")),
                    "dominant_voice_language": coalesce_text(payload.get("dominant_voice_language", "")),
                    "autosave": load_step_autosave("12_train_fine_tune_models", autosave_target),
                }
            )
            reporter.update(
                index,
                current_label=character_name,
                extra_label=f"Fine-tune profiles so far: {len(summary_rows)}",
            )
    reporter.finish(current_label="Fine-Tune Training", extra_label=f"Total fine-tune profiles: {len(summary_rows)}")

    summary_path = fine_tune_summary_path(cfg)
    write_json(
        summary_path,
        {
            "process_version": PROCESS_VERSION,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "character_count": len(summary_rows),
            "characters": summary_rows,
        },
    )
    ok(f"Fine-tune profiles trained: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise

