#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from pipeline_common import (
    SCRIPT_DIR,
    error,
    headline,
    load_config,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    rerun_in_runtime,
    runtime_python,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Erzeugt mehrere neue sichtbare Preview-Episoden am Stück.")
    parser.add_argument("--count", type=int, default=2, help="Anzahl neuer Episoden. Standard: 2")
    return parser.parse_args()


def story_dir() -> Path:
    return SCRIPT_DIR / "ai_series_project" / "generation" / "story_prompts"


def latest_episode_id() -> str | None:
    files = sorted(story_dir().glob("folge_*.md"))
    return files[-1].stem if files else None


def run_step(script_name: str, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(
        [str(runtime_python()), str(SCRIPT_DIR / script_name)],
        env=env,
        cwd=str(SCRIPT_DIR),
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    count = max(1, int(args.count))
    cfg = load_config()
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}

    headline("Mehrere sichtbare Preview-Episoden erzeugen")
    generated: list[str] = []
    autosave_target = f"count_{count}"
    mark_step_started("14_generate_preview_episodes", autosave_target, {"requested_count": count})
    try:
        run_step("07_train_series_model.py")
        if bool(foundation_cfg.get("required_before_generate", True)) or bool(foundation_cfg.get("required_before_render", True)):
            run_step("09_prepare_foundation_training.py")
            run_step("10_train_foundation_models.py")
        for index in range(count):
            before = latest_episode_id()
            run_step("11_generate_episode_from_trained_model.py")
            episode_id = latest_episode_id()
            if not episode_id or episode_id == before:
                raise RuntimeError("Neue Episode konnte nicht ermittelt werden.")
            env = os.environ.copy()
            env["SERIES_RENDER_EPISODE"] = episode_id
            run_step("13_render_episode.py", env=env)
            generated.append(episode_id)
            ok(f"{index + 1}/{count}: {episode_id} erzeugt und gerendert.")

        run_step("12_build_series_bible.py")
        mark_step_completed(
            "14_generate_preview_episodes",
            autosave_target,
            {"requested_count": count, "generated_episodes": generated, "generated_count": len(generated)},
        )
        ok(f"Fertig. Neue sichtbare Episoden: {', '.join(generated)}")
    except Exception as exc:
        mark_step_failed(
            "14_generate_preview_episodes",
            str(exc),
            autosave_target,
            {"requested_count": count, "generated_episodes": generated, "generated_count": len(generated)},
        )
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
