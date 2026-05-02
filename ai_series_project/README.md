# AI Series Training

## Table Of Contents

- [Purpose](#purpose)
- [Current Status](#current-status)
- [Project Layout](#project-layout)
- [Pipeline Order](#pipeline-order)
- [Quick Start](#quick-start)
- [Quality-First Mode](#quality-first-mode)
- [Configuration](#configuration)
- [Testing](#testing)
- [Known Limitations](#known-limitations)
- [Finished](#finished)
- [In Progress](#in-progress)
- [Planned](#planned)

## Purpose

This project turns existing TV episodes into a portable local pipeline for:

- importing and splitting source episodes
- transcription, diarization, face linking, and manual review
- dataset building and training preparation
- local generation of new episodes, storyboard assets, and renders
- preparation of project-local backend tools and model downloads
- quality-gated generation of finished synthetic episodes

The repository root contains the numbered pipeline scripts directly beside `ai_series_project/`. All project data, configs, tests, support code, runtime data, and generated assets stay inside `ai_series_project/`.

## Current Status

- preprocessing from source episode to reviewed character/speaker data is usable
- the training chain exists and is ordered before generation/rendering
- `00_prepare_runtime.py` now owns normal setup completely, including runtime prep, backend configuration, project-local downloads, model download verification, and folder creation
- the GitHub repo now treats `configs/project.template.json` as the tracked base template; the working `project.json` is generated locally
- quality-first generation is enforced more strictly than before
- final output quality is still limited when only the project-local fallback backends are used

## Project Layout

Run the numbered scripts from the repository root.

### Repository Root

- `00_prepare_runtime.py` to `57_process_next_episode.py`: numbered pipeline and orchestration scripts
- `ai_series_project/`: project-internal data, config, runtime, support code, tests, tools, training, and outputs

### Main Folders Inside `ai_series_project/`

- `characters/`: maps, previews, review queues, voice samples, voice models
- `configs/`: project configuration
- `configs/project.template.json`: tracked baseline config for new series/projects
- `configs/project.json`: local generated working config, not intended for GitHub
- `data/`: inbox, raw episodes, processed metadata, linked segments, datasets
- `generation/`: prompts, storyboard requests/assets, render packages, deliveries
- `runtime/`: autosaves, distributed leases, git helper state, backend staging, host runtime
- `runtime/host_runtime/`: venvs, install logs, package status, runtime-local downloads
- `support_scripts/`: shared helper modules and non-numbered helper scripts
- `tests/`: regression tests
- `tools/`: project-local backend runners, backend tools, model assets
- `training/`: foundation datasets, manifests, checkpoints, backend runs

### Numbered Scripts In Repository Root

- `00_prepare_runtime.py`: full setup, folder creation, runtime packages, backend config, backend/model downloads, download completeness checks, FFmpeg via `imageio-ffmpeg`
- `01_import_episode.py`
- `02_split_scenes.py`
- `03_diarize_and_transcribe.py`
- `04_link_faces_and_speakers.py`
- `05_review_unknowns.py`
- `06_build_dataset.py`
- `07_train_series_model.py`
- `08_prepare_foundation_training.py`
- `09_train_foundation_models.py`
- `10_train_adapter_models.py`
- `11_train_fine_tune_models.py`
- `12_run_backend_finetunes.py`
- `13_generate_episode.py`
- `14_generate_storyboard_assets.py`
- `15_run_storyboard_backend.py`
- `16_render_episode.py`
- `17_quality_gate.py`
- `18_regenerate_weak_scenes.py`
- `19_build_series_bible.py`
- `20_export_package.py`
- `21_refresh_after_manual_review.py`: rebuild from reviewed data after manual review changes
- `22_analyze_patterns.py` to `53_social_media_clips.py`: analysis, archive, export, and helper stages
- `54_backup_project.py`
- `55_restore_project.py`
- `56_generate_finished_episodes.py`: batch finished-episode generation
- `57_process_next_episode.py`: full inbox-to-finished-episode orchestrator

### Support Scripts Inside `ai_series_project/support_scripts/`

- `support_scripts/pipeline_common.py`: shared path handling, runtime helpers, orchestration, leases, quality gating
- `support_scripts/console_colors.py`: console formatting
- `support_scripts/configure_quality_backends.py`: helper called by `00` to write portable quality-backend config
- `support_scripts/prepare_quality_backends.py`: helper called by `00` to download/update project-local backend assets
- `support_scripts/backend_preset_benchmark.py`: benchmark helper for backend presets

## Pipeline Order

### Full Setup First

`00_prepare_runtime.py` must come first. It now handles:

- project structure creation
- generation of `configs/project.json` from `configs/project.template.json` when missing
- runtime environment setup
- project-local FFmpeg resolution from Python
- quality-backend config writing
- project-local backend tool/model downloads
- download completeness and revision checks

### Main Production Chain

If your review data is already ready, the intended main order is:

`00 -> 06 -> 07 -> 08 -> 09 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16 -> 17 -> 18 -> 19 -> 20`

That means:

- downloads and setup happen in `00`
- training happens before generation
- backend fine-tune runs happen before new episode generation
- storyboard backend materialization happens before render
- render, quality gate, regeneration, bible, and export happen only after generation

### Full Inbox Pipeline

For a raw new episode source, the full order is:

`00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16 -> 17 -> 18 -> 19 -> 20`

`57_process_next_episode.py` runs that complete chain and starts with `00` automatically.

`21_refresh_after_manual_review.py` and `56_generate_finished_episodes.py` also start with `00` automatically before their own main work.

## Quick Start

### 1. Open The Repository Root

```powershell
cd <repo-root>
```

### 2. Run Full Setup

```powershell
python 00_prepare_runtime.py
```

On Linux, the script uses the active `python3` interpreter and `pip --break-system-packages` when supported.

If optional runtime packages such as `face_recognition/facenet-pytorch`, `speechbrain`, or `TTS` cannot be installed, `00_prepare_runtime.py` now reports `ready with limitations` instead of pretending the full stack is complete. In that case, later steps can still run, but face linking, speaker embedding quality, or higher-quality voice cloning may stay reduced until the missing package installs cleanly.

`00_prepare_runtime.py` now also validates the Torch runtime stack more strictly: `torchvision` is treated as required for the face-recognition path, so a half-installed Torch setup is repaired before `facenet-pytorch` is checked.

If you only want to validate the current local asset state without new downloads:

```powershell
python 00_prepare_runtime.py --skip-downloads
```

### 3. Process New Source Episodes

Manual step-by-step:

```powershell
python 01_import_episode.py
python 02_split_scenes.py
python 03_diarize_and_transcribe.py
python 04_link_faces_and_speakers.py
python 05_review_unknowns.py
```

Or full orchestration:

```powershell
python 57_process_next_episode.py
```

### 4. Generate A Finished Episode From Reviewed Data

```powershell
python 56_generate_finished_episodes.py --count 1
```

### 5. Rebuild After Manual Review

```powershell
python 21_refresh_after_manual_review.py
```

## Quality-First Mode

The project is configured for quality-first finished-episode generation:

- release mode enabled
- original-line reuse enabled
- system TTS fallback disabled in the quality-first path
- lip-sync expected
- external backend runner hooks configured through project-local defaults

Important:

- if the real backend path is not ready, the pipeline should fail early instead of pretending a fallback render is a final TV-quality episode
- the project-local fallback backends are useful for plumbing and iteration, but they are still not equal to real original-episode-quality image/video/lip-sync generation

## Configuration

Main config:

- `configs/project.template.json`: tracked GitHub baseline
- `configs/project.json`: local working copy created by `00_prepare_runtime.py`

Recommended workflow:

- commit reusable defaults to `project.template.json`
- let each machine or series create its own local `project.json`
- avoid treating `project.json` as the public baseline for the repository

Important areas:

- `paths.*`: project folders
- `runtime.*`: device, FFmpeg GPU preference, torch index URL
- `distributed.*`: NAS/shared-worker lease timing
- `foundation_training.*`, `adapter_training.*`, `fine_tune_training.*`, `backend_fine_tune.*`
- `external_backends.*`: runner templates and project-local backend commands
- `release_mode.*`: quality gate thresholds and retry behavior
- `quality_backend_assets.*`: project-local backend tool/model targets

## Testing

Run the main regression suite from the repository root:

```powershell
python -m unittest discover -s ai_series_project\tests -v
```

Useful smoke checks:

```powershell
python -m py_compile 00_prepare_runtime.py 21_refresh_after_manual_review.py 56_generate_finished_episodes.py 57_process_next_episode.py ai_series_project\support_scripts\pipeline_common.py ai_series_project\support_scripts\configure_quality_backends.py ai_series_project\support_scripts\prepare_quality_backends.py
```

## Known Limitations

- project-local fallback image/video generation still does not equal strong dedicated TV-quality generation backends
- project-local lip-sync is still weaker than a full dedicated production lip-sync stack
- if external runners fail repeatedly, the quality gate will keep rejecting the episode even when the render technically finishes
- shared NAS runs still depend on file-system stability and can be slower for large backend/model downloads
- large Hugging Face model downloads are more reliable with authentication via `HF_TOKEN`

## Finished

- the numbered main scripts now live directly in the repository root and are continuous with no gaps: `00` through `57`
- `ai_series_project/` now contains the project internals only: configs, data, runtime state, tests, support scripts, backend tools, training artifacts, and generated outputs
- `00_prepare_runtime.py` now owns the normal setup flow completely, including folder creation, backend config, and project-local downloads
- the public repo now uses `project.template.json` as the tracked baseline, while `project.json` is generated locally and ignored
- `support_scripts/configure_quality_backends.py` and `support_scripts/prepare_quality_backends.py` remain available as internal helpers, but they are no longer part of the numbered main sequence
- project-local FFmpeg now comes from the Python runtime path instead of a separate external FFmpeg download
- `21_refresh_after_manual_review.py`, `56_generate_finished_episodes.py`, and `57_process_next_episode.py` now begin with `00_prepare_runtime.py`
- the documented order is now setup/downloads first, then training, then backend fine-tunes, then generate/render/gate/export

## In Progress

- improving the real quality of project-local fallback image generation so scenes stop looking like weak placeholder composites
- improving project-local voice generation and timing so scenes have more natural speech coverage
- improving the project-local lip-sync path beyond simple fallback mux behavior
- reducing external backend task failures so the quality gate can pass more often with real generated assets
- continuing cleanup of leftover legacy wording in some logs and helper comments

## Planned

- stronger automatic worker-capability routing for NAS multi-PC runs
- better scene-selection logic for regeneration retries after quality-gate failures
- stronger real local XTTS/lip-sync runtime integration when the backend assets are fully available
- higher-quality project-local image/video backends to move closer to original-episode visual consistency
