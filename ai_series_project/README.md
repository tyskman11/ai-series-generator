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

The project root is `KI Serien Training`. The numbered pipeline scripts live directly in that root beside `ai_series_project/`. All project data, configs, tests, support code, runtime data, and generated assets stay inside `ai_series_project/`.

## Current Status

- preprocessing from source episode to reviewed character/speaker data is usable
- the training chain exists and is ordered before generation/rendering
- `00_prepare_runtime.py` now owns normal setup completely, including runtime prep, backend configuration, project-local downloads, model download verification, and folder creation
- quality-first generation is enforced more strictly than before
- final output quality is still limited when only the project-local fallback backends are used

## Project Layout

Run the numbered scripts from the project root `KI Serien Training`.

### Project Root

- `00_prepare_runtime.py` to `57_process_next_episode.py`: numbered main pipeline scripts in execution order
- `ai_series_project/`: project-internal data, config, runtime, support code, tests, tools, training, and outputs

### Main Folders Inside `ai_series_project/`

- `characters/`: maps, previews, review queues, voice samples, voice models
- `configs/`: project configuration
- `data/`: inbox, raw episodes, processed metadata, linked segments, datasets
- `generation/`: prompts, storyboard requests/assets, render packages, deliveries
- `runtime/`: autosaves, distributed leases, git helper state, backend staging, host runtime
- `runtime/host_runtime/`: venvs, install logs, package status, runtime-local downloads
- `support_scripts/`: shared helper modules and non-numbered helper scripts
- `tests/`: regression tests
- `tools/`: project-local backend runners, backend tools, model assets
- `training/`: foundation datasets, manifests, checkpoints, backend runs

### Numbered Scripts In Project Root

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
- `12_generate_episode.py`
- `13_generate_storyboard_assets.py`
- `14_render_episode.py`
- `15_build_series_bible.py`
- `16_analyze_patterns.py` to `47_social_media_clips.py`: analysis, export, and helper stages
- `48_refresh_after_manual_review.py`: rebuild from reviewed data
- `49_run_backend_finetunes.py`
- `50_export_package.py`
- `51_quality_gate.py`
- `52_regenerate_weak_scenes.py`
- `53_run_storyboard_backend.py`
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
- runtime environment setup
- project-local FFmpeg resolution from Python
- quality-backend config writing
- project-local backend tool/model downloads
- download completeness and revision checks

### From Reviewed Data To Finished Episode

If your review data is already ready, the intended main order is:

`00 -> 06 -> 07 -> 08 -> 09 -> 10 -> 11 -> 49 -> 12 -> 13 -> 53 -> 14 -> 51 -> 15`

That means:

- downloads and setup happen in `00`
- training happens before generation
- generation and render happen only after training

### Full Inbox Pipeline

For a raw new episode source, the full order is:

`00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09 -> 10 -> 11 -> 49 -> 12 -> 13 -> 53 -> 14 -> 51 -> 15`

`57_process_next_episode.py` runs that complete chain and starts with `00` automatically.

`48_refresh_after_manual_review.py` and `56_generate_finished_episodes.py` also start with `00` automatically before their own main work.

## Quick Start

### 1. Enter The Project Root

```powershell
cd "B:\PROJEKTE\ai\KI Serien Training"
```

### 2. Run Full Setup

```powershell
python 00_prepare_runtime.py
```

On Linux, the script uses the active `python3` interpreter and `pip --break-system-packages` when supported.

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
python 48_refresh_after_manual_review.py
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

- `configs/project.json`

Important areas:

- `paths.*`: project folders
- `runtime.*`: device, FFmpeg GPU preference, torch index URL
- `distributed.*`: NAS/shared-worker lease timing
- `foundation_training.*`, `adapter_training.*`, `fine_tune_training.*`, `backend_fine_tune.*`
- `external_backends.*`: runner templates and project-local backend commands
- `release_mode.*`: quality gate thresholds and retry behavior
- `quality_backend_assets.*`: project-local backend tool/model targets

## Testing

Run the main regression suite from inside `ai_series_project/`:

```powershell
python -m unittest discover -s ai_series_project\tests -v
```

Useful smoke checks:

```powershell
python -m py_compile 00_prepare_runtime.py 48_refresh_after_manual_review.py 56_generate_finished_episodes.py 57_process_next_episode.py support_scripts\\pipeline_common.py support_scripts\\configure_quality_backends.py support_scripts\\prepare_quality_backends.py
```

## Known Limitations

- project-local fallback image/video generation still does not equal strong dedicated TV-quality generation backends
- project-local lip-sync is still weaker than a full dedicated production lip-sync stack
- if external runners fail repeatedly, the quality gate will keep rejecting the episode even when the render technically finishes
- shared NAS runs still depend on file-system stability and can be slower for large backend/model downloads
- large Hugging Face model downloads are more reliable with authentication via `HF_TOKEN`

## Finished

- the visible repository root now only contains `ai_series_project/` plus hidden Git metadata
- all tests, support scripts, runtime state, and backend tools now live inside `ai_series_project/`
- the numbered main scripts are now continuous with no gaps: `00` through `57`
- `00_prepare_runtime.py` now owns the normal setup flow completely
- `support_scripts/configure_quality_backends.py` and `support_scripts/prepare_quality_backends.py` remain available as internal helpers, but they are no longer part of the numbered main sequence
- project-local FFmpeg now comes from the Python runtime path instead of a separate external FFmpeg download
- `48_refresh_after_manual_review.py`, `56_generate_finished_episodes.py`, and `57_process_next_episode.py` now begin with `00_prepare_runtime.py`
- the documented order is now setup/downloads first, then training, then generate/render

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
