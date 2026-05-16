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
- `04_link_faces_and_speakers.py` now keeps portable segment-audio and language metadata in linked speaker rows
- `09_prepare_foundation_training.py` now backfills missing voice-reference audio from `speaker_transcripts` for older datasets
- the GitHub repo now treats `ai_series_project/configs/project.template.json` as the tracked base template; the working `project.json` is generated locally
- quality-first generation is enforced more strictly than before
- generated episode runtime now follows the average length of the input episodes instead of collapsing toward short dialogue-only timing
- finished-episode batches now force fresh storyboard-backend and render outputs by default so stale fallback artifacts do not silently pass through
- storyboard backend and render now refuse to treat local seed-frame/color-grade or still-motion fallback clips as production-generated outputs in quality-first mode
- quality-first generation now treats every fallback switch as an error: generated frames, scene video, cloned dialogue audio, lip-sync, and mastering must be produced by the configured backend runners
- the project-local voice backend now treats original dialogue audio as XTTS reference material, not as a finished generated dialogue line, when voice cloning is forced
- automatic transcription now cross-checks Whisper language detection against transcript text across common source languages instead of assuming German or any other fixed language
- `03_diarize_and_transcribe.py` now detects the episode language from several larger probe scenes, aggregates forced-language probe scores across those scenes, lets clear transcript text override wrong file-name/Whisper hints, and invalidates older transcription caches
- SpeechBrain speaker-model downloads now stay under `ai_series_project/runtime/models/speechbrain/ecapa` instead of creating a separate root-level `runtime/` folder
- generated story keywords now filter filler/function words such as `halt`, `weswegen`, `eigentlich`, and `irgendwie` instead of treating them as story topics
- episode generation now carries the detected series language into titles, dialogue templates, voice packages, and visual backend prompts instead of forcing German/English defaults
- fresh GitHub clones include the required `ai_series_project/tmp` placeholder so the local test suite can run without manual folder creation
- optional analysis/export tools are now registered as a generation toolkit and run automatically at pre-training, pre-generation, post-story, post-render, post-quality-gate, and post-export phases
- `05_review_unknowns.py` shows known-character quick-assign counts from the actual identity total, including automatically merged/recognized face clusters
- `06_manage_character_relationships.py` is now a main pipeline step after review; it opens a Tk GUI for character groups, relationships, and per-series input groups
- manual relationship data is stored in `ai_series_project/characters/relationships.json` and is fed into the trained series model, episode prompts, and series bible
- `18_quality_gate.py` keeps running regeneration/rerender cycles until the gate passes or the configured retry-cycle limit is reached
- final output quality now depends on the real local backend stack being ready: SDXL/diffusers for frames, LTX/diffusers for video, XTTS for cloned dialogue, and Wav2Lip for lip-sync

## Project Layout

Run the numbered scripts from the repository root.

### Repository Root

- `00_prepare_runtime.py` to `24_process_next_episode.py`: numbered pipeline and orchestration scripts
- `README.md`: public root overview for the whole project
- `ai_series_project/`: project-internal data, config, runtime, support code, tests, tools, training, and outputs

### Main Folders Inside `ai_series_project/`

- `characters/`: maps, relationships, previews, review queues, voice samples, voice models
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
- `06_manage_character_relationships.py`: Tk GUI for character groups, relationships, and multiple source inputs
- `07_build_dataset.py`
- `08_train_series_model.py`
- `09_prepare_foundation_training.py`
- `10_train_foundation_models.py`
- `11_train_adapter_models.py`
- `12_train_fine_tune_models.py`
- `13_run_backend_finetunes.py`
- `14_generate_episode.py`
- `15_generate_storyboard_assets.py`
- `16_run_storyboard_backend.py`
- `17_render_episode.py`
- `18_quality_gate.py`
- `19_regenerate_weak_scenes.py`
- `20_build_series_bible.py`
- `21_export_package.py`
- `22_refresh_after_manual_review.py`: rebuild from reviewed data after manual review changes
- `23_generate_finished_episodes.py`: batch finished-episode generation
- `24_process_next_episode.py`: full inbox-to-finished-episode orchestrator

### Support Scripts Inside `ai_series_project/support_scripts/`

- `support_scripts/pipeline_common.py`: shared path handling, runtime helpers, orchestration, leases, quality gating
- `support_scripts/console_colors.py`: console formatting
- `support_scripts/configure_quality_backends.py`: helper called by `00` to write portable quality-backend config
- `support_scripts/prepare_quality_backends.py`: helper called by `00` to download/update project-local backend assets
- `support_scripts/manage_character_relationships.py`: helper for manual character groups, relationships, and multiple source-series inputs
- `support_scripts/backend_preset_benchmark.py`: benchmark helper for backend presets
- `support_scripts/optional_tools/`: non-main optional analysis, archive, subtitle, trailer, recap, backup/restore, and export helpers that should not be confused with the required pipeline order

## Pipeline Order

### Full Setup First

`00_prepare_runtime.py` must come first. It now handles:

- project structure creation
- generation of `ai_series_project/configs/project.json` from `ai_series_project/configs/project.template.json` when missing
- runtime environment setup
- project-local FFmpeg staging into `ai_series_project/runtime/host_runtime/ffmpeg/bin`
- quality-backend config writing
- project-local backend tool/model downloads
- download completeness and revision checks

### Main Production Chain

If your review data is already ready, the intended main order is:

`00 -> 06 -> 07 -> 08 -> 09 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16 -> 17 -> 18 -> (19 if weak scenes are queued) -> 20 -> 21`

That means:

- downloads and setup happen in `00`
- character groups and relationships are reviewed in `06` after faces/speakers have names
- dataset building and all training stages happen after review and relationship setup
- backend fine-tune runs happen before new episode generation
- storyboard backend materialization happens before render
- render, quality gate, conditional regeneration, bible, and export happen only after generation
- `19_regenerate_weak_scenes.py` is not a mandatory normal step; `18_quality_gate.py` calls it automatically when the quality gate queues weak scenes and auto-retry is enabled

### Full Inbox Pipeline

For a raw new episode source, the full order is:

`00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16 -> 17 -> 18 -> (19 if weak scenes are queued) -> 20 -> 21`

`24_process_next_episode.py` runs that complete chain and starts with `00` automatically.

`22_refresh_after_manual_review.py` and `23_generate_finished_episodes.py` also start with `00` automatically before their own main work.

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

If optional runtime packages such as `face_recognition/facenet-pytorch`, `speechbrain`, or `TTS` cannot be installed, `00_prepare_runtime.py` reports `ready with limitations` instead of pretending the full stack is complete. In that case, later steps can still run, but face linking, speaker embedding quality, or higher-quality voice cloning may stay reduced until the missing package installs cleanly.

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
python 06_manage_character_relationships.py
```

Or full orchestration:

```powershell
python 24_process_next_episode.py
```

### 4. Generate A Finished Episode From Reviewed Data

```powershell
python 23_generate_finished_episodes.py --count 1
```

### 5. Rebuild After Manual Review

```powershell
python 22_refresh_after_manual_review.py
```

## Quality-First Mode

The project is configured for quality-first finished-episode generation:

- release mode enabled
- original-line reuse enabled
- system TTS fallback disabled in the quality-first path
- lip-sync expected
- external backend runner hooks configured through portable project-local wrappers
- `00_prepare_runtime.py` writes real local backend commands for SDXL image generation, LTX video generation, XTTS voice cloning, and Wav2Lip lip-sync
- the old `project_local_*` still-frame/mux scripts are treated as diagnostics and are rejected by the quality-first guard
- the XTTS voice runner now clones from character reference audio and does not use system TTS in quality-first mode
- `cloning.force_voice_cloning` is enabled by default, so quality-first dialogue lines must be synthesized as cloned speech instead of copied from original episode audio
- XTTS still needs `xtts_license_accepted = true` in the local `configs/project.json` and usable character reference audio
- render-time dialogue planning now falls back directly to `characters/voice_models/<character>_voice_model.json` when `voice_map` is stale or missing a named speaker entry
- delegated quality-backend runners now resolve project-local backend scripts from the project root even when scene packages run from nested working directories
- project-local quality backends now prefer the platform-correct FFmpeg binary from `ai_series_project/runtime/host_runtime/ffmpeg/bin` before falling back to older tool copies
- render-time scene duration now respects the planned per-scene runtime from episode generation instead of compressing most scenes into a short 8 to 22 second window
- batch and quality-gate output messages now include the generated display title, for example `Folge 19: ... (folge_19)`, instead of relying only on the technical episode ID
- the quality score can reach `100%` only when generated scene video/lip-sync, cloned dialogue audio, scene mastering, and style/continuity support are all complete
- `16_run_storyboard_backend.py` now fails if `external_backends.storyboard_scene_runner` does not create a real frame; it no longer creates blue/filter-style seed-frame stand-ins by default
- `17_render_episode.py` no longer writes local motion fallback clips into the same production paths used by the real video backend, so backend runners are not skipped by pre-existing fake outputs
- `17_render_episode.py` now defers quality-first audio to `finished_episode_voice_runner`; local preview TTS/silence is not used as the final voice path
- final delivery is blocked when required generated scene video, cloned dialogue audio, lip-sync output, or the master runner is missing
- quality-gate auto-retry now loops through weak-scene regeneration and forced rerender cycles until the release gate passes, bounded by `release_mode.max_auto_retry_cycles`
- generation prompts now include source-series style locks, auto-detected dialogue language, and stronger negative prompts against blue placeholders, filtered still-frame slideshows, and generic cartoon output

Important:

- if the real backend path is not ready, the pipeline should fail early instead of pretending a fallback render is a final TV-quality episode
- `24_process_next_episode.py` refreshes `00_prepare_runtime.py` automatically when an old autosave/config still points at rejected fallback backends
- Wav2Lip still requires a real checkpoint file such as `wav2lip_gan.pth` under `ai_series_project/tools/quality_models/lipsync/` or via `SERIES_WAV2LIP_CHECKPOINT`
- the diagnostic fallback backends are useful for plumbing and iteration, but they are still not equal to real original-episode-quality image/video/lip-sync generation
- if you intentionally want the old fallback behavior for debugging, set `release_mode.allow_project_local_fallback_backends` locally, but do not use that for final episode quality checks

## Configuration

Main config:

- `ai_series_project/configs/project.template.json`: tracked GitHub baseline
- `ai_series_project/configs/project.json`: local working copy created by `00_prepare_runtime.py`

Recommended workflow:

- commit reusable defaults to `project.template.json`
- let each machine or series create its own local `project.json`
- avoid treating `project.json` as the public baseline for the repository

Important areas:

- `paths.*`: project folders
- `paths.character_relationships`: portable manual relationship file, defaulting to `characters/relationships.json`
- `runtime.*`: device, FFmpeg GPU preference, torch index URL
- `distributed.*`: NAS/shared-worker lease timing
- `generation_toolkit.*`: automatic use of optional analysis, continuity, pacing, subtitle, metadata, review, and export helper tools during episode generation
- `transcription.language`: keep `auto` for new series unless you intentionally want to force a specific source language
- `transcription.auto_language_forced_probe`: enabled by default so language detection compares candidate transcriptions instead of trusting one Whisper hint
- `generation.allow_fallbacks`: must stay `false` for final generation
- `storyboard_backend.*`: controls whether local storyboard seed-frame fallbacks are allowed; quality-first defaults keep them disabled
- `render.allow_local_motion_fallback` and `render.require_*`: keep production output paths reserved for real generated video, voice clone, and lip-sync outputs
- `foundation_training.*`, `adapter_training.*`, `fine_tune_training.*`, `backend_fine_tune.*`
- `external_backends.*`: runner templates and project-local backend commands
- `external_backends.*.environment`: set real generation commands for storyboard/image/video/lip-sync; fallback scripts such as `project_local_video_backend.py` are blocked by default in quality-first mode
- `release_mode.*`: quality gate thresholds and retry behavior, including `retry_until_pass`, `max_auto_retry_cycles`, and full-rerender retries when no weak-scene queue remains
- `quality_backend_assets.*`: project-local backend tool/model targets

### Character Groups And Relationships

Use `06_manage_character_relationships.py` when a training project contains multiple character groups or multiple source-series inputs. It writes `ai_series_project/characters/relationships.json`, which is portable and contains no absolute paths by default.

Examples:

```powershell
python 06_manage_character_relationships.py
python 06_manage_character_relationships.py --set-group game_shakers --label "Game Shakers" --characters "Babe,Kenzie,Hudson,Triple G" --description "Main app team"
python 06_manage_character_relationships.py --add-relationship Babe Kenzie --type "best friends" --group game_shakers --tone "fast loyal banter" --story-rule "Keep their teamwork central"
python 06_manage_character_relationships.py --series-input game_shakers_s01 --label "Game Shakers Season 1" --default-group game_shakers --episode-glob "data/inbox/episodes/game_shakers/*.mp4"
python 06_manage_character_relationships.py --list
```

Running `06_manage_character_relationships.py` without arguments opens a Tk window similar to the character review flow: check known names, save the checked names as a group, select a relationship type, and add relationships for the checked pair or selected set.

During `08_train_series_model.py`, these entries are copied into `generation/model/series_model.json`. `14_generate_episode.py` then uses the selected group and relationship context for focus-character selection, scene prompts, dialogue planning metadata, and `20_build_series_bible.py`.

For a specific generated run, set one of these local config values in `ai_series_project/configs/project.json`:

- `generation.active_character_group`: choose a group id directly
- `generation.active_series_input`: choose a source-series input whose `default_group` should drive focus characters

## Testing

Run the main regression suite from the repository root:

```powershell
python -m unittest discover -s ai_series_project\tests -v
```

Useful smoke checks:

```powershell
python -m py_compile 00_prepare_runtime.py 06_manage_character_relationships.py 22_refresh_after_manual_review.py 23_generate_finished_episodes.py 24_process_next_episode.py ai_series_project\support_scripts\pipeline_common.py ai_series_project\support_scripts\generation_toolkit.py ai_series_project\support_scripts\configure_quality_backends.py ai_series_project\support_scripts\prepare_quality_backends.py ai_series_project\support_scripts\manage_character_relationships.py
```

## Known Limitations

- project-local fallback image/video generation still does not equal strong dedicated TV-quality generation backends
- project-local lip-sync is still weaker than a full dedicated production lip-sync stack
- project-local XTTS voice cloning still depends on clean speaker mapping and real reference segments; speakers with zero linked voice data still cannot clone correctly
- when backend scene-video or scene-audio generation still fails in quality-first mode, rendering stops with explicit missing-output details instead of exporting a fake final episode
- if external runners fail repeatedly, the quality gate will keep rejecting the episode even when the render technically finishes
- shared NAS runs still depend on file-system stability and can be slower for large backend/model downloads
- large Hugging Face model downloads are more reliable with authentication via `HF_TOKEN`

## Finished

- the numbered main scripts now live directly in the repository root and are continuous with no gaps: `00` through `24`
- `06_manage_character_relationships.py` is now the official relationship/group review step between character review and dataset building
- `ai_series_project/` now contains the project internals only: configs, data, runtime state, tests, support scripts, backend tools, training artifacts, and generated outputs
- `00_prepare_runtime.py` now owns the normal setup flow completely, including folder creation, backend config, and project-local downloads
- the public repo now uses `project.template.json` as the tracked baseline, while `project.json` is generated locally and ignored
- `support_scripts/configure_quality_backends.py` and `support_scripts/prepare_quality_backends.py` remain available as internal helpers, but they are no longer part of the numbered main sequence
- project-local FFmpeg now comes from the Python runtime path instead of a separate external FFmpeg download
- `22_refresh_after_manual_review.py`, `23_generate_finished_episodes.py`, and `24_process_next_episode.py` now begin with `00_prepare_runtime.py`
- the documented order is now setup/downloads first, then review, relationships, dataset/training, backend fine-tunes, then generate/render/gate/export
- the orchestrators now export `21_export_package.py` after `20_build_series_bible.py` so finished packages are the final pipeline artifact
- release-gate auto-retry now calls `19_regenerate_weak_scenes.py` from the repository root layout correctly after the root/`ai_series_project` split
- `23_generate_finished_episodes.py` and `24_process_next_episode.py` now call the generation toolkit so optional tools actively feed continuity, voice, pacing, subtitle, review, metadata, and export quality signals into finished-episode generation
- `03_diarize_and_transcribe.py` now stores Whisper and SpeechBrain model data only inside `ai_series_project/` and refreshes stale language caches with process version `12`
- quality-first render now blocks local seed-frame, local motion, system-TTS, and silent fallback paths from being labeled as finished generated episodes
- forced voice cloning ignores stale per-line audio files and regenerates dialogue through XTTS/voice-clone output instead of reusing old TTS artifacts

## In Progress

- improving the real external image/video backend path so generated scenes replace still-frame composites completely
- improving project-local XTTS voice cloning coverage so scenes have natural speech from cloned character voices
- improving render-time diagnostics so missing backend outputs point directly to the failing runner logs and expected target files
- improving the project-local lip-sync path beyond simple fallback mux behavior
- reducing external backend task failures so the quality gate can pass more often with real generated assets
- converting more toolkit outputs from passive reports into direct prompt, pacing, and regeneration inputs

## Planned

- stronger automatic worker-capability routing for NAS multi-PC runs
- better scene-selection logic for regeneration retries after quality-gate failures
- stronger real local XTTS/lip-sync runtime integration when the backend assets are fully available
- higher-quality project-local image/video backends to move closer to original-episode visual consistency
