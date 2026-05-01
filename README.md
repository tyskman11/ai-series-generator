# AI Series Training

## Table Of Contents

- [Purpose](#purpose)
- [At A Glance](#at-a-glance)
- [What Already Works Well](#what-already-works-well)
- [Current Focus](#current-focus)
- [In Progress](#in-progress)
- [Finished](#finished)
- [Planned](#planned)
- [Documentation Rule](#documentation-rule)
- [Project Layout](#project-layout)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Workflow Summary](#workflow-summary)
- [Testing And Smoke Runs](#testing-and-smoke-runs)
- [Known Limitations](#known-limitations)
- [Typical Daily Use](#typical-daily-use)

## Purpose

This project turns existing TV episodes into a local AI training and episode-generation pipeline.

Starting from real source episodes, the pipeline can:

- import and split episodes into scenes
- transcribe dialogue and cluster speakers
- detect faces and link recurring characters to speakers
- let multiple PCs cooperate on the same NAS-backed workspace through lease files
- build reviewed datasets and a local heuristic series model
- prepare and train local foundation, adapter, fine-tune, and backend-prep artifacts
- train per-character local voice profiles from original dialogue material
- generate new episode blueprints and shotlists
- create storyboard assets and backend-ready scene payloads
- render locally finished episodes with dialogue audio, scene masters, and delivery bundles
- expose backend-ready production packages for later image, video, voice, and lip-sync runners

The default path stays local-first and license-light. The project already produces a finished local episode bundle, but the long-term target remains higher-quality fully generated TV-style episodes.

## At A Glance

| Area | Status | Summary |
| --- | --- | --- |
| Import / split / transcription | usable | Source episodes can be imported, split, transcribed, and speaker-clustered in batch mode. |
| Face / speaker linking | actively tuned | Main characters are prioritized, and review load is reduced with auto-resolution helpers. |
| Dataset / model | usable | Reviewed material can be rebuilt into a dataset and heuristic series model. |
| Training stack | usable | Foundation, adapter, fine-tune, and backend-prep stages exist as explicit local steps. |
| Episode generation | usable | New synthetic episodes, shotlists, storyboard assets, and final local renders can be produced. |
| Finished-episode path | quality-first | Production packages, delivery bundles, backend hooks, and quality scoring are in place, and the batch path now rejects obvious fallback-quality episodes instead of silently passing them through. |

## What Already Works Well

- full local preprocessing from inbox episode to linked scene data
- NAS/shared-worker leases across the numbered pipeline, including recovery when one worker disappears
- generated JSON metadata now stores project-local relative paths and can rebase legacy absolute paths after moving the workspace
- automatic language detection during transcription, with filename language hints preferred when present
- per-character voice training preparation from original dialogue recordings instead of one generic online voice base
- local heuristic series model generation plus synthetic episode blueprints
- storyboard seed assets and backend-ready scene request payloads
- optional external runner hooks for storyboard, image, video, voice, lip-sync, and episode-master stages
- scene generation now carries series-style guidance plus remembered character continuity traits through prompts, storyboard backend payloads, and render packages
- per-scene production packages and one package-level full generated-episode contract
- dialogue-aware final audio assembly with original-segment reuse, loudness normalization, trim/pad timing, and short fades
- local multi-shot fallback scene videos when dedicated generated scene video is still missing
- per-scene mastered clips and package-level `*_full_generated_episode.mp4` masters
- dedicated delivery bundles under `generation/renders/deliveries/<episode>` plus stable `generation/renders/deliveries/latest`
- production-readiness summaries, backend-runner summaries, and heuristic scene/episode quality scoring in orchestration outputs and the series bible
- exportable external handoff packages under `exports/packages/<episode>/<format>`, now including render profile plus release-gate and regeneration metadata
- release-style quality-gate reports with regeneration queues, preserved retry metadata across rerenders, and enforced gating in the main finished-episode batch path

## Current Focus

- finish main-character review quality in `06_review_unknowns.py`
- keep the default path stable for large NAS-backed runs
- improve the new project-local quality backend defaults so they move closer to real original-episode output
- keep backend tools and model downloads strictly project-local instead of depending on external folders
- keep the generation chain clearly ordered as train first, then generate/render
- raise external-runner quality so the fully generated episode path can become the real default

## In Progress

- `05_link_faces_and_speakers.py` now opens assignment previews more robustly from NAS/UNC paths by preferring a self-contained browser preview plus exact image links before falling back to the montage JPG
- `06_review_unknowns.py` keeps reducing open manual review work through known-face matching, iterative naming propagation, and conservative `statist` auto-marking
- `06_review_unknowns.py` now rebases stored NAS preview paths, mirrors the active review images into a local temp preview cache for GUI/local-image opening, and avoids auto-opening multiple preview windows for one case
- `05_link_faces_and_speakers.py`, `06_review_unknowns.py`, and `pipeline_common.py` now surface shared interactive display diagnostics so desktop, NAS, and headless-session behavior is easier to see before manual review starts
- shared NAS lease handling now auto-recovers same-host stale worker locks when the recorded PID is no longer alive
- `05_link_faces_and_speakers.py`, `06_review_unknowns.py`, and the shared path helpers now normalize stored project paths back to relative metadata so the same workspace can move between Windows, Linux, NAS mounts, and different drive letters more safely
- `04_diarize_and_transcribe.py` keeps extending `speaker_unknown` rescue logic and language handling
- `99_process_next_episode.py` is being hardened for long resumable inbox runs with autosaves and live status files
- `99_process_next_episode.py` now consistently uses the shared project root helpers for batch-job bookkeeping, avoiding resume crashes in the completed-episode handoff path
- `13_generate_episode.py` writes multi-reference storyboard plans and backend-ready request exports
- backend fine-tune freshness checks now accept newer per-character backend run files even when the shared summary file is older, which avoids false stale-state blocks on NAS and mixed-OS workspaces
- `08_train_series_model.py`, `13_generate_episode.py`, `14_generate_storyboard_assets.py`, and `15_render_episode.py` now carry style constraints, remembered character continuity hints, and explicit quality targets deeper into the generated-episode path
- `09_prepare_foundation_training.py` now finishes cleanly when no candidates are found and rebuilds its final plan from all manifest files, so NAS/shared-worker runs no longer publish a partial local subset as the final result
- `09_prepare_foundation_training.py` now also ignores invalid directory-like voice reference paths such as `.` in stored metadata, so Linux/NAS runs do not crash while copying foundation voice samples
- `14_generate_storyboard_assets.py` emits backend-ready scene input payloads and rebases moved artifact paths
- `54_run_storyboard_backend.py` can materialize local scene packs and optionally call configured external storyboard runners first
- `15_render_episode.py` now also preserves camera/control hint dictionaries correctly instead of dropping them in the render/package path, then reuses backend frames/clips when present, assembles voiced scene masters, writes delivery bundles, and keeps improving the final generated-episode package
- `15_render_episode.py` now tolerates missing `scene_dialogue_outputs` keys in audio fallback metadata, so draft/final render does not crash on Linux/NAS when dialogue audio generation partially falls back
- `15_render_episode.py` is being tightened so draft storyboard cards stay preview-only artifacts while the finished-episode fallback render uses clean scene frames instead of text-labeled debug cards
- `54_run_storyboard_backend.py` now ignores directory placeholders such as `.` when choosing source images, so Linux/NAS runs do not crash if moved or incomplete metadata leaves a directory-like reference behind
- `57_generate_finished_episodes.py` now imports the shared logging helper correctly, so shared-worker batch generation no longer aborts immediately on startup
- `51_export_package.py` now exports real generated-episode packages for JSON, DaVinci-style, and Premiere-style handoff folders, including resolved render profile plus release/delivery/regeneration metadata
- `52_quality_gate.py` now writes persistent quality-gate reports, regeneration queues, and feeds that state back into episode artifacts
- `53_regenerate_weak_scenes.py` turns quality-gate queues into retry manifests and can rerun the current full-episode retry chain while preserving scene retry state; storyboard backend stage now supports `--scene-ids` for scene-selective reruns
- `52_quality_gate.py` and `53_regenerate_weak_scenes.py` now ignore empty stored artifact paths before resolving reports, packages, or delivery folders, so missing metadata cannot accidentally point at the current working directory
- `52_quality_gate.py` now supports `--auto-retry` plus `release_mode.auto_retry_*` config so failed gates can launch one automatic retry loop, while `53_regenerate_weak_scenes.py` suppresses nested auto-retries both during its gate refresh and inside the inner rerun
- `52_quality_gate.py` now also accepts the shared NAS worker CLI flags passed through by `57_generate_finished_episodes.py`, so release-mode batch runs do not fail just because internal worker arguments are forwarded automatically
- `52_quality_gate.py` no longer warns when `production_readiness` is already `fully_generated_episode_ready`; only genuinely weaker readiness states still show that warning
- `53_regenerate_weak_scenes.py` now treats exit code `1` from `52_quality_gate.py` as a valid failed-gate result during retry runs instead of aborting the whole regeneration process as if the script had crashed
- `15_render_episode.py`, `49_refresh_after_manual_review.py`, `57_generate_finished_episodes.py`, and `99_process_next_episode.py` now prepare project-local quality backend assets automatically and then stop early with a clear quality-first preflight error if release mode, original-line reuse, lipsync, or the required runners are still not ready
- `58_configure_quality_backends.py` now writes portable `{python}`-based runner templates plus project-local default backend commands into `project.json`, so Linux does not depend on a `python` PATH alias or manually exported shell variables
- `59_prepare_quality_backends.py` now prepares backend tools and model assets only inside the project folder, automatically reruns itself inside the project runtime on Windows, stages Hugging Face downloads locally before copying them into UNC/NAS project folders on Windows, disables `hf_xet` and downloads Hugging Face snapshots serially for more stable Windows/NAS model fetches, forces Git ownership checks open for this personal multi-device NAS setup, rebuilds corrupted project-local Git checkouts from a project-local GitHub ZIP staging folder and then moves them into place, uses the active runtime interpreter for package installs, tracks revisions, verifies required files, detects incomplete Hugging Face downloads, skips re-downloads when the newest local revision is already complete, and falls back to GitHub API plus ZIP downloads when Linux/NAS workers do not have `git` installed
- `pipeline_common.py` now checks configured runner prerequisites such as required commands, Python modules, and environment variables before the quality-first path is allowed to start
- the tracked regression suite covers export handoffs, release-gate reports, retry queues, strict warning handling, and regeneration metadata so those finished-episode contracts stay guarded
- `16_build_series_bible.py`, `57_generate_finished_episodes.py`, and `99_process_next_episode.py` surface readiness, backend coverage, and quality scoring for generated episodes
- `49_refresh_after_manual_review.py` and `57_generate_finished_episodes.py` now follow the real train-then-generate/render order against the current script names
- `49_refresh_after_manual_review.py`, `57_generate_finished_episodes.py`, and `99_process_next_episode.py` can now run `52_quality_gate.py` automatically after render when `release_mode.enabled` is active
- `50_run_backend_finetunes.py` remains the backend-prep bridge between `12_train_fine_tune_models.py` and the actual generation/render path
- real fully generated image/video/lip-sync quality is still active work even though the orchestration and package plumbing already exist
- `backend_preset_benchmark.py` provides a ranked comparison of backend runner presets with composite scoring and test-scene evaluation
- worker capability helpers exist in `pipeline_common.py`; full orchestration integration for GPU-heavy NAS scheduling is still being connected

## Finished

These items are implemented and should stay guarded by README updates and tests when they change.

- script numbering is numeric only, with training before generated-episode render in the documented orchestration paths
- OS-specific FFmpeg detection and runtime setup are in place for Windows and Linux/NAS runs
- `57_generate_finished_episodes.py` is the finished-episode entry point and defaults to one episode unless `--count 0` or `--endless` is used
- the default project config now targets `original_episode_quality_first`: release mode on, strict thresholds, original-line reuse on, system TTS fallback off, and lipsync on
- generated project metadata uses portable relative paths for previews, review queues, linked speaker reference frames, and render handoff fields while legacy absolute paths are still accepted on read
- the shared file-opening helper now uses stronger Windows/NAS fallbacks so interactive preview files are more likely to open even from UNC shares
- shared interactive display diagnostics now warn clearly when review is started from a headless or non-desktop session
- render-side camera/control hints no longer get lost when `camera_plan` and `control_hints` are stored as dictionaries in generated scene plans
- foundation-training plan generation now aggregates all existing manifests before writing the final plan in shared NAS runs
- backend fine-tune summaries are rebuilt from all existing per-character run files, and stale summary timestamps no longer block generation when the actual backend runs are newer
- render audio fallback metadata is now normalized defensively so missing scene-dialogue maps no longer abort the final encode phase
- finished-episode fallback clips no longer reuse the text-labeled storyboard preview card as their visual source, so local fallback renders stay visually cleaner
- finished-episode batch startup no longer crashes in shared-worker mode because the shared `info(...)` logger import is present again
- `57_generate_finished_episodes.py` now always runs `52_quality_gate.py` and refuses to accept placeholder-heavy, local-motion-fallback, or `pyttsx3` fallback episodes as finished output
- the resumable `99_process_next_episode.py` batch handoff path now resolves batch jobs through the shared project root import instead of crashing during resume
- foundation-training voice sample preparation now skips invalid directory paths instead of crashing on Linux/NAS when old metadata points at `.`
- storyboard backend scene materialization now skips directory placeholders like `.` instead of trying to open them as images on Linux/NAS
- `tools/quality_backends/master_runner.py` provides a built-in FFmpeg-based episode master runner that now prefers the project-local FFmpeg binary instead of requiring a global installation
- backend tool/model preparation now has a dedicated numbered step and stores project-local summaries under `ai_series_project/tools/quality_backends`
- quality-first runner templates now resolve the active interpreter automatically and no longer rely on `python` being present as a shell command on Linux/NAS systems
- release-gate, export-package, regeneration-queue, backend-benchmark, review-preview, display-diagnostics, and generation-quality behavior have tracked regression tests
- release-gate CLI handling now tolerates shared-worker passthrough arguments from the finished-episode batch runner
- regeneration reruns now tolerate a still-failing quality gate result after rerender and only stop on actual `52_quality_gate.py` execution errors

## Planned

Only untouched follow-up work belongs here. If implementation has already started, it belongs in `In Progress`.

- tighter worker-capability routing so GPU-heavy generation stages can prefer stronger NAS workers automatically
- stronger backend-side scene selection so regeneration can choose better alternates automatically after quality-gate failures
- real project-local XTTS and lip-sync runtime integration instead of metadata-only records for those backend assets
- stronger project-local voice generation so the default voice backend does not have to rely on already materialized line audio

## Documentation Rule

This file is mandatory documentation.

Whenever you change any of the following, update `README.md` in the same task:

- `00_prepare_runtime.py` through `59_prepare_quality_backends.py`
- `99_process_next_episode.py`
- `pipeline_common.py`
- `ai_series_project/configs/project.json`
- CLI options
- folder structure
- output formats
- environment variables
- known limitations or workarounds

Also keep the `In Progress` and `Planned` sections current.

## Project Layout

### Core Numbered Scripts

- `00_prepare_runtime.py` to `06_review_unknowns.py`: import, split, transcription, linking, and review
- `07_build_dataset.py` to `12_train_fine_tune_models.py`: dataset and training chain
- `13_generate_episode.py`: generate a new synthetic episode blueprint and shotlist
- `14_generate_storyboard_assets.py`: build storyboard seed assets and scene backend payloads
- `15_render_episode.py`: render draft/final episodes and write full generated-episode packages
- `16_build_series_bible.py`: rebuild the current bible and generated-episode snapshot
- `17_analyze_patterns.py` to `48_social_media_clips.py`: analysis, utility, and content-export helpers
- `49_refresh_after_manual_review.py`: one-command rebuild after heavy review work
- `50_run_backend_finetunes.py`: backend-oriented fine-tune/materialization bridge after `12`
- `51_export_package.py`: export generated-episode bundles for external tools with render-profile and release-gate handoff metadata
- `52_quality_gate.py`: evaluate finished episodes against release-style thresholds and write regeneration hints
- `53_regenerate_weak_scenes.py` to `56_restore_project.py`: regeneration, backup, and maintenance helpers
- `57_generate_finished_episodes.py`: batch or endless finished-episode generation
- `58_configure_quality_backends.py`: write portable quality-first runner templates into the project config
- `59_prepare_quality_backends.py`: download/update project-local backend tools and model assets with revision checks, including GitHub ZIP fallback when `git` is missing and Xet-disabled Hugging Face downloads for Windows/NAS stability
- `99_process_next_episode.py`: full end-to-end coordinator
- `backend_preset_benchmark.py`: compare backend runner presets and produce a ranked recommendation report

### Important Folders

- `ai_series_project/data/inbox/episodes`: new source episode files
- `ai_series_project/data/raw/episodes`: imported working copies
- `ai_series_project/data/processed/scene_clips`: split scene clips
- `ai_series_project/data/processed/speaker_transcripts`: merged transcript outputs
- `ai_series_project/data/processed/linked_segments`: linked face/speaker/scene data
- `ai_series_project/characters/maps`: shared character and voice maps
- `ai_series_project/characters/review`: review queue artifacts
- `ai_series_project/characters/voice_models`: local per-character voice profiles
- `ai_series_project/generation/model`: trained local series model
- `ai_series_project/generation/story_prompts`: generated markdown episodes
- `ai_series_project/generation/shotlists`: generated shotlists
- `ai_series_project/generation/storyboard_requests`: backend-ready storyboard request payloads
- `ai_series_project/generation/storyboard_assets`: scene seed frames and backend assets
- `ai_series_project/generation/final_episode_packages/<episode>`: full generated-episode package with per-scene targets and outputs
- `ai_series_project/generation/renders/deliveries/<episode>`: final handoff bundle per episode
- `ai_series_project/generation/renders/deliveries/latest`: stable latest-finished-episode handoff folder
- `ai_series_project/exports/packages/<episode>/<format>`: export packages for external editing or handoff tools
- `ai_series_project/runtime/autosaves`: autosaves and resumable run state
- `ai_series_project/runtime/distributed`: NAS/shared-worker lease files
- `ai_series_project/tools/ffmpeg/bin`: OS-specific FFmpeg binaries
- `tests`: tracked unit tests for export packages, quality gates, and regeneration/retry behavior

## Requirements

- Windows/PowerShell or Linux with `python3`
- enough disk space for scenes, audio, models, and render outputs
- patience: transcription, linking, training, and render steps can take a long time
- for NAS/shared-worker mode, all participating PCs must work on the same shared project contents and the same `ai_series_project/runtime/distributed` lease files; generated metadata is portable across different local mount points or drive letters
- optional NVIDIA GPU for faster transcription, embeddings, and rendering

`00_prepare_runtime.py` installs the runtime. On Linux/NAS it uses the active `python3` runtime directly with `python3 -m pip install --break-system-packages`. It also downloads the matching FFmpeg build for the current OS into `ai_series_project/tools/ffmpeg/bin`.

`ai_series_project/configs/project.json` now also exposes:

- `paths.export_packages`: root for `51_export_package.py`
- `paths.quality_backend_tools`: project-local root for downloaded backend tools such as ComfyUI
- `paths.quality_backend_models`: project-local root for backend model snapshots
- `paths.quality_backend_asset_summary`: summary JSON written by `59_prepare_quality_backends.py`
- `release_mode.*`: optional release thresholds and strictness for `52_quality_gate.py` plus the main orchestrators
- `release_mode.max_regeneration_retries`: cap for how often one weak scene may stay in the retry queue before `53_regenerate_weak_scenes.py` stops requesting another rerender
- `release_mode.auto_retry_failed_gate`: optionally lets `52_quality_gate.py` launch one automatic retry loop after a failed gate
- `release_mode.auto_retry_update_bible`: optionally appends `16_build_series_bible.py` to that automatic retry loop
- `generation.quality_mode`: labels the current generated-episode prompt/continuity strategy; default is `original_episode_quality_first`

Generated JSON artifacts under `characters`, `data/processed`, `generation`, and related handoff folders now prefer project-relative paths instead of machine-specific absolute paths. Older absolute paths are still rebased automatically when the workspace has been moved.

The default config is now intentionally quality-first:

- `release_mode.enabled = true`
- `release_mode.min_episode_quality = 0.9`
- `release_mode.max_weak_scenes = 0`
- `release_mode.watch_threshold = 0.82`
- `release_mode.strict_warnings = true`
- `release_mode.auto_retry_failed_gate = true`
- `cloning.allow_system_tts_fallback = false`
- `cloning.enable_original_line_reuse = true`
- `cloning.enable_lipsync = true`

The pipeline entry points `15`, `49`, `57`, and `99` now require configured quality runners for:

- `storyboard_scene_runner`
- `finished_episode_image_runner`
- `finished_episode_video_runner`
- `finished_episode_voice_runner`
- `finished_episode_lipsync_runner`
- `finished_episode_master_runner`

If those runners are not enabled with real `command_template` values, the quality-first path stops immediately with a clear configuration error instead of generating a fake “final” episode. `58_configure_quality_backends.py` now writes portable defaults for those runner entries automatically.

## Quick Start

1. Drop new source episodes into `ai_series_project/data/inbox/episodes`.
2. Run `python 00_prepare_runtime.py`
3. Run `python 01_setup_project.py`
4. Run `python 58_configure_quality_backends.py`
5. Run `python 59_prepare_quality_backends.py`
6. Run `python 99_process_next_episode.py --skip-downloads`
7. Review outputs in:
   - `ai_series_project/generation/story_prompts`
   - `ai_series_project/generation/shotlists`
   - `ai_series_project/generation/final_episode_packages`
   - `ai_series_project/generation/renders/deliveries/latest`
   - `ai_series_project/series_bible/episode_summaries`

If you already have reviewed and trained data and want one finished local episode directly:

```powershell
python 57_generate_finished_episodes.py --count 1 --skip-downloads
```

## Workflow Summary

Important: the logical run order is not strictly numeric anymore. Training must finish before final episode generation/render starts. The real finished-episode path is:

`07 -> 08 -> 09 -> 10 -> 11 -> 12 -> 50 -> 13 -> 14 -> 54 -> 15 -> 16`

The finished-episode batch path now always includes the quality gate:

`07 -> 08 -> 09 -> 10 -> 11 -> 12 -> 50 -> 13 -> 14 -> 54 -> 15 -> 52 -> 16`

Project-local backend preparation belongs before the quality-first render path, and the main entry scripts now re-run step `59` automatically:

`58 -> 59 -> 07 -> 08 -> 09 -> 10 -> 11 -> 12 -> 50 -> 13 -> 14 -> 54 -> 15 -> 52 -> 16`

If `release_mode.auto_retry_failed_gate` is also turned on and weak scenes are queued, the gate can append one retry loop:

`52 -> 53 -> 54(scene-selective when possible) -> 15 -> 52`

### 00 - Prepare Runtime

Prepares Python dependencies, installs Torch in the correct order for Linux/NAS reliability, and downloads the matching FFmpeg binary for the current OS.

### 01 - Set Up Project

Creates the expected folder structure and config.

### 02 - Import Episode

Imports the next unprocessed source episode into the raw workspace and removes the inbox copy after verification.

### 03 - Split Scenes

Splits imported episodes into scene clips and writes the scene index.

### 04 - Diarize And Transcribe

Extracts audio, runs Whisper, auto-detects language unless a fixed language is configured, and builds speaker clusters. NAS mode can split one episode across multiple workers at scene level.

### 05 - Link Faces And Speakers

Detects faces, links them to speaker clusters, and applies rescue logic for certain unresolved speaker cases. When interactive assignment is enabled, the script now prints exact preview file paths plus clickable `file://` links, prefers a self-contained HTML/browser preview for NAS or UNC shares, and still keeps the montage JPG as a fallback. The script also prints shared display/session diagnostics first, so it is obvious when a run has no interactive console or no desktop-capable GUI context.

### 06 - Review Unknowns

Interactive review for unknown or weakly linked face clusters. This step tries to match known characters first and can auto-mark safe low-activity background clusters as `statist`. Stored preview paths are rebased for NAS/moved workspaces and persisted back as relative project metadata; the exact selected face JPG files (`*_crop.jpg` first, then matching `*_context.jpg`) are mirrored into a local temp preview cache so the Tk window can launch from a local path even when the project itself lives on a NAS share. Automatic preview opening now prefers the Tk window and falls back to only one local image file instead of spawning multiple viewer windows. The terminal output prints both the original preview files and the local launch targets explicitly, while contact-sheet helpers remain available for manual opening. Before review starts, the script also prints shared display/session diagnostics so headless NAS, SSH, or non-desktop runs are easier to distinguish from real GUI bugs. Use `--no-open-previews` only when you want terminal-only review.

### 07 - Build Dataset

Builds the consolidated training dataset from linked and reviewed data.

### 08 - Train Series Model

Builds the local heuristic series model used for episode generation and series-bible summaries.

### 09 - Prepare Foundation Training

Prepares image, clip, and voice assets for later training stages. This step prioritizes original dialogue recordings and language-aware voice manifests. In shared NAS runs, the final plan is now rebuilt from all written manifest files instead of only the current worker's local subset.

### 10 - Train Foundation Models

Builds local foundation packs and per-character local voice-model profiles.

### 11 - Train Adapter Models

Builds local adapter profiles for image, voice, and clip dynamics.

### 12 - Train Fine-Tune Models

Builds local fine-tune profiles on top of the adapter stage.

### 50 - Run Backend Finetunes

Turns fine-tune profiles into backend-oriented runs/materialized artifacts and acts as the bridge from training into the later generated-episode backend path.

### 51 - Export Package

Builds handoff folders under `exports/packages/<episode>/<format>` from the real production package and can optionally copy the referenced media for external editing tools.

Export manifests now also carry the configured render profile plus delivery, quality-gate, and regeneration-manifest references so external tools or later handoff steps can see the actual release state of the exported episode.

### 52 - Quality Gate

Evaluates the latest or requested generated episode against `release_mode` thresholds, writes a report JSON plus regeneration queue, and updates the recorded episode artifacts with the result. With `--auto-retry` or `release_mode.auto_retry_failed_gate`, it can launch one automatic retry loop and only exits successfully when the refreshed gate actually passes. Internal gate calls from `53_regenerate_weak_scenes.py` use `--no-auto-retry` so config-driven auto-retry cannot recursively retrigger itself. Empty artifact path fields are ignored before report/package/delivery paths are resolved.

### 53 - Regenerate Weak Scenes

Turns the current quality-gate queue into a persistent retry manifest and can optionally apply the current retry loop:

`54 -> 15 -> 52`

Retry state is written back into the production package, scene packages, shotlist, and render manifest so later rerenders keep the same retry counters. When it is launched from `52_quality_gate.py --auto-retry`, the current gate thresholds are forwarded into the retry loop. If you call `53_regenerate_weak_scenes.py` directly with threshold overrides, it refreshes `52_quality_gate.py` first so the queue matches those overrides, but that refresh explicitly disables another automatic retry. Missing production-package metadata now fails explicitly instead of falling through to the current working directory.

### 13 - Generate Episode

Generates a new synthetic episode blueprint and shotlist from the trained local model. It also writes per-scene storyboard plans and backend-ready storyboard request exports. Those scene plans now carry remembered character continuity traits, series-style constraints, and explicit quality-target metadata.

### 14 - Generate Storyboard Assets

Builds scene-level storyboard seed assets and backend input payloads under `generation/storyboard_assets/<episode>`. The backend payloads now keep style constraints, character continuity hints, and quality targets so later external runners see the same guidance as the local pipeline.

### 54 - Run Storyboard Backend

Materializes local backend-style scene packs from the storyboard backend payloads. Each scene can produce graded frames, alternates, posters, and optional short local clips.

### 15 - Render Episode

Builds the draft and final local episode render, assembles dialogue audio, writes per-scene masters, refreshes the production package, and creates the final delivery bundle. Camera/control hint dictionaries are now preserved correctly through this stage instead of silently collapsing to empty lists.

### 16 - Build Series Bible

Rebuilds the compact series bible and appends the latest generated-episode production snapshot.

### 49 - Refresh After Manual Review

One-command rebuild path after significant character-review changes. It reruns the real current chain in the correct order:

`07 -> 08 -> 09 -> 10 -> 11 -> 12 -> 50 -> 13 -> 14 -> 54 -> 15 -> 16`

### 57 - Generate Finished Episodes

Runs the visible finished-episode batch flow. Without `--count`, it generates exactly one episode. `--count 0` or `--endless` keeps going until stopped manually.

### 99 - Full Pipeline

Runs the main end-to-end path from setup/import through review gate, training, backend prep, episode generation, storyboard backend, render, and bible refresh. It writes autosaves and live status files for long runs.

## Testing And Smoke Runs

Run the automated tests:

```powershell
python -m unittest discover -s tests -v
```

The tracked test suite currently focuses on the finished-episode handoff path: export packages, release quality gates, retry queues, strict warnings, and regeneration metadata.

Quick syntax check for all pipeline and test files:

```powershell
python _check_syntax.py
```

`_check_syntax.py` recursively validates every `*.py` file in the project root and the `tests/` directory, excluding `__pycache__`, `runtime/`, `.venv/`, `venv/`, and `ai_series_project/`. It is a fast pre-flight check before running the full test suite or smoke runs.

Recommended smoke run after larger pipeline changes:

```powershell
python 05_link_faces_and_speakers.py
python 07_build_dataset.py
python 08_train_series_model.py
python 09_prepare_foundation_training.py --skip-downloads
python 10_train_foundation_models.py
python 11_train_adapter_models.py
python 12_train_fine_tune_models.py
python 50_run_backend_finetunes.py
python 13_generate_episode.py
python 14_generate_storyboard_assets.py
python 54_run_storyboard_backend.py
python 15_render_episode.py
python 52_quality_gate.py
python 53_regenerate_weak_scenes.py
python 16_build_series_bible.py
```

One-command finished-episode run from already reviewed data:

```powershell
python 57_generate_finished_episodes.py --count 1 --skip-downloads
```

Export the latest generated episode for external tools:

```powershell
python 51_export_package.py --format davinci --copy-media
python 51_export_package.py --format premiere
```

Run the standalone release-style quality gate:

```powershell
python 52_quality_gate.py
python 52_quality_gate.py --strict
python 52_quality_gate.py --auto-retry
```

Write or apply a weak-scene regeneration manifest:

```powershell
python 53_regenerate_weak_scenes.py
python 53_regenerate_weak_scenes.py --apply
```

Benchmark backend runner presets:

```powershell
python backend_preset_benchmark.py
python backend_preset_benchmark.py --preset-file my_presets.json --output reports/benchmark.json
```

Shared NAS transcription example:

```powershell
python 04_diarize_and_transcribe.py --worker-id pc1
python 04_diarize_and_transcribe.py --worker-id pc2
```

Interactive NAS review example:

```powershell
python 06_review_unknowns.py --review-faces --open-previews
```

`--open-previews` is the default now, so `python 06_review_unknowns.py --review-faces` is enough in normal desktop use. Run this from a desktop session on the PC that should show the images. On Windows, the script now mirrors the active preview set into a local temp cache, tries the Tk preview window first, and only falls back to opening one local image file if Tk is not available for that case. The terminal still prints the original source images plus the local launch paths so you can open either manually if needed. Shared display diagnostics are printed before review starts. On headless Linux/NAS or SSH sessions without `DISPLAY`/`WAYLAND_DISPLAY`, the script prints the preview paths and keeps terminal assignment available.

## Known Limitations

- face/speaker quality still depends heavily on review quality in `06_review_unknowns.py`
- the local series model is heuristic, not a large multimodal frontier model
- without strong external image/video/voice/lip-sync backends, the project now prefers to fail the finished-episode gate instead of presenting a low-quality fallback render as a true final TV-style episode
- quality backend tools and model snapshots are now expected inside the project tree; external folders are no longer considered part of the supported setup
- the local finished episode path is structurally complete, but TV-grade visual consistency still depends on real backend generation quality and well-trained character-specific voice models
- by default, direct render/generation entry points now auto-run project-local backend preparation and then refuse to run in “final episode” mode until the required quality runners are configured
- the built-in project-local image/video/lipsync defaults are portable and runnable, but they still fall back to local asset reuse and FFmpeg composition rather than true series-grade model inference
- the current project-local voice backend can only assemble already materialized line audio or reused source audio; full project-local XTTS generation is still unfinished
- fully new dialogue lines still depend on generated speech when no strong original segment can be reused
- lip-sync and generated scene video quality still depend on later backend tuning
- `53_regenerate_weak_scenes.py` now reruns only the flagged scenes in the storyboard backend stage (`54_run_storyboard_backend.py --scene-ids`), but the render step (`15_render_episode.py`) still rebuilds the full episode package
- automatic gate retry now performs at most one extra retry loop per failing run; if the refreshed gate still fails, the pipeline stops and leaves the queue/report for manual follow-up
- orchestration-heavy scripts mainly use exclusive leases; the fine-grained parallelism lives in worker-heavy numbered steps underneath
- interactive image review in `06_review_unknowns.py` needs a local desktop session for embedded Tk windows; Windows review now launches the exact selected face JPG files first, prints those exact paths, and reports session/display diagnostics before the first review case
- `release_mode` now defaults to enabled quality-first mode, so the finished-episode entry points block immediately until the required backend commands and prerequisites are really present
- continuity/style guidance is now propagated much more consistently, but final visual quality still depends heavily on the actual external image/video/lip-sync backends you connect
- test runs may show harmless FFmpeg warnings like `moov atom not found` on stderr; these do not indicate test failures and can be ignored when exit code is 0

## Typical Daily Use

### Process New Source Episodes

```powershell
python 00_prepare_runtime.py
python 01_setup_project.py
python 99_process_next_episode.py --skip-downloads
```

### Generate One New Finished Episode From Reviewed Data

```powershell
python 08_train_series_model.py
python 09_prepare_foundation_training.py --skip-downloads
python 10_train_foundation_models.py
python 11_train_adapter_models.py
python 12_train_fine_tune_models.py
python 50_run_backend_finetunes.py
python 13_generate_episode.py
python 14_generate_storyboard_assets.py
python 54_run_storyboard_backend.py
python 15_render_episode.py
python 52_quality_gate.py
python 16_build_series_bible.py
```

### Rebuild After Manual Character Review

```powershell
python 49_refresh_after_manual_review.py --skip-downloads
```

### Generate Multiple Finished Episodes

```powershell
python 57_generate_finished_episodes.py --count 2 --skip-downloads
```

### Export Or Gate A Finished Episode

```powershell
python 51_export_package.py --format davinci --copy-media
python 52_quality_gate.py
python 53_regenerate_weak_scenes.py --apply
```

Endless mode:

```powershell
python 57_generate_finished_episodes.py --endless --skip-downloads
python 57_generate_finished_episodes.py --count 0 --skip-downloads
```
