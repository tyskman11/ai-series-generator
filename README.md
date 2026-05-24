# AI Series Training

## Table Of Contents

- [Purpose](#purpose)
- [Current Status](#current-status)
- [Development Direction](#development-direction)
- [Project Layout](#project-layout)
- [Pipeline Order](#pipeline-order)
- [Quick Start](#quick-start)
- [Quality-First Mode](#quality-first-mode)
- [Finished Episode Mode](#finished-episode-mode)
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

The project is intentionally strict about the difference between an intermediate preview and a finished synthetic episode. Planning metadata, storyboard packages, fallback motion, or a technically encoded render are useful development artifacts, but they are not release-ready output unless the configured media backends produce the required video, voice, lip-sync, audio-mix, manifest, and quality evidence.

## Current Status

- preprocessing from source episode to reviewed character/speaker data is usable
- the training chain exists and is ordered before generation/rendering
- `00_prepare_runtime.py` now owns normal setup completely, including runtime prep, backend configuration, project-local downloads, model download verification, and folder creation
- `04_link_faces_and_speakers.py` now keeps portable segment-audio and language metadata in linked speaker rows and uses a high-recall face scan by default so group shots and shorter recurring appearances are less likely to be capped away
- `09_prepare_foundation_training.py` now backfills missing voice-reference audio from `speaker_transcripts` for older datasets
- voice assignment is now speech-gated end to end: `03_diarize_and_transcribe.py` labels speech confidence/content type, `04`/`05` only auto-link eligible speech segments, and `09`/`10`/`17` only use clean speech as voice-clone references
- the GitHub repo now treats `ai_series_project/configs/project.template.json` as the tracked base template; the working `project.json` is generated locally
- quality-first generation is enforced more strictly than before
- generated episode runtime now follows the average length of the input episodes instead of collapsing toward short dialogue-only timing
- finished-episode batches now force fresh storyboard-backend and render outputs by default so stale fallback artifacts do not silently pass through
- storyboard backend and render now refuse to treat local seed-frame/color-grade or still-motion fallback clips as production-generated outputs in quality-first mode
- quality-first generation now treats every fallback switch as an error: generated frames, scene video, cloned dialogue audio, lip-sync, and mastering must be produced by the configured backend runners
- the project-local voice backend now treats original dialogue audio as XTTS reference material, not as a finished generated dialogue line, when voice cloning is forced
- automatic transcription now cross-checks Whisper language detection against transcript text across common source languages instead of assuming German or any other fixed language
- `03_diarize_and_transcribe.py` now detects the episode language from several larger probe scenes, aggregates forced-language probe scores across those scenes, lets clear transcript text override wrong file-name/Whisper hints, prefers confident audio probability over forced-probe hallucinations, and invalidates older transcription caches
- SpeechBrain speaker-model downloads now stay under `ai_series_project/runtime/models/speechbrain/ecapa` instead of creating a separate root-level `runtime/` folder
- generated story keywords now filter filler/function words such as `halt`, `weswegen`, `eigentlich`, and `irgendwie` instead of treating them as story topics
- episode generation now carries the detected series language into titles, dialogue templates, voice packages, and visual backend prompts instead of forcing German/English defaults
- fresh GitHub clones include the required `ai_series_project/tmp` placeholder so the local test suite can run without manual folder creation
- optional analysis/export tools are now registered as a generation toolkit and run automatically at pre-training, pre-generation, post-story, post-render, post-quality-gate, and post-export phases
- `05_review_unknowns.py` labels known-character quick assignments with both merged face-cluster count and available sample evidence so one identity cluster is not mistaken for one appearance
- `05_review_unknowns.py` now performs a public metadata/name lookup before manual review to complete partial labels such as `Babe` to `Babe Carano`; name completion is only written at `95%` confidence or higher, and older lower-confidence completions are rolled back on the next run
- `05_review_unknowns.py` can optionally upload preview face crops to a configured lookup command or HTTP endpoint before manual review; when no private service is configured, it also tries a built-in public-image lookup that downloads public character images and compares them locally without login/API credentials
- `05_review_unknowns.py --edit-names` opens a Tk name editor for correcting existing face and speaker names directly
- `05_review_unknowns.py --edit-names` also synchronizes slugged character artifacts after a rename: voice-model JSON files, foundation manifests, voice/reference folders, checkpoints, adapters, fine-tunes, and backend-run folders are moved to the new character slug, while conflicting old JSON files are archived under `ai_series_project/training/foundation/logs/renamed_character_artifacts/`
- `05_review_unknowns.py` also scans existing `statist`/background clusters against known local identity embeddings before opening manual review, so wrongly parked known faces can be rescued automatically
- `05_review_unknowns.py` now uses direct `speaker_face_cluster` evidence from linked segments to assign speaker entries more reliably, instead of relying only on single-visible-face scenes
- `04_link_faces_and_speakers.py` and `05_review_unknowns.py` ignore music, applause, laughter, silence, and low-confidence audio rows for speaker auto-naming so visible non-speakers do not get assigned to the wrong voice cluster
- `06_manage_character_relationships.py` is now a main pipeline step after review; it opens a Tk GUI for character groups, relationships, and per-series input groups, repairs stale rejected public-metadata names in old relationship rows, and shows face/voice evidence with explicit cluster/sample/segment labels
- manual relationship data is stored in `ai_series_project/characters/relationships.json` and is fed into the trained series model, episode prompts, and series bible
- `08b_analyze_behavior_model.py` now writes behavior-model schema version `2`, including sentence starts/ends, dialogue functions, relationship tempo, conflict repair, beat sequence, and callback candidates
- generated scenes now carry a `writer_room_plan`, `scene_purpose`, `conflict`, `character_intents`, behavior constraints, dialogue-style constraints, comedy/callback hints, and richer per-line voice metadata for downstream voice/lip-sync backends
- `17_render_episode.py` now resolves the configured lip-sync backend priority and records `selected_backend`, `backend_candidates`, and `backend_reason` in scene packages so Wav2Lip, MuseTalk, LatentSync, or future runners can be selected predictably
- `18_quality_gate.py` now produces per-scene Realism scores and regeneration hints for missing behavior context, template-heavy dialogue, missing voice metadata, missing voice-clone output, missing lip-sync output, and missing character reference data
- `19_regenerate_weak_scenes.py` now reads those hints and chooses a narrower retry plan for story/dialogue, voice/lip-sync, or scene-render problems instead of always treating every weak scene the same way
- `18_quality_gate.py` keeps running regeneration/rerender cycles until the gate passes or the configured retry-cycle limit is reached
- final output quality now depends on the real local backend stack being ready: SDXL/diffusers for frames, LTX/diffusers for video, XTTS for cloned dialogue, and Wav2Lip for lip-sync
- generation now creates a finished-episode blueprint with act structure, A/B story, callbacks, scene functions, set continuity, shot plans, dialogue-line acting metadata, audio-mix plans, and an edit decision list
- render packages now carry per-shot package targets, character continuity locks, set context, scene audio-mix targets, and standard backend manifest paths
- the local SDXL and LTX quality backends now execute shot packages, write per-shot manifests, and assemble scene video from generated shot clips instead of treating the shot plan as metadata only
- the master runner now follows shot/EDL clip order, writes dialogue-stem/final-mix targets, records audio-mix metrics, and muxes the final mix into the episode master when available
- the local Wav2Lip runner now writes measured sync metric payloads when duration evidence is available; the quality gate prefers backend metrics over unavailable placeholders
- `00_prepare_runtime.py`, `05_review_unknowns.py --reference-audit`, and quality reports now write reference-quality dashboards, backend-readiness reports, and worker-capability snapshots under `ai_series_project/generation/quality_reports/readiness/`
- distributed worker metadata now advertises capability hints and task scheduling can rank GPU memory, backend readiness, package readiness, backend health, and storage latency before choosing a worker
- generation show profiles and direct generation-toolkit guidance now feed style, camera, continuity, pacing, and reference-repair hints into scene prompts
- weak-scene regeneration now enforces a configurable cost budget per cycle so expensive retries are deferred while blocked or cheap high-impact repair scopes stay visible
- `18_quality_gate.py` now writes a Finished Episode Gate plus JSON/Markdown Realism Reports under `ai_series_project/generation/quality_reports/`
- `21_export_package.py` distinguishes `preview_export`, `review_export`, and `finished_episode_export`; finished exports are blocked unless the Finished Episode Gate passes

## Development Direction

The current codebase is a portable production-aware pipeline foundation, not a one-click original-episode-quality model. It already knows how to import source episodes, collect reviewed character/speaker context, prepare training artifacts, plan behavior-aware scenes, request backend media, reject placeholder output, and explain why a generated episode is not yet release-ready.

The next quality gains come from four connected loops:

- improve source evidence before generation: clean voice references, reliable speaker/face linking, language-aware transcripts, character relationships, set references, and rights-safe identity material
- improve generation contracts: episode blueprints, writer-room plans, shot packages, continuity locks, acting metadata, voice delivery metadata, and edit/audio plans that real backends can execute
- improve real media backends: shot-level image/video generation, cloned dialogue, lip-sync, mastering, manifests, and backend metrics instead of still-frame or mux-only stand-ins
- improve evaluation and retry decisions: measurable identity, ASR, lip-sync, audio, pacing, continuity, backend-integrity, and story/dialogue realism checks that select the smallest useful regeneration scope

The roadmap below keeps `Finished`, `In Progress`, and `Planned` explicit. The current implementation pass moved the previously listed repo-local roadmap items into executable contracts, reports, routing hints, backend outputs, tests, or documentation. Remaining real-media quality risk is tracked as a limitation when it depends on external checkpoints, rights-safe reference material, or backend-specific measurements rather than an unimplemented repository hook.

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
- `05_review_unknowns.py`: review GUI plus full local scan, false-face cleanup, public metadata name completion, built-in public-image face lookup, optional custom online face lookup, speaker-face auto-linking, GUI name editor, and known-face/statist rescue before manual input
- `06_manage_character_relationships.py`: Tk GUI for character groups, relationships, and multiple source inputs
- `07_build_dataset.py`
- `08_train_series_model.py`
- `08b_analyze_behavior_model.py`: analyzes speaking style, relationship behavior, scene pacing, episode structure, and dialogue patterns
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
- `21_export_package.py`: exports preview/review/finished episode packages, including realism reports, backend manifests, disclosure metadata, and edit data
- `22_refresh_after_manual_review.py`: rebuild from reviewed data after manual review changes
- `23_generate_finished_episodes.py`: batch finished-episode generation
- `24_process_next_episode.py`: full inbox-to-finished-episode orchestrator

### Support Scripts Inside `ai_series_project/support_scripts/`

- `support_scripts/pipeline_common.py`: shared path handling, runtime helpers, orchestration, leases, quality gating
- `support_scripts/console_colors.py`: console formatting
- `support_scripts/configure_quality_backends.py`: helper called by `00` to write portable quality-backend config
- `support_scripts/prepare_quality_backends.py`: helper called by `00` to download/update project-local backend assets
- `support_scripts/manage_character_relationships.py`: helper for manual character groups, relationships, and multiple source-series inputs
- `support_scripts/production_diagnostics.py`: reference-quality dashboards, backend readiness reports, and worker capability snapshots
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

`00 -> 06 -> 07 -> 08 -> 08b -> 09 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16 -> 17 -> 18 -> (19 if weak scenes are queued) -> 20 -> 21`

That means:

- downloads and setup happen in `00`
- character groups and relationships are reviewed in `06` after faces/speakers have names
- dataset building and all training stages happen after review and relationship setup
- `08b` analyzes behavior after the base series model exists and before foundation/backend training and generation
- backend fine-tune runs happen before new episode generation
- storyboard backend materialization happens before render
- render, quality gate, conditional regeneration, bible, and export happen only after generation
- `19_regenerate_weak_scenes.py` is not a mandatory normal step; `18_quality_gate.py` calls it automatically when the quality gate queues weak scenes and auto-retry is enabled

### Full Inbox Pipeline

For a raw new episode source, the full order is:

`00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 08b -> 09 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16 -> 17 -> 18 -> (19 if weak scenes are queued) -> 20 -> 21`

`24_process_next_episode.py` runs that complete chain and starts with `00` automatically. It now builds a source backlog from:

- new files in `data/inbox/episodes`
- already imported raw working files in `data/raw/episodes`
- partially processed scene folders in `data/processed/scene_clips`
- unfinished autosaves from a previous `24` run

This means a failed or interrupted `01` to `04` run can be resumed without moving the source episode back into the inbox. Fully completed `01` to `04` artifacts are detected and skipped.

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

`04_link_faces_and_speakers.py` defaults to a high-recall face scan. It samples source scenes more densely, accepts more valid faces per frame, keeps more per-scene face candidates, and raises the old early scene cutoff. This costs more scan time, but helps recurring side characters in group shots. Rerun `04` after changing these face-scan settings; process-version `11` also invalidates older low-recall step-04 scene caches automatically.

To force a completely offline review run:

```powershell
python 05_review_unknowns.py --offline
```

Without `--offline`, `05_review_unknowns.py` is online-first: it tries public text metadata, the built-in no-login public-image face lookup, and any configured face-image lookup backend before local/offline review checks. If the internet or lookup backend is unavailable, the script warns and continues with local review data. `--no-internet-lookup` only disables the public text metadata lookup; use `--offline` when no online lookup should be attempted at all. Deep public-image matching is enabled by default, so Torch/FaceNet may be loaded during this step.

To correct existing face and speaker names in a small GUI:

```powershell
python 05_review_unknowns.py --edit-names
```

To audit voice/face reference coverage and backend readiness before training or a long finished-episode run:

```powershell
python 05_review_unknowns.py --reference-audit
```

The audit writes JSON and Markdown readiness reports under `ai_series_project/generation/quality_reports/readiness/`. `00_prepare_runtime.py` also refreshes those reports after normal setup.

If `05_review_unknowns.py` reports thousands of open review cases, inspect the summary instead of printing the full queue:

```powershell
python 05_review_unknowns.py --show-queue
```

The queue is segment-based, so thousands of rows usually mean repeated speaker IDs or visible face IDs. Rename or assign the most repeated IDs first, then run `python 22_refresh_after_manual_review.py`; one correct assignment can clear many segment cases.

Or full orchestration:

```powershell
python 24_process_next_episode.py
```

By default, `24_process_next_episode.py` processes all pending/imported/partial source episodes it can find before rebuilding the dataset, training artifacts, generated episode, render, quality gate, bible, and export. To intentionally process only one backlog item, use:

```powershell
python 24_process_next_episode.py --single
```

To process a fixed batch size:

```powershell
python 24_process_next_episode.py --max-source-episodes 3
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
- voice references are accepted only when they pass the speech/content gate; `[music]`, applause/laughter, silence, ambience, low-confidence noise, and explicitly ineligible rows are excluded from speaker clustering, voice maps, foundation voice packs, render reference segments, and the project-local XTTS reference list
- render-time dialogue planning now falls back directly to `characters/voice_models/<character>_voice_model.json` when `voice_map` is stale or missing a named speaker entry
- delegated quality-backend runners now resolve project-local backend scripts from the project root even when scene packages run from nested working directories
- project-local quality backends now prefer the platform-correct FFmpeg binary from `ai_series_project/runtime/host_runtime/ffmpeg/bin` before falling back to older tool copies
- render-time scene duration now respects the planned per-scene runtime from episode generation instead of compressing most scenes into a short 8 to 22 second window
- scene packages now include behavior-model guidance and per-line voice metadata (`emotion`, `pace`, `energy`, `target_duration_seconds`, `pause_after_seconds`, `overlap_with_next_seconds`, `delivery_notes`, and `voice_reference_priority`) so voice runners can clone speech with clearer intent and timing
- show-profile rules plus the latest successful generation-toolkit roles now become prompt/camera/continuity guidance instead of passive reports
- local image/video quality runners now consume shot packages; scene masters can be assembled from generated shot clips and the EDL prefers shot clips when all required shot outputs exist
- the voice runner now records per-line delivery/reference diagnostics and respects planned pauses in generated dialogue audio
- lip-sync packages now resolve `lipsync_backends.preferred_order`, `allow_fallback`, and `min_sync_score`; Quality-First keeps fallback lip-sync blocked unless explicitly allowed locally and records clear backend diagnostics when a preferred runner is missing
- local Wav2Lip output can publish duration-alignment sync metrics and the quality gate merges measured backend metrics into Realism Reports when a backend returns them
- batch and quality-gate output messages now include the generated display title, for example `Folge 19: ... (folge_19)`, instead of relying only on the technical episode ID
- the quality score can reach `100%` only when generated scene video/lip-sync, cloned dialogue audio, scene mastering, and style/continuity support are all complete
- the quality gate also checks behavior constraints, scene purpose/conflict, relationship context, generic template-dialogue share, voice metadata, voice references, voice-clone output, lip-sync output, and a weighted Realism score with regeneration hints
- `16_run_storyboard_backend.py` now fails if `external_backends.storyboard_scene_runner` does not create a real frame; it no longer creates blue/filter-style seed-frame stand-ins by default
- `17_render_episode.py` no longer writes local motion fallback clips into the same production paths used by the real video backend, so backend runners are not skipped by pre-existing fake outputs
- `17_render_episode.py` now defers quality-first audio to `finished_episode_voice_runner`; local preview TTS/silence is not used as the final voice path
- final delivery is blocked when required generated scene video, cloned dialogue audio, lip-sync output, or the master runner is missing
- quality-gate auto-retry now loops through weak-scene regeneration and forced rerender cycles until the release gate passes, bounded by `release_mode.max_auto_retry_cycles`
- `release_mode.max_regeneration_cost_per_cycle` bounds each weak-scene queue cycle so missing references or expensive low-priority rerenders do not consume the whole retry budget
- generation prompts now include source-series style locks, auto-detected dialogue language, and stronger negative prompts against blue placeholders, filtered still-frame slideshows, and generic cartoon output

## Finished Episode Mode

`finished_episode_mode` is the stricter production layer on top of Quality-First mode. It is enabled in the template config and treats still-frame previews, slideshow motion, system TTS, missing lip-sync, missing audio mix, stale outputs, and missing backend manifests as blockers.

Finished Episode Mode adds these structures to generated packages:

- `episode_blueprint`: logline, theme, cold open, acts, resolution, A/B story, running gag, callbacks, character arcs, and continuity requirements
- per-scene `scene_function`: `cold_open`, `setup`, `inciting_incident`, `plan`, `complication`, `misunderstanding`, `escalation`, `midpoint_turn`, `low_point`, `reveal`, `resolution`, or `tag_joke`
- expanded `writer_room_plan`: emotional goal, conflict source, comedy engine, required information, scene button, previous dependency, next hook, and who drives/resists/resolves/gets the punchline
- per-line `dialogue_line_metadata`: dialogue function, subtext, reaction to previous line, interruption flag, callback setup/payoff, physical action, facial expression, and camera focus
- `shot_plan` and `shot_packages`: establishing shots, two-shots, close-ups, insert/reaction/button shots, target durations, dialogue line mapping, and per-shot output targets
- `character_continuity_lock`: identity, outfit, hair, body-shape, voice, allowed variations, forbidden variations, and reference-image slots per character
- `set_bible`: reusable set IDs, descriptions, props, lighting, camera axis, allowed angles, and reference slots
- `audio_mix_plan`: dialogue, ambience, music, SFX, final mix stems, LUFS target, and scene mix targets
- `edit_decision_list`: shot order, start/end timing, audio layers, and cut type for final assembly
- backend manifests: every real image/video/voice/lip-sync/master output gets a machine-readable manifest with command, inputs, outputs, hashes, status, and fallback/placeholder/stale flags

The Finished Episode Gate is written into the quality gate report:

```json
{
  "finished_episode_gate": {
    "passed": false,
    "readiness": "blocked",
    "blockers": [],
    "warnings": [],
    "required_actions": []
  }
}
```

`21_export_package.py --export-type finished_episode_export` refuses to mark an export as a real finished episode until this gate passes. Use `preview_export` or `review_export` for intermediate handoff packages.

Important:

- if the real backend path is not ready, the pipeline should fail early instead of pretending a fallback render is a final TV-quality episode
- use voice cloning and lip sync for real people only when you have the required rights, consent, and legal basis for the source material and generated output
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
- `paths.behavior_model` and `paths.behavior_model_summary`: outputs written by `08b_analyze_behavior_model.py`
- `paths.set_bible`: reusable set continuity output written during generation
- `paths.quality_reports`: JSON/Markdown Realism Reports written by `18_quality_gate.py`
- `runtime.*`: device, FFmpeg GPU preference, torch index URL
- `distributed.*`: NAS/shared-worker lease timing
- `character_detection.internet_name_lookup`: public metadata lookup in `05_review_unknowns.py` for completing partial character labels; only suggestions with at least `95%` confidence are written
- `character_detection.high_recall_face_scan`: uses recall floors for frame sampling, faces per frame, and per-scene clusters even when an older local `project.json` still contains the previous low caps
- `character_detection.high_recall_max_sample_every_n_frames`, `high_recall_min_faces_per_frame`, `high_recall_min_scene_clusters`, and `max_scene_face_detections`: tune the step-04 recall/performance tradeoff; disable `high_recall_face_scan` locally only when the faster old caps are intentional
- `character_detection.internet_name_lookup_context_terms`: optional show/title terms to improve name completion when filenames do not clearly contain the series name
- `character_detection.internet_face_lookup`: optional online face-image lookup in `05_review_unknowns.py`; configure either `SERIES_FACE_LOOKUP_COMMAND`/`character_detection.internet_face_lookup_command` or `SERIES_FACE_LOOKUP_URL`/`character_detection.internet_face_lookup_url`
- `character_detection.internet_face_lookup_token_env`: optional bearer-token environment variable for private face-lookup APIs, defaulting to `SERIES_FACE_LOOKUP_TOKEN`
- `character_detection.internet_face_lookup_builtin_public_images`: built-in no-login public-image lookup; it downloads public character images from public metadata results, embeds them locally, and only assigns a face when the local similarity/margin rules are met
- `character_detection.review_match_background_faces`: lets the local embedding scan rescue known faces that were previously marked as `statist`
- `character_detection.speaker_face_cluster_vote_weight` and `speaker_face_link_*`: tune how strongly direct face/speaker evidence can auto-name speaker clusters
- `generation_toolkit.*`: automatic use of optional analysis, continuity, pacing, subtitle, metadata, review, and export helper tools during episode generation
- `generation.show_profile.*`: reusable public style, camera, continuity, subtitle, and export-disclosure defaults that are copied into scene-generation prompts without absolute paths
- `transcription.language`: keep `auto` for new or mixed-language series; set it to a language code such as `de` for a known monolingual source project so wrong Auto-Detection caches are rejected and rebuilt
- `transcription.auto_language_forced_probe`: enabled by default so language detection compares candidate transcriptions instead of trusting one Whisper hint
- `transcription.speaker_cluster_min_speech_confidence`, `voice_reference_min_speech_confidence`, `voice_reference_min_duration_seconds`, `voice_reference_max_duration_seconds`, and `voice_reference_min_words`: control how strict the speech gate is before audio may become a speaker cluster or clone reference
- `foundation_training.visual_samples_require_character_visible`: when `true`, foundation frame/video samples are exported only from rows or transcript segments where the character is visibly present
- `foundation_training.voice_reference_require_character_evidence`: when `true`, automatic voice-clone references require character evidence such as a linked `speaker_face_cluster` instead of trusting a loose speaker name alone
- `foundation_training.allow_visible_only_voice_fallback`: keep `false` for quality-first work; when enabled, a single visible character can rescue old datasets, but this is weaker evidence and can mislabel off-screen speakers
- explicit `transcription.language`/`generation.language` values also constrain automatic foundation voice references, so stale wrong-language transcript rows are not copied into character voice packs
- `generation.allow_fallbacks`: must stay `false` for final generation
- `storyboard_backend.*`: controls whether local storyboard seed-frame fallbacks are allowed; quality-first defaults keep them disabled
- `render.allow_local_motion_fallback` and `render.require_*`: keep production output paths reserved for real generated video, voice clone, and lip-sync outputs
- `lipsync_backends.preferred_order`: priority list for lip-sync runners, defaulting to `musetalk`, `latentsync`, then `wav2lip`
- `lipsync_backends.allow_fallback`: keep `false` for Quality-First so missing lip-sync is reported instead of silently muxed
- `lipsync_backends.min_sync_score`: reserved threshold for backend sync metrics; current local checks expose the field even when a backend cannot yet calculate it
- `finished_episode_mode.*`: final gate policy for real motion video, cloned voice audio, lip-sync, audio mix, scene continuity, backend manifests, stale-output blocking, and minimum Realism scores
- `audio_mastering.*`: target LUFS, true-peak limit, required dialogue/final mix stems, and placeholder policy for music/SFX
- `foundation_training.*`, `adapter_training.*`, `fine_tune_training.*`, `backend_fine_tune.*`
- `external_backends.*`: runner templates and project-local backend commands
- `external_backends.*.environment`: set real generation commands for storyboard/image/video/lip-sync; fallback scripts such as `project_local_video_backend.py` are blocked by default in quality-first mode
- `release_mode.*`: quality gate thresholds and retry behavior, including `retry_until_pass`, `max_auto_retry_cycles`, and full-rerender retries when no weak-scene queue remains
- `release_mode.max_regeneration_cost_per_cycle`: per-cycle weak-scene regeneration cost budget; low-cost and blocked diagnostics remain visible while expensive work can be deferred
- `quality_backend_assets.*`: project-local backend tool/model targets

### Source Language Detection

The public template keeps source language detection portable:

```json
{
  "transcription": {
    "language": "auto"
  },
  "generation": {
    "language": "auto"
  }
}
```

With `transcription.language = "auto"`, step `03_diarize_and_transcribe.py` probes several source scenes, compares Whisper audio-language probabilities with candidate transcriptions, and locks segment language to the selected episode language by default. Confident audio-language evidence is preferred over forced-probe transcript text so one hallucinated probe is less likely to label a German episode as Spanish, Polish, Turkish, or another candidate language.

For a known single-language project, edit the local generated file `ai_series_project/configs/project.json` and set the language code explicitly. Example for German source episodes:

```json
{
  "transcription": {
    "language": "de"
  },
  "generation": {
    "language": "de"
  }
}
```

`transcription.language` controls source transcription and cached segment-language validation. `generation.language` controls generated titles, dialogue templates, voice packages, and generation prompts. Keep both on `auto` for mixed-language projects, or set both to the same language code when the whole source series is known to use one language.

After changing `transcription.language`, rebuild the affected source episodes so old cached language choices are replaced:

```powershell
python 03_diarize_and_transcribe.py
python 04_link_faces_and_speakers.py
```

The local `project.json` is intentionally ignored by Git. Change `project.template.json` only when you want to change the public default for future clones.

### Behavior Model

`08b_analyze_behavior_model.py` reads available dataset rows, transcripts, speaker libraries, character names, and `characters/relationships.json`. It writes schema version `2`:

- `ai_series_project/generation/model/behavior_model.json`
- `ai_series_project/generation/model/behavior_model_summary.md`

The model contains per-character speaking style, relationship behavior, scene pacing, episode-structure estimates, and dialogue patterns. Version `2` adds typical sentence openings/endings, short-vs-long answer ratios, dialogue-function tendencies, preferred partners, conflict reaction patterns, pair tempo, conversation starter/resolver signals, beat sequences, scene-type distribution, new-information/reaction balance, and callback candidates.

`14_generate_episode.py` uses this model to create a small `writer_room_plan` per scene. That plan decides which character sets up the scene, who opposes, who gets the joke, who resolves the beat, the likely turn order, whether a callback should appear, and which relationship dynamic is active. Missing data does not stop the pipeline; diagnostics are written into the summary and safe defaults are used.

Per-line voice metadata is also behavior-aware. Scene packages include emotion, pace, energy, target duration, pause/overlap placeholders, delivery notes, dialogue function, and voice-reference priority so XTTS-compatible runners can choose better reference material without breaking older packages.

### Realism Quality Gate

`18_quality_gate.py` combines the existing completeness checks with a per-scene Realism score. The score includes:

- `behavior_score`
- `dialogue_style_score`
- `relationship_score`
- `voice_metadata_score`
- `reference_coverage_score`
- `template_penalty`
- `lipsync_backend_score`
- `scene_structure_score`

Every weak scene now carries `failed_reasons`, `regeneration_hints`, and a `regeneration_scope`. `19_regenerate_weak_scenes.py` uses that scope to prefer story/dialogue refreshes for generic writing issues, voice/lip-sync reruns for audio or sync issues, and scene-selective visual reruns for render issues.

When a backend writes technical metric JSON into its manifest targets, `18_quality_gate.py` uses that measured metric instead of leaving a placeholder `unavailable` metric in the Realism Report. The project-local Wav2Lip path currently exposes a duration-alignment sync proxy and the master runner exposes audio-mix output metrics; stronger ASR, identity, and continuity metrics can be added by backend manifests without changing the report schema.

Finished Episode Mode extends this with additional scopes:

- `shot_plan`
- `visual_rerender`
- `voice_only`
- `lipsync_only`
- `audio_mix_only`
- `scene_master_only`
- `full_episode_master`
- `blocked_missing_references`
- `blocked_missing_backend`
- `blocked_low_identity_score`
- `blocked_low_lipsync_score`

Blocked scopes stop auto-retry loops. They require better references, backend setup, or measurable backend output before rerendering can help.

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

Running `06_manage_character_relationships.py` without arguments opens a Tk window similar to the character review flow: check known names, save the checked names as a group, select a relationship type, and add relationships for the checked pair or selected set. The known-character rows distinguish `face clusters`, `face samples`, face `detections`, `voice clusters`, named `voice segments`, and explicitly eligible `voice refs`; a single face cluster can therefore still represent many observed samples.

Existing relationship rows in the Tk window use click-to-toggle multi-selection, so several incorrect rows can be selected and removed with one `Remove selected` action.

If an older `05` metadata lookup briefly wrote a rejected public title or pair name such as `Character A & Character B` into `relationships.json`, `06` repairs it only when the Character Map contains a unique `internet_name_lookup_rejected` mapping back to the current reviewed identity. Manual relationship names without that safe rollback evidence are kept.

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
python -m py_compile 00_prepare_runtime.py 03_diarize_and_transcribe.py 04_link_faces_and_speakers.py 05_review_unknowns.py 06_manage_character_relationships.py 08_train_series_model.py 08b_analyze_behavior_model.py 09_prepare_foundation_training.py 10_train_foundation_models.py 14_generate_episode.py 15_generate_storyboard_assets.py 16_run_storyboard_backend.py 17_render_episode.py 18_quality_gate.py 19_regenerate_weak_scenes.py 20_build_series_bible.py 21_export_package.py 22_refresh_after_manual_review.py 23_generate_finished_episodes.py 24_process_next_episode.py ai_series_project\support_scripts\pipeline_common.py ai_series_project\support_scripts\generation_toolkit.py ai_series_project\support_scripts\production_diagnostics.py ai_series_project\support_scripts\configure_quality_backends.py ai_series_project\support_scripts\prepare_quality_backends.py ai_series_project\support_scripts\manage_character_relationships.py ai_series_project\tools\quality_backends\local_diffusion_image_backend.py ai_series_project\tools\quality_backends\local_ltx_video_backend.py ai_series_project\tools\quality_backends\local_wav2lip_backend.py ai_series_project\tools\quality_backends\master_runner.py ai_series_project\tools\quality_backends\project_local_voice_backend.py
```

## Known Limitations

- project-local fallback image/video generation still does not equal strong dedicated TV-quality generation backends
- project-local lip-sync is still weaker than a full dedicated production lip-sync stack
- project-local XTTS voice cloning still depends on clean speaker mapping and real reference segments; speakers with zero linked eligible speech data still cannot clone correctly
- existing bad voice maps or voice models created before the speech gate should be regenerated by rerunning `03` through `10` for the affected episodes/characters
- behavior analysis is heuristic and depends on reviewed transcript/speaker quality; it improves scene planning metadata but is not a replacement for a real large generative model or dedicated acting/performance evaluation
- automatic source-language detection is multi-probe and configurable, but short, noisy, musical, or multilingual source material can still need a local `transcription.language` override and a rerun of `03`/`04`
- public metadata and public-image lookup in `05_review_unknowns.py` are assistive only; ambiguous character matches and low-confidence names still require manual review
- character-style similarity, post-render ASR matching, and lip-sync confidence are reported as `unavailable` metrics until real external backends return measurable scores
- Finished Episode Mode can plan real TV-episode structure, shots, manifests, and gates, but it still depends on actual external image/video/voice/lip-sync backends producing real media
- the quality gate can reject placeholder or incomplete media, but it cannot invent missing reference voices, rights-safe face anchors, set assets, checkpoints, or backend outputs
- when backend scene-video or scene-audio generation still fails in quality-first mode, rendering stops with explicit missing-output details instead of exporting a fake final episode
- if external runners fail repeatedly, the quality gate will keep rejecting the episode even when the render technically finishes
- shared NAS runs still depend on file-system stability and can be slower for large backend/model downloads
- higher-recall face scanning keeps strict face validation but can take longer and can expose more unknown clusters for manual review when crowded or low-quality source shots contain many candidate faces
- large Hugging Face model downloads are more reliable with authentication via `HF_TOKEN`

## Finished

- the numbered main scripts now live directly in the repository root; `08b_analyze_behavior_model.py` is an explicit substep between base model training and foundation/backend training
- `06_manage_character_relationships.py` is now the official relationship/group review step between character review and dataset building
- `ai_series_project/` now contains the project internals only: configs, data, runtime state, tests, support scripts, backend tools, training artifacts, and generated outputs
- `00_prepare_runtime.py` now owns the normal setup flow completely, including folder creation, backend config, and project-local downloads
- the public repo now uses `project.template.json` as the tracked baseline, while `project.json` is generated locally and ignored
- `support_scripts/configure_quality_backends.py` and `support_scripts/prepare_quality_backends.py` remain available as internal helpers, but they are no longer part of the numbered main sequence
- project-local FFmpeg now comes from the Python runtime path instead of a separate external FFmpeg download
- `22_refresh_after_manual_review.py`, `23_generate_finished_episodes.py`, and `24_process_next_episode.py` now begin with `00_prepare_runtime.py`
- the documented order is now setup/downloads first, then review, relationships, dataset/training, backend fine-tunes, then generate/render/gate/export
- the orchestrators now export `21_export_package.py` after `20_build_series_bible.py` so finished packages are the final pipeline artifact
- `24_process_next_episode.py` now processes the whole pending source backlog, including raw files imported before the run and scene folders left behind by aborted `02/03/04` steps; `--single` or `--max-source-episodes` can cap the batch when needed
- release-gate auto-retry now calls `19_regenerate_weak_scenes.py` from the repository root layout correctly after the root/`ai_series_project` split
- `23_generate_finished_episodes.py` and `24_process_next_episode.py` now call the generation toolkit so optional tools actively feed continuity, voice, pacing, subtitle, review, metadata, and export quality signals into finished-episode generation
- `03_diarize_and_transcribe.py` now stores Whisper and SpeechBrain model data only inside `ai_series_project/`, refreshes stale language caches with process version `14`, rejects cached scene languages that conflict with an explicit project transcription language, and records speech-confidence/content-type fields used by downstream voice filtering
- source-language handling now defaults to portable `auto` detection in the public template while supporting explicit local language codes for known single-language projects
- quality-first render now blocks local seed-frame, local motion, system-TTS, and silent fallback paths from being labeled as finished generated episodes
- forced voice cloning ignores stale per-line audio files and regenerates dialogue through XTTS/voice-clone output instead of reusing old TTS artifacts
- voice-clone references are now filtered across transcription, face/speaker linking, manual-review auto-linking, foundation-pack creation, render voice plans, and the project-local XTTS backend so music/noise cannot silently become a character voice
- foundation-pack creation now requires visible-character evidence for frame/video exports and direct character evidence for automatic voice exports; subtitle/channel boilerplate such as credits, URLs, "thanks for watching", uploaded subtitle markers, or wrong-language rows in explicit-language projects are rejected before they can become voice samples
- step `04` now defaults to high-recall face detection with a denser frame sample, more valid faces per frame, a higher per-scene cluster budget, and a versioned scene-cache refresh; step `05` shows cluster count separately from review sample evidence
- `05_review_unknowns.py` now supports `--offline`, `--edit-names`, built-in public-image face lookup without login/API credentials, optional uploaded face-crop lookup, `95%` minimum public name completion, cleanup/rollback of older low-confidence public metadata renames, and stronger speaker auto-linking from direct face/speaker evidence
- character-name changes now keep active Voice/Training artifact names aligned with the canonical slug; stale active files such as `babe_voice_model.json` or `babe_manifest.json` are removed from the live pipeline when `Babe` is renamed to `Babe Carano`, and any collision is kept in the rename archive instead of being overwritten
- `08b_analyze_behavior_model.py`, behavior-aware scene metadata, voice metadata propagation, and stricter content checks in `18_quality_gate.py` are now part of the main generation path
- behavior-model schema version `2`, writer-room scene plans, richer delivery metadata, lip-sync backend priority resolution, and scope-aware weak-scene regeneration are now part of the main generation path
- Finished Episode Mode now adds episode blueprints, set bible, character continuity locks, multi-shot plans, EDL, audio mix planning, backend manifests, Finished Episode Gate, Realism Reports, export-type separation, and blocked-scope retry handling
- local SDXL and LTX runners now execute shot packages, write shot manifests, and let scene video be assembled from generated shot clips instead of relying only on scene-level placeholders
- the master runner now prefers complete EDL shot coverage, builds dialogue/final-mix stems, writes audio-mix metrics, and adds the final audio mix to the master export path when available
- the voice runner now writes line-delivery/reference diagnostics and respects line pauses; the Wav2Lip runner can write measured sync metric payloads for the quality gate
- reference-quality dashboards, backend-readiness reports, worker-capability snapshots, capability-ranked task scheduling, and `05_review_unknowns.py --reference-audit` are available before long training/render runs
- reusable `generation.show_profile` rules and successful generation-toolkit output roles now feed prompt, camera, continuity, pacing, and repair guidance into scene generation
- weak-scene retry manifests now include regeneration cost, deferred queue entries, and a configurable per-cycle budget
- finished exports now carry disclosure metadata so real-person voice, face, and lip-sync use stays visible to downstream packaging
- the regression suite covers behavior metadata, finished-episode blocking, voice-reference filtering, language selection, backlog orchestration, regeneration scopes/budgets, capability routing, readiness diagnostics, show profiles, backend metric ingestion, toolkit guidance, and export-type separation

## In Progress

- no repo-local roadmap item from the previous `In Progress` list is intentionally left as a documentation-only promise after this implementation pass
- active validation is now evidence-driven: run real rights-safe source material through the configured image, video, voice, lip-sync, and mastering backends, inspect the readiness reports, and fix the blocker that the Finished Episode Gate names
- backend-specific quality still improves only when those backends return stronger media, manifests, reference coverage, and metrics than the project-local compatibility paths can measure

## Planned

- no named repo-local feature from the previous `Planned` list remains unimplemented in the public pipeline contract
- future ideas should be added here only after a measured backend, reference, review, or quality-report bottleneck shows that the existing diagnostics, manifests, budgets, profiles, evaluators, benchmark helpers, review flows, and disclosure packaging need another concrete extension
