# AI Series Training

## Table Of Contents

- [Purpose](#purpose)
- [Current Status](#current-status)
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
- `05_review_unknowns.py` now performs a public metadata/name lookup before manual review to complete partial labels such as `Babe` to `Babe Carano`; name completion is only written at `95%` confidence or higher, and older lower-confidence completions are rolled back on the next run
- `05_review_unknowns.py` can optionally upload preview face crops to a configured lookup command or HTTP endpoint before manual review; when no private service is configured, it also tries a built-in public-image lookup that downloads public character images and compares them locally without login/API credentials
- `05_review_unknowns.py --edit-names` opens a Tk name editor for correcting existing face and speaker names directly
- `05_review_unknowns.py` also scans existing `statist`/background clusters against known local identity embeddings before opening manual review, so wrongly parked known faces can be rescued automatically
- `05_review_unknowns.py` now uses direct `speaker_face_cluster` evidence from linked segments to assign speaker entries more reliably, instead of relying only on single-visible-face scenes
- `06_manage_character_relationships.py` is now a main pipeline step after review; it opens a Tk GUI for character groups, relationships, and per-series input groups
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
- `18_quality_gate.py` now writes a Finished Episode Gate plus JSON/Markdown Realism Reports under `ai_series_project/generation/quality_reports/`
- `21_export_package.py` distinguishes `preview_export`, `review_export`, and `finished_episode_export`; finished exports are blocked unless the Finished Episode Gate passes

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

To force a completely offline review run:

```powershell
python 05_review_unknowns.py --offline
```

Without `--offline`, `05_review_unknowns.py` is online-first: it tries public text metadata, the built-in no-login public-image face lookup, and any configured face-image lookup backend before local/offline review checks. If the internet or lookup backend is unavailable, the script warns and continues with local review data. `--no-internet-lookup` only disables the public text metadata lookup; use `--offline` when no online lookup should be attempted at all. Deep public-image matching is enabled by default, so Torch/FaceNet may be loaded during this step.

To correct existing face and speaker names in a small GUI:

```powershell
python 05_review_unknowns.py --edit-names
```

If `05_review_unknowns.py` reports thousands of open review cases, inspect the summary instead of printing the full queue:

```powershell
python 05_review_unknowns.py --show-queue
```

The queue is segment-based, so thousands of rows usually mean repeated speaker IDs or visible face IDs. Rename or assign the most repeated IDs first, then run `python 22_refresh_after_manual_review.py`; one correct assignment can clear many segment cases.

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
- scene packages now include behavior-model guidance and per-line voice metadata (`emotion`, `pace`, `energy`, `target_duration_seconds`, `pause_after_seconds`, `overlap_with_next_seconds`, `delivery_notes`, and `voice_reference_priority`) so voice runners can clone speech with clearer intent and timing
- lip-sync packages now resolve `lipsync_backends.preferred_order`, `allow_fallback`, and `min_sync_score`; Quality-First keeps fallback lip-sync blocked unless explicitly allowed locally and records clear backend diagnostics when a preferred runner is missing
- batch and quality-gate output messages now include the generated display title, for example `Folge 19: ... (folge_19)`, instead of relying only on the technical episode ID
- the quality score can reach `100%` only when generated scene video/lip-sync, cloned dialogue audio, scene mastering, and style/continuity support are all complete
- the quality gate also checks behavior constraints, scene purpose/conflict, relationship context, generic template-dialogue share, voice metadata, voice references, voice-clone output, lip-sync output, and a weighted Realism score with regeneration hints
- `16_run_storyboard_backend.py` now fails if `external_backends.storyboard_scene_runner` does not create a real frame; it no longer creates blue/filter-style seed-frame stand-ins by default
- `17_render_episode.py` no longer writes local motion fallback clips into the same production paths used by the real video backend, so backend runners are not skipped by pre-existing fake outputs
- `17_render_episode.py` now defers quality-first audio to `finished_episode_voice_runner`; local preview TTS/silence is not used as the final voice path
- final delivery is blocked when required generated scene video, cloned dialogue audio, lip-sync output, or the master runner is missing
- quality-gate auto-retry now loops through weak-scene regeneration and forced rerender cycles until the release gate passes, bounded by `release_mode.max_auto_retry_cycles`
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
- `character_detection.internet_name_lookup_context_terms`: optional show/title terms to improve name completion when filenames do not clearly contain the series name
- `character_detection.internet_face_lookup`: optional online face-image lookup in `05_review_unknowns.py`; configure either `SERIES_FACE_LOOKUP_COMMAND`/`character_detection.internet_face_lookup_command` or `SERIES_FACE_LOOKUP_URL`/`character_detection.internet_face_lookup_url`
- `character_detection.internet_face_lookup_token_env`: optional bearer-token environment variable for private face-lookup APIs, defaulting to `SERIES_FACE_LOOKUP_TOKEN`
- `character_detection.internet_face_lookup_builtin_public_images`: built-in no-login public-image lookup; it downloads public character images from public metadata results, embeds them locally, and only assigns a face when the local similarity/margin rules are met
- `character_detection.review_match_background_faces`: lets the local embedding scan rescue known faces that were previously marked as `statist`
- `character_detection.speaker_face_cluster_vote_weight` and `speaker_face_link_*`: tune how strongly direct face/speaker evidence can auto-name speaker clusters
- `generation_toolkit.*`: automatic use of optional analysis, continuity, pacing, subtitle, metadata, review, and export helper tools during episode generation
- `transcription.language`: keep `auto` for new series unless you intentionally want to force a specific source language
- `transcription.auto_language_forced_probe`: enabled by default so language detection compares candidate transcriptions instead of trusting one Whisper hint
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
- `quality_backend_assets.*`: project-local backend tool/model targets

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
python -m py_compile 00_prepare_runtime.py 06_manage_character_relationships.py 08_train_series_model.py 08b_analyze_behavior_model.py 14_generate_episode.py 15_generate_storyboard_assets.py 16_run_storyboard_backend.py 17_render_episode.py 18_quality_gate.py 19_regenerate_weak_scenes.py 20_build_series_bible.py 21_export_package.py 22_refresh_after_manual_review.py 23_generate_finished_episodes.py 24_process_next_episode.py ai_series_project\support_scripts\pipeline_common.py ai_series_project\support_scripts\generation_toolkit.py ai_series_project\support_scripts\configure_quality_backends.py ai_series_project\support_scripts\prepare_quality_backends.py ai_series_project\support_scripts\manage_character_relationships.py
```

## Known Limitations

- project-local fallback image/video generation still does not equal strong dedicated TV-quality generation backends
- project-local lip-sync is still weaker than a full dedicated production lip-sync stack
- project-local XTTS voice cloning still depends on clean speaker mapping and real reference segments; speakers with zero linked voice data still cannot clone correctly
- behavior analysis is heuristic and depends on reviewed transcript/speaker quality; it improves scene planning metadata but is not a replacement for a real large generative model or dedicated acting/performance evaluation
- character-style similarity, post-render ASR matching, and lip-sync confidence are reported as `unavailable` metrics until real external backends return measurable scores
- Finished Episode Mode can plan real TV-episode structure, shots, manifests, and gates, but it still depends on actual external image/video/voice/lip-sync backends producing real media
- when backend scene-video or scene-audio generation still fails in quality-first mode, rendering stops with explicit missing-output details instead of exporting a fake final episode
- if external runners fail repeatedly, the quality gate will keep rejecting the episode even when the render technically finishes
- shared NAS runs still depend on file-system stability and can be slower for large backend/model downloads
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
- release-gate auto-retry now calls `19_regenerate_weak_scenes.py` from the repository root layout correctly after the root/`ai_series_project` split
- `23_generate_finished_episodes.py` and `24_process_next_episode.py` now call the generation toolkit so optional tools actively feed continuity, voice, pacing, subtitle, review, metadata, and export quality signals into finished-episode generation
- `03_diarize_and_transcribe.py` now stores Whisper and SpeechBrain model data only inside `ai_series_project/` and refreshes stale language caches with process version `12`
- quality-first render now blocks local seed-frame, local motion, system-TTS, and silent fallback paths from being labeled as finished generated episodes
- forced voice cloning ignores stale per-line audio files and regenerates dialogue through XTTS/voice-clone output instead of reusing old TTS artifacts
- `05_review_unknowns.py` now supports `--offline`, `--edit-names`, built-in public-image face lookup without login/API credentials, optional uploaded face-crop lookup, `95%` minimum public name completion, cleanup/rollback of older low-confidence public metadata renames, and stronger speaker auto-linking from direct face/speaker evidence
- `08b_analyze_behavior_model.py`, behavior-aware scene metadata, voice metadata propagation, and stricter content checks in `18_quality_gate.py` are now part of the main generation path
- behavior-model schema version `2`, writer-room scene plans, richer delivery metadata, lip-sync backend priority resolution, and scope-aware weak-scene regeneration are now part of the main generation path
- Finished Episode Mode now adds episode blueprints, set bible, character continuity locks, multi-shot plans, EDL, audio mix planning, backend manifests, Finished Episode Gate, Realism Reports, export-type separation, and blocked-scope retry handling

## In Progress

- improving the real external image/video backend path so generated scenes replace still-frame composites completely
- wiring real measurable lip-sync scores and character-style similarity scores back from external backends into the Quality Gate
- wiring shot-level backends so `shot_packages` become real per-shot generated clips rather than only production-ready contracts
- producing real dialogue/ambience/music/SFX stems and measured LUFS/true-peak audio reports from the mastering backend
- improving ready-made adapters for online face recognition services; the project currently exposes a portable command/API hook, but does not ship private service credentials
- improving project-local XTTS voice cloning coverage so scenes have natural speech from cloned character voices
- improving render-time diagnostics so missing backend outputs point directly to the failing runner logs and expected target files
- improving the project-local lip-sync path beyond simple fallback mux behavior
- reducing external backend task failures so the quality gate can pass more often with real generated assets
- converting more toolkit outputs from passive reports into direct prompt, pacing, and regeneration inputs

## Planned

- stronger automatic worker-capability routing for NAS multi-PC runs
- stronger real local XTTS/lip-sync runtime integration when the backend assets are fully available
- higher-quality project-local image/video backends to move closer to original-episode visual consistency
