# AI Series Training

## Purpose

This project turns existing TV episodes into a local AI training and episode-generation pipeline.

Starting from real episode files, the pipeline can:

- import and split source episodes into scenes
- transcribe dialogue and build speaker clusters
- let multiple PCs cooperate on the same NAS-backed pipeline through shared leases, so batch steps can divide episodes, characters, or scenes across workers and automatically recover when one worker disappears
- detect the spoken language automatically per source segment instead of forcing one fixed transcription language
- detect faces and link recurring characters to speakers
- build a local series dataset and heuristic series model
- train local foundation, adapter, fine-tune, and backend preparation artifacts
- train per-character local voice models from original dialogue recordings so later voice cloning can follow the real character voices and their detected languages
- generate new synthetic episodes as markdown and shotlists
- render draft previews and final voiced storyboard episodes
- export backend-ready production packages for fully generated episodes with new frames, cloned voices, and lip-sync
- reuse original dialogue segments in final episode audio whenever matching source material is available, while falling back to local TTS only for truly new missing lines
- automatically compose a more complete final episode from already generated per-scene video or lip-sync clips whenever those backend outputs exist

The default path stays local-first and license-light. Stronger voice cloning backends remain optional follow-up paths, but the final render path now aims to produce a complete voiced storyboard episode instead of only a silent preview.

All scripts in this repository are AI-generated and maintained with `GPT-5.4`.

## At A Glance

| Area | Status | Summary |
| --- | --- | --- |
| Import / scene split | stable | New source episodes can be imported, registered, and split into scenes in batch mode. |
| Transcription / speaker clustering | usable | Whisper transcription and speaker clustering work locally, with GPU support when available. |
| Face / character linking | actively tuned | Recurring characters are prioritized over one-off background faces. |
| Review workflow | active | Manual review is still important, but the tooling now reduces the open unknowns much more aggressively. |
| Dataset / model | usable | A local series dataset and heuristic series model can be rebuilt from reviewed material. |
| Training stack | expanding | Foundation, adapter, fine-tune, and backend-prep stages now exist as explicit steps. |
| Generation / render | voiced-storyboard-plus-production-package | New episodes, shotlists, draft videos, final voiced storyboard episodes, and backend-ready full-episode production packages can be produced. |

## What Already Works Well

- full local preprocessing from inbox episode to linked scene data
- batch processing over multiple new source episodes
- NAS-based shared-worker processing across the numbered pipeline, so several PCs can cooperate on inbox imports, per-episode steps, per-character training stages, storyboard scene work, and selected orchestration paths without duplicating the same target
- synthetic episode generation from the trained local model
- automatic language detection during transcription, so mixed-language material no longer depends on one hard-coded transcription language
- model-native storyboard planning per scene, with multi-reference slots, continuity anchors, and camera/control hints for later image-model backends
- voice-training preparation now prioritizes original per-character dialogue segments from the source material and writes language-aware local voice-model profiles for each trained character
- storyboard scene asset generation now writes backend-ready per-scene seed packages and can reuse moved training artifacts after workspace path changes
- local storyboard backend materialization can turn those seed packages into reusable per-scene backend frames before render
- real episode titles for generated outputs instead of raw `folge_0x` placeholders
- draft previews and final voiced storyboard episode rendering
- full-episode production packages with per-scene image generation, video generation, voice clone, and lip-sync plans for later real episode backends
- mixed final-audio rendering that already reuses matching original dialogue segments and materializes per-line/per-scene audio into the full-episode package
- final-episode composition that already prefers generated scene videos and lip-sync clips from the production package over static storyboard cards whenever those scene outputs exist
- per-scene mastered episode clips inside the production package, so each scene already has a reusable endclip with its local dialogue timing
- scene and master package JSONs that now track the real already-generated outputs, not only planned target paths
- autosaves and live progress dashboards for long-running pipelines, now including concrete finished-episode output paths, production-readiness labels, coverage ratios, and remaining backend tasks instead of only step completion flags
- the series bible can now summarize the most recent generated episodes together with their final render, full generated master, and production-package progress
- generated episodes now also carry a central production-readiness label plus scene-video, scene-dialogue, and scene-master coverage ratios for easier follow-up automation
- local voice fallback with German Windows voices instead of old English default fallbacks

## Current Focus

- separate main recurring characters from background faces faster and more reliably
- auto-merge already known faces before manual review so review mostly sees true unknowns
- keep the default path robust, local, and free of unnecessary license prompts
- harden the end-to-end batch pipeline for large inbox runs
- keep the new NAS/shared-worker mode stable across all numbered batch steps, including graceful takeover when one PC disappears
- improve the voice path so future local cloning backends can replace simple fallback TTS where appropriate
- keep the voice path centered on original per-character dialogue material instead of generic online voice base models
- improve the model-side storyboard planning so future image/video backends can use original-series references without relying on any graphical node editor or UI-specific workflow
- move the path after `17_render_episode.py` from voiced storyboard previews toward truly newly generated episodes with new frames, new shot video, original-voice cloning, and lip-sync

## In Progress

- `06_review_unknowns.py` now tries to re-identify known faces before manual naming, using multi-reference matching per character
- `06_review_unknowns.py` keeps iterating after each manual naming step, so newly named characters immediately help resolve more open clusters
- `04_diarize_and_transcribe.py` now applies extra rescue passes for remaining `speaker_unknown` segments using neighborhood and embedding agreement
- `04_diarize_and_transcribe.py` now also auto-detects spoken language per transcription run and stores that language on the emitted source segments
- the numbered batch scripts now support shared NAS workers through lease files under `ai_series_project/runtime/distributed`, with step-specific work units such as inbox files, episodes, characters, scenes, or full orchestration scopes
- `00_prepare_runtime.py`, `01_setup_project.py`, `06_review_unknowns.py`, `08_train_series_model.py`, and `14_generate_episode_from_trained_model.py` now also use the same shared-worker lease layer, but intentionally as exclusive leases because concurrent writes there would be unsafe or ambiguous
- `04_diarize_and_transcribe.py` still has the deepest split, leasing scenes dynamically so multiple PCs can help on the same episode and stale leases get reclaimed automatically
- `19_generate_finished_episodes.py`, `20_refresh_after_manual_review.py`, `18_build_series_bible.py`, and `99_process_next_episode.py` now use shared-worker leases mainly to prevent conflicting duplicate orchestration runs, while the underlying numbered worker steps do the actual distributed work
- the orchestration scripts now also forward `--worker-id` and `--no-shared-workers` into the numbered child steps, so one NAS run keeps one consistent worker identity and an explicitly local-only run stays local-only all the way down
- `99_process_next_episode.py` is being hardened for real inbox batch workflows with autosaves, resumable checkpoints, and live status files
- `17_render_episode.py` continues to improve long Windows render runs, especially for large segment stacks
- the new training chain around `09` through `13` is being observed on real project data, especially where voice material is still weak
- `14_generate_episode_from_trained_model.py` now writes per-scene multi-reference storyboard plans inspired by shot-by-shot image-edit workflows, but stays model-only and does not depend on any GUI workflow
- `16_run_storyboard_backend.py` can now materialize local backend-style scene frames from the seed payloads emitted by `15_generate_storyboard_assets.py`
- `17_render_episode.py` now carries storyboard plans into the render manifest, reuses backend frames from `16` when they exist, and falls back to seed assets or generated placeholder cards when needed
- `14_generate_episode_from_trained_model.py` now also exports backend-ready storyboard request files per episode and per scene under `generation/storyboard_requests`
- `17_render_episode.py` now automatically picks up already generated storyboard scene frames from `generation/storyboard_assets/<episode>` when they exist, so local backend materialization and later model backends can feed visuals back into the render path without changing script order
- `15_generate_storyboard_assets.py` now emits backend-ready scene input payloads beside each generated seed frame and rebases older stored artifact paths after project moves, so NAS relocations do not silently break later backend use
- `15_generate_storyboard_assets.py`, `16_run_storyboard_backend.py`, `17_render_episode.py`, and `19_generate_finished_episodes.py` now resolve the newest generated artifact by timestamp instead of assuming a `folge_*` filename pattern
- `06_review_unknowns.py`, `18_build_series_bible.py`, and `99_process_next_episode.py` continue to standardize user-facing CLI help, review prompts, and progress output in English so the numbered pipeline reads consistently end to end
- `pipeline_common.py` now also uses the same English-first tool and runtime error wording as the numbered scripts, so shared failures read consistently in logs and consoles
- `09_prepare_foundation_training.py` and `18_build_series_bible.py` now write English-first markdown summaries, while `pipeline_common.py` continues to standardize live dashboard labels and dependency-staleness warnings across the numbered training chain
- `08_train_series_model.py` now also emits English-first story markdown summaries and generic fallback beat dialogue, so the model-training and generation outputs match the rest of the numbered pipeline
- `05_link_faces_and_speakers.py`, `06_review_unknowns.py`, and `07_build_dataset.py` continue the same English-first cleanup for interactive naming prompts, named-character listings, and dataset completion messages
- `13_run_backend_finetunes.py`, `14_generate_episode_from_trained_model.py`, and `18_build_series_bible.py` now also use English-first status and missing-model messages so the mid-pipeline generation path reads consistently in logs and dashboards
- `07_build_dataset.py` now also uses English-first progress labels inside the live reporter, so dataset rebuild runs match the rest of the numbered training chain
- `10_train_foundation_models.py`, `11_train_adapter_models.py`, `12_train_fine_tune_models.py`, and `13_run_backend_finetunes.py` now also use English-first training/existing-artifact status lines so the numbered training chain reads consistently from preparation through backend runs
- `00_prepare_runtime.py` now also uses English-first package-install and already-present status messages so the numbered pipeline starts with the same console tone as the later steps
- `04_diarize_and_transcribe.py` and `09_prepare_foundation_training.py` now also use English-first completion and remote-revision fallback messages so transcription and model-prep logs stay consistent with the rest of the numbered pipeline
- `00_prepare_runtime.py` now also uses English-first install-failure and runtime-Python status lines so startup diagnostics match the rest of the numbered pipeline
- `00_prepare_runtime.py` now installs the torch stack before torch-dependent packages like Whisper, SpeechBrain, facenet-pytorch, and optional XTTS, which avoids Linux install failures caused by dependency resolution in the wrong order
- `00_prepare_runtime.py` now uses the active `python3` interpreter on Linux/NAS and installs with `python3 -m pip install --break-system-packages` instead of relying on a separate venv there, which avoids Synology-style runtime installs silently landing outside the usable environment
- `00_prepare_runtime.py` now also downloads and installs the matching FFmpeg build for the current OS into `ai_series_project/tools/ffmpeg/bin`, and `pipeline_common.py` only resolves the platform-valid binary there so Linux never tries to launch a Windows `.exe`
- `09_prepare_foundation_training.py` now skips online voice-base downloads by default when local character voice models are enabled, extracts original per-character dialogue segments from the reviewed dataset, and prepares language-aware voice manifests from those original recordings
- `10_train_foundation_models.py`, `11_train_adapter_models.py`, `12_train_fine_tune_models.py`, and `13_run_backend_finetunes.py` now carry per-character voice-model paths plus dominant-language metadata through the full training chain, so the downstream render/backends can follow original-character voices instead of one generic voice language
- `17_render_episode.py` now carries those character-language and voice-model hints into the production package and uses detected character language when choosing fallback system voices
- `03_split_scenes.py`, `04_diarize_and_transcribe.py`, `05_link_faces_and_speakers.py`, and `07_build_dataset.py` now also use English-first live progress scope labels and segment/cluster counters so long batch runs read consistently across the early numbered pipeline
- `05_link_faces_and_speakers.py` now also derives its live face-cluster counter directly from each processed scene payload, so long linking runs no longer break on the progress display after scene analysis starts
- `17_render_episode.py` now also turns the timed dialogue voice-plan into a final dialogue audio track and muxes it into the final storyboard episode, while keeping the draft render as a lighter silent check
- `17_render_episode.py` now also exports a full generated-episode production package under `generation/final_episode_packages/<episode>`, with stable per-scene targets for image generation, shot video generation, voice cloning, and lip-sync backends
- `17_render_episode.py` now also reuses original dialogue segments in the final episode audio when matching source audio or scene clips are available, materializes per-speaker line audio into `generation/final_episode_packages/<episode>/audio/<speaker>/`, and writes per-scene dialogue tracks beside the production package
- `17_render_episode.py` now also normalizes already generated per-scene video or lip-sync clips from `generation/final_episode_packages/<episode>` into one full final episode and falls back scene-by-scene to storyboard cards only where real generated video is still missing
- `17_render_episode.py` now also writes per-scene mastered clips under `generation/final_episode_packages/<episode>/master/scenes`, muxing each scene clip with its own scene dialogue track when available before assembling the package-level full episode master
- `17_render_episode.py` now also writes the real generated-output state back into the scene package JSONs and the episode master package, including ready counts for scene videos, scene dialogue tracks, and scene master clips
- the numbered order now keeps all training in `07-13`, then generation/render in `14-17`, and only then rebuilds the series bible in `18`, so the pipeline follows the requested train-before-generate/render sequence more clearly
- `19_generate_finished_episodes.py` now rebuilds the series bible once after the full generated/rendered batch instead of after every single episode, so multi-episode runs stay closer to the intended train-then-generate/render flow
- `19_generate_finished_episodes.py` now also supports an endless generation mode: `--count 0` or `--endless` keeps generating episodes until stopped, and updates the series bible after each newly rendered episode in that mode
- `20_refresh_after_manual_review.py` now derives its rebuild order from one explicit planned-step list, keeps the same train-then-generate/render-then-bible sequence, and now also respects the configured optional foundation/adapter/fine-tune/backend stages instead of always forcing `09` through `13`
- `19_generate_finished_episodes.py` now also records explicit planned/completed batch-step lists in its completion metadata, so autosaves and orchestration logs keep the same ordering contract as the refreshed rebuild path
- `19_generate_finished_episodes.py` now also uses the same configurable foundation/adapter/fine-tune/backend stage toggles as `99_process_next_episode.py` and `20_refresh_after_manual_review.py`, including the `prepare_after_batch` / `auto_train_after_prepare` split, so planned and executed batch steps stay aligned
- `19_generate_finished_episodes.py` now also supports `--skip-downloads` for step `09`, so multi-episode rebuild runs can reuse existing model downloads like the manual refresh path
- `99_process_next_episode.py` now also supports `--skip-downloads` for step `09`, so the full end-to-end inbox pipeline can reuse existing model downloads without changing the train-then-generate/render order
- `99_process_next_episode.py` now also stores explicit planned/completed global step metadata in its autosaves and status files, so resume state and live status stay aligned with the real train-then-generate/render plan
- `19_generate_finished_episodes.py`, `20_refresh_after_manual_review.py`, and `99_process_next_episode.py` now block only on actionable open face-review clusters from step `06`, while `06_review_unknowns.py` explicitly points to `--show-queue` when only speaker/segment review_queue entries remain
- `pipeline_common.py`, `19_generate_finished_episodes.py`, and `99_process_next_episode.py` now also collect the real render/package outputs written by `17_render_episode.py`, so batch metadata and live status files point directly to the latest final render, full generated episode master, render manifest, and production package
- `18_build_series_bible.py` now also pulls in the most recent generated-episode outputs, so the bible itself reflects the current finished-episode production state instead of only the trained model summary
- `pipeline_common.py`, `17_render_episode.py`, and `18_build_series_bible.py` now also derive and expose a central production-readiness summary with coverage ratios and remaining backend tasks per generated episode
- `99_process_next_episode.py` now also shows that same production-readiness summary directly in its live markdown status, so the full pipeline dashboard reveals how close the latest generated episode is to a fully generated master

## Planned

- finish naming main characters in `06_review_unknowns.py`
- mark minor characters as `statist` where appropriate so they stay visible but do not become main roles
- run `20_refresh_after_manual_review.py` on the fully reviewed set to rebuild datasets, model, training packs, generated episodes, bible, and renders
- keep reducing `speaker_unknown` cases across full seasons
- expand the fine-tune and backend stages from local preparation artifacts into real model-weight training later on
- connect the new backend-ready storyboard seed packages from `15_generate_storyboard_assets.py` and the full generated-episode production packages from `17_render_episode.py` to actual local image/video/voice/lip-sync model runners later on
- improve render quality, character consistency, and synthetic episode quality after the review and training loop stabilizes
- improve the new voiced storyboard episode path so it sounds more natural and character-specific until the full generated-episode backends replace the fallback render
- only let the full generated-episode path become the default once image/video generation and lip-sync actually look series-quality

## Documentation Rule

This file is mandatory documentation.

Whenever you change any of the following, update `README.md` in the same task:

- `00_prepare_runtime.py` through `20_refresh_after_manual_review.py`
- `99_process_next_episode.py`
- `pipeline_common.py`
- `ai_series_project/configs/project.json`
- CLI options
- folder structure
- output formats
- environment variables
- known limitations or workarounds

Also keep the `In Progress` and `Planned` sections current. If priorities change, this README must be updated in the same edit.

## Project Layout

### Root Scripts

- `00_prepare_runtime.py`: create the runtime and install the required packages
- `01_setup_project.py`: create the project structure and config
- `02_import_episode.py`: import the next source episode from the inbox
- `03_split_scenes.py`: split imported source episodes into scene clips
- `04_diarize_and_transcribe.py`: extract audio, transcribe, and cluster speakers
- `05_link_faces_and_speakers.py`: detect faces and link them to speakers
- `06_review_unknowns.py`: review unknown face clusters and character assignments
- `07_build_dataset.py`: build the local training dataset
- `08_train_series_model.py`: train the local heuristic series model
- `09_prepare_foundation_training.py`: prepare training assets and plans for later fine-tuning
- `10_train_foundation_models.py`: train local foundation packs
- `11_train_adapter_models.py`: train local adapter profiles
- `12_train_fine_tune_models.py`: train local fine-tune profiles
- `13_run_backend_finetunes.py`: materialize backend-oriented fine-tune runs
- `14_generate_episode_from_trained_model.py`: generate a new synthetic episode blueprint
- `15_generate_storyboard_assets.py`: build scene-level storyboard seed assets and backend-ready per-scene input payloads from exported storyboard requests
- `16_run_storyboard_backend.py`: materialize local backend-style storyboard scene frames from the per-scene backend input payloads
- `17_render_episode.py`: render a draft preview plus a final voiced episode from seed assets, backend frames, generated scene videos, or placeholder scene cards, reuse matching original dialogue where possible, and export/update a backend-ready production package for a fully generated episode
- `18_build_series_bible.py`: rebuild the series bible
- `19_generate_finished_episodes.py`: generate multiple finished episodes in one run or keep generating endlessly until stopped
- `20_refresh_after_manual_review.py`: rebuild the pipeline after manual character review
- `99_process_next_episode.py`: run the full end-to-end workflow
- `pipeline_common.py`: shared helpers for paths, config, runtime, progress reporting, and status handling

### Important Project Folders

- `ai_series_project/data/inbox/episodes`: new source episode files
- `ai_series_project/data/raw/episodes`: imported working copies
- `ai_series_project/data/raw/audio`: extracted audio
- `ai_series_project/data/processed/scene_clips`: scene clips per episode
- `ai_series_project/data/processed/scene_index`: scene CSV indexes
- `ai_series_project/data/processed/speaker_segments`: segmented speaker caches
- `ai_series_project/data/processed/speaker_transcripts`: merged transcript outputs
- `ai_series_project/data/processed/faces`: face detection caches
- `ai_series_project/data/processed/linked_segments`: linked face / speaker / scene data
- `ai_series_project/data/datasets/video_training`: local training datasets
- `ai_series_project/characters/maps`: `character_map.json` and `voice_map.json`
- `ai_series_project/characters/review`: review queue artifacts
- `ai_series_project/characters/voice_samples`: per-character voice references
- `ai_series_project/characters/voice_models`: per-character local voice profiles
- `ai_series_project/generation/model`: trained local series model
- `ai_series_project/generation/story_prompts`: generated markdown episodes
- `ai_series_project/generation/shotlists`: generated shotlists
- `ai_series_project/generation/storyboard_requests`: backend-ready storyboard request payloads per episode and per scene
- `ai_series_project/generation/storyboard_assets`: optional generated scene frames that the render step can reuse
- `ai_series_project/generation/storyboard_assets/<episode>/*_backend_input.json`: per-scene backend-ready seed payloads for later local image/video model runners
- `ai_series_project/generation/final_episode_packages/<episode>`: backend-ready full-episode package with per-scene image/video/voice/lip-sync plans and stable output targets for actual generated episodes
- `ai_series_project/generation/renders/drafts`: draft renders
- `ai_series_project/generation/renders/final`: final voiced storyboard episode renders
- `ai_series_project/generation/renders/final/*_dialogue_audio.wav`: assembled dialogue audio track for the voiced final episode
- `ai_series_project/generation/renders/final/*_voice_plan.json`: timed dialogue and speaker plan for later voiced render backends
- `ai_series_project/generation/renders/final/*_dialogue_preview.srt`: subtitle-style dialogue preview aligned to the silent storyboard render timeline
- `ai_series_project/generation/final_episode_packages/<episode>/audio/<speaker>/line_*.wav`: materialized per-line voice assets, now filled automatically from original dialogue reuse or local fallback TTS during step `17`
- `ai_series_project/generation/final_episode_packages/<episode>/audio/<scene>/<scene>_dialogue.wav`: per-scene dialogue tracks aligned to the generated scene timing, ready for later lip-sync or video backends
- `ai_series_project/generation/final_episode_packages/<episode>/master/scenes/*_master.mp4`: per-scene mastered clips that combine each scene video with its own dialogue track when available
- `ai_series_project/generation/final_episode_packages/<episode>/master/*_production_package.json`: master production package for a later full generated episode
- `ai_series_project/generation/final_episode_packages/<episode>/master/*_full_generated_episode.mp4`: full episode master that `17_render_episode.py` now writes from generated scene clips when available, with storyboard-card fallback for missing scenes
- `ai_series_project/characters/voice_models/*_voice_model.json`: local per-character voice-model profiles with reference audio, language counts, dominant language, and original source segments
- `ai_series_project/series_bible/episode_summaries`: generated series bible files
- `ai_series_project/runtime/autosaves`: autosaves and resumable run state
- `ai_series_project/runtime/distributed`: shared NAS lease files for cooperative multi-PC workers such as step `04`
- `ai_series_project/tools/ffmpeg/bin`: OS-specific local FFmpeg binaries used by split, transcription, training prep, storyboard asset generation, and render steps
- `runtime/venv_<os>_<arch>_<bitness>`: local Python environment for the current machine and runtime architecture on Windows; Linux/NAS uses the active `python3` runtime directly

## Requirements

- Windows / PowerShell or Linux with `python3`
- working local Python
- enough disk space for scenes, audio, model assets, renders, and the local FFmpeg download
- patience: `04`, `05`, and the training stages can take a long time
- for NAS-based shared-worker mode in `04`, all participating PCs must see the same project path and the same `ai_series_project/runtime/distributed` lease files
- for project-wide NAS/shared-worker mode, all participating PCs must see the same project path and the same `ai_series_project/runtime/distributed` lease files
- optional NVIDIA GPU for faster transcription, embeddings, and rendering

`00_prepare_runtime.py` prepares the runtime. Depending on availability, the stack may include:

- `torch`
- `torchvision`
- `torchaudio`
- `numpy`
- `Pillow`
- `opencv-python`
- `librosa`
- `openai-whisper`
- `scenedetect[opencv]`
- `facenet-pytorch`
- `speechbrain`
- `pyttsx3`
- optional `TTS` only when explicitly enabled

## Quick Start

1. Drop new source episodes into `ai_series_project/data/inbox/episodes`.
2. Run `python 00_prepare_runtime.py`
3. Run `python 01_setup_project.py`
4. Run `python 99_process_next_episode.py`
5. Review results in:
   - `generation/story_prompts`
   - `generation/shotlists`
   - `generation/final_episode_packages`
   - `generation/renders/drafts`
   - `generation/renders/final`
   - `generation/final_episode_packages/<episode>/master/*_full_generated_episode.mp4`
   - `series_bible/episode_summaries`

## Workflow Summary

### 00 - Prepare Runtime

Creates `runtime/venv_<os>_<arch>_<bitness>` on Windows, updates packaging tools, installs dependencies, prepares the project structure, and prefers CUDA-capable Torch when possible. The runtime step now installs base packages first, then torch, and only afterwards torch-dependent packages such as Whisper and SpeechBrain so Linux setups do not fail on dependency order. On Linux/NAS it now uses the active `python3` interpreter directly and installs via `python3 -m pip install --break-system-packages`, which avoids Synology-style cases where a separate venv still installs packages outside the actually usable runtime. It also downloads the matching FFmpeg build for the current OS into `ai_series_project/tools/ffmpeg/bin`, so Windows gets `.exe` binaries while Linux gets native Linux binaries.

The default path stays license-light. Optional XTTS / Coqui is only prepared when explicitly enabled. XTTS must never be enabled implicitly through a hidden license acceptance.

In shared NAS mode, this step uses one exclusive global lease so multiple PCs do not try to recreate the same runtime or tool folder at the same time.

### 01 - Set Up Project

Ensures that the expected folder structure and config file exist.

In shared NAS mode, this step uses one exclusive global lease so several PCs cannot rewrite the same project structure simultaneously.

### 02 - Import Episode

Imports the next unprocessed source episode from the inbox into `data/raw/episodes`, writes metadata, and removes the inbox source after the working copy has been safely created. In shared NAS mode, inbox files are leased one by one so multiple PCs can import different source episodes without colliding.

### 03 - Split Scenes

Splits imported episodes into scene clips, writes a scene index CSV, and now works through the available batch instead of only one episode. In shared NAS mode, whole episodes are leased so several PCs can split different imported episodes in parallel.

### 04 - Diarize And Transcribe

Extracts audio, runs Whisper, auto-detects the spoken language unless a fixed language is explicitly configured, builds speaker segments, computes speaker embeddings, and clusters speakers. It now also includes additional rescue passes for unresolved `speaker_unknown` cases.

On a NAS-backed workspace, multiple PCs can now run `04_diarize_and_transcribe.py` at the same time. They coordinate through shared lease files under `ai_series_project/runtime/distributed/04_diarize_and_transcribe/...`:

- workers claim one scene at a time instead of locking the whole episode
- several PCs can therefore split one episode across many scenes in parallel
- if one PC exits or crashes, its lease expires and another PC can take over that scene automatically
- the final speaker-clustering/merge step is also protected by a short shared finalize lease so only one worker writes the final episode outputs at a time

Useful flags:

- `--no-shared-workers`: force local single-worker behavior
- `--worker-id <name>`: assign a stable readable worker id for NAS runs

### 05 - Link Faces And Speakers

Detects faces, clusters them, links visible faces to dialogue segments, and writes linked segment outputs. One-off background faces are filtered more aggressively so recurring characters are easier to review. In shared NAS mode, whole episodes are leased so several PCs can link different episodes in parallel.

### 06 - Review Unknowns

Interactive review stage for unknown or auto-named face clusters. It now tries to recognize already known characters first, supports stronger role hints, propagates new manual names back into open review state, and keeps its CLI/help and interactive prompts aligned with the English-first numbered pipeline.

In shared NAS mode, this step uses one exclusive global lease because face review changes the shared character and voice maps directly.

### 07 - Build Dataset

Builds the consolidated training dataset from linked segments and reviewed character data. In shared NAS mode, whole episodes are leased so several PCs can build different dataset episodes in parallel.

### 08 - Train Series Model

Builds the local heuristic series model from the reviewed datasets. This is not a large neural model; it is a structured local model used for preview generation, and its fallback summaries and beat templates now follow the same English-first output style as the downstream steps.

In shared NAS mode, this step uses one exclusive global lease so the shared `series_model.json` is only written by one worker at a time.

### 09 - Prepare Foundation Training

Prepares 720p frame, clip, and voice assets for later training stages. It can also manage model downloads and update checks for configured base assets, but now skips online voice-base downloads by default when local character voice models are enabled. The voice preparation path now prioritizes original per-character dialogue recordings from the reviewed dataset, keeps their detected languages, and writes those language-aware original voice references into the training manifests and markdown summary. In shared NAS mode, characters are leased individually so several PCs can prepare different character manifests in parallel.

### 10 - Train Foundation Models

Builds local foundation packs from the prepared assets and now also writes one local per-character voice-model profile under `characters/voice_models`, using original dialogue material, detected language counts, dominant language, and reference-audio paths. In shared NAS mode, characters are leased individually so several PCs can train different foundation packs in parallel.

### 11 - Train Adapter Models

Builds local adapter profiles for image, voice, and clip dynamics. The voice branch now carries the local voice-model path plus dominant-language metadata forward from the foundation stage. In shared NAS mode, characters are leased individually so several PCs can train different adapter profiles in parallel.

### 12 - Train Fine-Tune Models

Builds local fine-tune profiles on top of the adapter stage and keeps the per-character voice-model path and dominant language available for downstream backends. In shared NAS mode, characters are leased individually so several PCs can train different fine-tune profiles in parallel.

### 13 - Run Backend Finetunes

Turns the local fine-tune profiles into backend-oriented fine-tune runs and materialized local backend artifacts, now including the per-character voice-model path and dominant language for the voice backend. In shared NAS mode, characters are leased individually so several PCs can prepare different backend runs in parallel.

### 14 - Generate Episode From Trained Model

Generates a new synthetic episode blueprint from the trained local model. It restores real episode titles, blocks hard if required training stages are missing or outdated, and now writes a per-scene storyboard generation plan with:

- up to three reference slots
- previous-scene continuity anchors
- camera and composition guidance
- control hints for pose and staging
- model-oriented positive and negative prompts

In addition, `14` now exports backend-ready storyboard request files under `generation/storyboard_requests/<episode>`, including one episode-level request, one scene-level request per scene, and a prompt preview text file.

When no explicit episode ID is provided, the downstream storyboard, backend, render, and multi-episode helpers now select the newest matching generated artifact by timestamp rather than relying on a `folge_*` filename prefix.

In shared NAS mode, this step uses one exclusive lease per requested episode target. `--episode-id <id>` leases that concrete episode; the default `auto_next` mode is also protected so two PCs do not generate the same next episode simultaneously.

### 15 - Generate Storyboard Assets

Builds scene-level storyboard asset files from the exported storyboard requests. This step prepares `generation/storyboard_assets/<episode>` so later render passes can already reuse scene frames when they exist. In shared NAS mode, scenes are leased individually so several PCs can build different storyboard scene assets for the same episode.

### 16 - Run Storyboard Backend

Materializes local backend-style scene frames from the per-scene backend input payloads written by `15`. This is the bridge between the seed-package export and later render reuse. In shared NAS mode, scenes are leased individually so several PCs can materialize different backend frames for the same episode.

### 17 - Render Episode

Builds the draft render from storyboard cards and the final episode from the best currently available scene material in this order: generated lip-sync clips, generated scene videos, backend/seed storyboard frames, and only then placeholder cards. It reuses matching original dialogue segments when available, fills only missing lines with local fallback TTS, writes per-line and per-scene audio into the production package, muxes the full dialogue track into the final render, and now also writes per-scene mastered clips plus a `*_full_generated_episode.mp4` master into `generation/final_episode_packages/<episode>/master`.

Renders a draft local preview plus a final voiced storyboard episode. The current default path:

- reuses materialized backend frames from `16_run_storyboard_backend.py` when they exist
- otherwise reuses generated storyboard seed assets from `generation/storyboard_assets/<episode>`
- generates readable placeholder scene cards when no visual asset exists yet
- preserves the per-scene storyboard plan in the render manifest for future model backends
- uses FFmpeg concat lists to avoid long Windows command line failures
- keeps the draft render as a fast silent preview
- reuses matching original dialogue segments from stored source audio or scene clips whenever they exist
- fills only the remaining missing lines with local `pyttsx3` speech synthesis while preferring the detected dominant language of each character when selecting a system voice
- muxes that mixed dialogue track into the final episode render when local audio assembly succeeds
- falls back to a silent final video only if local audio synthesis fails
- also writes a timed dialogue voice-plan JSON plus an `.srt` subtitle preview beside the final outputs
- materializes per-speaker line audio and per-scene dialogue tracks into `generation/final_episode_packages/<episode>/audio/...` so later lip-sync/video backends already receive concrete audio assets, not only planned paths
- writes `generation/final_episode_packages/<episode>/master/scenes/*_master.mp4` so every scene already has a reusable mastered clip for later assembly, replacement, or QA
- updates the production-package JSONs with the real currently available outputs and ready-count summaries, so later runners can distinguish between planned targets and already materialized assets

At the same time, `17` now exports `generation/final_episode_packages/<episode>/master/<episode>_production_package.json` plus per-scene production JSONs. Those packages define the next stage for a real newly generated episode:

- new scene keyframes instead of reused old-series frames
- new shot video clips per scene
- cloned original-character dialogue audio per line
- lip-sync composites per scene
- one stable target path contract that later local backends can fill without changing the numbered pipeline order

`17` uses a shared NAS lease at episode level. That means only one worker renders one specific episode at a time, but different PCs can still render different episodes without colliding.

### 18 - Build Series Bible

Rebuilds the compact series bible from the trained series model and current reviewed data, including an English-first markdown summary for downstream review output. It now also appends the most recent generated episodes with their render mode, production-readiness label, final render path, full generated episode master, production package, render manifest, scene-output counters, coverage ratios, and remaining backend tasks so the bible doubles as a lightweight production snapshot. In shared NAS mode, this step uses an exclusive global lease so the bible is not rewritten concurrently by several PCs.

### 19 - Generate Finished Episodes

Runs a full visible multi-episode generation flow, including rebuild, training, generation, storyboard backend materialization, and render stages. It now targets finished episodes instead of merely preview-labeled batches, rebuilds the series bible once after the full batch instead of repeating that step after every generated episode, records the planned/completed batch-step order in its step metadata for easier resume/debug inspection, stores the concrete output bundle per generated episode from step `17` for easier follow-up automation, respects the same optional foundation/adapter/fine-tune/backend training toggles used by the other orchestration scripts, supports `--skip-downloads` for the foundation-prepare step, and can also run endlessly with `--count 0` or `--endless`. In endless mode it updates the series bible after each new rendered episode because there is no final batch end. In shared NAS mode, `19` uses an orchestration lease so the same endless/batch run is not started twice by different PCs.

Its shared-worker flags are now propagated into every child step as well, so `--worker-id <name>` stays consistent across the whole batch and `--no-shared-workers` really disables NAS leasing for the complete nested run.

### 20 - Refresh After Manual Review

One-command rebuild path after manual character cleanup. This is the preferred way to rebuild dependent outputs after heavy review work, including refreshed storyboard assets, backend frames, render output, and the final series bible update. Its planned-step list now keeps the same train-then-generate/render ordering as the other orchestration scripts, `--stop-after-training` cuts the run cleanly after the active training block, and the refresh path now respects the same optional foundation/adapter/fine-tune/backend stage toggles as the other numbered orchestrators instead of always forcing `09` through `13`. In shared NAS mode, `20` uses an orchestration lease so the same rebuild run is not started twice by different PCs.

Its shared-worker flags are also forwarded into the nested numbered steps, so `--worker-id` and `--no-shared-workers` apply to the whole rebuild, not only to the coordinator itself.

### 99 - Full Pipeline

Runs the main end-to-end flow:

1. setup
2. import and preprocess every new inbox episode
3. stop for manual review if open review items remain
4. rebuild dataset and model
5. prepare and train downstream local packs
6. generate a new episode
7. generate storyboard seed assets and optional local backend frames
8. render finished episode outputs
9. rebuild the bible

The pipeline now writes autosaves, resumable checkpoints, and live status files for long-running batch work.
It also supports `--skip-downloads` for the foundation-prepare stage when existing model downloads should be reused, stores the planned/completed global step order in the autosave state so resume and status output reflect the real run plan, and now carries the latest generated episode output bundle through those status files so the current final render, full generated episode master, render manifest, production package, production-readiness label, coverage ratios, and remaining backend tasks are visible without digging through folders. In shared NAS mode, `99` uses an exclusive orchestration lease so there is only one full end-to-end coordinator at a time while the underlying numbered worker scripts still distribute the actual step work.

The coordinator now also forwards the shared-worker flags into setup, per-episode processing, and all later global steps. That means one `99` run keeps one stable NAS worker identity across its child steps, and `--no-shared-workers` truly forces the entire nested pipeline into local single-worker mode.

## Testing And Smoke Runs

Run the automated tests:

```powershell
python -m unittest discover -s tests -v
```

The current tests cover, among other things:

- placeholder name normalization
- face / speaker linking behavior
- generic episode fallbacks
- `statist` as minor-character status
- character priority behavior
- review-related regression cases

Recommended smoke run after larger changes:

```powershell
python 05_link_faces_and_speakers.py
python 07_build_dataset.py
python 08_train_series_model.py
python 09_prepare_foundation_training.py --skip-downloads
python 10_train_foundation_models.py
python 11_train_adapter_models.py
python 12_train_fine_tune_models.py
python 13_run_backend_finetunes.py
python 14_generate_episode_from_trained_model.py
python 15_generate_storyboard_assets.py
python 16_run_storyboard_backend.py
python 17_render_episode.py
python 18_build_series_bible.py
```

Shared NAS transcription example:

```powershell
python 04_diarize_and_transcribe.py --worker-id pc1
python 04_diarize_and_transcribe.py --worker-id pc2
```

Both workers will cooperate on the same pending episodes. They do not double-process the same scene unless a stale lease has to be recovered after a worker disappears.

Project-wide shared-worker flags:

- `--worker-id <name>`: optional readable worker id on NAS runs
- `--no-shared-workers`: disable NAS leasing for the current run

After major manual review work:

```powershell
python 20_refresh_after_manual_review.py --skip-downloads
```

## Known Limitations

- face / speaker quality still depends heavily on manual review quality in `06_review_unknowns.py`
- `character_map.json` and `voice_map.json` are still global, not per episode
- speaker linking remains heuristic rather than production-grade diarization
- the local series model is not comparable to a large multimodal frontier model
- even the voiced `final` episode is still a local storyboard production, not a true TV-grade final production
- voice similarity is strongest when original segment retrieval can be reused; fully new lines still fall back to generated speech
- lip-sync remains a local fallback path, not production-quality facial performance generation
- XTTS requires explicit installation, explicit license acceptance, and sufficient voice reference quality
- `large-v3` on CPU is possible but slow
- shared-worker mode in `04` parallelizes at scene level, not inside one single scene; one especially long scene is still handled by one worker at a time
- orchestration-heavy scripts such as `18`, `19`, `20`, and `99` mostly use exclusive leases to prevent conflicting duplicate runs; the fine-grained parallelism lives in the worker-heavy numbered steps underneath

## Maintenance Checklist

When a script changes, at minimum check whether the README must also be updated for:

- run commands
- new or removed CLI flags
- new environment variables
- changed input paths
- changed output paths
- new cache or reset behavior
- changed workflow order
- changed artifact filename assumptions
- new limitations or caveats

## Typical Daily Use

### Process New Source Episodes

```powershell
python 00_prepare_runtime.py
python 01_setup_project.py
python 99_process_next_episode.py --skip-downloads
```

### Generate New Finished Episodes From Existing Reviewed Data

```powershell
python 08_train_series_model.py
python 09_prepare_foundation_training.py --skip-downloads
python 10_train_foundation_models.py
python 11_train_adapter_models.py
python 12_train_fine_tune_models.py
python 13_run_backend_finetunes.py
python 14_generate_episode_from_trained_model.py
python 15_generate_storyboard_assets.py
python 16_run_storyboard_backend.py
python 17_render_episode.py
python 18_build_series_bible.py
```

### Rebuild After Manual Character Review

```powershell
python 20_refresh_after_manual_review.py --skip-downloads
```

### Generate Multiple Finished Episodes

```powershell
python 19_generate_finished_episodes.py --count 2 --skip-downloads
```

Endless mode:

```powershell
python 19_generate_finished_episodes.py
python 19_generate_finished_episodes.py --endless --skip-downloads
```
