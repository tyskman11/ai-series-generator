# AI Series Training

## Purpose

This project turns existing TV episodes into a local AI training and preview-generation pipeline.

Starting from real episode files, the pipeline can:

- import and split source episodes into scenes
- transcribe dialogue and build speaker clusters
- detect faces and link recurring characters to speakers
- build a local series dataset and heuristic series model
- train local foundation, adapter, fine-tune, and backend preparation artifacts
- generate new synthetic preview episodes as markdown and shotlists
- render draft and final storyboard preview videos

The default path stays local-first and license-light. Voice cloning and stronger TTS backends remain optional follow-up paths rather than the default preview flow.

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
| Generation / render | preview-ready | New episodes, shotlists, draft videos, and final preview videos can be produced. |
| GitHub sync | active | All root Python scripts, including the sync helper itself, plus `README.md` are mirrored to GitHub, never downloaded back. |

## What Already Works Well

- full local preprocessing from inbox episode to linked scene data
- batch processing over multiple new source episodes
- synthetic preview episode generation from the trained local model
- model-native storyboard planning per scene, with multi-reference slots, continuity anchors, and camera/control hints for later image-model backends
- storyboard scene asset generation now writes backend-ready per-scene seed packages and can reuse moved training artifacts after workspace path changes
- local storyboard backend materialization can turn those seed packages into reusable per-scene backend frames before render
- real episode titles for generated outputs instead of raw `folge_0x` placeholders
- draft and final silent storyboard preview rendering
- autosaves and live progress dashboards for long-running pipelines
- local voice fallback with German Windows voices instead of old English default fallbacks
- GitHub mirroring that updates the repo `About` text, includes the sync helper itself, and never uses `clone`, `fetch`, or `pull`

## Current Focus

- separate main recurring characters from background faces faster and more reliably
- auto-merge already known faces before manual review so review mostly sees true unknowns
- keep the default path robust, local, and free of unnecessary license prompts
- harden the end-to-end batch pipeline for large inbox runs
- improve the voice path so future local cloning backends can replace simple fallback TTS where appropriate
- improve the model-side storyboard planning so future image/video backends can use original-series references without relying on any graphical node editor or UI-specific workflow

## In Progress

- `06_review_unknowns.py` now tries to re-identify known faces before manual naming, using multi-reference matching per character
- `06_review_unknowns.py` keeps iterating after each manual naming step, so newly named characters immediately help resolve more open clusters
- `04_diarize_and_transcribe.py` now applies extra rescue passes for remaining `speaker_unknown` segments using neighborhood and embedding agreement
- `99_process_next_episode.py` is being hardened for real inbox batch workflows with autosaves, resumable checkpoints, and live status files
- `18_render_episode.py` continues to improve long Windows render runs, especially for large segment stacks
- the new training chain around `09` through `13` is being observed on real project data, especially where voice material is still weak
- `14_generate_episode_from_trained_model.py` now writes per-scene multi-reference storyboard plans inspired by shot-by-shot image-edit workflows, but stays model-only and does not depend on any GUI workflow
- `16_run_storyboard_backend.py` can now materialize local backend-style scene frames from the seed payloads emitted by `15_generate_storyboard_assets.py`
- `18_render_episode.py` now carries storyboard plans into the render manifest, reuses backend frames from `16` when they exist, and falls back to seed assets or generated placeholder cards when needed
- `14_generate_episode_from_trained_model.py` now also exports backend-ready storyboard request files per episode and per scene under `generation/storyboard_requests`
- `18_render_episode.py` now automatically picks up already generated storyboard scene frames from `generation/storyboard_assets/<episode>` when they exist, so local backend materialization and later model backends can feed visuals back into the render path without changing script order
- `15_generate_storyboard_assets.py` now emits backend-ready scene input payloads beside each generated seed frame and rebases older stored artifact paths after project moves, so NAS relocations do not silently break later backend use
- `15_generate_storyboard_assets.py`, `16_run_storyboard_backend.py`, `18_render_episode.py`, and `19_generate_preview_episodes.py` now resolve the newest generated artifact by timestamp instead of assuming a `folge_*` filename pattern

## Planned

- finish naming main characters in `06_review_unknowns.py`
- mark minor characters as `statist` where appropriate so they stay visible but do not become main roles
- run `20_refresh_after_manual_review.py` on the fully reviewed set to rebuild datasets, model, training packs, generated episodes, bible, and renders
- keep reducing `speaker_unknown` cases across full seasons
- expand the fine-tune and backend stages from local preparation artifacts into real model-weight training later on
- connect the new backend-ready storyboard seed packages from `15_generate_storyboard_assets.py` to an actual local image/video model runner later on
- improve render quality, character consistency, and synthetic episode quality after the review and training loop stabilizes
- add an optional voiced render path on top of the current silent storyboard preview video flow
- only re-enable stronger lip-sync paths when they actually look series-quality

## Documentation Rule

This file is mandatory documentation.

Whenever you change any of the following, update `README.md` in the same task:

- `00_prepare_runtime.py` through `21_sync_to_github.py`
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
- `14_generate_episode_from_trained_model.py`: generate a new synthetic preview episode
- `15_generate_storyboard_assets.py`: build scene-level storyboard seed assets and backend-ready per-scene input payloads from exported storyboard requests
- `16_run_storyboard_backend.py`: materialize local backend-style storyboard scene frames from the per-scene backend input payloads
- `17_build_series_bible.py`: rebuild the series bible
- `18_render_episode.py`: render draft and final storyboard preview videos from seed assets, backend frames, or placeholder scene cards
- `19_generate_preview_episodes.py`: generate multiple visible preview episodes in one run
- `20_refresh_after_manual_review.py`: rebuild the pipeline after manual character review
- `21_sync_to_github.py`: mirror all root local Python scripts plus `README.md` to GitHub
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
- `ai_series_project/generation/renders/drafts`: draft renders
- `ai_series_project/generation/renders/final`: final preview renders
- `ai_series_project/series_bible/episode_summaries`: generated series bible files
- `ai_series_project/runtime/autosaves`: autosaves and resumable run state
- `runtime/venv_<os>_<arch>_<bitness>`: local Python environment for the current machine and runtime architecture

## Requirements

- Windows / PowerShell
- working local Python
- enough disk space for scenes, audio, model assets, and renders
- patience: `04`, `05`, and the training stages can take a long time
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
   - `generation/renders/drafts`
   - `generation/renders/final`
   - `series_bible/episode_summaries`

To mirror the current code to GitHub:

```powershell
python 21_sync_to_github.py
```

## Workflow Summary

### 00 - Prepare Runtime

Creates `runtime/venv_<os>_<arch>_<bitness>`, updates packaging tools, installs dependencies, prepares the project structure, and prefers CUDA-capable Torch when possible.

The default path stays license-light. Optional XTTS / Coqui is only prepared when explicitly enabled. XTTS must never be enabled implicitly through a hidden license acceptance.

### 01 - Set Up Project

Ensures that the expected folder structure and config file exist.

### 02 - Import Episode

Imports the next unprocessed source episode from the inbox into `data/raw/episodes`, writes metadata, and removes the inbox source after the working copy has been safely created.

### 03 - Split Scenes

Splits imported episodes into scene clips, writes a scene index CSV, and now works through the available batch instead of only one episode.

### 04 - Diarize And Transcribe

Extracts audio, runs Whisper, builds speaker segments, computes speaker embeddings, and clusters speakers. It now also includes additional rescue passes for unresolved `speaker_unknown` cases.

### 05 - Link Faces And Speakers

Detects faces, clusters them, links visible faces to dialogue segments, and writes linked segment outputs. One-off background faces are filtered more aggressively so recurring characters are easier to review.

### 06 - Review Unknowns

Interactive review stage for unknown or auto-named face clusters. It now tries to recognize already known characters first, supports stronger role hints, and propagates new manual names back into open review state.

### 07 - Build Dataset

Builds the consolidated training dataset from linked segments and reviewed character data.

### 08 - Train Series Model

Builds the local heuristic series model from the reviewed datasets. This is not a large neural model; it is a structured local model used for preview generation.

### 09 - Prepare Foundation Training

Prepares 720p frame, clip, and voice assets for later training stages. It can also manage model downloads and update checks for configured base assets.

### 10 - Train Foundation Models

Builds local foundation packs from the prepared assets.

### 11 - Train Adapter Models

Builds local adapter profiles for image, voice, and clip dynamics.

### 12 - Train Fine-Tune Models

Builds local fine-tune profiles on top of the adapter stage.

### 13 - Run Backend Finetunes

Turns the local fine-tune profiles into backend-oriented fine-tune runs and materialized local backend artifacts.

### 14 - Generate Episode From Trained Model

Generates a new synthetic preview episode from the trained local model. It restores real episode titles, blocks hard if required training stages are missing or outdated, and now writes a per-scene storyboard generation plan with:

- up to three reference slots
- previous-scene continuity anchors
- camera and composition guidance
- control hints for pose and staging
- model-oriented positive and negative prompts

In addition, `14` now exports backend-ready storyboard request files under `generation/storyboard_requests/<episode>`, including one episode-level request, one scene-level request per scene, and a prompt preview text file.

When no explicit episode ID is provided, the downstream storyboard, backend, render, and multi-episode helpers now select the newest matching generated artifact by timestamp rather than relying on a `folge_*` filename prefix.

### 15 - Generate Storyboard Assets

Builds scene-level storyboard asset files from the exported storyboard requests. This step prepares `generation/storyboard_assets/<episode>` so later render passes can already reuse scene frames when they exist.

### 16 - Run Storyboard Backend

Materializes local backend-style scene frames from the per-scene backend input payloads written by `15`. This is the bridge between the seed-package export and later render reuse.

### 17 - Build Series Bible

Rebuilds the compact series bible from the trained series model and current reviewed data.

### 18 - Render Episode

Renders a draft and final local storyboard preview video. The current default path:

- reuses materialized backend frames from `16_run_storyboard_backend.py` when they exist
- otherwise reuses generated storyboard seed assets from `generation/storyboard_assets/<episode>`
- generates readable placeholder scene cards when no visual asset exists yet
- preserves the per-scene storyboard plan in the render manifest for future model backends
- uses FFmpeg concat lists to avoid long Windows command line failures
- currently renders a silent storyboard preview rather than a voiced final mix

### 19 - Generate Preview Episodes

Runs a full visible multi-episode generation flow, including rebuild, training, generation, storyboard backend materialization, and render stages.

### 20 - Refresh After Manual Review

One-command rebuild path after manual character cleanup. This is the preferred way to rebuild dependent outputs after heavy review work, including refreshed storyboard assets, backend frames, bible output, and render output.

### 21 - Sync To GitHub

Mirrors all root `*.py` files plus `README.md` to GitHub, including `21_sync_to_github.py` itself.

Key rules:

- never downloads from GitHub
- never uses `clone`, `fetch`, or `pull`
- updates the repository `About` text on every run
- keeps local as the source of truth
- ends cleanly when there are no allowed file changes to commit

The GitHub `About` text states that the project is a local AI pipeline for learning from TV episodes and generating new preview episodes, and that all scripts are AI-generated with `GPT-5.4`.

### 99 - Full Pipeline

Runs the main end-to-end flow:

1. setup
2. import and preprocess every new inbox episode
3. stop for manual review if open review items remain
4. rebuild dataset and model
5. prepare and train downstream local packs
6. generate a new episode
7. generate storyboard seed assets and optional local backend frames
8. rebuild the bible
9. render preview outputs

The pipeline now writes autosaves, resumable checkpoints, and live status files for long-running batch work.

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
python 17_build_series_bible.py
python 18_render_episode.py
```

After major manual review work:

```powershell
python 20_refresh_after_manual_review.py --skip-downloads
```

## Known Limitations

- face / speaker quality still depends heavily on manual review quality in `06_review_unknowns.py`
- `character_map.json` and `voice_map.json` are still global, not per episode
- speaker linking remains heuristic rather than production-grade diarization
- the local series model is not comparable to a large multimodal frontier model
- even the `final` video is still a local preview output, not a true TV-grade final production
- voice similarity is strongest when original segment retrieval can be reused; fully new lines still fall back to generated speech
- lip-sync remains a local fallback path, not production-quality facial performance generation
- XTTS requires explicit installation, explicit license acceptance, and sufficient voice reference quality
- `large-v3` on CPU is possible but slow

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
python 99_process_next_episode.py
```

### Generate New Preview Episodes From Existing Reviewed Data

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
python 17_build_series_bible.py
python 18_render_episode.py
```

### Rebuild After Manual Character Review

```powershell
python 20_refresh_after_manual_review.py --skip-downloads
```

### Generate Multiple Visible Preview Episodes

```powershell
python 19_generate_preview_episodes.py --count 2
```
