#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from support_scripts.pipeline_common import (
    PROJECT_ROOT,
    add_shared_worker_arguments,
    coalesce_text,
    derive_prompt_constraints_from_bible,
    display_person_name,
    distributed_item_lease,
    distributed_step_runtime_root,
    error,
    extract_keywords,
    has_primary_person_name,
    is_background_person_name,
    headline,
    info,
    keyword_token_allowed,
    load_character_continuity_memory,
    load_config,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    read_json,
    rerun_in_runtime,
    resolve_project_path,
    shared_worker_id_for_args,
    shared_workers_enabled_for_args,
    tokens_from_text,
    write_json,
    write_text,
)

GENERIC_FOCUS_CHARACTERS = ["Hauptfigur A", "Hauptfigur B", "Hauptfigur C"]
TITLE_PREFIXES = [
    "Das",
    "Der",
    "Die",
    "Chaos um",
    "Alarm um",
    "Verwechslung um",
    "Das große",
]
TITLE_SUFFIXES = [
    "Spiel",
    "Geheimnis",
    "Plan",
    "Code",
    "Trick",
    "Durcheinander",
]
TITLE_WEAK_TERMS = {
    "alter",
    "babe",
    "denn",
    "echt",
    "game",
    "gott",
    "kinder",
    "meinem",
    "meine",
    "meiner",
    "meines",
    "soll",
    "teil",
    "weiß",
    "weiss",
    "wieder",
}
SAFE_SHORT_TITLE_TERMS = {"app", "code", "chat", "deal", "geld", "plan", "song", "spiel", "test", "web"}
GENERIC_TITLE_FALLBACKS = [
    "Der große Plan",
    "Das geheime Projekt",
    "Die doppelte Verwechslung",
    "Das totale Durcheinander",
    "Der falsche Trick",
    "Alarm im Studio",
    "Die verrückte Challenge",
    "Das geheime Update",
    "Der App-Plan",
    "Das Missverständnis",
]
TITLE_PHRASE_PATTERN = re.compile(
    r"\b(?P<article>Der|Die|Das)\s+(?P<noun>[A-ZÄÖÜ][A-Za-zÄÖÜäöüß-]{3,})\b"
)
TITLE_TOKEN_PATTERN = re.compile(r"\b([A-ZÄÖÜ][A-Za-zÄÖÜäöüß-]{3,})\b")
TITLE_PHRASE_STOPWORDS = {
    "Danke",
    "Dankeschön",
    "Entschuldigung",
    "Erstens",
    "Genau",
    "Vielen",
    "Videos",
}
CAMERA_PRESETS = {
    "Cold Open": [
        {"shot_type": "wide establishing shot", "composition": "two-subject frame", "camera_move": "static", "lens_hint": "24mm", "pose_hint": "introduce characters in place"},
        {"shot_type": "medium two-shot", "composition": "balanced eye-level framing", "camera_move": "slow push-in", "lens_hint": "35mm", "pose_hint": "conversation starter beat"},
    ],
    "Plan": [
        {"shot_type": "medium shot", "composition": "over-table planning frame", "camera_move": "subtle dolly-in", "lens_hint": "40mm", "pose_hint": "characters pointing at plan or prop"},
        {"shot_type": "over-the-shoulder shot", "composition": "focus on problem-solving detail", "camera_move": "static", "lens_hint": "50mm", "pose_hint": "one character explains while the other reacts"},
    ],
    "Komplikation": [
        {"shot_type": "dynamic medium shot", "composition": "off-center tension framing", "camera_move": "handheld energy", "lens_hint": "35mm", "pose_hint": "sudden reaction or interruption"},
        {"shot_type": "close reaction shot", "composition": "tight emotional framing", "camera_move": "quick push-in", "lens_hint": "70mm", "pose_hint": "surprised or stressed expression"},
    ],
    "Verwechslung": [
        {"shot_type": "split-focus two-shot", "composition": "misaligned eyelines", "camera_move": "static", "lens_hint": "50mm", "pose_hint": "characters talking past each other"},
        {"shot_type": "comedic close-up", "composition": "punchline framing", "camera_move": "snap zoom feel", "lens_hint": "85mm", "pose_hint": "confused or incredulous expression"},
    ],
    "Wendepunkt": [
        {"shot_type": "hero medium close-up", "composition": "resolved center frame", "camera_move": "slow reveal push-in", "lens_hint": "65mm", "pose_hint": "moment of realization"},
        {"shot_type": "insert and reaction pair", "composition": "detail then face", "camera_move": "rack-focus feel", "lens_hint": "55mm", "pose_hint": "show clue or fix clearly"},
    ],
    "Auflösung": [
        {"shot_type": "wide payoff shot", "composition": "clean resolved staging", "camera_move": "gentle pull-back", "lens_hint": "28mm", "pose_hint": "show outcome and group energy"},
        {"shot_type": "warm medium shot", "composition": "friendly end beat", "camera_move": "static", "lens_hint": "45mm", "pose_hint": "relaxed success pose"},
    ],
}


def next_episode_id(story_dir) -> str:
    numbers = []
    for file in sorted(story_dir.glob("folge_*.md")):
        try:
            numbers.append(int(file.stem.split("_")[1]))
        except Exception:
            pass
    return f"folge_{((max(numbers) + 1) if numbers else 1):02d}"


def parse_episode_index(episode_id: str) -> int:
    try:
        return max(1, int(str(episode_id).split("_")[1]))
    except Exception:
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the series model from the available datasets")
    add_shared_worker_arguments(parser)
    return parser.parse_args()


def useful_character(name: str) -> bool:
    return has_primary_person_name(name)


def background_character(name: str) -> bool:
    return is_background_person_name(name)


def fallback_focus_characters(count: int = 2) -> list[str]:
    return GENERIC_FOCUS_CHARACTERS[: max(2, min(count, len(GENERIC_FOCUS_CHARACTERS)))]


def humanize_keyword(keyword: str) -> str:
    cleaned = coalesce_text(keyword).replace("_", " ").replace("-", " ")
    tokens = [token for token in cleaned.split() if token]
    if not tokens:
        return "Geheimnis"
    return " ".join(token.capitalize() for token in tokens[:3])


def format_episode_number(episode_index: int) -> str:
    return f"Folge {max(1, int(episode_index)):02d}"


def build_episode_title(
    focus_characters: list[str],
    keywords: list[str],
    rng: random.Random,
    episode_index: int,
    model: dict | None = None,
) -> str:
    def focus_name_tokens() -> set[str]:
        tokens: set[str] = set()
        for name in focus_characters:
            for token in tokens_from_text(name):
                cleaned = token.lower()
                if cleaned:
                    tokens.add(cleaned)
        return tokens

    def title_keyword_allowed_local(keyword: str, blocked_tokens: set[str]) -> bool:
        cleaned = coalesce_text(keyword).replace("_", " ").replace("-", " ").strip()
        if not cleaned:
            return False
        tokens = [token.lower() for token in tokens_from_text(cleaned)]
        if not tokens:
            return False
        if any(token in blocked_tokens for token in tokens):
            return False
        if any(token in TITLE_WEAK_TERMS for token in tokens):
            return False
        if any(not keyword_token_allowed(token) for token in tokens):
            return False
        if len(tokens) == 1 and len(tokens[0]) < 4 and tokens[0] not in SAFE_SHORT_TITLE_TERMS:
            return False
        return True

    def keyword_root_forms(keyword: str) -> set[str]:
        cleaned = coalesce_text(keyword).replace("_", " ").replace("-", " ").lower()
        roots: set[str] = set()
        for token in tokens_from_text(cleaned):
            token = token.lower()
            if len(token) >= 4:
                roots.add(token)
            if len(token) >= 6 and token.endswith("en"):
                roots.add(token[:-2])
            if len(token) >= 6 and token.endswith("er"):
                roots.add(token[:-2])
            if len(token) >= 6 and token.endswith("e"):
                roots.add(token[:-1])
        return {root for root in roots if len(root) >= 4}

    def title_word_allowed(word: str, blocked_tokens: set[str]) -> bool:
        lower = coalesce_text(word).lower()
        if not lower:
            return False
        if lower in blocked_tokens or lower in TITLE_WEAK_TERMS:
            return False
        if lower in {token.lower() for token in TITLE_PHRASE_STOPWORDS}:
            return False
        if any(not keyword_token_allowed(token) for token in tokens_from_text(lower)):
            return False
        return True

    def collect_title_anchor_candidates(
        model_payload: dict | None,
        blocked_tokens: set[str],
        usable_title_keywords: list[str],
    ) -> list[tuple[float, str]]:
        if not model_payload or not usable_title_keywords:
            return []
        root_map = {keyword: keyword_root_forms(keyword) for keyword in usable_title_keywords}
        candidates: dict[str, float] = {}
        source_texts: list[str] = []
        for entries in (model_payload.get("speaker_line_library", {}) or {}).values():
            for entry in entries[:60]:
                text = coalesce_text(str(entry.get("text", "")))
                if text:
                    source_texts.append(text)
        for text in source_texts[:800]:
            lower_text = text.lower()
            for keyword, roots in root_map.items():
                if roots and not any(root in lower_text for root in roots):
                    continue
                for match in TITLE_PHRASE_PATTERN.finditer(text):
                    article = match.group("article")
                    noun = match.group("noun")
                    noun_lower = noun.lower()
                    if not title_word_allowed(noun, blocked_tokens):
                        continue
                    if roots and not any(root in noun_lower for root in roots):
                        continue
                    score = 2.4
                    if noun_lower != keyword.lower() and any(root in noun_lower for root in roots):
                        score += 0.8
                    if "-" in noun:
                        score += 0.15
                    phrase = f"{article} {noun}"
                    candidates[phrase] = max(candidates.get(phrase, 0.0), score)
                for noun in TITLE_TOKEN_PATTERN.findall(text):
                    noun_lower = noun.lower()
                    if not title_word_allowed(noun, blocked_tokens):
                        continue
                    if roots and not any(root in noun_lower for root in roots):
                        continue
                    score = 1.6
                    if noun_lower != keyword.lower() and any(root in noun_lower for root in roots):
                        score += 1.1
                    if "-" in noun:
                        score += 0.1
                    candidates[noun] = max(candidates.get(noun, 0.0), score)
        return sorted(candidates.items(), key=lambda item: (-item[1], item[0]))

    blocked_tokens = focus_name_tokens()
    usable_keywords: list[str] = []
    seen_keywords: set[str] = set()
    for keyword in keywords:
        if not title_keyword_allowed_local(keyword, blocked_tokens):
            continue
        humanized = humanize_keyword(keyword)
        normalized = humanized.lower()
        if normalized in seen_keywords:
            continue
        seen_keywords.add(normalized)
        usable_keywords.append(humanized)

    title_seed = episode_index + len(focus_characters) + len(usable_keywords)
    if not usable_keywords:
        return GENERIC_TITLE_FALLBACKS[title_seed % len(GENERIC_TITLE_FALLBACKS)]

    main_keyword = usable_keywords[0]
    secondary_keyword = usable_keywords[1] if len(usable_keywords) > 1 else ""
    suffix = TITLE_SUFFIXES[(title_seed + len(main_keyword)) % len(TITLE_SUFFIXES)]
    main_compound = main_keyword.replace(" ", "-")
    anchor_candidates = collect_title_anchor_candidates(model, blocked_tokens, usable_keywords[:3])
    if anchor_candidates:
        best_anchor = anchor_candidates[0][0]
        if " " in best_anchor:
            return best_anchor
        return f"Das Geheimnis um {best_anchor}"

    candidate_titles: list[str] = []
    if " " in main_keyword:
        candidate_titles.extend(
            [
                f"Die Sache mit {main_keyword}",
                f"Das Geheimnis um {main_keyword}",
                f"Das Chaos um {main_keyword}",
            ]
        )
    else:
        candidate_titles.extend(
            [
                f"Das {main_compound}-Geheimnis",
                f"Der {main_compound}-{suffix}",
                f"Die Sache mit {main_keyword}",
                f"Die {main_compound}-Challenge",
            ]
        )

    if secondary_keyword:
        if " " not in main_keyword and " " not in secondary_keyword:
            candidate_titles.append(f"Das {main_compound}-{humanize_keyword(secondary_keyword).replace(' ', '-')}-Problem")
        else:
            candidate_titles.append(f"Das Geheimnis um {main_keyword}")

    cleaned_titles = [coalesce_text(title).strip() for title in candidate_titles if coalesce_text(title).strip()]
    deduped_titles: list[str] = []
    for title in cleaned_titles:
        if title not in deduped_titles:
            deduped_titles.append(title)
    return deduped_titles[title_seed % len(deduped_titles)]


def build_markov_chain(lines: list[str], order: int = 2) -> dict[str, dict[str, int]]:
    chain: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for line in lines:
        words = ["<s>"] * order + tokens_from_text(line) + ["</s>"]
        if len(words) <= order:
            continue
        for index in range(len(words) - order):
            state = " ".join(words[index : index + order]).lower()
            next_token = words[index + order]
            chain[state][next_token] += 1
    return {state: dict(next_tokens) for state, next_tokens in chain.items()}


def build_character_directory(char_map: dict) -> dict[str, dict]:
    directory: dict[str, dict] = {}
    for cluster_id, payload in char_map.get("clusters", {}).items():
        if payload.get("ignored"):
            continue
        name = display_person_name(str(payload.get("name", "")), cluster_id)
        if not name:
            continue
        entry = directory.setdefault(
            name,
            {
                "priority": False,
                "background_role": False,
                "scene_count_hint": 0,
                "detection_count_hint": 0,
                "face_clusters": [],
            },
        )
        entry["priority"] = entry["priority"] or bool(payload.get("priority", False))
        entry["background_role"] = entry["background_role"] or background_character(name)
        entry["scene_count_hint"] += int(payload.get("scene_count", 0) or 0)
        entry["detection_count_hint"] += int(payload.get("detection_count", 0) or 0)
        if cluster_id not in entry["face_clusters"]:
            entry["face_clusters"].append(cluster_id)
        entry["face_cluster_count"] = len(entry["face_clusters"])
    return directory


def extract_scene_characters(row: dict, character_directory: dict[str, dict]) -> list[str]:
    names: list[str] = []
    for name in row.get("characters_visible", []) or []:
        if useful_character(name) or background_character(name):
            if name not in names:
                names.append(name)

    cluster_to_name: dict[str, str] = {}
    for name, payload in character_directory.items():
        for cluster_id in payload.get("face_clusters", []):
            cluster_to_name[cluster_id] = name

    for cluster in row.get("face_clusters", []) or []:
        cluster_id = ""
        if isinstance(cluster, dict):
            cluster_id = str(cluster.get("cluster_id", ""))
        elif isinstance(cluster, str):
            cluster_id = cluster
        mapped = cluster_to_name.get(cluster_id, "")
        if mapped and mapped not in names:
            names.append(mapped)

    for segment in row.get("transcript_segments", []) or []:
        for cluster_id in segment.get("visible_face_clusters", []) or []:
            mapped = cluster_to_name.get(str(cluster_id), "")
            if mapped and mapped not in names:
                names.append(mapped)
        speaker_name = coalesce_text(segment.get("speaker_name", ""))
        if useful_character(speaker_name) and speaker_name not in names:
            names.append(speaker_name)
    return names


def resolve_segment_character_name(segment: dict, row: dict) -> str:
    speaker_name = coalesce_text(segment.get("speaker_name", ""))
    if useful_character(speaker_name):
        return speaker_name
    for name in segment.get("visible_character_names", []) or []:
        candidate = coalesce_text(name)
        if useful_character(candidate):
            return candidate
    for name in row.get("characters_visible", []) or []:
        candidate = coalesce_text(name)
        if useful_character(candidate):
            return candidate
    return speaker_name


def append_unique_line_entry(target: dict[str, list[dict]], key: str, entry: dict) -> None:
    if not key:
        return
    segment_id = str(entry.get("segment_id", "")).strip()
    existing_ids = {str(item.get("segment_id", "")).strip() for item in target.get(key, [])}
    if segment_id and segment_id in existing_ids:
        return
    target.setdefault(key, []).append(entry)


def collect_preview_assets(preview_dir: str, patterns: list[str], limit: int = 3) -> list[str]:
    root = Path(preview_dir)
    if not root.exists():
        return []
    results: list[str] = []
    for pattern in patterns:
        for candidate in sorted(root.glob(pattern)):
            if candidate.exists():
                resolved = str(candidate.resolve())
                if resolved not in results:
                    results.append(resolved)
                if len(results) >= limit:
                    return results
    return results


def build_character_reference_library(char_map: dict) -> dict[str, dict]:
    library: dict[str, dict] = {}
    for cluster_id, payload in (char_map.get("clusters", {}) or {}).items():
        if payload.get("ignored"):
            continue
        name = display_person_name(str(payload.get("name", "")), cluster_id)
        if not name:
            continue
        preview_dir = str(payload.get("preview_dir", "") or "")
        entry = library.setdefault(
            name,
            {
                "context_images": [],
                "portrait_images": [],
                "priority": False,
                "background_role": False,
            },
        )
        entry["priority"] = entry["priority"] or bool(payload.get("priority", False))
        entry["background_role"] = entry["background_role"] or background_character(name)
        for image_path in collect_preview_assets(preview_dir, ["*_context.jpg", "*_speaker_frame_*.jpg", "*.jpg"], limit=4):
            if image_path not in entry["context_images"]:
                entry["context_images"].append(image_path)
        for image_path in collect_preview_assets(preview_dir, ["*_crop.jpg", "*.jpg"], limit=4):
            if image_path not in entry["portrait_images"]:
                entry["portrait_images"].append(image_path)
    return library


def build_linked_segment_line_library(cfg: dict) -> tuple[dict[str, list[str]], dict[str, list[dict]]]:
    linked_root = resolve_project_path(cfg["paths"]["linked_segments"])
    scene_root = resolve_project_path(cfg["paths"]["scene_clips"])
    audio_root = resolve_project_path("data/raw/audio")
    speaker_samples: dict[str, list[str]] = defaultdict(list)
    speaker_line_library: dict[str, list[dict]] = defaultdict(list)

    for linked_path in sorted(linked_root.glob("*_linked_segments.json")):
        episode_id = linked_path.stem.replace("_linked_segments", "")
        rows = read_json(linked_path, [])
        for row in rows:
            resolved_name = resolve_segment_character_name(row, {"characters_visible": row.get("visible_character_names", []) or []})
            text = coalesce_text(row.get("text", ""))
            if not useful_character(resolved_name) or not text:
                continue
            speaker_samples[resolved_name].append(text)
            append_unique_line_entry(
                speaker_line_library,
                resolved_name,
                {
                    "episode_id": episode_id,
                    "scene_id": row.get("scene_id", ""),
                    "segment_id": row.get("segment_id", ""),
                    "text": text,
                    "start": float(row.get("start", 0.0) or 0.0),
                    "end": float(row.get("end", 0.0) or 0.0),
                    "audio_file": str(audio_root / episode_id / f"{row.get('segment_id', '')}.wav"),
                    "video_file": str(scene_root / episode_id / f"{row.get('scene_id', '')}.mp4"),
                    "keywords": [],
                },
            )

    return speaker_samples, speaker_line_library


def build_series_model(dataset_files: list, cfg: dict, char_map: dict) -> dict:
    character_directory = build_character_directory(char_map)
    character_reference_library = build_character_reference_library(char_map)
    character_stats: dict[str, dict[str, int | bool]] = defaultdict(
        lambda: {"scene_count": 0, "line_count": 0, "priority": False, "face_cluster_count": 0}
    )
    speaker_samples: dict[str, list[str]] = defaultdict(list)
    speaker_line_library: dict[str, list[dict]] = defaultdict(list)
    transcripts = []
    scene_library = []
    speaker_counter: Counter[str] = Counter()
    keyword_counter: Counter[str] = Counter()
    episode_duration_counter: defaultdict[str, float] = defaultdict(float)
    segment_durations: list[float] = []
    scene_durations: list[float] = []

    for name, payload in character_directory.items():
        if useful_character(name):
            character_stats[name]["scene_count"] += int(payload.get("scene_count_hint", 0))
            character_stats[name]["priority"] = bool(payload.get("priority", False))
            character_stats[name]["face_cluster_count"] = int(payload.get("face_cluster_count", 0) or 0)

    for dataset_file in dataset_files:
        rows = read_json(dataset_file, [])
        for row in rows:
            transcript = coalesce_text(row.get("transcript", ""))
            if transcript:
                transcripts.append(transcript)
            scene_keywords = row.get("scene_keywords") or extract_keywords([transcript], limit=8)
            keyword_counter.update(scene_keywords)
            row_duration = float(row.get("duration_seconds", 0.0) or 0.0)
            episode_duration_counter[str(row.get("episode_id", ""))] += row_duration
            if row_duration > 0.0:
                scene_durations.append(row_duration)
            characters = [name for name in extract_scene_characters(row, character_directory) if useful_character(name)]
            for character in characters:
                character_stats[character]["scene_count"] += 1
                if character in character_directory:
                    character_stats[character]["priority"] = bool(character_directory[character].get("priority", False))
                    character_stats[character]["face_cluster_count"] = int(
                        character_directory[character].get("face_cluster_count", 0) or 0
                    )
            for segment in row.get("transcript_segments", []):
                speaker_name = segment.get("speaker_name", "")
                line_character_name = resolve_segment_character_name(segment, row)
                text = coalesce_text(segment.get("text", ""))
                if speaker_name and text:
                    segment_duration = max(0.0, float(segment.get("end", 0.0) or 0.0) - float(segment.get("start", 0.0) or 0.0))
                    if segment_duration > 0.0:
                        segment_durations.append(segment_duration)
                    speaker_counter[speaker_name] += 1
                    sample_key = line_character_name or speaker_name
                    speaker_samples[sample_key].append(text)
                    speaker_line_library[sample_key].append(
                        {
                            "episode_id": row.get("episode_id", ""),
                            "scene_id": row.get("scene_id", ""),
                            "segment_id": segment.get("segment_id", ""),
                            "text": text,
                            "start": float(segment.get("start", 0.0) or 0.0),
                            "end": float(segment.get("end", 0.0) or 0.0),
                            "audio_file": segment.get("audio_file", ""),
                            "video_file": row.get("video_file", ""),
                            "keywords": scene_keywords,
                        }
                    )
                    if useful_character(sample_key):
                        character_stats[sample_key]["line_count"] += 1
            scene_library.append(
                {
                    "episode_id": row["episode_id"],
                    "scene_id": row["scene_id"],
                    "characters": characters,
                    "speaker_names": row.get("speaker_names", []),
                    "keywords": scene_keywords,
                    "transcript": transcript,
                    "video_file": row.get("video_file", ""),
                    "duration_seconds": row_duration,
                }
            )

    top_characters = sorted(
        character_stats.items(),
        key=lambda item: (-int(bool(item[1].get("priority", False))), -int(item[1]["scene_count"]), -int(item[1]["line_count"]), item[0]),
    )
    linked_samples, linked_line_library = build_linked_segment_line_library(cfg)
    for speaker, samples in linked_samples.items():
        for sample in samples:
            if sample not in speaker_samples[speaker]:
                speaker_samples[speaker].append(sample)
    for speaker, entries in linked_line_library.items():
        for entry in entries:
            append_unique_line_entry(speaker_line_library, speaker, entry)
    top_keywords = extract_keywords(transcripts, limit=20)
    if not top_keywords:
        top_keywords = [keyword for keyword, _ in keyword_counter.most_common(20)]
    markov_chain = build_markov_chain(transcripts)
    average_scene_duration = sum(scene_durations) / max(1, len(scene_durations))
    average_segment_duration = sum(segment_durations) / max(1, len(segment_durations))

    return {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_files": [str(path) for path in dataset_files],
        "scene_count": len(scene_library),
        "characters": [
            {"name": name, **stats}
            for name, stats in top_characters
        ],
        "speakers": dict(speaker_counter),
        "keywords": top_keywords,
        "speaker_samples": {speaker: samples[:20] for speaker, samples in speaker_samples.items()},
        "speaker_line_library": {speaker: rows[:160] for speaker, rows in speaker_line_library.items()},
        "character_reference_library": character_reference_library,
        "scene_library": scene_library,
        "source_episode_durations": dict(episode_duration_counter),
        "average_scene_duration_seconds": round(average_scene_duration, 3),
        "average_segment_duration_seconds": round(average_segment_duration, 3),
        "markov_order": 2,
        "markov_chain": markov_chain,
        "generation_defaults": cfg.get("generation", {}),
    }


def clean_callback(line: str) -> str:
    line = coalesce_text(line).strip()
    if not line:
        return ""
    if len(line.split()) > 16:
        line = " ".join(line.split()[:16]).strip()
    if line and line[-1] not in ".!?":
        line += "."
    return line


def choose_speaker_sample(
    speaker: str,
    template_line: str,
    keyword: str,
    speaker_samples: dict[str, list[str]],
    used_lines: set[str],
    rng: random.Random,
) -> str:
    template_tokens = set(tokens_from_text(template_line))
    keyword_token = keyword.lower().strip()
    candidates: list[tuple[float, str]] = []
    for raw_line in speaker_samples.get(speaker, []):
        candidate = clean_callback(raw_line)
        normalized = candidate.lower()
        if not candidate or normalized in used_lines:
            continue
        sample_tokens = set(tokens_from_text(candidate))
        if len(sample_tokens) < 2:
            continue
        token_overlap = len(template_tokens & sample_tokens) / max(1, len(template_tokens | sample_tokens))
        keyword_bonus = 0.35 if keyword_token and keyword_token in normalized else 0.0
        length_bonus = min(0.2, len(candidate.split()) / 50.0)
        score = token_overlap + keyword_bonus + length_bonus
        candidates.append((score, candidate))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (-item[0], item[1]))
    top_candidates = [line for score, line in candidates[: min(5, len(candidates))] if score >= 0.08]
    if not top_candidates:
        return ""
    return rng.choice(top_candidates)


def choose_original_line_entry(
    speaker: str,
    keyword: str,
    speaker_line_library: dict[str, list[dict]],
    used_segment_ids: set[str],
    rng: random.Random,
    preferred_terms: list[str] | None = None,
) -> dict | None:
    keyword_token = keyword.lower().strip()
    preferred_tokens = {token.lower() for token in (preferred_terms or []) if token}
    scored: list[tuple[float, dict]] = []
    for entry in speaker_line_library.get(speaker, []):
        segment_id = str(entry.get("segment_id", "")).strip()
        text = coalesce_text(entry.get("text", ""))
        if not segment_id or segment_id in used_segment_ids or not text:
            continue
        text_tokens_set = {token.lower() for token in tokens_from_text(text)}
        if len(text_tokens_set) < 2:
            continue
        duration = max(0.0, float(entry.get("end", 0.0) or 0.0) - float(entry.get("start", 0.0) or 0.0))
        if duration < 0.45 or duration > 8.0:
            continue
        score = 0.1 + min(0.25, len(text.split()) / 40.0)
        if keyword_token and keyword_token in text.lower():
            score += 0.55
        keyword_hits = sum(1 for token in entry.get("keywords", []) if str(token).lower() in text.lower())
        score += min(0.25, keyword_hits * 0.08)
        preferred_hits = len(text_tokens_set & preferred_tokens)
        score += min(0.2, preferred_hits * 0.05)
        if any(punct in text for punct in ("?", "!")):
            score += 0.06
        scored.append((score, entry))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1].get("segment_id", "")))
    top = [entry for score, entry in scored[: min(8, len(scored))] if score >= 0.16]
    if not top:
        return None
    return dict(rng.choice(top))


def beat_dialogue_templates(
    beat: str,
    speaker_a: str,
    speaker_b: str,
    keyword: str,
    callback: str,
) -> list[str]:
    callback_line = callback or f"{speaker_a}: Wir halten uns diesmal wirklich an den Plan."
    templates = {
        "Cold Open": [
            f"{speaker_a}: {keyword.capitalize()} was supposed to be the easiest task today.",
            f"{speaker_b}: Which is exactly why it already feels like trouble.",
            f"{speaker_a}: If we start improvising right away, this turns into complete chaos again.",
            callback_line,
        ],
        "Plan": [
            f"{speaker_a}: We need a clear plan before {keyword} throws us off balance.",
            f"{speaker_b}: Then we break the problem into small steps and test each one.",
            f"{speaker_a}: Fine, but this time without any spontaneous shortcuts.",
            callback_line,
        ],
        "Komplikation": [
            f"{speaker_b}: Now {keyword} is becoming bigger even though it looked simple a moment ago.",
            f"{speaker_a}: Then someone must have skipped a crucial step somewhere.",
            f"{speaker_b}: And of course we only notice that right in the middle of the stress.",
            callback_line,
        ],
        "Verwechslung": [
            f"{speaker_a}: Wait, we are talking about two completely different versions of {keyword}.",
            f"{speaker_b}: That explains why both of us were convinced we were right.",
            f"{speaker_a}: Then we sort the facts first and continue arguing later.",
            callback_line,
        ],
        "Wendepunkt": [
            f"{speaker_b}: I think I finally see why {keyword} was blocked the whole time.",
            f"{speaker_a}: Please tell me the solution does not require even more improvisation.",
            f"{speaker_b}: No, just one clean step back and a better setup.",
            callback_line,
        ],
        "Auflösung": [
            f"{speaker_a}: See, with a little calm {keyword} suddenly works after all.",
            f"{speaker_b}: And we still have enough time to present the result properly.",
            f"{speaker_a}: Let's remember this sequence for the next round of chaos.",
            callback_line,
        ],
    }
    return templates.get(beat, templates["Plan"])


def build_dialogue(
    beat: str,
    focus_characters: list[str],
    model: dict,
    rng: random.Random,
    cfg: dict,
    keyword: str,
    target_length: int | None = None,
) -> tuple[list[str], list[dict]]:
    speaker_samples = model.get("speaker_samples", {})
    speaker_line_library = model.get("speaker_line_library", {})
    speaker_a = focus_characters[0]
    speaker_b = focus_characters[1] if len(focus_characters) > 1 else fallback_focus_characters(2)[1]
    used_segment_ids: set[str] = set()
    callback_candidates = [
        clean_callback(line)
        for speaker in (speaker_a, speaker_b)
        for line in speaker_samples.get(speaker, [])
    ]
    callback_candidates = [line for line in callback_candidates if len(line.split()) >= 3]
    callback = f"{speaker_a}: {rng.choice(callback_candidates)}" if callback_candidates else ""
    base_lines = beat_dialogue_templates(beat, speaker_a, speaker_b, keyword, callback)
    used_samples = {callback.split(":", 1)[1].strip().lower()} if callback else set()
    enriched_lines: list[str] = []
    line_sources: list[dict] = []
    preferred_terms = [beat, keyword, *focus_characters]
    prefer_original_dialogue = bool(cfg.get("generation", {}).get("prefer_original_dialogue_remix", True))
    line_rounds = 0
    desired_length = target_length or int(cfg["generation"].get("max_dialogue_lines_per_scene", 7))
    while len(enriched_lines) < desired_length:
        line_rounds += 1
        round_lines = list(base_lines)
        if line_rounds > 1:
            rng.shuffle(round_lines)
        for line_index, line in enumerate(round_lines):
            if len(enriched_lines) >= desired_length:
                break
            if ":" not in line:
                enriched_lines.append(line)
                line_sources.append({})
                continue
            speaker, line_text = line.split(":", 1)
            speaker = speaker.strip()
            line_text = line_text.strip()
            original_entry = None
            if prefer_original_dialogue:
                original_entry = choose_original_line_entry(
                    speaker,
                    keyword,
                    speaker_line_library,
                    used_segment_ids,
                    rng,
                    preferred_terms=preferred_terms,
                )
            if original_entry:
                used_segment_ids.add(str(original_entry.get("segment_id", "")))
                selected_text = coalesce_text(original_entry.get("text", "")) or line_text
                enriched_lines.append(f"{speaker}: {selected_text}")
                line_sources.append(
                    {
                        "type": "original_line",
                        "speaker": speaker,
                        "text": selected_text,
                        "episode_id": original_entry.get("episode_id", ""),
                        "scene_id": original_entry.get("scene_id", ""),
                        "segment_id": original_entry.get("segment_id", ""),
                        "start": float(original_entry.get("start", 0.0) or 0.0),
                        "end": float(original_entry.get("end", 0.0) or 0.0),
                        "audio_file": original_entry.get("audio_file", ""),
                        "video_file": original_entry.get("video_file", ""),
                    }
                )
                continue
            sample_line = choose_speaker_sample(speaker, line_text, keyword, speaker_samples, used_samples, rng)
            if sample_line:
                used_samples.add(sample_line.lower())
                enriched_lines.append(f"{speaker}: {sample_line}")
            else:
                suffix = f" ({line_rounds})" if line_rounds > 1 and line_text and line_text[-1] not in ".!?" else ""
                enriched_lines.append(f"{speaker}: {line_text}{suffix}" if suffix else line)
            line_sources.append({})
        if line_rounds >= 12:
            break
    min_lines = int(cfg["generation"].get("min_dialogue_lines_per_scene", 4))
    max_lines = int(cfg["generation"].get("max_dialogue_lines_per_scene", 7))
    final_length = max(min_lines, min(max_lines, len(enriched_lines))) if target_length is None else max(min_lines, len(enriched_lines))
    return enriched_lines[:final_length], line_sources[:final_length]


def select_target_runtime_seconds(model: dict, cfg: dict) -> int:
    generation_cfg = cfg.get("generation", {})
    durations = [int(float(value or 0.0)) for value in (model.get("source_episode_durations", {}) or {}).values() if float(value or 0.0) > 0.0]
    if durations and bool(generation_cfg.get("match_source_episode_runtime", True)):
        average = int(sum(durations) / max(1, len(durations)))
        return max(300, average)
    fallback_seconds = int(float(generation_cfg.get("target_episode_minutes_fallback", 22.0)) * 60.0)
    return max(300, fallback_seconds)


def planning_targets(model: dict, cfg: dict, target_runtime_seconds: int) -> tuple[int, int]:
    generation_cfg = cfg.get("generation", {})
    target_scene_duration = max(18.0, float(generation_cfg.get("target_scene_duration_seconds", 42.0)))
    estimated_line_seconds = max(
        1.4,
        float(model.get("average_segment_duration_seconds", 0.0) or 0.0)
        or float(generation_cfg.get("estimated_dialogue_line_seconds", 2.7)),
    )
    target_scene_count = max(12, int(math.ceil(target_runtime_seconds / target_scene_duration)))
    target_line_count = max(target_scene_count * 6, int(math.ceil(target_runtime_seconds / estimated_line_seconds)))
    per_scene_lines = max(8, int(math.ceil(target_line_count / max(1, target_scene_count))))
    return target_scene_count, per_scene_lines


def select_focus_characters(model: dict, rng: random.Random, count: int = 3) -> list[str]:
    prioritized = [row["name"] for row in model.get("characters", []) if useful_character(row["name"]) and bool(row.get("priority", False))]
    candidates = prioritized or [row["name"] for row in model.get("characters", []) if useful_character(row["name"])]
    if not candidates:
        candidates = [speaker for speaker in model.get("speakers", {}).keys() if useful_character(speaker)]
    if not candidates:
        return fallback_focus_characters(min(count, 2))
    if len(candidates) == 1:
        return [candidates[0], fallback_focus_characters(2)[1]]
    if len(candidates) <= count:
        return candidates
    return candidates[:count]


def select_background_characters(model: dict, count: int = 1) -> list[str]:
    candidates = [row["name"] for row in model.get("characters", []) if background_character(row["name"])]
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped[:count]


def scene_title(beat: str, keyword: str) -> str:
    return f"{beat}: {keyword.capitalize()}"


def summary_line(characters: list[str], keyword: str, beat: str) -> str:
    joined = ", ".join(characters)
    verb = "treibt" if len(characters) == 1 else "treiben"
    return f"{joined} {verb} das Thema '{keyword}' voran und geraten in eine typische {beat.lower()}-Situation."


def choose_camera_plan(beat: str, scene_index: int, rng: random.Random) -> dict:
    presets = CAMERA_PRESETS.get(beat) or CAMERA_PRESETS["Plan"]
    preset = presets[scene_index % len(presets)]
    return dict(preset)


def choose_environment_reference(scene_characters: list[str], keyword: str, model: dict, used_scene_refs: set[str]) -> dict:
    best_entry: dict | None = None
    best_score = -1.0
    keyword_token = keyword.lower().strip()
    for entry in model.get("scene_library", []) or []:
        video_file = coalesce_text(entry.get("video_file", ""))
        entry_scene_id = coalesce_text(entry.get("scene_id", ""))
        if not video_file or not entry_scene_id:
            continue
        if entry_scene_id in used_scene_refs:
            continue
        score = 0.0
        entry_characters = {coalesce_text(name) for name in entry.get("characters", []) or []}
        overlap = len(entry_characters & set(scene_characters))
        score += overlap * 2.0
        if keyword_token and keyword_token in {coalesce_text(token).lower() for token in entry.get("keywords", []) or []}:
            score += 1.25
        if float(entry.get("duration_seconds", 0.0) or 0.0) >= 8.0:
            score += 0.2
        if score > best_score:
            best_score = score
            best_entry = entry
    if not best_entry:
        return {}
    used_scene_refs.add(coalesce_text(best_entry.get("scene_id", "")))
    return {
        "type": "environment",
        "episode_id": coalesce_text(best_entry.get("episode_id", "")),
        "scene_id": coalesce_text(best_entry.get("scene_id", "")),
        "video_file": coalesce_text(best_entry.get("video_file", "")),
        "keywords": best_entry.get("keywords", []) or [],
        "characters": best_entry.get("characters", []) or [],
    }


def scene_character_continuity_hints(
    scene_characters: list[str],
    continuity_memory: dict | None,
) -> list[dict]:
    memory = continuity_memory if isinstance(continuity_memory, dict) else {}
    memory_characters = memory.get("characters", {}) if isinstance(memory.get("characters"), dict) else {}
    hints: list[dict] = []
    for character in scene_characters:
        entry = memory_characters.get(character, {}) if isinstance(memory_characters.get(character), dict) else {}
        continuity = entry.get("continuity", {}) if isinstance(entry.get("continuity"), dict) else {}
        appearances = entry.get("appearances", {})
        appearance_count = len(appearances) if isinstance(appearances, dict) else len(appearances) if isinstance(appearances, list) else 0
        hint = {
            "character": character,
            "outfit": coalesce_text(continuity.get("outfit", "")),
            "hairstyle": coalesce_text(continuity.get("hairstyle", "")),
            "hair_color": coalesce_text(continuity.get("hair_color", "")),
            "accessories": coalesce_text(continuity.get("accessories", "")),
            "voice_traits": coalesce_text(continuity.get("voice_traits", "")),
            "last_episode_id": coalesce_text(entry.get("last_episode_id", "")),
            "appearance_count": appearance_count,
        }
        if any(
            hint[key]
            for key in ("outfit", "hairstyle", "hair_color", "accessories", "voice_traits", "last_episode_id")
        ):
            hints.append(hint)
    return hints


def continuity_prompt_fragments(character_hints: list[dict]) -> list[str]:
    fragments: list[str] = []
    for hint in character_hints[:2]:
        character = coalesce_text(hint.get("character", ""))
        if not character:
            continue
        details: list[str] = []
        for key in ("outfit", "hairstyle", "hair_color", "accessories"):
            value = coalesce_text(hint.get(key, ""))
            if value:
                details.append(value)
        if details:
            fragments.append(f"{character} keeps {'; '.join(details[:3])}")
    return fragments


def normalized_style_constraints(raw_constraints: dict | None) -> dict:
    constraints = raw_constraints if isinstance(raw_constraints, dict) else {}
    positive = [coalesce_text(value) for value in constraints.get("positive", []) if coalesce_text(value)]
    negative = [coalesce_text(value) for value in constraints.get("negative", []) if coalesce_text(value)]
    guidance = constraints.get("guidance", {}) if isinstance(constraints.get("guidance"), dict) else {}
    return {
        "positive": positive[:6],
        "negative": negative[:8],
        "guidance": {key: coalesce_text(value) for key, value in guidance.items() if coalesce_text(value)},
    }


def build_scene_generation_plan(
    scene_id: str,
    scene_index: int,
    beat: str,
    keyword: str,
    scene_characters: list[str],
    summary: str,
    model: dict,
    used_scene_refs: set[str],
    previous_scene_id: str,
    continuity_memory: dict | None = None,
    style_constraints: dict | None = None,
    quality_mode: str = "series_consistency",
) -> dict:
    continuity_memory = continuity_memory if isinstance(continuity_memory, dict) else load_character_continuity_memory(PROJECT_ROOT)
    style_constraints = style_constraints if isinstance(style_constraints, dict) else derive_prompt_constraints_from_bible(PROJECT_ROOT, {})
    reference_library = model.get("character_reference_library", {}) or {}
    camera_plan = choose_camera_plan(beat, scene_index, random.Random(scene_index + len(scene_characters)))
    camera_plan["camera"] = camera_plan.get("shot_type", "")
    camera_plan["focus"] = camera_plan.get("composition", "")
    camera_plan["movement"] = camera_plan.get("camera_move", "")
    camera_plan["lens"] = camera_plan.get("lens_hint", "")
    reference_slots: list[dict] = []
    for slot_index, character in enumerate(scene_characters[:2], start=1):
        reference_row = reference_library.get(character, {}) if isinstance(reference_library, dict) else {}
        reference_slots.append(
            {
                "slot": f"subject_{slot_index}",
                "type": "character",
                "name": character,
                "context_images": list((reference_row.get("context_images", []) or [])[:2]),
                "portrait_images": list((reference_row.get("portrait_images", []) or [])[:2]),
                "priority": bool(reference_row.get("priority", False)),
            }
        )
    environment_reference = choose_environment_reference(scene_characters, keyword, model, used_scene_refs)
    if environment_reference:
        reference_slots.append({"slot": "scene_reference", **environment_reference})
    character_continuity = scene_character_continuity_hints(scene_characters, continuity_memory)
    style_profile = normalized_style_constraints(style_constraints)
    continuity_fragments = continuity_prompt_fragments(character_continuity)
    style_positive = ", ".join(style_profile.get("positive", [])[:3])
    style_negative = ", ".join(style_profile.get("negative", [])[:5])
    style_guidance = style_profile.get("guidance", {}) if isinstance(style_profile.get("guidance", {}), dict) else {}

    positive_prompt = (
        f"storyboard frame, animated sitcom look, {camera_plan['shot_type']}, {camera_plan['composition']}, "
        f"{camera_plan['camera_move']}, {camera_plan['lens_hint']}, "
        f"characters {', '.join(scene_characters[:2])}, keyword {keyword}, beat {beat}, "
        f"{camera_plan['pose_hint']}, keep identity, wardrobe and environment continuity, "
        f"16:9 frame, clean readable staging"
    )
    if style_positive:
        positive_prompt += f", style cues: {style_positive}"
    if continuity_fragments:
        positive_prompt += f", continuity notes: {'; '.join(continuity_fragments)}"
    if style_guidance.get("camera"):
        positive_prompt += f", series camera preference: {style_guidance['camera']}"
    if style_guidance.get("angle"):
        positive_prompt += f", preferred angle: {style_guidance['angle']}"
    negative_prompt = (
        "no collage, no split panel, no duplicate characters, no extra fingers, no warped face, "
        "no mismatched outfit, no GUI overlay, no text box, no watermark"
    )
    if style_negative:
        negative_prompt += f", avoid {style_negative}"
    batch_prompt_line = (
        f"{beat} | {', '.join(scene_characters[:2])} | {keyword} | {camera_plan['shot_type']} | "
        f"{camera_plan['composition']} | continuity from {previous_scene_id or 'none'}"
    )
    if style_guidance.get("camera"):
        batch_prompt_line += f" | style camera {style_guidance['camera']}"
    return {
        "scene_id": scene_id,
        "model_mode": "multi_reference_storyboard",
        "reference_slots": reference_slots,
        "continuity": {
            "previous_scene_id": previous_scene_id,
            "use_previous_scene_as_reference": bool(previous_scene_id),
            "carry_forward_characters": [row.get("character", "") for row in character_continuity if row.get("character")],
        },
        "camera_plan": camera_plan,
        "control_hints": {
            "pose_guidance": camera_plan["pose_hint"],
            "pose_emphasis": camera_plan["pose_hint"],
            "composition_guidance": camera_plan["composition"],
            "composition_emphasis": camera_plan["composition"],
            "motion_guidance": camera_plan["camera_move"],
            "motion_emphasis": camera_plan["camera_move"],
            "style_camera": style_guidance.get("camera", ""),
            "style_angle": style_guidance.get("angle", ""),
        },
        "style_constraints": style_profile,
        "character_continuity": character_continuity,
        "quality_targets": {
            "quality_mode": quality_mode,
            "minimum_reference_slots": max(1, min(3, len(reference_slots))),
            "prefer_previous_scene_reference": bool(previous_scene_id),
            "prefer_backend_character_models": bool(scene_characters),
            "style_guidance_available": bool(style_profile.get("positive") or style_guidance),
            "continuity_character_count": len(character_continuity),
        },
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "batch_prompt_line": batch_prompt_line,
        "summary": summary,
    }


def generate_episode_package(model: dict, cfg: dict, episode_index: int = 1) -> tuple[dict, str]:
    generation_cfg = cfg.get("generation", {})
    rng_seed = int(generation_cfg.get("seed", 42)) + len(model.get("dataset_files", [])) + (episode_index * 101)
    rng = random.Random(rng_seed)
    keywords = model.get("keywords", []) or ["idee", "chaos", "plan", "showdown"]
    if len(keywords) > 1:
        keyword_offset = episode_index % len(keywords)
        keywords = keywords[keyword_offset:] + keywords[:keyword_offset]
    target_runtime_seconds = select_target_runtime_seconds(model, cfg)
    scene_count, target_lines_per_scene = planning_targets(model, cfg, target_runtime_seconds)
    continuity_memory = load_character_continuity_memory(PROJECT_ROOT)
    style_constraints = derive_prompt_constraints_from_bible(PROJECT_ROOT, {})
    quality_mode = coalesce_text(generation_cfg.get("quality_mode", "")) or "series_consistency"
    focus_characters = select_focus_characters(model, rng)
    background_characters = select_background_characters(model, count=1)
    beats = ["Cold Open", "Plan", "Komplikation", "Verwechslung", "Wendepunkt", "Auflösung"]
    if len(beats) > 1:
        beat_offset = episode_index % len(beats)
        beats = beats[beat_offset:] + beats[:beat_offset]
    episode_title = build_episode_title(focus_characters, keywords, rng, episode_index, model=model)
    episode_label = format_episode_number(episode_index)
    display_title = f"{episode_label}: {episode_title}"

    scenes = []
    markdown_lines = [
        f"# {display_title}",
        "",
        "## Trained Series Model Basis",
        "",
        f"- Episode title: {episode_title}",
        f"- Evaluated scenes: {model.get('scene_count', 0)}",
        f"- Main characters: {', '.join(focus_characters)}",
        f"- Recurring themes: {', '.join(keywords[:6])}",
        "",
        "## Scene Plan",
        "",
        f"- Episode seed: {rng_seed}",
        f"- Target runtime: about {round(target_runtime_seconds / 60.0, 1)} minutes",
        f"- Target scenes: {scene_count}",
        f"- Target dialogue lines per scene: {target_lines_per_scene}",
        "- Storyboard-Plan: multi-reference, continuity-aware, model-native prompts",
        "",
    ]
    used_scene_refs: set[str] = set()
    previous_scene_id = ""
    for scene_index in range(scene_count):
        beat = beats[scene_index % len(beats)]
        keyword = keywords[scene_index % len(keywords)]
        scene_characters = focus_characters[: max(2, min(len(focus_characters), 2 + (scene_index % 2)))]
        if background_characters and scene_index % 3 == 1:
            background_name = background_characters[0]
            if background_name not in scene_characters:
                scene_characters.append(background_name)
        summary = summary_line(scene_characters, keyword, beat)
        dialogue, dialogue_sources = build_dialogue(
            beat,
            scene_characters,
            model,
            rng,
            cfg,
            keyword,
            target_length=target_lines_per_scene,
        )
        scene_id = f"scene_{scene_index + 1:04d}"
        generation_plan = build_scene_generation_plan(
            scene_id,
            scene_index,
            beat,
            keyword,
            scene_characters,
            summary,
            model,
            used_scene_refs,
            previous_scene_id,
            continuity_memory=continuity_memory,
            style_constraints=style_constraints,
            quality_mode=quality_mode,
        )
        scenes.append(
            {
                "scene_id": scene_id,
                "title": scene_title(beat, keyword),
                "beat": beat,
                "summary": summary,
                "characters": scene_characters,
                "location": f"Set {((scene_index % 3) + 1)}",
                "mood": "energetisch" if scene_index < scene_count - 1 else "auflösend",
                "dialogue_lines": dialogue,
                "dialogue_sources": dialogue_sources,
                "prompt": (
                    f"{beat} mit {', '.join(scene_characters)}. Fokus auf {keyword}, schnelle Pointen "
                    f"und klaren Konfliktbogen."
                ),
                "generation_plan": generation_plan,
                "estimated_runtime_seconds": round(len(dialogue) * float(model.get("average_segment_duration_seconds", 2.7) or 2.7), 2),
            }
        )
        markdown_lines.extend(
            [
                f"### {scene_id} - {scene_title(beat, keyword)}",
                "",
                summary,
                "",
                f"Shot Plan: {generation_plan['camera_plan']['shot_type']} | {generation_plan['camera_plan']['composition']} | {generation_plan['camera_plan']['camera_move']}",
                f"Continuity: previous scene = {generation_plan['continuity']['previous_scene_id'] or 'none'}",
                "Reference Slots:",
                *[
                    f"- {slot['slot']}: {slot.get('type', '')} {slot.get('name', slot.get('scene_id', ''))}".strip()
                    for slot in generation_plan["reference_slots"]
                ],
                "",
                "Dialog:",
                "",
                *[f"- {line}" for line in dialogue],
                "",
            ]
        )
        previous_scene_id = scene_id

    return {
        "episode_title": episode_title,
        "episode_label": episode_label,
        "display_title": display_title,
        "generation_mode": "synthetic_preview",
        "quality_mode": quality_mode,
        "storyboard_plan_mode": "multi_reference_storyboard",
        "target_runtime_seconds": target_runtime_seconds,
        "target_scene_count": scene_count,
        "target_dialogue_lines_per_scene": target_lines_per_scene,
        "scenes": scenes,
        "focus_characters": focus_characters,
        "keywords": keywords[:10],
    }, "\n".join(markdown_lines).strip() + "\n"


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Train Series Model")
    cfg = load_config()
    worker_id = shared_worker_id_for_args(args)
    shared_workers = shared_workers_enabled_for_args(cfg, args)
    mark_step_started("08_train_series_model", "global")
    if shared_workers:
        info(f"Shared NAS workers: enabled ({worker_id})")
    lease_manager = distributed_item_lease(
        root=distributed_step_runtime_root("08_train_series_model", "global"),
        lease_name="global",
        cfg=cfg,
        worker_id=worker_id,
        enabled=shared_workers,
        meta={"step": "08_train_series_model", "scope": "global", "worker_id": worker_id},
    )
    acquired = lease_manager.__enter__()
    if not acquired:
        info("Series model training is already running on another worker.")
        lease_manager.__exit__(None, None, None)
        return
    dataset_root = resolve_project_path(cfg["paths"]["datasets_video_training"])
    dataset_files = sorted(dataset_root.glob("*_dataset.json"))
    try:
        if not dataset_files:
            info("No datasets found.")
            mark_step_completed("08_train_series_model", "global", {"series_model": "", "dataset_count": 0})
            return
        char_map = read_json(resolve_project_path(cfg["paths"]["character_map"]), {"clusters": {}, "aliases": {}})
        model = build_series_model(dataset_files, cfg, char_map)
        model_path = resolve_project_path(cfg["paths"]["series_model"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(model_path, model)
        mark_step_completed(
            "08_train_series_model",
            "global",
            {"series_model": str(model_path), "dataset_count": len(dataset_files)},
        )
        ok(f"Series Model trainiert: {model_path}")
    except Exception as exc:
        mark_step_failed("08_train_series_model", str(exc), "global", {"dataset_count": len(dataset_files)})
        raise
    finally:
        lease_manager.__exit__(None, None, None)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise


