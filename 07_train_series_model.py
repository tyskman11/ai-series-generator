#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import time
from collections import Counter, defaultdict

from pipeline_common import (
    coalesce_text,
    display_person_name,
    error,
    extract_keywords,
    has_primary_person_name,
    is_background_person_name,
    headline,
    info,
    load_config,
    ok,
    read_json,
    rerun_in_runtime,
    resolve_project_path,
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
    parser = argparse.ArgumentParser(description="Serienmodell aus den vorhandenen Datensaetzen trainieren")
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
) -> str:
    usable_keywords = [humanize_keyword(keyword) for keyword in keywords if coalesce_text(keyword)]
    main_keyword = usable_keywords[0] if usable_keywords else "Geheimnis"
    secondary_keyword = usable_keywords[1] if len(usable_keywords) > 1 else main_keyword
    title_seed = episode_index + len(focus_characters) + len(usable_keywords)
    prefix = TITLE_PREFIXES[title_seed % len(TITLE_PREFIXES)]
    suffix = TITLE_SUFFIXES[(title_seed + len(main_keyword)) % len(TITLE_SUFFIXES)]
    if " " in main_keyword:
        base = main_keyword
    elif prefix in {"Das", "Der", "Die", "Das große"}:
        base = f"{main_keyword}-{suffix}"
    else:
        base = main_keyword
    candidate_titles = [
        f"{prefix} {base}",
        f"{main_keyword} ohne Plan",
        f"{main_keyword} gegen {secondary_keyword}",
    ]
    if focus_characters:
        lead = focus_characters[0].split()[0]
        candidate_titles.append(f"{lead} und {base}")
    title = candidate_titles[title_seed % len(candidate_titles)]
    return title.strip()


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
    character_stats: dict[str, dict[str, int | bool]] = defaultdict(lambda: {"scene_count": 0, "line_count": 0, "priority": False})
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
            f"{speaker_a}: {keyword.capitalize()} sollte heute eigentlich die leichteste Aufgabe sein.",
            f"{speaker_b}: Genau deshalb fühlt es sich schon jetzt nach Ärger an.",
            f"{speaker_a}: Wenn wir sofort improvisieren, wird daraus wieder ein komplettes Chaos.",
            callback_line,
        ],
        "Plan": [
            f"{speaker_a}: Wir brauchen einen klaren Plan, bevor {keyword} uns aus dem Takt bringt.",
            f"{speaker_b}: Dann teilen wir das Problem in kleine Schritte auf und testen jeden davon.",
            f"{speaker_a}: Gut, aber diesmal ohne spontane Abkürzungen.",
            callback_line,
        ],
        "Komplikation": [
            f"{speaker_b}: Jetzt wird {keyword} größer, obwohl es eben noch ganz einfach aussah.",
            f"{speaker_a}: Dann hat irgendwo jemand einen entscheidenden Schritt übersprungen.",
            f"{speaker_b}: Und genau das fällt uns natürlich erst mitten im Stress auf.",
            callback_line,
        ],
        "Verwechslung": [
            f"{speaker_a}: Warte mal, wir reden gerade über zwei völlig verschiedene Versionen von {keyword}.",
            f"{speaker_b}: Das erklärt, warum wir beide überzeugt waren, im Recht zu sein.",
            f"{speaker_a}: Dann sortieren wir erst die Fakten und streiten später weiter.",
            callback_line,
        ],
        "Wendepunkt": [
            f"{speaker_b}: Ich glaube, ich sehe endlich, warum {keyword} die ganze Zeit blockiert war.",
            f"{speaker_a}: Sag bitte, dass die Lösung nicht noch mehr Improvisation braucht.",
            f"{speaker_b}: Nein, nur einen sauberen Schritt zurück und einen besseren Aufbau.",
            callback_line,
        ],
        "Auflösung": [
            f"{speaker_a}: Siehst du, mit Ruhe funktioniert {keyword} plötzlich doch.",
            f"{speaker_b}: Und wir haben sogar noch genug Zeit, um das Ergebnis ordentlich zu präsentieren.",
            f"{speaker_a}: Merken wir uns diesen Ablauf bitte für das nächste Chaos.",
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
    focus_characters = select_focus_characters(model, rng)
    background_characters = select_background_characters(model, count=1)
    beats = ["Cold Open", "Plan", "Komplikation", "Verwechslung", "Wendepunkt", "Auflösung"]
    if len(beats) > 1:
        beat_offset = episode_index % len(beats)
        beats = beats[beat_offset:] + beats[:beat_offset]
    episode_title = build_episode_title(focus_characters, keywords, rng, episode_index)
    episode_label = format_episode_number(episode_index)
    display_title = f"{episode_label}: {episode_title}"

    scenes = []
    markdown_lines = [
        f"# {display_title}",
        "",
        "## Basis des trainierten Serienmodells",
        "",
        f"- Episodentitel: {episode_title}",
        f"- Ausgewertete Szenen: {model.get('scene_count', 0)}",
        f"- Hauptfiguren: {', '.join(focus_characters)}",
        f"- Wiederkehrende Themen: {', '.join(keywords[:6])}",
        "",
        "## Szenenplan",
        "",
        f"- Episoden-Seed: {rng_seed}",
        f"- Ziel-Laufzeit: ca. {round(target_runtime_seconds / 60.0, 1)} Minuten",
        f"- Ziel-Szenen: {scene_count}",
        f"- Ziel-Dialogzeilen pro Szene: {target_lines_per_scene}",
        "",
    ]
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
                "estimated_runtime_seconds": round(len(dialogue) * float(model.get("average_segment_duration_seconds", 2.7) or 2.7), 2),
            }
        )
        markdown_lines.extend(
            [
                f"### {scene_id} - {scene_title(beat, keyword)}",
                "",
                summary,
                "",
                "Dialog:",
                "",
                *[f"- {line}" for line in dialogue],
                "",
            ]
        )

    return {
        "episode_title": episode_title,
        "episode_label": episode_label,
        "display_title": display_title,
        "generation_mode": "synthetic_preview",
        "target_runtime_seconds": target_runtime_seconds,
        "target_scene_count": scene_count,
        "target_dialogue_lines_per_scene": target_lines_per_scene,
        "scenes": scenes,
        "focus_characters": focus_characters,
        "keywords": keywords[:10],
    }, "\n".join(markdown_lines).strip() + "\n"


def main() -> None:
    rerun_in_runtime()
    parse_args()
    headline("Serienmodell trainieren")
    cfg = load_config()
    dataset_root = resolve_project_path(cfg["paths"]["datasets_video_training"])
    dataset_files = sorted(dataset_root.glob("*_dataset.json"))
    if not dataset_files:
        info("Keine Datensätze gefunden.")
        return

    char_map = read_json(resolve_project_path(cfg["paths"]["character_map"]), {"clusters": {}, "aliases": {}})
    model = build_series_model(dataset_files, cfg, char_map)
    model_path = resolve_project_path(cfg["paths"]["series_model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(model_path, model)
    ok(f"Serienmodell trainiert: {model_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
