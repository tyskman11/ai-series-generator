#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent / "ai_series_project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from support_scripts.pipeline_common import (
    add_shared_worker_arguments,
    coalesce_text,
    distributed_item_lease,
    distributed_step_runtime_root,
    dominant_language,
    error,
    extract_keywords,
    headline,
    info,
    load_character_relationships,
    load_config,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    normalize_language_code,
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


STOPWORDS = {
    "aber",
    "also",
    "and",
    "are",
    "auf",
    "das",
    "den",
    "der",
    "die",
    "ein",
    "eine",
    "for",
    "halt",
    "ich",
    "ist",
    "mit",
    "nicht",
    "nur",
    "oder",
    "that",
    "the",
    "und",
    "was",
    "wir",
    "you",
    "zur",
}
REACTION_MARKERS = {
    "aber",
    "doch",
    "moment",
    "nein",
    "okay",
    "stopp",
    "warte",
    "wait",
    "what",
    "wirklich",
    "wow",
}
CONFLICT_MARKERS = {
    "aber",
    "doch",
    "falsch",
    "nein",
    "nicht",
    "problem",
    "stimmt",
    "stop",
    "wrong",
}
RESOLUTION_MARKERS = {
    "alles klar",
    "done",
    "geschafft",
    "gut",
    "okay",
    "solved",
    "together",
    "zusammen",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze character behavior patterns for episode generation.")
    parser.add_argument("--force", action="store_true", help="Rebuild the behavior model even if it already exists.")
    add_shared_worker_arguments(parser)
    return parser.parse_args()


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float], default: float = 0.0) -> float:
    return round(sum(values) / len(values), 3) if values else round(default, 3)


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def behavior_model_path(cfg: dict[str, Any]) -> Path:
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    return resolve_project_path(str(paths.get("behavior_model", "generation/model/behavior_model.json")))


def behavior_summary_path(cfg: dict[str, Any]) -> Path:
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    return resolve_project_path(str(paths.get("behavior_model_summary", "generation/model/behavior_model_summary.md")))


def dataset_files(cfg: dict[str, Any]) -> list[Path]:
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    root = resolve_project_path(str(paths.get("datasets_video_training", "data/datasets/video_training")))
    return sorted(path for path in root.glob("*_dataset.json") if path.is_file())


def load_dataset_rows(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    diagnostics: list[str] = []
    rows: list[dict[str, Any]] = []
    files = dataset_files(cfg)
    if not files:
        diagnostics.append("No dataset files found; behavior model uses defaults.")
        return rows, diagnostics
    for path in files:
        try:
            payload = read_json(path, [])
        except Exception as exc:
            diagnostics.append(f"Could not read dataset {path.name}: {exc}")
            continue
        if isinstance(payload, dict):
            payload = payload.get("rows", payload.get("items", []))
        if not isinstance(payload, list):
            diagnostics.append(f"Dataset {path.name} is not a row list.")
            continue
        for row in payload:
            if isinstance(row, dict):
                rows.append(row)
    if not rows:
        diagnostics.append("Dataset files were present, but no usable rows were found.")
    return rows, diagnostics


def load_optional_series_model(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    path = resolve_project_path(str(paths.get("series_model", "generation/model/series_model.json")))
    payload = read_json(path, {}) if path.exists() else {}
    return payload if isinstance(payload, dict) else {}


def line_speaker(segment: dict[str, Any], row: dict[str, Any]) -> str:
    for key in (
        "speaker_name",
        "character_name",
        "assigned_character",
        "resolved_character_name",
        "speaker",
        "name",
    ):
        value = coalesce_text(segment.get(key, ""))
        if value:
            return value
    speakers = row.get("speaker_names", []) if isinstance(row.get("speaker_names"), list) else []
    return coalesce_text(speakers[0] if speakers else "")


def parse_transcript_dialogue(row: dict[str, Any]) -> list[dict[str, Any]]:
    transcript = str(row.get("transcript", "") or "").strip()
    if not transcript:
        return []
    parsed: list[dict[str, Any]] = []
    for index, raw_line in enumerate(transcript.splitlines()):
        line = coalesce_text(raw_line)
        if ":" not in line:
            continue
        speaker, text = line.split(":", 1)
        speaker = coalesce_text(speaker)
        text = coalesce_text(text)
        if speaker and text:
            parsed.append({"speaker_name": speaker, "text": text, "start": float(index), "end": float(index + 1)})
    return parsed


def collect_line_records(rows: list[dict[str, Any]], series_model: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        segments = row.get("transcript_segments", [])
        if not isinstance(segments, list) or not segments:
            segments = parse_transcript_dialogue(row)
        row_language = normalize_language_code(row.get("detected_language", "") or row.get("language", ""))
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            speaker = line_speaker(segment, row)
            text = coalesce_text(segment.get("text", ""))
            if not speaker or not text:
                continue
            start = safe_float(segment.get("start", 0.0))
            end = safe_float(segment.get("end", start))
            duration = max(0.0, end - start)
            records.append(
                {
                    "episode_id": coalesce_text(row.get("episode_id", "")),
                    "scene_id": coalesce_text(row.get("scene_id", "")),
                    "speaker": speaker,
                    "text": text,
                    "start": start,
                    "end": end,
                    "duration_seconds": duration,
                    "language": normalize_language_code(segment.get("language", "") or row_language),
                }
            )
    if records or not isinstance(series_model, dict):
        return records
    for speaker, entries in (series_model.get("speaker_line_library", {}) or {}).items():
        if not isinstance(entries, list):
            continue
        for entry in entries[:120]:
            if not isinstance(entry, dict):
                continue
            text = coalesce_text(entry.get("text", ""))
            if text:
                records.append(
                    {
                        "episode_id": coalesce_text(entry.get("episode_id", "")),
                        "scene_id": coalesce_text(entry.get("scene_id", "")),
                        "speaker": coalesce_text(speaker),
                        "text": text,
                        "start": safe_float(entry.get("start", 0.0)),
                        "end": safe_float(entry.get("end", 0.0)),
                        "duration_seconds": max(0.0, safe_float(entry.get("end", 0.0)) - safe_float(entry.get("start", 0.0))),
                        "language": normalize_language_code(entry.get("language", "")),
                    }
                )
    return records


def collect_scene_records(rows: list[dict[str, Any]], line_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    line_index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for line in line_records:
        line_index[(coalesce_text(line.get("episode_id", "")), coalesce_text(line.get("scene_id", "")))].append(line)
    scenes: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (coalesce_text(row.get("episode_id", "")), coalesce_text(row.get("scene_id", "")))
        if key in seen:
            continue
        seen.add(key)
        lines = line_index.get(key, [])
        speaker_names = sorted({coalesce_text(line.get("speaker", "")) for line in lines if coalesce_text(line.get("speaker", ""))})
        scenes.append(
            {
                "episode_id": key[0],
                "scene_id": key[1],
                "duration_seconds": safe_float(row.get("duration_seconds", 0.0)),
                "line_count": len(lines),
                "speaker_count": len(speaker_names),
                "speakers": speaker_names,
                "keywords": row.get("scene_keywords", []) if isinstance(row.get("scene_keywords"), list) else [],
                "transcript": coalesce_text(row.get("transcript", "")),
                "language": normalize_language_code(row.get("detected_language", "") or row.get("language", "")),
            }
        )
    if scenes:
        return scenes
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for line in line_records:
        grouped[(coalesce_text(line.get("episode_id", "")), coalesce_text(line.get("scene_id", "")))].append(line)
    for key, lines in grouped.items():
        speakers = sorted({coalesce_text(line.get("speaker", "")) for line in lines if coalesce_text(line.get("speaker", ""))})
        duration = max((safe_float(line.get("end", 0.0)) for line in lines), default=0.0)
        scenes.append(
            {
                "episode_id": key[0],
                "scene_id": key[1],
                "duration_seconds": duration,
                "line_count": len(lines),
                "speaker_count": len(speakers),
                "speakers": speakers,
                "keywords": [],
                "transcript": "\n".join(coalesce_text(line.get("text", "")) for line in lines),
                "language": dominant_language(Counter(coalesce_text(line.get("language", "")) for line in lines)),
            }
        )
    return scenes


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"[.!?]+", coalesce_text(text)) if sentence.strip()]


def token_list(text: str) -> list[str]:
    return [token.lower() for token in tokens_from_text(text) if len(token) > 1]


def meaningful_tokens(text: str) -> list[str]:
    return [token for token in token_list(text) if token not in STOPWORDS]


def top_phrases(lines: list[str], limit: int = 8) -> list[str]:
    counter: Counter[str] = Counter()
    for line in lines:
        tokens = meaningful_tokens(line)
        for size in (2, 3):
            for chunk in zip(*(tokens[index:] for index in range(size))):
                if any(token in STOPWORDS for token in chunk):
                    continue
                counter[" ".join(chunk)] += 1
    return [phrase for phrase, _count in counter.most_common(limit)]


def reaction_patterns(lines: list[str], limit: int = 8) -> list[str]:
    counter: Counter[str] = Counter()
    for line in lines:
        tokens = token_list(line)
        if not tokens:
            continue
        if "?" in line or "!" in line or tokens[0] in REACTION_MARKERS or any(token in REACTION_MARKERS for token in tokens[:3]):
            counter[" ".join(tokens[: min(5, len(tokens))])] += 1
    return [phrase for phrase, _count in counter.most_common(limit)]


def humor_hints(lines: list[str]) -> dict[str, Any]:
    text = "\n".join(lines).lower()
    markers = [marker for marker in ("haha", "witz", "joke", "seriously", "wirklich", "klar", "great") if marker in text]
    punctuation_rate = sum(1 for line in lines if "!" in line or "?" in line) / max(1, len(lines))
    return {
        "markers": markers[:8],
        "punctuation_comedy_rate": round(punctuation_rate, 3),
        "likely_sarcasm": bool({"klar", "great", "seriously"} & set(markers)) or punctuation_rate >= 0.35,
    }


def speaking_style_for_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    lines = [coalesce_text(record.get("text", "")) for record in records if coalesce_text(record.get("text", ""))]
    word_counts = [len(token_list(line)) for line in lines]
    sentence_lengths: list[float] = []
    for line in lines:
        for sentence in split_sentences(line):
            sentence_lengths.append(float(len(token_list(sentence))))
    question_rate = sum(1 for line in lines if "?" in line) / max(1, len(lines))
    exclamation_rate = sum(1 for line in lines if "!" in line) / max(1, len(lines))
    durations = [safe_float(record.get("duration_seconds", 0.0)) for record in records if safe_float(record.get("duration_seconds", 0.0)) > 0.0]
    words_per_second = sum(word_counts) / max(0.1, sum(durations)) if durations else 0.0
    energy = clamp(0.32 + (question_rate * 0.16) + (exclamation_rate * 0.26) + min(0.26, words_per_second * 0.08))
    token_counter = Counter(token for line in lines for token in meaningful_tokens(line))
    return {
        "line_count": len(lines),
        "average_words_per_line": round(mean([float(count) for count in word_counts], 6.0), 2),
        "typical_sentence_length": round(mean(sentence_lengths, 7.0), 2),
        "question_rate": round(question_rate, 3),
        "exclamation_rate": round(exclamation_rate, 3),
        "typical_words": [token for token, _count in token_counter.most_common(12)],
        "typical_phrases": top_phrases(lines),
        "recurring_reactions": reaction_patterns(lines),
        "energy_level": round(energy, 3),
        "energy_label": "high" if energy >= 0.7 else "low" if energy < 0.42 else "medium",
        "humor_sarcasm_hints": humor_hints(lines),
        "defaulted": not bool(lines),
    }


def build_speaking_style(line_records: list[dict[str, Any]], series_model: dict[str, Any]) -> dict[str, Any]:
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in line_records:
        speaker = coalesce_text(record.get("speaker", ""))
        if speaker:
            by_speaker[speaker].append(record)
    if not by_speaker:
        for character in series_model.get("characters", []) if isinstance(series_model.get("characters"), list) else []:
            if isinstance(character, dict) and coalesce_text(character.get("name", "")):
                by_speaker[coalesce_text(character.get("name", ""))] = []
    return {speaker: speaking_style_for_records(records) for speaker, records in sorted(by_speaker.items())}


def relationship_key(a: str, b: str) -> str:
    left, right = sorted([coalesce_text(a), coalesce_text(b)])
    return f"{left}||{right}"


def scene_line_groups(line_records: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for line in line_records:
        groups[(coalesce_text(line.get("episode_id", "")), coalesce_text(line.get("scene_id", "")))].append(line)
    for lines in groups.values():
        lines.sort(key=lambda row: (safe_float(row.get("start", 0.0)), safe_float(row.get("end", 0.0))))
    return groups


def relationship_label(row: dict[str, Any]) -> str:
    for key in ("dynamic", "relationship", "type", "label", "notes"):
        value = coalesce_text(row.get(key, ""))
        if value:
            return value
    return "observed recurring interaction"


def build_relationship_behavior(
    line_records: list[dict[str, Any]],
    relationship_payload: dict[str, Any],
) -> dict[str, Any]:
    pairs: dict[str, dict[str, Any]] = {}
    for row in relationship_payload.get("relationships", []) if isinstance(relationship_payload.get("relationships"), list) else []:
        if not isinstance(row, dict):
            continue
        source = coalesce_text(row.get("source", ""))
        target = coalesce_text(row.get("target", ""))
        if source and target:
            pairs[relationship_key(source, target)] = {
                "characters": sorted([source, target]),
                "configured_dynamic": relationship_label(row),
                "typical_dynamic": relationship_label(row),
                "conversation_leader": "",
                "conversation_leader_counts": {},
                "contradiction_counts": {},
                "typical_conflict_patterns": [],
                "resolution_patterns": [],
                "co_scene_count": 0,
            }
    groups = scene_line_groups(line_records)
    for lines in groups.values():
        speakers = [coalesce_text(line.get("speaker", "")) for line in lines if coalesce_text(line.get("speaker", ""))]
        unique_speakers = sorted(set(speakers))
        for a, b in itertools.combinations(unique_speakers, 2):
            key = relationship_key(a, b)
            entry = pairs.setdefault(
                key,
                {
                    "characters": sorted([a, b]),
                    "configured_dynamic": "",
                    "typical_dynamic": "co-present dialogue pair",
                    "conversation_leader": "",
                    "conversation_leader_counts": {},
                    "contradiction_counts": {},
                    "typical_conflict_patterns": [],
                    "resolution_patterns": [],
                    "co_scene_count": 0,
                },
            )
            entry["co_scene_count"] = int(entry.get("co_scene_count", 0) or 0) + 1
            first = speakers[0] if speakers else ""
            if first:
                leader_counts = Counter(entry.get("conversation_leader_counts", {}) or {})
                leader_counts[first] += 1
                entry["conversation_leader_counts"] = dict(leader_counts)
                entry["conversation_leader"] = leader_counts.most_common(1)[0][0]
        previous_speaker = ""
        conflict_counter: Counter[str] = Counter()
        resolution_counter: Counter[str] = Counter()
        for line in lines:
            speaker = coalesce_text(line.get("speaker", ""))
            text = coalesce_text(line.get("text", ""))
            tokens = set(token_list(text))
            if previous_speaker and speaker and speaker != previous_speaker and tokens & CONFLICT_MARKERS:
                key = relationship_key(previous_speaker, speaker)
                if key in pairs:
                    contradiction_counts = Counter(pairs[key].get("contradiction_counts", {}) or {})
                    contradiction_counts[f"{speaker}_against_{previous_speaker}"] += 1
                    pairs[key]["contradiction_counts"] = dict(contradiction_counts)
                    conflict_counter[f"{speaker} challenges {previous_speaker}"] += 1
            lowered = text.lower()
            for marker in RESOLUTION_MARKERS:
                if marker in lowered and previous_speaker and speaker and speaker != previous_speaker:
                    resolution_counter[f"{previous_speaker} -> {speaker}: {marker}"] += 1
            previous_speaker = speaker or previous_speaker
        for key in pairs:
            if set(pairs[key].get("characters", [])) <= set(unique_speakers):
                pairs[key]["typical_conflict_patterns"] = [item for item, _ in conflict_counter.most_common(5)]
                pairs[key]["resolution_patterns"] = [item for item, _ in resolution_counter.most_common(5)]
    return dict(sorted(pairs.items()))


def build_scene_behavior(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [safe_float(scene.get("duration_seconds", 0.0)) for scene in scenes if safe_float(scene.get("duration_seconds", 0.0)) > 0.0]
    line_counts = [float(scene.get("line_count", 0) or 0) for scene in scenes]
    speaker_counts = [float(scene.get("speaker_count", 0) or 0) for scene in scenes]
    densities = [
        (float(scene.get("line_count", 0) or 0) / max(0.1, safe_float(scene.get("duration_seconds", 0.0)))) * 60.0
        for scene in scenes
        if safe_float(scene.get("duration_seconds", 0.0)) > 0.0
    ]
    transcripts = [coalesce_text(scene.get("transcript", "")) for scene in scenes]
    conflict_scenes = sum(1 for text in transcripts if set(token_list(text)) & CONFLICT_MARKERS)
    return {
        "typical_scene_length_seconds": round(mean(durations, 42.0), 2),
        "dialogue_density_lines_per_minute": round(mean(densities, 7.0), 2),
        "average_speakers_per_scene": round(mean(speaker_counts, 2.0), 2),
        "average_dialogue_lines_per_scene": round(mean(line_counts, 6.0), 2),
        "typical_beat_sequence": ["Cold Open", "Plan", "Complication", "Mix-up", "Turning Point", "Resolution"],
        "conflict_escalation": {
            "conflict_scene_ratio": round(conflict_scenes / max(1, len(scenes)), 3),
            "default_curve": ["setup", "pressure", "misread", "reversal", "repair"],
        },
        "defaulted": not bool(scenes),
    }


def build_episode_structure(scenes: list[dict[str, Any]], series_model: dict[str, Any]) -> dict[str, Any]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scene in scenes:
        episode_id = coalesce_text(scene.get("episode_id", "")) or "episode"
        by_episode[episode_id].append(scene)
    for episode_scenes in by_episode.values():
        episode_scenes.sort(key=lambda row: coalesce_text(row.get("scene_id", "")))
    scene_counts = [float(len(episode_scenes)) for episode_scenes in by_episode.values()]
    first_scene_lengths = [
        safe_float(episode_scenes[0].get("duration_seconds", 0.0))
        for episode_scenes in by_episode.values()
        if episode_scenes and safe_float(episode_scenes[0].get("duration_seconds", 0.0)) > 0.0
    ]
    source_durations = [
        safe_float(value)
        for value in (series_model.get("source_episode_durations", {}) if isinstance(series_model, dict) else {}).values()
        if safe_float(value) > 0.0
    ]
    avg_runtime = mean(source_durations, 22.0 * 60.0)
    avg_scene_count = mean(scene_counts, float(series_model.get("scene_count", 6) or 6) if isinstance(series_model, dict) else 6.0)
    cold_open = mean(first_scene_lengths, max(45.0, avg_runtime * 0.06))
    return {
        "cold_open_length_seconds": round(cold_open, 2),
        "conflict_intro_position": 0.16,
        "turning_point_position": 0.68,
        "resolution_duration_seconds": round(max(45.0, avg_runtime * 0.12), 2),
        "average_scene_count": round(avg_scene_count, 2),
        "average_episode_runtime_seconds": round(avg_runtime, 2),
        "defaulted": not bool(scenes),
    }


def build_dialogue_patterns(line_records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = scene_line_groups(line_records)
    turn_rates: list[float] = []
    short_count = 0
    long_count = 0
    interruption_count = 0
    setup_reaction_count = 0
    callback_counter: Counter[str] = Counter()
    for lines in groups.values():
        if not lines:
            continue
        speaker_changes = 0
        previous_speaker = ""
        previous_end = -1.0
        for line in lines:
            speaker = coalesce_text(line.get("speaker", ""))
            text = coalesce_text(line.get("text", ""))
            words = len(token_list(text))
            short_count += 1 if words <= 5 else 0
            long_count += 1 if words >= 14 else 0
            if previous_speaker and speaker and speaker != previous_speaker:
                speaker_changes += 1
            if previous_end >= 0 and safe_float(line.get("start", 0.0)) < previous_end - 0.08:
                interruption_count += 1
            if "?" in text:
                setup_reaction_count += 1
            for phrase in top_phrases([text], limit=2):
                callback_counter[phrase] += 1
            previous_speaker = speaker or previous_speaker
            previous_end = max(previous_end, safe_float(line.get("end", previous_end)))
        turn_rates.append(speaker_changes / max(1, len(lines) - 1))
    total_lines = max(1, len(line_records))
    return {
        "turn_taking": {
            "speaker_change_rate": round(mean(turn_rates, 0.65), 3),
            "default_pattern": "alternating A/B dialogue with reaction beats",
        },
        "short_answer_ratio": round(short_count / total_lines, 3),
        "long_answer_ratio": round(long_count / total_lines, 3),
        "interruption_ratio": round(interruption_count / total_lines, 3),
        "callback_structure": [phrase for phrase, count in callback_counter.most_common(8) if count > 1],
        "setup_reaction_punchline": {
            "question_setup_ratio": round(setup_reaction_count / total_lines, 3),
            "default_pattern": "setup -> reaction -> complication -> punchline/callback",
        },
        "defaulted": not bool(line_records),
    }


def language_counts(line_records: list[dict[str, Any]], scenes: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in itertools.chain(line_records, scenes):
        language = normalize_language_code(row.get("language", ""))
        if language:
            counter[language] += 1
    return dict(counter)


def build_behavior_model(
    cfg: dict[str, Any],
    rows: list[dict[str, Any]] | None = None,
    relationship_payload: dict[str, Any] | None = None,
    series_model: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    diagnostics: list[str] = []
    if rows is None:
        rows, row_diagnostics = load_dataset_rows(cfg)
        diagnostics.extend(row_diagnostics)
    else:
        rows = [row for row in rows if isinstance(row, dict)]
    series_model = series_model if isinstance(series_model, dict) else load_optional_series_model(cfg)
    relationship_payload = relationship_payload if isinstance(relationship_payload, dict) else load_character_relationships(cfg)
    line_records = collect_line_records(rows, series_model)
    scenes = collect_scene_records(rows, line_records)
    if not line_records:
        diagnostics.append("No usable speaker lines found; speaking_style uses defaults.")
    if not scenes:
        diagnostics.append("No usable scene records found; scene and episode structure use defaults.")
    if not relationship_payload.get("relationships"):
        diagnostics.append("No configured character relationships found; relationship_behavior is inferred from co-scenes only.")
    languages = language_counts(line_records, scenes)
    model = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_counts": {
            "dataset_rows": len(rows),
            "line_records": len(line_records),
            "scene_records": len(scenes),
            "configured_relationships": len(relationship_payload.get("relationships", []) or []),
        },
        "dominant_language": dominant_language(languages, series_model.get("dominant_language", "") if isinstance(series_model, dict) else ""),
        "language_counts": languages,
        "speaking_style": build_speaking_style(line_records, series_model),
        "relationship_behavior": build_relationship_behavior(line_records, relationship_payload),
        "scene_behavior": build_scene_behavior(scenes),
        "episode_structure": build_episode_structure(scenes, series_model),
        "dialogue_patterns": build_dialogue_patterns(line_records),
        "diagnostics": diagnostics,
        "defaults": {
            "line_pace": "natural",
            "line_energy": 0.52,
            "line_emotion": "focused",
            "scene_conflict": "small misunderstanding escalates and resolves within the scene",
            "comedy_pattern": "setup -> reaction -> complication -> punchline/callback",
        },
    }
    return model, render_summary(model)


def render_summary(model: dict[str, Any]) -> str:
    source_counts = model.get("source_counts", {}) if isinstance(model.get("source_counts"), dict) else {}
    lines = [
        "# Behavior Model Summary",
        "",
        f"- Generated at: {coalesce_text(model.get('generated_at', ''))}",
        f"- Dominant language: {coalesce_text(model.get('dominant_language', 'auto')) or 'auto'}",
        f"- Dataset rows: {int(source_counts.get('dataset_rows', 0) or 0)}",
        f"- Speaker lines: {int(source_counts.get('line_records', 0) or 0)}",
        f"- Scene records: {int(source_counts.get('scene_records', 0) or 0)}",
        f"- Configured relationships: {int(source_counts.get('configured_relationships', 0) or 0)}",
        "",
        "## Diagnostics",
        "",
    ]
    diagnostics = model.get("diagnostics", []) if isinstance(model.get("diagnostics"), list) else []
    if diagnostics:
        lines.extend(f"- {coalesce_text(item)}" for item in diagnostics if coalesce_text(item))
    else:
        lines.append("- No blocking diagnostics; behavior model was built from available data.")
    speaking_style = model.get("speaking_style", {}) if isinstance(model.get("speaking_style"), dict) else {}
    if speaking_style:
        lines.extend(["", "## Characters", ""])
        for name, style in sorted(speaking_style.items()):
            if not isinstance(style, dict):
                continue
            lines.append(
                f"- {name}: {style.get('line_count', 0)} lines, "
                f"{style.get('average_words_per_line', 0)} words/line, "
                f"energy {style.get('energy_label', 'medium')}"
            )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Analyze Behavior Model")
    cfg = load_config()
    worker_id = shared_worker_id_for_args(args)
    shared_workers = shared_workers_enabled_for_args(cfg, args)
    mark_step_started("08b_analyze_behavior_model", "global")
    if shared_workers:
        info(f"Shared NAS workers: enabled ({worker_id})")
    lease_manager = distributed_item_lease(
        root=distributed_step_runtime_root("08b_analyze_behavior_model", "global"),
        lease_name="global",
        cfg=cfg,
        worker_id=worker_id,
        enabled=shared_workers,
        meta={"step": "08b_analyze_behavior_model", "scope": "global", "worker_id": worker_id},
    )
    acquired = lease_manager.__enter__()
    if not acquired:
        info("Behavior model analysis is already running on another worker.")
        lease_manager.__exit__(None, None, None)
        return
    target = behavior_model_path(cfg)
    summary_target = behavior_summary_path(cfg)
    try:
        if target.exists() and not args.force:
            ok(f"Behavior model already exists: {target}")
            mark_step_completed(
                "08b_analyze_behavior_model",
                "global",
                {"behavior_model": str(target), "behavior_model_summary": str(summary_target), "skipped": True},
            )
            return
        model, summary = build_behavior_model(cfg)
        target.parent.mkdir(parents=True, exist_ok=True)
        summary_target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, model)
        write_text(summary_target, summary)
        mark_step_completed(
            "08b_analyze_behavior_model",
            "global",
            {
                "behavior_model": str(target),
                "behavior_model_summary": str(summary_target),
                "line_records": model.get("source_counts", {}).get("line_records", 0),
                "scene_records": model.get("source_counts", {}).get("scene_records", 0),
            },
        )
        ok(f"Behavior model written: {target}")
        ok(f"Behavior summary written: {summary_target}")
    except Exception as exc:
        mark_step_failed("08b_analyze_behavior_model", str(exc), "global", {"behavior_model": str(target)})
        raise
    finally:
        lease_manager.__exit__(None, None, None)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
