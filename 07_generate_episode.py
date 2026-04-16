#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Serienmodell trainieren und neue Folge erzeugen")
    parser.add_argument("--episode-id", help="Zielt auf eine konkrete Folge-ID wie folge_02.")
    return parser.parse_args()


def useful_character(name: str) -> bool:
    return has_primary_person_name(name)


def background_character(name: str) -> bool:
    return is_background_person_name(name)


def fallback_focus_characters(count: int = 2) -> list[str]:
    return GENERIC_FOCUS_CHARACTERS[: max(2, min(count, len(GENERIC_FOCUS_CHARACTERS)))]


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


def build_series_model(dataset_files: list, cfg: dict, char_map: dict) -> dict:
    character_directory = build_character_directory(char_map)
    character_stats: dict[str, dict[str, int | bool]] = defaultdict(lambda: {"scene_count": 0, "line_count": 0, "priority": False})
    speaker_samples: dict[str, list[str]] = defaultdict(list)
    transcripts = []
    scene_library = []
    speaker_counter: Counter[str] = Counter()
    keyword_counter: Counter[str] = Counter()

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
            characters = [name for name in extract_scene_characters(row, character_directory) if useful_character(name)]
            for character in characters:
                character_stats[character]["scene_count"] += 1
                if character in character_directory:
                    character_stats[character]["priority"] = bool(character_directory[character].get("priority", False))
            for segment in row.get("transcript_segments", []):
                speaker_name = segment.get("speaker_name", "")
                text = coalesce_text(segment.get("text", ""))
                if speaker_name and text:
                    speaker_counter[speaker_name] += 1
                    speaker_samples[speaker_name].append(text)
                    if useful_character(speaker_name):
                        character_stats[speaker_name]["line_count"] += 1
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
    top_keywords = extract_keywords(transcripts, limit=20)
    if not top_keywords:
        top_keywords = [keyword for keyword, _ in keyword_counter.most_common(20)]
    markov_chain = build_markov_chain(transcripts)

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
        "scene_library": scene_library,
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
) -> list[str]:
    speaker_samples = model.get("speaker_samples", {})
    speaker_a = focus_characters[0]
    speaker_b = focus_characters[1] if len(focus_characters) > 1 else fallback_focus_characters(2)[1]
    callback_candidates = [
        clean_callback(line)
        for speaker in (speaker_a, speaker_b)
        for line in speaker_samples.get(speaker, [])
    ]
    callback_candidates = [line for line in callback_candidates if len(line.split()) >= 3]
    callback = f"{speaker_a}: {rng.choice(callback_candidates)}" if callback_candidates else ""
    base_lines = beat_dialogue_templates(beat, speaker_a, speaker_b, keyword, callback)
    min_lines = int(cfg["generation"].get("min_dialogue_lines_per_scene", 4))
    max_lines = int(cfg["generation"].get("max_dialogue_lines_per_scene", 7))
    target_length = max(min_lines, min(max_lines, len(base_lines)))
    return base_lines[:target_length]


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
    scene_count = int(generation_cfg.get("default_scene_count", 6))
    focus_characters = select_focus_characters(model, rng)
    background_characters = select_background_characters(model, count=1)
    beats = ["Cold Open", "Plan", "Komplikation", "Verwechslung", "Wendepunkt", "Auflösung"]
    if len(beats) > 1:
        beat_offset = episode_index % len(beats)
        beats = beats[beat_offset:] + beats[:beat_offset]

    scenes = []
    markdown_lines = [
        "# Neue Folge",
        "",
        "## Basis des trainierten Serienmodells",
        "",
        f"- Ausgewertete Szenen: {model.get('scene_count', 0)}",
        f"- Hauptfiguren: {', '.join(focus_characters)}",
        f"- Wiederkehrende Themen: {', '.join(keywords[:6])}",
        "",
        "## Szenenplan",
        "",
        f"- Episoden-Seed: {rng_seed}",
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
        dialogue = build_dialogue(beat, scene_characters, model, rng, cfg, keyword)
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
                "prompt": (
                    f"{beat} mit {', '.join(scene_characters)}. Fokus auf {keyword}, schnelle Pointen "
                    f"und klaren Konfliktbogen."
                ),
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

    return {"scenes": scenes, "focus_characters": focus_characters, "keywords": keywords[:10]}, "\n".join(markdown_lines).strip() + "\n"


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Serienmodell trainieren und neue Folge erzeugen")
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

    story_dir = resolve_project_path("generation/story_prompts")
    shotlist_dir = resolve_project_path("generation/shotlists")
    story_dir.mkdir(parents=True, exist_ok=True)
    shotlist_dir.mkdir(parents=True, exist_ok=True)

    episode_id = (args.episode_id or "").strip() or next_episode_id(story_dir)
    episode_package, markdown = generate_episode_package(model, cfg, parse_episode_index(episode_id))
    markdown = markdown.replace("# Neue Folge", f"# {episode_id}", 1)
    write_text(story_dir / f"{episode_id}.md", markdown)
    write_json(
        shotlist_dir / f"{episode_id}.json",
        {
            "episode_id": episode_id,
            "trained_model": str(model_path),
            **episode_package,
        },
    )
    ok(f"Serienmodell trainiert und neue Folge erzeugt: {episode_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
