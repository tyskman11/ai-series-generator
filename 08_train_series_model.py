#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "ai_series_project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import argparse
import itertools
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
    dominant_language,
    error,
    extract_keywords,
    has_primary_person_name,
    is_background_person_name,
    headline,
    info,
    keyword_token_allowed,
    character_groups_for_names,
    load_character_relationships,
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
    normalize_language_code,
    relationship_prompt_fragments,
    relationships_for_characters,
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
LOCALIZED_EPISODE_LABELS = {
    "de": "Folge",
    "en": "Episode",
    "fr": "Episode",
    "es": "Episodio",
    "it": "Episodio",
    "pt": "Episodio",
    "nl": "Aflevering",
    "tr": "Bolum",
    "pl": "Odcinek",
}
LOCALIZED_TITLE_WORDS = {
    "de": {"the": "Das", "secret": "Geheimnis", "case": "Sache", "chaos": "Chaos", "challenge": "Challenge", "problem": "Problem", "of": "um"},
    "en": {"the": "The", "secret": "Secret", "case": "Thing", "chaos": "Chaos", "challenge": "Challenge", "problem": "Problem", "of": "of"},
    "fr": {"the": "Le", "secret": "Secret", "case": "Affaire", "chaos": "Chaos", "challenge": "Defi", "problem": "Probleme", "of": "de"},
    "es": {"the": "El", "secret": "Secreto", "case": "Asunto", "chaos": "Caos", "challenge": "Reto", "problem": "Problema", "of": "de"},
    "it": {"the": "Il", "secret": "Segreto", "case": "Caso", "chaos": "Caos", "challenge": "Sfida", "problem": "Problema", "of": "di"},
    "pt": {"the": "O", "secret": "Segredo", "case": "Caso", "chaos": "Caos", "challenge": "Desafio", "problem": "Problema", "of": "de"},
    "nl": {"the": "Het", "secret": "Geheim", "case": "Verhaal", "chaos": "Chaos", "challenge": "Challenge", "problem": "Probleem", "of": "rond"},
    "tr": {"the": "", "secret": "Sir", "case": "Mesele", "chaos": "Kaos", "challenge": "Meydan Okuma", "problem": "Sorun", "of": ""},
    "pl": {"the": "", "secret": "Tajemnica", "case": "Sprawa", "chaos": "Chaos", "challenge": "Wyzwanie", "problem": "Problem", "of": ""},
}
LOCALIZED_GENERIC_TITLE_FALLBACKS = {
    "en": ["The Big Plan", "The Secret Project", "The Double Mix-Up", "Total Chaos", "The Wrong Trick"],
    "fr": ["Le Grand Plan", "Le Projet Secret", "Le Double Malentendu", "Le Chaos Total", "Le Mauvais Tour"],
    "es": ["El Gran Plan", "El Proyecto Secreto", "La Doble Confusion", "El Caos Total", "El Truco Equivocado"],
    "it": ["Il Grande Piano", "Il Progetto Segreto", "Il Doppio Malinteso", "Il Caos Totale", "Il Trucco Sbagliato"],
    "pt": ["O Grande Plano", "O Projeto Secreto", "A Dupla Confusao", "O Caos Total", "O Truque Errado"],
    "nl": ["Het Grote Plan", "Het Geheime Project", "De Dubbele Verwarring", "Totale Chaos", "De Verkeerde Truc"],
    "tr": ["Buyuk Plan", "Gizli Proje", "Iki Kat Karisiklik", "Tam Kaos", "Yanlis Numara"],
    "pl": ["Wielki Plan", "Tajny Projekt", "Podwojne Zamieszanie", "Totalny Chaos", "Zly Trik"],
}
LOCALIZED_DIALOGUE_LABELS = {
    "de": "Dialog",
    "en": "Dialogue",
    "fr": "Dialogue",
    "es": "Dialogo",
    "it": "Dialogo",
    "pt": "Dialogo",
    "nl": "Dialoog",
    "tr": "Diyalog",
    "pl": "Dialog",
}
LOCALIZED_BEAT_LABELS = {
    "de": {
        "Cold Open": "Cold Open",
        "Plan": "Plan",
        "Komplikation": "Komplikation",
        "Verwechslung": "Verwechslung",
        "Wendepunkt": "Wendepunkt",
        "Auflösung": "Auflösung",
    },
    "en": {
        "Cold Open": "Cold Open",
        "Plan": "Plan",
        "Komplikation": "Complication",
        "Verwechslung": "Mix-up",
        "Wendepunkt": "Turning Point",
        "Auflösung": "Resolution",
    },
    "fr": {
        "Cold Open": "Ouverture",
        "Plan": "Plan",
        "Komplikation": "Complication",
        "Verwechslung": "Malentendu",
        "Wendepunkt": "Tournant",
        "Auflösung": "Resolution",
    },
    "es": {
        "Cold Open": "Apertura",
        "Plan": "Plan",
        "Komplikation": "Complicacion",
        "Verwechslung": "Confusion",
        "Wendepunkt": "Giro",
        "Auflösung": "Resolucion",
    },
}
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


def language_family(language: str) -> str:
    normalized = normalize_language_code(language)
    return normalized.split("-", 1)[0] if normalized else ""


def localized_title_words(language: str) -> dict[str, str]:
    family = language_family(language)
    return LOCALIZED_TITLE_WORDS.get(family, LOCALIZED_TITLE_WORDS["en"])


def localized_beat_label(beat: str, language: str) -> str:
    family = language_family(language)
    labels = LOCALIZED_BEAT_LABELS.get(family) or LOCALIZED_BEAT_LABELS.get("en", {})
    return labels.get(beat, beat)


def localized_dialogue_label(language: str) -> str:
    return LOCALIZED_DIALOGUE_LABELS.get(language_family(language), LOCALIZED_DIALOGUE_LABELS["en"])


def humanize_keyword(keyword: str, language: str = "") -> str:
    cleaned = coalesce_text(keyword).replace("_", " ").replace("-", " ")
    tokens = [token for token in cleaned.split() if token]
    if not tokens:
        return localized_title_words(language)["secret"]
    return " ".join(token.capitalize() for token in tokens[:3])


def clean_generation_keywords(keywords: object, limit: int = 20) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    if isinstance(keywords, str):
        iterable = [keywords]
    elif isinstance(keywords, (list, tuple, set)):
        iterable = list(keywords)
    else:
        iterable = []
    for raw_keyword in iterable:
        keyword = coalesce_text(raw_keyword).replace("_", " ").replace("-", " ")
        tokens = [token.lower() for token in tokens_from_text(keyword)]
        if not tokens:
            continue
        if any(not keyword_token_allowed(token) for token in tokens):
            continue
        normalized = " ".join(tokens)
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
        if len(cleaned) >= limit:
            break
    return cleaned


def format_episode_number(episode_index: int, language: str = "") -> str:
    prefix = LOCALIZED_EPISODE_LABELS.get(language_family(language), "Episode")
    return f"{prefix} {max(1, int(episode_index)):02d}"


def build_episode_title(
    focus_characters: list[str],
    keywords: list[str],
    rng: random.Random,
    episode_index: int,
    model: dict | None = None,
    language: str = "",
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
        humanized = humanize_keyword(keyword, language)
        normalized = humanized.lower()
        if normalized in seen_keywords:
            continue
        seen_keywords.add(normalized)
        usable_keywords.append(humanized)

    title_seed = episode_index + len(focus_characters) + len(usable_keywords)
    family = language_family(language)
    words = localized_title_words(language)
    if not usable_keywords:
        fallbacks = GENERIC_TITLE_FALLBACKS if family == "de" else LOCALIZED_GENERIC_TITLE_FALLBACKS.get(family, LOCALIZED_GENERIC_TITLE_FALLBACKS["en"])
        return fallbacks[title_seed % len(fallbacks)]

    main_keyword = usable_keywords[0]
    secondary_keyword = usable_keywords[1] if len(usable_keywords) > 1 else ""
    suffix = TITLE_SUFFIXES[(title_seed + len(main_keyword)) % len(TITLE_SUFFIXES)]
    main_compound = main_keyword.replace(" ", "-")
    anchor_candidates = collect_title_anchor_candidates(model, blocked_tokens, usable_keywords[:3])
    if anchor_candidates:
        best_anchor = anchor_candidates[0][0]
        if " " in best_anchor:
            return best_anchor
        if family == "de":
            return f"Das Geheimnis um {best_anchor}"
        title_prefix = f"{words['the']} " if words.get("the") else ""
        of_word = f" {words['of']} " if words.get("of") else " "
        return f"{title_prefix}{words['secret']}{of_word}{best_anchor}".strip()

    candidate_titles: list[str] = []
    if family != "de":
        title_prefix = f"{words['the']} " if words.get("the") else ""
        of_word = f" {words['of']} " if words.get("of") else " "
        if " " in main_keyword:
            candidate_titles.extend(
                [
                    f"{title_prefix}{words['case']}{of_word}{main_keyword}".strip(),
                    f"{title_prefix}{words['secret']}{of_word}{main_keyword}".strip(),
                    f"{title_prefix}{words['chaos']}{of_word}{main_keyword}".strip(),
                ]
            )
        else:
            candidate_titles.extend(
                [
                    f"{title_prefix}{main_compound} {words['secret']}".strip(),
                    f"{title_prefix}{main_compound} {words['problem']}".strip(),
                    f"{title_prefix}{words['case']}{of_word}{main_keyword}".strip(),
                    f"{title_prefix}{main_compound} {words['challenge']}".strip(),
                ]
            )
        if secondary_keyword:
            candidate_titles.append(f"{title_prefix}{main_compound} {humanize_keyword(secondary_keyword, language).replace(' ', '-')} {words['problem']}".strip())
    elif " " in main_keyword:
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

    if family == "de" and secondary_keyword:
        if " " not in main_keyword and " " not in secondary_keyword:
            candidate_titles.append(f"Das {main_compound}-{humanize_keyword(secondary_keyword, language).replace(' ', '-')}-Problem")
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
    language_counter: Counter[str] = Counter()
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
            row_language = normalize_language_code(row.get("detected_language", "") or row.get("language", ""))
            if row_language:
                language_counter[row_language] += 1
            scene_keywords = clean_generation_keywords(row.get("scene_keywords") or extract_keywords([transcript], limit=8), limit=8)
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
                segment_language = normalize_language_code(segment.get("language", "") or row_language)
                if segment_language:
                    language_counter[segment_language] += 1
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
                            "language": segment_language,
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
                    "language": row_language,
                }
            )

    top_characters = sorted(
        character_stats.items(),
        key=lambda item: (-int(bool(item[1].get("priority", False))), -int(item[1]["scene_count"]), -int(item[1]["line_count"]), item[0]),
    )
    relationship_payload = load_character_relationships(cfg)
    group_membership_by_character: dict[str, list[str]] = defaultdict(list)
    for group_id, group in relationship_payload.get("groups", {}).items():
        for character_name in group.get("characters", []) or []:
            if group_id not in group_membership_by_character[character_name]:
                group_membership_by_character[character_name].append(group_id)
    relationship_count_by_character: Counter[str] = Counter()
    for relation in relationship_payload.get("relationships", []) or []:
        source = str(relation.get("source", "")).strip()
        target = str(relation.get("target", "")).strip()
        if source:
            relationship_count_by_character[source] += 1
        if target:
            relationship_count_by_character[target] += 1
    linked_samples, linked_line_library = build_linked_segment_line_library(cfg)
    for speaker, samples in linked_samples.items():
        for sample in samples:
            if sample not in speaker_samples[speaker]:
                speaker_samples[speaker].append(sample)
    for speaker, entries in linked_line_library.items():
        for entry in entries:
            append_unique_line_entry(speaker_line_library, speaker, entry)
    top_keywords = clean_generation_keywords(extract_keywords(transcripts, limit=40), limit=20)
    if not top_keywords:
        top_keywords = clean_generation_keywords([keyword for keyword, _ in keyword_counter.most_common(40)], limit=20)
    markov_chain = build_markov_chain(transcripts)
    average_scene_duration = sum(scene_durations) / max(1, len(scene_durations))
    average_segment_duration = sum(segment_durations) / max(1, len(segment_durations))

    generation_cfg = cfg.get("generation", {}) if isinstance(cfg.get("generation", {}), dict) else {}
    configured_language = normalize_language_code(generation_cfg.get("language", ""))
    return {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_files": [str(path) for path in dataset_files],
        "scene_count": len(scene_library),
        "characters": [
            {
                "name": name,
                **stats,
                "groups": group_membership_by_character.get(name, []),
                "relationship_count": int(relationship_count_by_character.get(name, 0)),
            }
            for name, stats in top_characters
        ],
        "speakers": dict(speaker_counter),
        "keywords": top_keywords,
        "character_groups": relationship_payload.get("groups", {}),
        "character_relationships": relationship_payload,
        "series_inputs": relationship_payload.get("series_inputs", {}),
        "language_counts": dict(language_counter),
        "dominant_language": dominant_language(dict(language_counter), configured_language),
        "speaker_samples": {speaker: samples[:20] for speaker, samples in speaker_samples.items()},
        "speaker_line_library": {speaker: rows[:160] for speaker, rows in speaker_line_library.items()},
        "character_reference_library": character_reference_library,
        "scene_library": scene_library,
        "source_episode_durations": dict(episode_duration_counter),
        "average_scene_duration_seconds": round(average_scene_duration, 3),
        "average_segment_duration_seconds": round(average_segment_duration, 3),
        "markov_order": 2,
        "markov_chain": markov_chain,
        "behavior_model_path": str(behavior_model_path(cfg)),
        "behavior_model": load_behavior_model(cfg),
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
    language: str = "",
) -> list[str]:
    topic = humanize_keyword(keyword, language)
    family = language_family(language)
    multilingual_templates = {
        "de": {
            "callback": f"{speaker_a}: Wir halten uns diesmal wirklich an den Plan.",
            "Cold Open": [
                f"{speaker_a}: Die Sache mit {topic} sollte heute eigentlich ganz leicht werden.",
                f"{speaker_b}: Genau deshalb fühlt es sich jetzt schon nach Ärger an.",
                f"{speaker_a}: Wenn wir sofort improvisieren, endet das wieder im Chaos.",
            ],
            "Plan": [
                f"{speaker_a}: Wir brauchen einen klaren Plan, bevor {topic} alles durcheinanderbringt.",
                f"{speaker_b}: Dann teilen wir das Problem auf und prüfen jeden Schritt.",
                f"{speaker_a}: Einverstanden, aber diesmal ohne spontane Abkürzungen.",
            ],
            "Komplikation": [
                f"{speaker_b}: Jetzt wird {topic} größer, obwohl es eben noch harmlos aussah.",
                f"{speaker_a}: Dann hat irgendwo jemand einen wichtigen Schritt übersprungen.",
                f"{speaker_b}: Und natürlich merken wir das erst mitten im Stress.",
            ],
            "Verwechslung": [
                f"{speaker_a}: Moment, wir reden über zwei völlig verschiedene Versionen von {topic}.",
                f"{speaker_b}: Das erklärt, warum wir beide überzeugt waren, recht zu haben.",
                f"{speaker_a}: Dann sortieren wir zuerst die Fakten und streiten danach weiter.",
            ],
            "Wendepunkt": [
                f"{speaker_b}: Ich glaube, ich sehe endlich, warum {topic} die ganze Zeit blockiert war.",
                f"{speaker_a}: Bitte sag mir, dass die Lösung nicht noch mehr Improvisation braucht.",
                f"{speaker_b}: Nein, nur einen sauberen Schritt zurück und einen besseren Aufbau.",
            ],
            "Auflösung": [
                f"{speaker_a}: Siehst du, mit etwas Ruhe funktioniert {topic} am Ende doch.",
                f"{speaker_b}: Und wir haben sogar noch Zeit, das Ergebnis ordentlich zu zeigen.",
                f"{speaker_a}: Diese Reihenfolge merken wir uns für das nächste Chaos.",
            ],
        },
        "en": {
            "callback": f"{speaker_a}: This time we are really sticking to the plan.",
            "Cold Open": [
                f"{speaker_a}: {topic} was supposed to be the easiest task today.",
                f"{speaker_b}: Which is exactly why it already feels like trouble.",
                f"{speaker_a}: If we start improvising right away, this turns into complete chaos again.",
            ],
            "Plan": [
                f"{speaker_a}: We need a clear plan before {topic} throws us off balance.",
                f"{speaker_b}: Then we break the problem into small steps and test each one.",
                f"{speaker_a}: Fine, but this time without any spontaneous shortcuts.",
            ],
            "Komplikation": [
                f"{speaker_b}: Now {topic} is becoming bigger even though it looked simple a moment ago.",
                f"{speaker_a}: Then someone must have skipped a crucial step somewhere.",
                f"{speaker_b}: And of course we only notice that right in the middle of the stress.",
            ],
            "Verwechslung": [
                f"{speaker_a}: Wait, we are talking about two completely different versions of {topic}.",
                f"{speaker_b}: That explains why both of us were convinced we were right.",
                f"{speaker_a}: Then we sort the facts first and continue arguing later.",
            ],
            "Wendepunkt": [
                f"{speaker_b}: I think I finally see why {topic} was blocked the whole time.",
                f"{speaker_a}: Please tell me the solution does not require even more improvisation.",
                f"{speaker_b}: No, just one clean step back and a better setup.",
            ],
            "Auflösung": [
                f"{speaker_a}: See, with a little calm {topic} suddenly works after all.",
                f"{speaker_b}: And we still have enough time to present the result properly.",
                f"{speaker_a}: Let's remember this sequence for the next round of chaos.",
            ],
        },
        "fr": {
            "callback": f"{speaker_a}: Cette fois, on suit vraiment le plan.",
            "Plan": [
                f"{speaker_a}: Il nous faut un plan clair avant que {topic} complique tout.",
                f"{speaker_b}: Alors on se calme et on vérifie chaque étape.",
                f"{speaker_a}: D'accord, mais sans raccourci improvisé cette fois.",
            ],
        },
        "es": {
            "callback": f"{speaker_a}: Esta vez vamos a seguir el plan de verdad.",
            "Plan": [
                f"{speaker_a}: Necesitamos un plan claro antes de que {topic} lo complique todo.",
                f"{speaker_b}: Entonces dividimos el problema y revisamos cada paso.",
                f"{speaker_a}: De acuerdo, pero sin atajos improvisados esta vez.",
            ],
        },
        "it": {
            "callback": f"{speaker_a}: Questa volta seguiamo davvero il piano.",
            "Plan": [
                f"{speaker_a}: Ci serve un piano chiaro prima che {topic} complichi tutto.",
                f"{speaker_b}: Allora controlliamo ogni passaggio con calma.",
                f"{speaker_a}: Va bene, ma niente scorciatoie improvvisate.",
            ],
        },
        "pt": {
            "callback": f"{speaker_a}: Desta vez vamos seguir mesmo o plano.",
            "Plan": [
                f"{speaker_a}: Precisamos de um plano claro antes que {topic} complique tudo.",
                f"{speaker_b}: Então dividimos o problema e conferimos cada passo.",
                f"{speaker_a}: Certo, mas sem atalhos improvisados desta vez.",
            ],
        },
        "nl": {
            "callback": f"{speaker_a}: Deze keer houden we ons echt aan het plan.",
            "Plan": [
                f"{speaker_a}: We hebben een duidelijk plan nodig voordat {topic} alles in de war schopt.",
                f"{speaker_b}: Dan delen we het probleem op en controleren we elke stap.",
                f"{speaker_a}: Goed, maar deze keer zonder spontane omwegen.",
            ],
        },
        "tr": {
            "callback": f"{speaker_a}: Bu kez gerçekten plana bağlı kalıyoruz.",
            "Plan": [
                f"{speaker_a}: {topic} her şeyi karıştırmadan önce net bir plana ihtiyacımız var.",
                f"{speaker_b}: O zaman sorunu parçalara ayırıp her adımı kontrol ederiz.",
                f"{speaker_a}: Tamam, ama bu kez doğaçlama kestirme yol yok.",
            ],
        },
        "pl": {
            "callback": f"{speaker_a}: Tym razem naprawdę trzymamy się planu.",
            "Plan": [
                f"{speaker_a}: Potrzebujemy jasnego planu, zanim {topic} wszystko skomplikuje.",
                f"{speaker_b}: Więc dzielimy problem na kroki i sprawdzamy każdy z nich.",
                f"{speaker_a}: Dobrze, ale tym razem bez improwizowanych skrótów.",
            ],
        },
    }
    templates = multilingual_templates.get(family, multilingual_templates["en"])
    callback_line = callback or templates["callback"]
    lines = templates.get(beat) or templates.get("Plan") or multilingual_templates["en"]["Plan"]
    return [*lines, callback_line]
    return templates.get(beat, templates["Plan"])


def build_dialogue(
    beat: str,
    focus_characters: list[str],
    model: dict,
    rng: random.Random,
    cfg: dict,
    keyword: str,
    target_length: int | None = None,
    language: str = "",
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
    base_lines = beat_dialogue_templates(beat, speaker_a, speaker_b, keyword, callback, language=language)
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
                line_sources.append({"type": "generated_template", "language": normalize_language_code(language)})
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
                        "language": normalize_language_code(original_entry.get("language", "") or language),
                    }
                )
                continue
            sample_line = choose_speaker_sample(speaker, line_text, keyword, speaker_samples, used_samples, rng)
            if sample_line:
                used_samples.add(sample_line.lower())
                enriched_lines.append(f"{speaker}: {sample_line}")
                line_sources.append({"type": "speaker_sample", "language": normalize_language_code(language)})
            else:
                suffix = f" ({line_rounds})" if line_rounds > 1 and line_text and line_text[-1] not in ".!?" else ""
                enriched_lines.append(f"{speaker}: {line_text}{suffix}" if suffix else line)
                line_sources.append({"type": "generated_template", "language": normalize_language_code(language)})
        if line_rounds >= 12:
            break
    min_lines = int(cfg["generation"].get("min_dialogue_lines_per_scene", 4))
    max_lines = int(cfg["generation"].get("max_dialogue_lines_per_scene", 7))
    final_length = max(min_lines, min(max_lines, len(enriched_lines))) if target_length is None else max(min_lines, len(enriched_lines))
    return enriched_lines[:final_length], line_sources[:final_length]


def behavior_model_path(cfg: dict) -> Path:
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    return resolve_project_path(str(paths.get("behavior_model", "generation/model/behavior_model.json")))


def load_behavior_model(cfg: dict) -> dict:
    path = behavior_model_path(cfg)
    payload = read_json(path, {}) if path.exists() else {}
    return payload if isinstance(payload, dict) else {}


def active_behavior_model(model: dict, cfg: dict) -> dict:
    embedded = model.get("behavior_model", {}) if isinstance(model.get("behavior_model", {}), dict) else {}
    if embedded:
        return embedded
    return load_behavior_model(cfg)


def behavior_relationship_key(a: str, b: str) -> str:
    left, right = sorted([coalesce_text(a), coalesce_text(b)])
    return f"{left}||{right}"


def speaking_style_for_character(behavior_model: dict, character: str) -> dict:
    styles = behavior_model.get("speaking_style", {}) if isinstance(behavior_model.get("speaking_style"), dict) else {}
    style = styles.get(character, {}) if character else {}
    return style if isinstance(style, dict) else {}


def relationship_behavior_for_scene(behavior_model: dict, characters: list[str]) -> list[dict]:
    relationships = (
        behavior_model.get("relationship_behavior", {})
        if isinstance(behavior_model.get("relationship_behavior"), dict)
        else {}
    )
    rows: list[dict] = []
    for left, right in itertools.combinations([name for name in characters if name], 2):
        entry = relationships.get(behavior_relationship_key(left, right), {})
        if isinstance(entry, dict) and entry:
            rows.append(entry)
    return rows


def line_text_from_dialogue_line(line: str) -> tuple[str, str]:
    text = coalesce_text(line)
    if ":" not in text:
        return "", text
    speaker, line_text = text.split(":", 1)
    return coalesce_text(speaker), coalesce_text(line_text)


def voice_metadata_for_line(
    speaker: str,
    text: str,
    source: dict,
    behavior_model: dict,
    average_segment_duration: float,
) -> dict:
    style = speaking_style_for_character(behavior_model, speaker)
    word_count = max(1, len(tokens_from_text(text)))
    average_words = float(style.get("average_words_per_line", 0.0) or 0.0)
    energy = float(style.get("energy_level", 0.52) or 0.52)
    if "!" in text:
        energy = max(energy, 0.72)
    if "?" in text:
        energy = max(energy, 0.58)
    if word_count <= max(4.0, average_words * 0.55):
        pace = "quick"
    elif average_words and word_count >= average_words * 1.45:
        pace = "measured"
    else:
        pace = "natural"
    emotion = "curious" if "?" in text else "excited" if "!" in text else "focused"
    source_type = coalesce_text(source.get("type", "")) if isinstance(source, dict) else ""
    duration_floor = max(1.1, word_count * (0.22 if pace == "quick" else 0.31 if pace == "measured" else 0.27))
    target_duration = max(duration_floor, min(8.0, float(average_segment_duration or 2.7) * max(0.8, word_count / max(5.0, average_words or 7.0))))
    return {
        "speaker": speaker or "Narrator",
        "text": text,
        "emotion": emotion,
        "pace": pace,
        "energy": round(max(0.0, min(1.0, energy)), 3),
        "target_duration_seconds": round(target_duration, 3),
        "voice_reference_priority": [
            value
            for value in (
                "matched_original_segment" if source_type == "original_line" else "",
                "trained_character_voice_model",
                "speaker_reference_samples",
                "closest_language_reference",
            )
            if value
        ],
    }


def build_dialogue_voice_metadata(
    dialogue: list[str],
    dialogue_sources: list[dict],
    behavior_model: dict,
    model: dict,
) -> list[dict]:
    average_segment_duration = float(model.get("average_segment_duration_seconds", 2.7) or 2.7)
    metadata: list[dict] = []
    for index, line in enumerate(dialogue):
        speaker, text = line_text_from_dialogue_line(line)
        source = dialogue_sources[index] if index < len(dialogue_sources) and isinstance(dialogue_sources[index], dict) else {}
        metadata.append(voice_metadata_for_line(speaker, text, source, behavior_model, average_segment_duration))
    return metadata


def build_behavior_scene_fields(
    *,
    beat: str,
    keyword: str,
    scene_characters: list[str],
    scene_index: int,
    scene_count: int,
    behavior_model: dict,
    relationship_context: list[dict],
    dialogue: list[str],
    language: str,
) -> dict:
    defaults = behavior_model.get("defaults", {}) if isinstance(behavior_model.get("defaults"), dict) else {}
    relationships = relationship_behavior_for_scene(behavior_model, scene_characters)
    styles = [speaking_style_for_character(behavior_model, character) for character in scene_characters]
    behavior_constraints: list[str] = []
    dialogue_constraints: list[str] = []
    character_intents: dict[str, str] = {}
    for character, style in zip(scene_characters, styles):
        words = style.get("average_words_per_line", "")
        energy = style.get("energy_label", "medium")
        phrases = style.get("typical_phrases", []) if isinstance(style.get("typical_phrases"), list) else []
        reactions = style.get("recurring_reactions", []) if isinstance(style.get("recurring_reactions"), list) else []
        if words:
            behavior_constraints.append(f"{character}: keep lines near {words} words with {energy} energy")
        if phrases:
            dialogue_constraints.append(f"{character}: allow natural callbacks like '{phrases[0]}'")
        if reactions:
            dialogue_constraints.append(f"{character}: reaction pattern '{reactions[0]}'")
        if beat == "Cold Open":
            intent = f"introduce pressure around {keyword}"
        elif beat in {"Komplikation", "Verwechslung"}:
            intent = f"push against the misunderstanding around {keyword}"
        elif beat == "Auflösung":
            intent = f"help resolve {keyword} without losing character voice"
        else:
            intent = f"advance the plan around {keyword}"
        character_intents[character] = intent
    if relationships:
        for row in relationships[:3]:
            characters = row.get("characters", []) if isinstance(row.get("characters"), list) else []
            dynamic = coalesce_text(row.get("typical_dynamic", "") or row.get("configured_dynamic", ""))
            leader = coalesce_text(row.get("conversation_leader", ""))
            if characters and dynamic:
                behavior_constraints.append(f"{' / '.join(characters)} dynamic: {dynamic}")
            if leader:
                behavior_constraints.append(f"{leader} tends to drive the exchange")
    elif relationship_context:
        behavior_constraints.append("Use the configured relationship context to motivate the scene conflict")
    dialogue_patterns = behavior_model.get("dialogue_patterns", {}) if isinstance(behavior_model.get("dialogue_patterns"), dict) else {}
    comedy_pattern = coalesce_text(
        (dialogue_patterns.get("setup_reaction_punchline", {}) if isinstance(dialogue_patterns.get("setup_reaction_punchline"), dict) else {}).get("default_pattern", "")
    ) or coalesce_text(defaults.get("comedy_pattern", "")) or "setup -> reaction -> complication -> punchline/callback"
    callback_targets = clean_generation_keywords(
        extract_keywords(["\n".join(dialogue), keyword, *scene_characters], limit=8),
        limit=5,
    )
    progress = scene_index / max(1, scene_count - 1)
    if progress < 0.18:
        arc = "curiosity rises into a clear problem"
    elif progress < 0.68:
        arc = "pressure escalates through conflicting assumptions"
    else:
        arc = "energy narrows toward repair and payoff"
    family = language_family(language)
    scene_purpose = (
        f"{localized_beat_label(beat, language)}: {keyword} im Episodenbogen vorantreiben"
        if family == "de"
        else f"{localized_beat_label(beat, language)}: move {keyword} through the episode arc"
    )
    conflict = behavior_constraints[0] if behavior_constraints else coalesce_text(defaults.get("scene_conflict", ""))
    if not conflict:
        conflict = f"{keyword} creates a small misunderstanding that the scene must escalate and clarify"
    return {
        "scene_purpose": scene_purpose,
        "conflict": conflict,
        "character_intents": character_intents,
        "behavior_constraints": behavior_constraints[:10],
        "dialogue_style_constraints": dialogue_constraints[:10],
        "comedy_pattern": comedy_pattern,
        "emotional_arc": arc,
        "callback_targets": callback_targets,
    }


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


def allocate_scene_runtime_seconds(
    target_runtime_seconds: int,
    scene_floor_seconds: list[float],
    scene_beats: list[str] | None = None,
) -> list[float]:
    if not scene_floor_seconds:
        return []
    floors = [max(12.0, float(value or 0.0)) for value in scene_floor_seconds]
    beats = scene_beats or []
    beat_weight_map = {
        "Cold Open": 0.92,
        "Plan": 1.0,
        "Komplikation": 1.08,
        "Verwechslung": 1.05,
        "Wendepunkt": 1.14,
        "Auflösung": 0.96,
    }
    target_total = max(300, int(target_runtime_seconds or 0))
    floor_total = float(sum(floors))

    def rebalance_allocations(values: list[float], minimums: list[float]) -> list[float]:
        adjusted = [round(float(value or 0.0), 2) for value in values]
        mins = [round(float(value or 0.0), 2) for value in minimums]
        if not adjusted:
            return adjusted
        delta = round(float(target_total) - sum(adjusted), 2)
        if abs(delta) < 0.01:
            return adjusted
        if delta > 0:
            adjusted[-1] = round(adjusted[-1] + delta, 2)
            return adjusted
        for index in range(len(adjusted) - 1, -1, -1):
            removable = round(adjusted[index] - mins[index], 2)
            if removable <= 0.0:
                continue
            step = min(removable, abs(delta))
            adjusted[index] = round(adjusted[index] - step, 2)
            delta = round(delta + step, 2)
            if abs(delta) < 0.01:
                break
        if abs(delta) >= 0.01:
            adjusted[-1] = round(max(mins[-1], adjusted[-1] + delta), 2)
        return adjusted

    if floor_total <= 0.0:
        even = max(12.0, float(target_total) / max(1, len(floors)))
        return [round(even, 2) for _ in floors]
    if floor_total >= float(target_total):
        scale = float(target_total) / floor_total
        scaled = [max(6.0, round(value * scale, 2)) for value in floors]
        return rebalance_allocations(scaled, [6.0 for _ in scaled])

    remaining = float(target_total) - floor_total
    weights: list[float] = []
    for index, floor in enumerate(floors):
        beat = coalesce_text(beats[index]) if index < len(beats) else ""
        beat_weight = beat_weight_map.get(beat, 1.0)
        weights.append(max(1.0, floor * 0.35) + beat_weight * 4.0)
    weight_total = sum(weights) or float(len(weights))
    allocations: list[float] = []
    for floor, weight in zip(floors, weights):
        allocations.append(round(floor + (remaining * (weight / weight_total)), 2))
    return rebalance_allocations(allocations, floors)


def configured_character_group_id(model: dict, cfg: dict | None) -> str:
    generation_cfg = cfg.get("generation", {}) if isinstance(cfg, dict) and isinstance(cfg.get("generation", {}), dict) else {}
    explicit_group = str(
        generation_cfg.get("active_character_group", "")
        or generation_cfg.get("character_group", "")
        or generation_cfg.get("default_character_group", "")
    ).strip()
    if explicit_group:
        return explicit_group
    active_input = str(generation_cfg.get("active_series_input", "") or generation_cfg.get("series_input", "")).strip()
    series_inputs = model.get("series_inputs", {}) if isinstance(model.get("series_inputs", {}), dict) else {}
    input_payload = series_inputs.get(active_input, {}) if active_input else {}
    if isinstance(input_payload, dict):
        return str(input_payload.get("default_group", "") or "").strip()
    return ""


def candidate_names_for_group(model: dict, group_id: str, candidates: list[str]) -> list[str]:
    groups = model.get("character_groups", {}) if isinstance(model.get("character_groups", {}), dict) else {}
    group = groups.get(group_id, {}) if group_id else {}
    if not isinstance(group, dict):
        return []
    candidate_set = set(candidates)
    selected: list[str] = []
    for name in group.get("characters", []) or []:
        if name in candidate_set and name not in selected and useful_character(name):
            selected.append(name)
    return selected


def select_focus_characters(model: dict, rng: random.Random, count: int = 3, cfg: dict | None = None) -> list[str]:
    prioritized = [row["name"] for row in model.get("characters", []) if useful_character(row["name"]) and bool(row.get("priority", False))]
    candidates = prioritized or [row["name"] for row in model.get("characters", []) if useful_character(row["name"])]
    if not candidates:
        candidates = [speaker for speaker in model.get("speakers", {}).keys() if useful_character(speaker)]
    if not candidates:
        return fallback_focus_characters(min(count, 2))
    explicit_group = configured_character_group_id(model, cfg)
    group_candidates = candidate_names_for_group(model, explicit_group, candidates)
    if not group_candidates:
        relationship_payload = model.get("character_relationships", {})
        candidate_groups = [
            group
            for group in character_groups_for_names(relationship_payload, candidates)
            if any(name in candidates for name in group.get("characters", []) or [])
        ]
        if candidate_groups:
            candidate_groups.sort(key=lambda row: (-len(row.get("characters", []) or []), row.get("label", row.get("id", ""))))
            group_candidates = candidate_names_for_group(model, str(candidate_groups[0].get("id", "")), candidates)
    if group_candidates:
        selected = group_candidates[:count]
        for candidate in candidates:
            if len(selected) >= count:
                break
            if candidate not in selected:
                selected.append(candidate)
        if len(selected) == 1:
            selected.append(fallback_focus_characters(2)[1])
        return selected
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


def scene_title(beat: str, keyword: str, language: str = "") -> str:
    return f"{localized_beat_label(beat, language)}: {keyword.capitalize()}"


def summary_line(characters: list[str], keyword: str, beat: str, language: str = "") -> str:
    joined = ", ".join(characters)
    family = language_family(language)
    beat_label = localized_beat_label(beat, language).lower()
    if family == "de":
        verb = "treibt" if len(characters) == 1 else "treiben"
        return f"{joined} {verb} das Thema '{keyword}' voran und geraten in eine typische {beat_label}-Situation."
    if family == "fr":
        return f"{joined} font avancer le sujet '{keyword}' et entrent dans une situation de {beat_label} typique."
    if family == "es":
        return f"{joined} impulsan el tema '{keyword}' y entran en una situacion tipica de {beat_label}."
    return f"{joined} push the theme '{keyword}' forward and land in a typical {beat_label} situation."


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
    series_language: str = "",
    style_descriptor: str = "",
    relationship_context: list[dict] | None = None,
    behavior_fields: dict | None = None,
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
    relationship_context = relationship_context if isinstance(relationship_context, list) else []
    behavior_fields = behavior_fields if isinstance(behavior_fields, dict) else {}
    behavior_constraints = [
        coalesce_text(value)
        for value in behavior_fields.get("behavior_constraints", [])
        if coalesce_text(value)
    ] if isinstance(behavior_fields.get("behavior_constraints", []), list) else []
    dialogue_style_constraints = [
        coalesce_text(value)
        for value in behavior_fields.get("dialogue_style_constraints", [])
        if coalesce_text(value)
    ] if isinstance(behavior_fields.get("dialogue_style_constraints", []), list) else []
    relationship_fragments = relationship_prompt_fragments(
        {"relationships": relationship_context},
        scene_characters,
        limit=4,
    )
    style_positive = ", ".join(style_profile.get("positive", [])[:3])
    style_negative = ", ".join(style_profile.get("negative", [])[:5])
    style_guidance = style_profile.get("guidance", {}) if isinstance(style_profile.get("guidance", {}), dict) else {}
    style_descriptor = coalesce_text(style_descriptor) or "source-series faithful TV episode frame"
    series_language = normalize_language_code(series_language)
    prompt_language = series_language or "auto-detected source language"

    positive_prompt = (
        f"{style_descriptor}, match original episode lighting, lensing, set design and color palette, "
        f"{camera_plan['shot_type']}, {camera_plan['composition']}, "
        f"{camera_plan['camera_move']}, {camera_plan['lens_hint']}, "
        f"characters {', '.join(scene_characters[:2])}, keyword {keyword}, beat {beat}, "
        f"{camera_plan['pose_hint']}, keep identity, wardrobe and environment continuity, "
        f"dialogue language {prompt_language}, 16:9 frame, clean readable TV staging"
    )
    if style_positive:
        positive_prompt += f", style cues: {style_positive}"
    if continuity_fragments:
        positive_prompt += f", continuity notes: {'; '.join(continuity_fragments)}"
    if relationship_fragments:
        positive_prompt += f", relationship dynamics: {'; '.join(relationship_fragments)}"
    if behavior_constraints:
        positive_prompt += f", behavior constraints: {'; '.join(behavior_constraints[:3])}"
    if dialogue_style_constraints:
        positive_prompt += f", dialogue style: {'; '.join(dialogue_style_constraints[:2])}"
    if style_guidance.get("camera"):
        positive_prompt += f", series camera preference: {style_guidance['camera']}"
    if style_guidance.get("angle"):
        positive_prompt += f", preferred angle: {style_guidance['angle']}"
    negative_prompt = (
        "no collage, no split panel, no duplicate characters, no extra fingers, no warped face, "
        "no mismatched outfit, no generic cartoon style, no blue placeholder frame, no filtered still-frame slideshow, "
        "no GUI overlay, no text box, no watermark"
    )
    if style_negative:
        negative_prompt += f", avoid {style_negative}"
    batch_prompt_line = (
        f"{beat} | {', '.join(scene_characters[:2])} | {keyword} | {camera_plan['shot_type']} | "
        f"{camera_plan['composition']} | match original episode style | continuity from {previous_scene_id or 'none'}"
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
        "relationship_context": relationship_context,
        "scene_purpose": coalesce_text(behavior_fields.get("scene_purpose", "")),
        "conflict": coalesce_text(behavior_fields.get("conflict", "")),
        "character_intents": behavior_fields.get("character_intents", {}) if isinstance(behavior_fields.get("character_intents"), dict) else {},
        "behavior_constraints": behavior_constraints,
        "dialogue_style_constraints": dialogue_style_constraints,
        "comedy_pattern": coalesce_text(behavior_fields.get("comedy_pattern", "")),
        "emotional_arc": coalesce_text(behavior_fields.get("emotional_arc", "")),
        "callback_targets": behavior_fields.get("callback_targets", []) if isinstance(behavior_fields.get("callback_targets"), list) else [],
        "quality_targets": {
            "quality_mode": quality_mode,
            "series_language": series_language or "auto",
            "source_series_style_locked": True,
            "minimum_reference_slots": max(1, min(3, len(reference_slots))),
            "prefer_previous_scene_reference": bool(previous_scene_id),
            "prefer_backend_character_models": bool(scene_characters),
            "style_guidance_available": bool(style_profile.get("positive") or style_guidance),
            "continuity_character_count": len(character_continuity),
            "relationship_context_count": len(relationship_context),
            "behavior_model_available": bool(behavior_fields),
            "behavior_constraints_count": len(behavior_constraints),
            "dialogue_style_constraints_count": len(dialogue_style_constraints),
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
    keywords = clean_generation_keywords(model.get("keywords", []), limit=20) or ["idee", "chaos", "plan", "showdown"]
    if len(keywords) > 1:
        keyword_offset = episode_index % len(keywords)
        keywords = keywords[keyword_offset:] + keywords[:keyword_offset]
    target_runtime_seconds = select_target_runtime_seconds(model, cfg)
    scene_count, target_lines_per_scene = planning_targets(model, cfg, target_runtime_seconds)
    continuity_memory = load_character_continuity_memory(PROJECT_ROOT)
    style_constraints = derive_prompt_constraints_from_bible(PROJECT_ROOT, {})
    quality_mode = coalesce_text(generation_cfg.get("quality_mode", "")) or "series_consistency"
    series_language = dominant_language(model.get("language_counts", {}), model.get("dominant_language", "") or generation_cfg.get("language", ""))
    package_language = series_language or "auto"
    style_descriptor = (
        coalesce_text(generation_cfg.get("style_descriptor", ""))
        or coalesce_text(model.get("style_descriptor", ""))
        or "source-series faithful TV episode frame"
    )
    behavior_model = active_behavior_model(model, cfg)
    if behavior_model:
        model = {**model, "behavior_model": behavior_model}
    relationship_payload = model.get("character_relationships") or load_character_relationships(cfg)
    if not model.get("character_groups") and isinstance(relationship_payload, dict):
        model = {
            **model,
            "character_groups": relationship_payload.get("groups", {}),
            "series_inputs": relationship_payload.get("series_inputs", {}),
        }
    focus_characters = select_focus_characters(model, rng, cfg=cfg)
    active_character_groups = character_groups_for_names(relationship_payload, focus_characters)
    relationship_context = relationships_for_characters(relationship_payload, focus_characters, include_group_members=True)
    background_characters = select_background_characters(model, count=1)
    beats = ["Cold Open", "Plan", "Komplikation", "Verwechslung", "Wendepunkt", "Auflösung"]
    if len(beats) > 1:
        beat_offset = episode_index % len(beats)
        beats = beats[beat_offset:] + beats[:beat_offset]
    episode_title = build_episode_title(focus_characters, keywords, rng, episode_index, model=model, language=series_language)
    episode_label = format_episode_number(episode_index, series_language)
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
        "## Character Relationship Context",
        "",
        f"- Active groups: {', '.join(group.get('label', group.get('id', '')) for group in active_character_groups) or 'none configured'}",
        *[
            f"- {fragment}"
            for fragment in relationship_prompt_fragments({"relationships": relationship_context}, focus_characters, limit=8)
        ],
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
    scene_runtime_floors: list[float] = []
    scene_beats: list[str] = []
    for scene_index in range(scene_count):
        beat = beats[scene_index % len(beats)]
        keyword = keywords[scene_index % len(keywords)]
        scene_characters = focus_characters[: max(2, min(len(focus_characters), 2 + (scene_index % 2)))]
        if background_characters and scene_index % 3 == 1:
            background_name = background_characters[0]
            if background_name not in scene_characters:
                scene_characters.append(background_name)
        scene_relationship_context = relationships_for_characters(
            {"relationships": relationship_context},
            scene_characters,
            include_group_members=False,
        )
        summary = summary_line(scene_characters, keyword, beat, series_language)
        dialogue, dialogue_sources = build_dialogue(
            beat,
            scene_characters,
            model,
            rng,
            cfg,
            keyword,
            target_length=target_lines_per_scene,
            language=series_language,
        )
        behavior_fields = build_behavior_scene_fields(
            beat=beat,
            keyword=keyword,
            scene_characters=scene_characters,
            scene_index=scene_index,
            scene_count=scene_count,
            behavior_model=behavior_model,
            relationship_context=scene_relationship_context,
            dialogue=dialogue,
            language=series_language,
        )
        dialogue_voice_metadata = build_dialogue_voice_metadata(dialogue, dialogue_sources, behavior_model, model)
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
            series_language=series_language,
            style_descriptor=style_descriptor,
            relationship_context=scene_relationship_context,
            behavior_fields=behavior_fields,
        )
        scenes.append(
            {
                "scene_id": scene_id,
                "title": scene_title(beat, keyword, series_language),
                "beat": beat,
                "language": package_language,
                "summary": summary,
                **behavior_fields,
                "characters": scene_characters,
                "relationship_context": scene_relationship_context,
                "location": f"Set {((scene_index % 3) + 1)}",
                "mood": ("energetisch" if scene_index < scene_count - 1 else "auflösend") if language_family(series_language) == "de" else ("energetic" if scene_index < scene_count - 1 else "resolved"),
                "dialogue_lines": dialogue,
                "dialogue_sources": dialogue_sources,
                "dialogue_voice_metadata": dialogue_voice_metadata,
                "prompt": (
                    (
                        f"{beat} mit {', '.join(scene_characters)}. Fokus auf {keyword}, schnelle Pointen "
                        f"und klaren Konfliktbogen."
                    )
                    if language_family(series_language) == "de"
                    else f"{localized_beat_label(beat, series_language)} with {', '.join(scene_characters)}. Focus on {keyword}, clear conflict and source-language dialogue."
                ),
                "generation_plan": generation_plan,
                "estimated_runtime_seconds": round(len(dialogue) * float(model.get("average_segment_duration_seconds", 2.7) or 2.7), 2),
            }
        )
        scene_runtime_floors.append(
            max(
                len(dialogue) * float(model.get("average_segment_duration_seconds", 2.7) or 2.7),
                float(generation_cfg.get("target_scene_duration_seconds", 42.0) or 42.0) * 0.35,
            )
        )
        scene_beats.append(beat)
        markdown_lines.extend(
            [
                f"### {scene_id} - {scene_title(beat, keyword, series_language)}",
                "",
                summary,
                "",
                f"Shot Plan: {generation_plan['camera_plan']['shot_type']} | {generation_plan['camera_plan']['composition']} | {generation_plan['camera_plan']['camera_move']}",
                f"Continuity: previous scene = {generation_plan['continuity']['previous_scene_id'] or 'none'}",
                f"Relationships: {'; '.join(relationship_prompt_fragments({'relationships': scene_relationship_context}, scene_characters, limit=4)) or 'none configured'}",
                "Reference Slots:",
                *[
                    f"- {slot['slot']}: {slot.get('type', '')} {slot.get('name', slot.get('scene_id', ''))}".strip()
                    for slot in generation_plan["reference_slots"]
                ],
                "",
                f"{localized_dialogue_label(series_language)}:",
                "",
                *[f"- {line}" for line in dialogue],
                "",
            ]
        )
        previous_scene_id = scene_id

    allocated_scene_runtimes = allocate_scene_runtime_seconds(target_runtime_seconds, scene_runtime_floors, scene_beats)
    for scene, runtime_seconds in zip(scenes, allocated_scene_runtimes):
        scene["estimated_runtime_seconds"] = round(float(runtime_seconds or 0.0), 2)

    return {
        "episode_title": episode_title,
        "episode_label": episode_label,
        "display_title": display_title,
        "generation_mode": "synthetic_preview",
        "quality_mode": quality_mode,
        "series_language": package_language,
        "style_descriptor": style_descriptor,
        "storyboard_plan_mode": "multi_reference_storyboard",
        "target_runtime_seconds": target_runtime_seconds,
        "target_scene_count": scene_count,
        "target_dialogue_lines_per_scene": target_lines_per_scene,
        "scenes": scenes,
        "focus_characters": focus_characters,
        "active_character_groups": active_character_groups,
        "character_relationship_context": relationship_context,
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


