#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

from pipeline_common import (
    add_shared_worker_arguments,
    canonical_person_name,
    cosine_similarity,
    current_os,
    display_person_name,
    distributed_item_lease,
    distributed_step_runtime_root,
    error,
    has_manual_person_name,
    headline,
    info,
    is_background_person_name,
    is_interactive_session,
    load_config,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    open_review_item_count,
    ok,
    read_json,
    rerun_in_runtime,
    resolve_project_path,
    shared_worker_id_for_args,
    shared_workers_enabled_for_args,
    write_json,
)

IGNORED_FACE_NAMES = {
    "noface",
    "no face",
    "ignore",
    "ignored",
    "falsepositive",
    "false positive",
    "kein gesicht",
}
EXAMPLE_FACE_HINTS = [
    "Babe Carano",
    "Mr. Sammich",
    "Teague/Busboy",
    "Hudson",
    "Triple G",
    "noface = ignore",
    "statist = minor character",
]
REVIEW_SKIP_TOKEN = "__skip__"
REVIEW_QUIT_TOKEN = "__quit__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review, assign, rename, or ignore face clusters")
    parser.add_argument("--list-faces", action="store_true", help="Show already named characters together with preview paths.")
    parser.add_argument("--created", action="store_true", help="Show only the names of already created recognized characters.")
    parser.add_argument("--show-queue", action="store_true", help="Show the open review_queue.json instead of the face review.")
    parser.add_argument("--assign-face", help="Face cluster ID such as face_001.")
    parser.add_argument("--name", help="Name for --assign-face, for example 'Babe Carano'.")
    parser.add_argument("--priority", action="store_true", help="Mark --assign-face or --rename-face as a prioritized main character.")
    parser.add_argument("--set-priority", help="Set main-character priority for an already named face cluster by ID or name.")
    parser.add_argument("--clear-priority", help="Clear main-character priority for an already named face cluster by ID or name.")
    parser.add_argument("--rename-face", help="Rename an already named face cluster by ID or current name.")
    parser.add_argument("--rename-to", help="New name for --rename-face.")
    parser.add_argument("--ignore", action="store_true", help="Set --assign-face to 'noface' and ignore the cluster from now on.")
    parser.add_argument(
        "--review-faces",
        action="store_true",
        help="Interactive naming for automatically named face clusters.",
    )
    parser.add_argument(
        "--include-named",
        action="store_true",
        help="Include already named clusters in --review-faces as well.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of clusters to process per start. Default: 20")
    parser.add_argument("--all", action="store_true", help="Process every currently open face cluster.")
    parser.add_argument("--open-previews", action="store_true", help="Open the contact sheet during interactive review.")
    parser.add_argument(
        "--auto-mark-statists",
        action="store_true",
        help="Mark safe low-activity unknown face clusters as 'statist' and stop.",
    )
    parser.add_argument(
        "--no-auto-mark-statists",
        action="store_true",
        help="Disable the automatic low-activity 'statist' cleanup for this run.",
    )
    parser.add_argument("--statist-max-scenes", type=int, default=None, help="Maximum scenes for automatic 'statist' candidates.")
    parser.add_argument("--statist-max-detections", type=int, default=None, help="Maximum detections for automatic 'statist' candidates.")
    parser.add_argument("--statist-max-samples", type=int, default=None, help="Maximum preview samples for automatic 'statist' candidates.")
    add_shared_worker_arguments(parser)
    return parser.parse_args()


def looks_auto_named(name: str) -> bool:
    return not has_manual_person_name(name)


def normalize_alias_name(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


def is_ignored_face_name(name: str) -> bool:
    return normalize_alias_name(name) in IGNORED_FACE_NAMES


def is_ignored_face_payload(payload: dict | None) -> bool:
    if not payload:
        return False
    return bool(payload.get("ignored")) or is_ignored_face_name(str(payload.get("name", "")))


def remove_cluster_aliases(char_map: dict, cluster_id: str) -> None:
    aliases = char_map.setdefault("aliases", {})
    stale_aliases = [alias for alias, alias_cluster in aliases.items() if alias_cluster == cluster_id]
    for alias in stale_aliases:
        aliases.pop(alias, None)


def resolve_face_reference(char_map: dict, reference: str) -> str:
    clusters = char_map.get("clusters", {})
    if reference in clusters:
        return reference

    normalized_reference = normalize_alias_name(reference)
    if not normalized_reference:
        raise ValueError("Leere Face-Referenz ist nicht erlaubt.")

    alias_cluster = char_map.get("aliases", {}).get(normalized_reference)
    if alias_cluster in clusters:
        return alias_cluster

    matches = []
    for cluster_id, payload in clusters.items():
        name = normalize_alias_name(str(payload.get("name", "")))
        aliases = [normalize_alias_name(alias) for alias in payload.get("aliases", [])]
        if normalized_reference == name or normalized_reference in aliases:
            matches.append(cluster_id)

    if not matches:
        raise FileNotFoundError(f"Face cluster or name not found: {reference}")
    if len(matches) > 1:
        raise ValueError(f"Face reference is ambiguous: {reference} -> {', '.join(sorted(matches))}")
    return matches[0]


def identity_has_priority(char_map: dict, identity_name: str) -> bool:
    final_name = canonical_person_name(identity_name)
    if not final_name or is_background_person_name(final_name) or is_ignored_face_name(final_name):
        return False
    identities = char_map.get("identities", {}) or {}
    identity_payload = identities.get(final_name)
    if identity_payload is not None:
        return bool(identity_payload.get("priority", False))
    for _cluster_id, payload in identity_clusters(char_map, final_name):
        if bool(payload.get("priority", False)):
            return True
    return False


def known_identity_button_options(char_map: dict, limit: int = 16) -> list[tuple[str, bool, int]]:
    options: list[tuple[str, bool, int]] = []
    identities = char_map.get("identities", {}) or {}
    if identities:
        for identity_name, payload in identities.items():
            final_name = canonical_person_name(identity_name)
            if not final_name or is_background_person_name(final_name) or is_ignored_face_name(final_name):
                continue
            if not has_manual_person_name(final_name):
                continue
            options.append(
                (
                    final_name,
                    bool(payload.get("priority", False)),
                    int(payload.get("cluster_count", 0) or 0),
                )
            )
    else:
        seen: set[str] = set()
        for cluster_id, payload in char_map.get("clusters", {}).items():
            if is_ignored_face_payload(payload):
                continue
            final_name = canonical_person_name(str(payload.get("name", cluster_id)))
            if not final_name or final_name in seen:
                continue
            if is_background_person_name(final_name) or not has_manual_person_name(final_name):
                continue
            seen.add(final_name)
            options.append((final_name, bool(payload.get("priority", False)), identity_cluster_count(char_map, final_name)))
    options.sort(key=lambda item: (0 if item[1] else 1, -int(item[2]), item[0]))
    return options[: max(0, limit)]


def assign_character_name(char_map: dict, cluster_id: str, assigned_name: str, priority: bool | None = None) -> dict:
    payload = char_map.setdefault("clusters", {}).setdefault(cluster_id, {})
    remove_cluster_aliases(char_map, cluster_id)

    final_name = canonical_person_name((assigned_name or cluster_id).strip() or cluster_id) or cluster_id
    ignored = is_ignored_face_name(final_name)
    if ignored:
        final_name = "noface"
    background_role = is_background_person_name(final_name)
    inherited_priority = False
    if not ignored and not background_role and has_manual_person_name(final_name):
        inherited_priority = identity_has_priority(char_map, final_name)
    effective_priority = (
        False
        if ignored or background_role
        else inherited_priority or bool(priority)
        if priority is not None
        else bool(payload.get("priority", False)) or inherited_priority
    )

    payload["name"] = final_name
    payload["ignored"] = ignored
    payload["background_role"] = background_role
    payload["priority"] = effective_priority
    payload["auto_named"] = False

    normalized_alias = normalize_alias_name(final_name)
    if ignored or background_role or not normalized_alias:
        payload["aliases"] = []
    else:
        payload["aliases"] = [normalized_alias]
        char_map.setdefault("aliases", {})[normalized_alias] = cluster_id
    return payload


def face_display_name(payload: dict | None, cluster_id: str) -> str:
    if payload is None:
        return cluster_id
    return display_person_name(str(payload.get("name", "")), cluster_id)


def preview_files(payload: dict) -> list[Path]:
    preview_dir = Path(payload.get("preview_dir", ""))
    if not preview_dir.is_dir():
        return []
    files = [
        path
        for path in sorted(preview_dir.glob("*.jpg"))
        if "_montage" not in path.name
    ]
    return files[:6]


def preview_pairs(payload: dict) -> list[tuple[Path | None, Path | None]]:
    preview_dir = Path(payload.get("preview_dir", ""))
    if not preview_dir.is_dir():
        return []

    contexts = sorted(preview_dir.glob("*_context.jpg"))
    pairs: list[tuple[Path | None, Path | None]] = []
    for context_path in contexts:
        crop_name = context_path.name.replace("_context.jpg", "_crop.jpg")
        crop_path = context_path.with_name(crop_name)
        pairs.append((context_path, crop_path if crop_path.exists() else None))

    if pairs:
        return pairs[:3]

    files = preview_files(payload)
    return [(path, None) for path in files[:3]]


def create_contact_sheet(image_paths: list[Path], output_path: Path, title: str | None = None) -> Path | None:
    if not image_paths:
        return None
    try:
        from PIL import Image, ImageDraw, ImageOps
    except Exception:
        return None

    cards = []
    header_height = 40 if title else 0
    for image_path in image_paths[:6]:
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.contain(image, (260, 260))
        canvas = Image.new("RGB", (280, 300), "white")
        canvas.paste(image, ((280 - image.width) // 2, 10))
        ImageDraw.Draw(canvas).text((10, 270), image_path.name[:28], fill="black")
        cards.append(canvas)

    rows = max(1, (len(cards) + 1) // 2)
    sheet = Image.new("RGB", (560, header_height + (300 * rows)), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.rectangle((0, 0, 560, header_height), fill="#1f2937")
        draw.text((12, 10), title[:56], fill="white")
    for index, card in enumerate(cards):
        sheet.paste(card, ((index % 2) * 280, header_height + ((index // 2) * 300)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def create_face_review_sheet(cluster_id: str, payload: dict) -> Path | None:
    preview_dir = Path(payload.get("preview_dir", ""))
    if not preview_dir:
        return None
    pairs = preview_pairs(payload)
    if not pairs:
        files = preview_files(payload)
        if not files:
            return None
        return create_contact_sheet(files, preview_dir / f"{cluster_id}_montage.jpg", title=f"{cluster_id} | Preview")

    try:
        from PIL import Image, ImageDraw, ImageOps
    except Exception:
        fallback_files = [path for pair in pairs for path in pair if path is not None]
    return create_contact_sheet(fallback_files, preview_dir / f"{cluster_id}_montage.jpg", title=f"{cluster_id} | Preview")

    row_height = 300
    row_width = 560
    header_height = 46
    rows = max(1, len(pairs))
    sheet = Image.new("RGB", (row_width, header_height + (rows * row_height)), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, row_width, header_height), fill="#1f2937")
    draw.text((12, 12), f"{cluster_id} | links Szene, rechts Ausschnitt", fill="white")

    for row_index, (context_path, crop_path) in enumerate(pairs):
        top = header_height + (row_index * row_height)
        draw.rectangle((0, top, row_width, top + row_height), fill="#f4f4f4")
        draw.text((14, top + 8), "Szene", fill="black")
        draw.text((294, top + 8), "Ausschnitt", fill="black")

        for column_index, image_path in enumerate((context_path, crop_path)):
            if image_path is None or not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            image = ImageOps.contain(image, (250, 250))
            x_offset = 14 + (column_index * 280) + ((250 - image.width) // 2)
            y_offset = top + 28 + ((250 - image.height) // 2)
            sheet.paste(image, (x_offset, y_offset))

    output_path = preview_dir / f"{cluster_id}_montage.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def poll_terminal_line(buffer: list[str]) -> str | None:
    if current_os() != "windows":
        return None
    try:
        import msvcrt
    except Exception:
        return None

    while msvcrt.kbhit():
        char = msvcrt.getwch()
        if char in ("\r", "\n"):
            print()
            line = "".join(buffer).strip()
            buffer.clear()
            return line
        if char == "\003":
            raise KeyboardInterrupt
        if char == "\b":
            if buffer:
                buffer.pop()
            continue
        if char in ("\x00", "\xe0"):
            try:
                msvcrt.getwch()
            except Exception:
                pass
            continue
        buffer.append(char)
        try:
            print(char, end="", flush=True)
        except Exception:
            pass
    return None


def parse_assignment_input(answer: str) -> tuple[str, bool | None]:
    raw = (answer or "").strip()
    if not raw:
        return "", None
    if raw.startswith("!"):
        return raw[1:].strip(), True
    return raw, None


def prompt_priority_for_name(char_map: dict, name: str) -> bool:
    final_name = canonical_person_name(name)
    if not final_name or is_ignored_face_name(final_name) or is_background_person_name(final_name) or not has_manual_person_name(final_name):
        return False
    if identity_has_priority(char_map, final_name):
        return True
    print("Prioritize as main character? [y/N]")
    try:
        decision = input("> ").strip().lower()
    except EOFError:
        return False
    return decision in {"j", "ja", "y", "yes", "1"}


def show_preview_assignment_window(
    image_path: Path,
    title: str,
    status_text: str = "",
    initial_priority: bool = False,
    quick_assignments: list[tuple[str, bool, int]] | None = None,
) -> dict[str, object] | None:
    if not image_path.exists():
        return None
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except Exception:
        return None

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((1200, 900))

    result: dict[str, object] = {"value": None, "priority": bool(initial_priority)}
    terminal_buffer: list[str] = []

    window = tk.Tk()
    window.title(title)
    window.attributes("-topmost", True)
    window.configure(bg="#1f2937")

    def finish(value: str | None, priority: bool | None = None) -> None:
        if result["value"] is not None:
            return
        result["value"] = None if value is None else value.strip()
        if priority is not None:
            result["priority"] = bool(priority)
        try:
            window.destroy()
        except Exception:
            pass

    window.protocol("WM_DELETE_WINDOW", lambda: finish(None))
    window.bind("<Escape>", lambda _event: finish(None))

    photo = ImageTk.PhotoImage(image)
    label = tk.Label(window, image=photo, bg="#1f2937")
    label.image = photo
    label.pack(padx=12, pady=(12, 8))

    if status_text:
        status_label = tk.Label(
            window,
            text=status_text,
            fg="#d1fae5",
            bg="#1f2937",
            justify="left",
            font=("Segoe UI", 10, "bold"),
        )
        status_label.pack(padx=12, pady=(0, 8), anchor="w")

    entry_var = tk.StringVar()
    entry = tk.Entry(window, textvariable=entry_var, width=42, font=("Segoe UI", 11))
    entry.pack(padx=12, pady=(0, 8), fill="x")
    entry.focus_set()
    priority_var = tk.BooleanVar(value=bool(initial_priority))
    entry.bind("<Return>", lambda _event: finish(entry_var.get(), priority_var.get()))

    hint = tk.Label(
        window,
        text="Type a name here, click a quick-assign button, or enter it in the terminal. Enter confirms immediately. 'noface' ignores the cluster, 'statist' marks a minor character, and '!Name' in the terminal marks a main character right away.",
        fg="white",
        bg="#1f2937",
        wraplength=900,
        justify="left",
    )
    hint.pack(padx=12, pady=(0, 8))

    if quick_assignments:
        quick_label = tk.Label(
            window,
            text="Known Character Quick Assign",
            fg="white",
            bg="#1f2937",
            justify="left",
        )
        quick_label.pack(padx=12, pady=(0, 4), anchor="w")
        quick_frame = tk.Frame(window, bg="#1f2937")
        quick_frame.pack(padx=12, pady=(0, 8), fill="x")
        columns = 4
        for index, (name, is_priority, cluster_count) in enumerate(quick_assignments):
            label = name if cluster_count <= 1 else f"{name} ({cluster_count})"
            if is_priority:
                label = f"{label} *"
            button = tk.Button(
                quick_frame,
                text=label,
                command=lambda selected=name, selected_priority=is_priority: finish(selected, selected_priority),
            )
            button.grid(row=index // columns, column=index % columns, padx=4, pady=4, sticky="ew")
        for column_index in range(columns):
            quick_frame.grid_columnconfigure(column_index, weight=1)

    priority_check = tk.Checkbutton(
        window,
        text="Prioritize as main character",
        variable=priority_var,
        fg="white",
        bg="#1f2937",
        selectcolor="#111827",
        activebackground="#1f2937",
        activeforeground="white",
    )
    priority_check.pack(padx=12, pady=(0, 8), anchor="w")

    button_row = tk.Frame(window, bg="#1f2937")
    button_row.pack(padx=12, pady=(0, 12), fill="x")
    tk.Button(button_row, text="Apply", command=lambda: finish(entry_var.get(), priority_var.get())).pack(side="left", padx=(0, 8))
    tk.Button(button_row, text="Minor", command=lambda: finish("statist", False)).pack(side="left", padx=(0, 8))
    tk.Button(button_row, text="NoFace", command=lambda: finish("noface", False)).pack(side="left", padx=(0, 8))
    tk.Button(button_row, text="Terminal", command=lambda: finish(None)).pack(side="left", padx=(0, 8))
    tk.Button(button_row, text="Quit", command=lambda: finish(REVIEW_QUIT_TOKEN, False)).pack(side="right")

    def poll_terminal() -> None:
        if result["value"] is not None:
            return
        line = poll_terminal_line(terminal_buffer)
        if line is not None:
            parsed_name, parsed_priority = parse_assignment_input(line)
            finish(parsed_name, parsed_priority)
            return
        try:
            window.after(80, poll_terminal)
        except Exception:
            pass

    if is_interactive_session() and current_os() == "windows":
        window.after(80, poll_terminal)

    window.mainloop()
    return result


def prompt_terminal_assignment(char_map: dict) -> tuple[str, bool | None]:
    print(f"Example names: {' | '.join(EXAMPLE_FACE_HINTS)}")
    quick_names = [name for name, _priority, _count in known_identity_button_options(char_map, limit=8)]
    if quick_names:
        print(f"Known characters: {' | '.join(quick_names)}")
    print("Enter a name. 'noface' ignores the match, 'statist' saves a minor character, '!Name' prioritizes as a main character, and 'q' quits the review.")
    raw = input("> ").strip()
    name, explicit_priority = parse_assignment_input(raw)
    if not name:
        return "", explicit_priority
    if explicit_priority is not None:
        return name, explicit_priority
    return name, prompt_priority_for_name(char_map, name)


def cluster_sort_key(item: tuple[str, dict]) -> tuple[int, int, int, str]:
    cluster_id, payload = item
    ignored_rank = 1 if is_ignored_face_payload(payload) else 0
    named_rank = 1 if not looks_auto_named(str(payload.get("name", ""))) else 0
    activity_rank = -face_activity_score(payload)
    scene_rank = -int(payload.get("scene_count", 0))
    detection_rank = -int(payload.get("detection_count", 0))
    samples_rank = -int(payload.get("samples", 0))
    return (ignored_rank, named_rank, activity_rank, scene_rank, detection_rank, samples_rank, cluster_id)


def face_activity_score(payload: dict) -> float:
    return (
        (float(payload.get("scene_count", 0)) * 4.0)
        + (float(payload.get("samples", 0)) * 2.0)
        + (float(payload.get("detection_count", 0)) * 0.2)
        + (10.0 if bool(payload.get("priority", False)) else 0.0)
    )


def suggested_face_role(payload: dict) -> str:
    if bool(payload.get("priority", False)):
        return "hauptfigur-kandidat"
    if is_ignored_face_payload(payload):
        return "ignorieren"
    if is_background_person_name(str(payload.get("name", ""))):
        return "statist"

    score = face_activity_score(payload)
    scene_count = int(payload.get("scene_count", 0) or 0)
    detection_count = int(payload.get("detection_count", 0) or 0)
    samples = int(payload.get("samples", 0) or 0)

    if score >= 42.0 or scene_count >= 10 or detection_count >= 60:
        return "hauptfigur-kandidat"
    if score <= 10.0 and scene_count <= 2 and detection_count <= 12 and samples <= 3:
        return "statist-kandidat"
    return "nebenfigur-kandidat"


def suggested_face_action_hint(payload: dict) -> str:
    role = suggested_face_role(payload)
    if role == "hauptfigur-kandidat":
        return "Hint: use a real name and optionally mark as a main character"
    if role == "statist-kandidat":
        return "Hint: consider 'statist' or 'noface'"
    if role == "statist":
        return "Hint: already saved as a minor character"
    if role == "ignorieren":
        return "Hint: already ignored"
    return "Hint: use a real name or consider 'statist'"


def auto_statist_thresholds(cfg: dict, args: argparse.Namespace | None = None) -> dict[str, int]:
    face_cfg = cfg.get("character_detection", {}) if isinstance(cfg.get("character_detection"), dict) else {}

    def threshold(name: str, fallback: int) -> int:
        cli_value = getattr(args, name, None) if args is not None else None
        if cli_value is not None:
            return max(0, int(cli_value))
        config_key = f"auto_statist_{name[8:]}"
        return max(0, int(face_cfg.get(config_key, fallback) or fallback))

    return {
        "max_scenes": threshold("statist_max_scenes", 2),
        "max_detections": threshold("statist_max_detections", 12),
        "max_samples": threshold("statist_max_samples", 3),
    }


def is_auto_statist_candidate(payload: dict, thresholds: dict[str, int]) -> bool:
    if is_ignored_face_payload(payload) or bool(payload.get("priority", False)):
        return False
    if not looks_auto_named(str(payload.get("name", ""))):
        return False
    if suggested_face_role(payload) != "statist-kandidat":
        return False
    return (
        int(payload.get("scene_count", 0) or 0) <= int(thresholds.get("max_scenes", 2))
        and int(payload.get("detection_count", 0) or 0) <= int(thresholds.get("max_detections", 12))
        and int(payload.get("samples", 0) or 0) <= int(thresholds.get("max_samples", 3))
    )


def plan_auto_statist_candidates(char_map: dict, thresholds: dict[str, int], limit: int = 0) -> list[tuple[str, dict]]:
    candidates = [
        (cluster_id, payload)
        for cluster_id, payload in char_map.get("clusters", {}).items()
        if is_auto_statist_candidate(payload, thresholds)
    ]
    candidates.sort(
        key=lambda item: (
            int(item[1].get("scene_count", 0) or 0),
            int(item[1].get("detection_count", 0) or 0),
            int(item[1].get("samples", 0) or 0),
            item[0],
        )
    )
    if limit > 0:
        return candidates[:limit]
    return candidates


def mark_auto_statist_candidates(char_map: dict, thresholds: dict[str, int], limit: int = 0) -> list[dict[str, object]]:
    marked: list[dict[str, object]] = []
    for cluster_id, payload in plan_auto_statist_candidates(char_map, thresholds, limit):
        marked.append(
            {
                "cluster_id": cluster_id,
                "previous_name": str(payload.get("name", cluster_id)),
                "scene_count": int(payload.get("scene_count", 0) or 0),
                "detection_count": int(payload.get("detection_count", 0) or 0),
                "samples": int(payload.get("samples", 0) or 0),
            }
        )
        assign_character_name(char_map, cluster_id, "statist", priority=False)
    return marked


def hydrate_face_clusters_from_previews(cfg: dict, char_map: dict) -> int:
    previews_root_value = cfg.get("paths", {}).get("character_previews", "characters/previews")
    previews_root = resolve_project_path(previews_root_value)
    if not previews_root.exists():
        return 0

    added = 0
    allow_new_clusters = not bool(char_map.get("clusters"))
    for preview_dir in sorted(previews_root.glob("face_*")):
        if not preview_dir.is_dir():
            continue
        cluster_id = preview_dir.name
        payload = char_map.setdefault("clusters", {}).setdefault(cluster_id, {})
        if payload:
            payload.setdefault("preview_dir", str(preview_dir))
            payload.setdefault("samples", max(1, len(list(preview_dir.glob("*_crop.jpg")))))
            continue
        if not allow_new_clusters:
            char_map["clusters"].pop(cluster_id, None)
            continue
        payload["name"] = cluster_id
        payload["preview_dir"] = str(preview_dir)
        payload["samples"] = max(1, len(list(preview_dir.glob("*_crop.jpg"))))
        payload["auto_named"] = True
        payload["ignored"] = False
        payload["aliases"] = []
        added += 1
    return added


def known_face_reference_clusters(char_map: dict) -> dict[str, dict]:
    references: dict[str, dict] = {}
    for cluster_id, payload in char_map.get("clusters", {}).items():
        if is_ignored_face_payload(payload):
            continue
        final_name = canonical_person_name(str(payload.get("name", cluster_id)))
        if looks_auto_named(final_name):
            continue
        if is_background_person_name(final_name):
            continue
        if not payload.get("embedding"):
            continue
        references[cluster_id] = payload
    return references


def normalize_embedding(vector: list[float]) -> list[float]:
    if not vector:
        return []
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if not norm:
        return []
    return [round(float(value) / norm, 6) for value in vector]


def primary_cluster_for_identity(clusters: list[tuple[str, dict]]) -> str:
    ranked = sorted(
        clusters,
        key=lambda item: (
            0 if bool(item[1].get("priority", False)) else 1,
            -int(item[1].get("scene_count", 0)),
            -int(item[1].get("detection_count", 0)),
            -int(item[1].get("samples", 0)),
            item[0],
        ),
    )
    return ranked[0][0]


def identity_clusters(char_map: dict, identity_name: str) -> list[tuple[str, dict]]:
    normalized_identity = canonical_person_name(identity_name)
    if not normalized_identity:
        return []
    clusters: list[tuple[str, dict]] = []
    for cluster_id, payload in char_map.get("clusters", {}).items():
        if is_ignored_face_payload(payload):
            continue
        payload_name = canonical_person_name(str(payload.get("name", cluster_id)))
        if payload_name != normalized_identity:
            continue
        clusters.append((cluster_id, payload))
    return clusters


def identity_cluster_count(char_map: dict, identity_name: str) -> int:
    return len(identity_clusters(char_map, identity_name))


def rebuild_character_map_identities(char_map: dict) -> None:
    aliases: dict[str, str] = {}
    identities: dict[str, dict] = {}
    grouped: dict[str, list[tuple[str, dict]]] = {}

    for cluster_id, payload in char_map.get("clusters", {}).items():
        payload.pop("identity_name", None)
        payload.pop("identity_primary_cluster", None)
        payload.pop("identity_cluster_ids", None)
        payload.pop("identity_cluster_count", None)
        if is_ignored_face_payload(payload):
            continue
        final_name = canonical_person_name(str(payload.get("name", cluster_id)))
        if not has_manual_person_name(final_name):
            continue
        grouped.setdefault(final_name, []).append((cluster_id, payload))

    for identity_name, clusters in grouped.items():
        primary_cluster = primary_cluster_for_identity(clusters)
        cluster_ids = [cluster_id for cluster_id, _payload in clusters]
        cluster_count = len(cluster_ids)
        priority = any(bool(payload.get("priority", False)) for _cluster_id, payload in clusters)
        background_role = is_background_person_name(identity_name)
        identities[identity_name] = {
            "name": identity_name,
            "primary_cluster": primary_cluster,
            "cluster_ids": cluster_ids,
            "cluster_count": cluster_count,
            "priority": False if background_role else priority,
            "background_role": background_role,
        }
        normalized_alias = normalize_alias_name(identity_name)
        if normalized_alias and not background_role:
            aliases[normalized_alias] = primary_cluster
        for cluster_id, payload in clusters:
            payload["identity_name"] = identity_name
            payload["identity_primary_cluster"] = primary_cluster
            payload["identity_cluster_ids"] = cluster_ids
            payload["identity_cluster_count"] = cluster_count

    char_map["aliases"] = aliases
    char_map["identities"] = identities


def face_cluster_quality_score(payload: dict) -> float:
    return (
        (float(payload.get("scene_count", 0)) * 3.0)
        + (float(payload.get("samples", 0)) * 2.0)
        + (float(payload.get("detection_count", 0)) * 0.2)
        + (20.0 if bool(payload.get("priority", False)) else 0.0)
    )


def identity_embedding(clusters: list[tuple[str, dict]]) -> list[float]:
    weighted: list[float] = []
    total_weight = 0
    for _cluster_id, payload in clusters:
        embedding = payload.get("embedding") or []
        if not embedding:
            continue
        weight = max(1.0, face_cluster_quality_score(payload))
        if not weighted:
            weighted = [0.0] * len(embedding)
        for index, value in enumerate(embedding):
            weighted[index] += float(value) * weight
        total_weight += weight
    if not total_weight:
        return []
    return normalize_embedding([value / total_weight for value in weighted])


def known_face_identity_strength(payload: dict) -> float:
    cluster_count = float(payload.get("cluster_count", 0) or 0)
    references = payload.get("references") or []
    average_quality = 0.0
    if references:
        average_quality = sum(float(reference.get("quality", 0.0) or 0.0) for reference in references) / len(references)
    priority_bonus = 1.0 if bool(payload.get("priority", False)) else 0.0
    return round((cluster_count * 0.8) + (average_quality * 0.06) + priority_bonus, 3)


def select_identity_reference_clusters(
    clusters: list[tuple[str, dict]],
    max_references: int,
    min_quality: float,
) -> list[tuple[str, dict]]:
    ranked = sorted(
        clusters,
        key=lambda item: (
            0 if bool(item[1].get("priority", False)) else 1,
            -face_cluster_quality_score(item[1]),
            item[0],
        ),
    )
    if not ranked:
        return []

    selected = [item for item in ranked if face_cluster_quality_score(item[1]) >= min_quality]
    if not selected:
        selected = ranked[:1]

    keep_count = min(max_references, max(1, min(len(ranked), max(len(selected), 2))))
    return ranked[:keep_count]


def known_face_reference_identities(char_map: dict, cfg: dict | None = None) -> dict[str, dict]:
    match_cfg = known_face_match_config(cfg or {})
    grouped: dict[str, list[tuple[str, dict]]] = {}
    for cluster_id, payload in known_face_reference_clusters(char_map).items():
        identity_name = canonical_person_name(str(payload.get("name", cluster_id)))
        grouped.setdefault(identity_name, []).append((cluster_id, payload))

    identities: dict[str, dict] = {}
    for identity_name, clusters in grouped.items():
        reference_clusters = select_identity_reference_clusters(
            clusters,
            int(match_cfg.get("reference_count", 8)),
            float(match_cfg.get("min_reference_quality", 5.0)),
        )
        embedding = identity_embedding(reference_clusters)
        if not embedding:
            continue
        identities[identity_name] = {
            "name": identity_name,
            "primary_cluster": primary_cluster_for_identity(clusters),
            "clusters": [cluster_id for cluster_id, _payload in clusters],
            "references": [
                {
                    "cluster_id": cluster_id,
                    "embedding": payload.get("embedding") or [],
                    "quality": round(face_cluster_quality_score(payload), 3),
                }
                for cluster_id, payload in reference_clusters
                if payload.get("embedding")
            ],
            "cluster_count": len(clusters),
            "embedding": embedding,
            "priority": identity_has_priority(char_map, identity_name),
        }
        identities[identity_name]["identity_strength"] = known_face_identity_strength(identities[identity_name])
    return identities


def unknown_face_candidates(char_map: dict) -> list[tuple[str, dict]]:
    candidates: list[tuple[str, dict]] = []
    for cluster_id, payload in sorted(char_map.get("clusters", {}).items(), key=cluster_sort_key):
        if is_ignored_face_payload(payload):
            continue
        if not looks_auto_named(str(payload.get("name", cluster_id))):
            continue
        if not payload.get("embedding"):
            continue
        candidates.append((cluster_id, payload))
    return candidates


def score_known_face_identity(embedding: list[float], payload: dict, cfg: dict | None = None) -> dict | None:
    match_cfg = known_face_match_config(cfg or {})
    references = payload.get("references") or []
    if not references:
        return None

    reference_scores = []
    for reference in references:
        reference_embedding = reference.get("embedding") or []
        if not reference_embedding:
            continue
        reference_scores.append(
            {
                "cluster_id": str(reference.get("cluster_id", "")),
                "score": cosine_similarity(embedding, reference_embedding),
                "quality": float(reference.get("quality", 0.0)),
            }
        )
    if not reference_scores:
        return None

    reference_scores.sort(key=lambda item: item["score"], reverse=True)
    top_k = max(1, int(match_cfg.get("top_k", 3)))
    top_scores = [float(item["score"]) for item in reference_scores[:top_k]]
    best_reference_score = top_scores[0]
    average_top_score = sum(top_scores) / len(top_scores)
    centroid_score = cosine_similarity(embedding, payload.get("embedding") or [])
    consensus_threshold = float(match_cfg.get("consensus_threshold", 0.66))
    consensus_count = sum(1 for item in reference_scores if float(item["score"]) >= consensus_threshold)
    consensus_bonus = 0.015 * min(consensus_count, max(2, top_k))
    composite_score = (best_reference_score * 0.50) + (average_top_score * 0.30) + (centroid_score * 0.20) + consensus_bonus

    return {
        "identity": str(payload.get("name", "")),
        "primary_cluster": str(payload.get("primary_cluster", "")),
        "identity_strength": float(payload.get("identity_strength", 0.0) or 0.0),
        "score": composite_score,
        "best_reference_score": best_reference_score,
        "average_top_score": average_top_score,
        "centroid_score": centroid_score,
        "consensus_count": consensus_count,
        "best_reference_cluster": str(reference_scores[0].get("cluster_id", "")),
        "reference_scores": reference_scores,
    }


def rank_known_face_matches(embedding: list[float], references: dict[str, dict], cfg: dict | None = None) -> list[dict]:
    ranked = []
    for identity_name, payload in references.items():
        scored = score_known_face_identity(embedding, payload, cfg)
        if not scored:
            continue
        if not scored.get("identity"):
            scored["identity"] = identity_name
        ranked.append(scored)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def best_known_face_match(embedding: list[float], references: dict[str, dict], cfg: dict | None = None) -> tuple[str | None, float, float]:
    ranked = rank_known_face_matches(embedding, references, cfg)
    if not ranked:
        return None, -1.0, -1.0
    best = ranked[0]
    second_score = ranked[1]["score"] if len(ranked) > 1 else -1.0
    return str(best.get("identity") or ""), float(best.get("score", -1.0)), float(second_score)


def known_face_match_config(cfg: dict) -> dict[str, float | int]:
    face_cfg = cfg.get("character_detection", {})
    threshold = float(
        face_cfg.get(
            "review_known_face_threshold",
            max(0.74, float(face_cfg.get("embedding_threshold", 0.80)) - 0.06),
        )
    )
    margin = float(face_cfg.get("review_known_face_margin", 0.08))
    return {
        "threshold": threshold,
        "margin": margin,
        "reference_count": int(face_cfg.get("review_known_face_reference_count", 8)),
        "top_k": int(face_cfg.get("review_known_face_top_k", 3)),
        "consensus_threshold": float(face_cfg.get("review_known_face_consensus_threshold", 0.66)),
        "min_consensus": int(face_cfg.get("review_known_face_min_consensus", 2)),
        "strong_match_threshold": float(face_cfg.get("review_known_face_strong_match_threshold", 0.84)),
        "min_reference_quality": float(face_cfg.get("review_known_face_min_reference_quality", 5.0)),
        "identity_relaxed_consensus_strength": float(face_cfg.get("review_known_face_identity_relaxed_consensus_strength", 5.0)),
        "identity_weak_strength": float(face_cfg.get("review_known_face_identity_weak_strength", 2.0)),
        "identity_threshold_bonus_max": float(face_cfg.get("review_known_face_identity_threshold_bonus_max", 0.03)),
        "identity_margin_bonus_max": float(face_cfg.get("review_known_face_identity_margin_bonus_max", 0.02)),
        "identity_weak_threshold_penalty": float(face_cfg.get("review_known_face_identity_weak_threshold_penalty", 0.02)),
        "identity_weak_margin_penalty": float(face_cfg.get("review_known_face_identity_weak_margin_penalty", 0.01)),
    }


def plan_known_face_matches(cfg: dict, char_map: dict) -> dict[str, dict]:
    identities = known_face_reference_identities(char_map, cfg)
    if not identities:
        return {}

    match_cfg = known_face_match_config(cfg)
    threshold = float(match_cfg.get("threshold", 0.72))
    margin = float(match_cfg.get("margin", 0.05))
    min_consensus = int(match_cfg.get("min_consensus", 2))
    strong_match_threshold = float(match_cfg.get("strong_match_threshold", 0.84))
    matches: dict[str, dict] = {}
    for cluster_id, payload in unknown_face_candidates(char_map):
        embedding = payload.get("embedding") or []
        ranked = rank_known_face_matches(embedding, identities, cfg)
        if not ranked:
            continue
        best_match = ranked[0]
        best_identity = str(best_match.get("identity", "")).strip()
        best_score = float(best_match.get("score", -1.0))
        second_score = float(ranked[1]["score"]) if len(ranked) > 1 else -1.0
        score_margin = best_score - max(second_score, -1.0)
        identity_strength = float(best_match.get("identity_strength", 0.0) or 0.0)
        relaxed_consensus_strength = float(match_cfg.get("identity_relaxed_consensus_strength", 5.0))
        weak_strength = float(match_cfg.get("identity_weak_strength", 2.0))
        threshold_bonus_max = float(match_cfg.get("identity_threshold_bonus_max", 0.03))
        margin_bonus_max = float(match_cfg.get("identity_margin_bonus_max", 0.02))
        weak_threshold_penalty = float(match_cfg.get("identity_weak_threshold_penalty", 0.02))
        weak_margin_penalty = float(match_cfg.get("identity_weak_margin_penalty", 0.01))

        if identity_strength >= relaxed_consensus_strength:
            effective_min_consensus = max(1, min_consensus - 1)
            strength_factor = min(1.0, max(0.0, (identity_strength - relaxed_consensus_strength) / max(1.0, relaxed_consensus_strength)))
            effective_threshold = threshold - (threshold_bonus_max * strength_factor)
            effective_margin = max(0.02, margin - (margin_bonus_max * strength_factor))
        elif identity_strength <= weak_strength:
            effective_min_consensus = min_consensus
            weakness_factor = min(1.0, max(0.0, (weak_strength - identity_strength) / max(1.0, weak_strength)))
            effective_threshold = threshold + (weak_threshold_penalty * weakness_factor)
            effective_margin = margin + (weak_margin_penalty * weakness_factor)
        else:
            effective_min_consensus = min_consensus
            effective_threshold = threshold
            effective_margin = margin

        has_consensus = int(best_match.get("consensus_count", 0)) >= effective_min_consensus
        strong_single_match = float(best_match.get("best_reference_score", -1.0)) >= strong_match_threshold
        if best_score < effective_threshold or score_margin < effective_margin or (not has_consensus and not strong_single_match):
            continue
        target_cluster = str(best_match.get("primary_cluster", "")).strip()
        if not target_cluster:
            continue
        matches[cluster_id] = {
            "target_cluster": target_cluster,
            "target_identity": best_identity,
            "score": round(best_score, 4),
            "margin": round(score_margin, 4),
            "best_reference_score": round(float(best_match.get("best_reference_score", -1.0)), 4),
            "average_top_score": round(float(best_match.get("average_top_score", -1.0)), 4),
            "centroid_score": round(float(best_match.get("centroid_score", -1.0)), 4),
            "consensus_count": int(best_match.get("consensus_count", 0)),
            "effective_min_consensus": effective_min_consensus,
            "effective_threshold": round(effective_threshold, 4),
            "effective_margin": round(effective_margin, 4),
            "identity_strength": round(identity_strength, 3),
            "best_reference_cluster": str(best_match.get("best_reference_cluster", "")),
        }
    return matches


def merge_cluster_embeddings(target_payload: dict, source_payload: dict) -> None:
    target_embedding = target_payload.get("embedding") or []
    source_embedding = source_payload.get("embedding") or []
    target_samples = max(1, int(target_payload.get("samples", 1)))
    source_samples = max(1, int(source_payload.get("samples", 1)))
    if target_embedding and source_embedding and len(target_embedding) == len(source_embedding):
        total_samples = target_samples + source_samples
        merged = [
            round(((target_embedding[index] * target_samples) + (source_embedding[index] * source_samples)) / total_samples, 6)
            for index in range(len(target_embedding))
        ]
        target_payload["embedding"] = merged
    elif source_embedding and not target_embedding:
        target_payload["embedding"] = source_embedding
    target_payload["samples"] = target_samples + source_samples


def merge_face_cluster_into_target(char_map: dict, source_cluster_id: str, target_cluster_id: str, match_meta: dict | None = None) -> None:
    if source_cluster_id == target_cluster_id:
        return

    clusters = char_map.setdefault("clusters", {})
    source_payload = clusters.get(source_cluster_id)
    target_payload = clusters.get(target_cluster_id)
    if source_payload is None or target_payload is None:
        return

    merge_cluster_embeddings(target_payload, source_payload)
    target_payload["scene_count"] = int(target_payload.get("scene_count", 0)) + int(source_payload.get("scene_count", 0))
    target_payload["detection_count"] = int(target_payload.get("detection_count", 0)) + int(source_payload.get("detection_count", 0))

    preview_dirs = []
    for payload in (target_payload, source_payload):
        preview_dir = str(payload.get("preview_dir", "")).strip()
        if preview_dir and preview_dir not in preview_dirs:
            preview_dirs.append(preview_dir)
        for extra_dir in payload.get("merged_preview_dirs", []) or []:
            extra_value = str(extra_dir).strip()
            if extra_value and extra_value not in preview_dirs:
                preview_dirs.append(extra_value)
    if preview_dirs:
        target_payload["preview_dir"] = preview_dirs[0]
        if len(preview_dirs) > 1:
            target_payload["merged_preview_dirs"] = preview_dirs

    merged_ids = [target_cluster_id]
    for payload in (target_payload, source_payload):
        for merged_id in payload.get("merged_cluster_ids", []) or []:
            merged_id_text = str(merged_id).strip()
            if merged_id_text and merged_id_text not in merged_ids:
                merged_ids.append(merged_id_text)
    if source_cluster_id not in merged_ids:
        merged_ids.append(source_cluster_id)
    target_payload["merged_cluster_ids"] = merged_ids

    if match_meta:
        history = list(target_payload.get("auto_merged_matches", []) or [])
        history.append(
            {
                "source_cluster": source_cluster_id,
                "score": float(match_meta.get("score", 0.0)),
                "margin": float(match_meta.get("margin", 0.0)),
            }
        )
        target_payload["auto_merged_matches"] = history

    remove_cluster_aliases(char_map, source_cluster_id)
    clusters.pop(source_cluster_id, None)


def remap_face_cluster_list(cluster_ids: list[str], replacements: dict[str, str]) -> list[str]:
    remapped: list[str] = []
    for cluster_id in cluster_ids or []:
        final_cluster_id = replacements.get(cluster_id, cluster_id)
        if final_cluster_id not in remapped:
            remapped.append(final_cluster_id)
    return remapped


def remap_voice_face_links(voice_map: dict, replacements: dict[str, str]) -> int:
    changed = 0
    for payload in voice_map.get("clusters", {}).values():
        linked_face = payload.get("linked_face_cluster")
        if linked_face in replacements:
            payload["linked_face_cluster"] = replacements[linked_face]
            changed += 1
    return changed


def remap_linked_segments_face_clusters(cfg: dict, replacements: dict[str, str]) -> int:
    if not replacements:
        return 0
    linked_root = resolve_project_path(cfg["paths"]["linked_segments"])
    changed_files = 0
    for linked_file in sorted(linked_root.glob("*_linked_segments.json")):
        rows = read_json(linked_file, [])
        changed = False
        for row in rows:
            for field_name in ("visible_face_clusters", "face_clusters"):
                cluster_ids = row.get(field_name)
                if isinstance(cluster_ids, list):
                    remapped = remap_face_cluster_list(cluster_ids, replacements)
                    if remapped != cluster_ids:
                        row[field_name] = remapped
                        changed = True
            speaker_face_cluster = row.get("speaker_face_cluster")
            if speaker_face_cluster in replacements:
                row["speaker_face_cluster"] = replacements[speaker_face_cluster]
                changed = True
        if changed:
            write_json(linked_file, rows)
            changed_files += 1
    return changed_files


def auto_match_known_faces(cfg: dict, char_map: dict, voice_map: dict) -> dict[str, object]:
    planned_matches = plan_known_face_matches(cfg, char_map)
    if not planned_matches:
        return {"matched": 0, "replacements": {}, "linked_files": 0, "voice_links": 0}

    replacements: dict[str, str] = {}
    for source_cluster_id, match_meta in planned_matches.items():
        target_cluster_id = str(match_meta.get("target_cluster", "")).strip()
        if not target_cluster_id:
            continue
        if source_cluster_id not in char_map.get("clusters", {}) or target_cluster_id not in char_map.get("clusters", {}):
            continue
        merge_face_cluster_into_target(char_map, source_cluster_id, target_cluster_id, match_meta)
        replacements[source_cluster_id] = target_cluster_id

    linked_files = remap_linked_segments_face_clusters(cfg, replacements)
    voice_links = remap_voice_face_links(voice_map, replacements)
    return {
        "matched": len(replacements),
        "replacements": replacements,
        "linked_files": linked_files,
        "voice_links": voice_links,
    }


def linked_segment_rows(cfg: dict) -> list[dict]:
    linked_root = resolve_project_path(cfg["paths"]["linked_segments"])
    rows: list[dict] = []
    for linked_file in sorted(linked_root.glob("*_linked_segments.json")):
        payload = read_json(linked_file, [])
        if isinstance(payload, list):
            rows.extend(payload)
    return rows


def auto_link_speakers_from_single_visible_faces(cfg: dict, char_map: dict, voice_map: dict) -> dict[str, int]:
    speaker_votes: dict[str, dict[str, int]] = {}
    for row in linked_segment_rows(cfg):
        speaker_cluster = str(row.get("speaker_cluster", "")).strip()
        if not speaker_cluster:
            continue
        visible_clusters = []
        for cluster_id in row.get("visible_face_clusters", []) or []:
            payload = char_map.get("clusters", {}).get(cluster_id, {})
            name = str(payload.get("name", cluster_id))
            if is_ignored_face_payload(payload) or not has_manual_person_name(name) or is_background_person_name(name):
                continue
            visible_clusters.append(cluster_id)
        visible_clusters = list(dict.fromkeys(visible_clusters))
        if len(visible_clusters) != 1:
            continue
        speaker_votes.setdefault(speaker_cluster, {})
        only_cluster = visible_clusters[0]
        speaker_votes[speaker_cluster][only_cluster] = speaker_votes[speaker_cluster].get(only_cluster, 0) + 1

    linked = 0
    for speaker_cluster, votes in speaker_votes.items():
        ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        if not ranked:
            continue
        best_cluster, best_votes = ranked[0]
        second_votes = ranked[1][1] if len(ranked) > 1 else 0
        total_votes = sum(votes.values())
        vote_share = (best_votes / total_votes) if total_votes else 0.0
        if best_votes < 3 or vote_share < 0.70 or (best_votes - second_votes) < 2:
            continue
        face_payload = char_map.get("clusters", {}).get(best_cluster, {})
        face_name = str(face_payload.get("name", "")).strip()
        if not has_manual_person_name(face_name) or is_background_person_name(face_name):
            continue
        speaker_payload = voice_map.setdefault("clusters", {}).setdefault(speaker_cluster, {})
        previous_cluster = str(speaker_payload.get("linked_face_cluster", "")).strip()
        previous_name = str(speaker_payload.get("name", "")).strip()
        inferred_name = display_person_name(face_name, speaker_cluster)
        if previous_cluster == best_cluster and previous_name == inferred_name and bool(speaker_payload.get("auto_named", True)):
            continue
        if previous_cluster and previous_cluster != best_cluster and not bool(speaker_payload.get("auto_named", True)):
            continue
        speaker_payload["linked_face_cluster"] = best_cluster
        if speaker_payload.get("auto_named", True) or looks_auto_named(previous_name):
            speaker_payload["name"] = inferred_name
            speaker_payload["auto_named"] = True
        linked += 1
    return {"matched": linked}


def auto_learn_remaining_reviews(cfg: dict, char_map: dict, voice_map: dict, max_rounds: int = 8) -> dict[str, int]:
    total_face_matches = 0
    total_voice_links = 0
    total_linked_files = 0
    rounds = 0
    for _ in range(max(1, max_rounds)):
        rounds += 1
        face_summary = auto_match_known_faces(cfg, char_map, voice_map)
        voice_summary = auto_link_speakers_from_single_visible_faces(cfg, char_map, voice_map)
        matched_faces = int(face_summary.get("matched", 0))
        matched_speakers = int(voice_summary.get("matched", 0))
        total_face_matches += matched_faces
        total_voice_links += matched_speakers
        total_linked_files += int(face_summary.get("linked_files", 0))
        if matched_faces <= 0 and matched_speakers <= 0:
            rounds -= 1
            break
    return {
        "rounds": max(0, rounds),
        "matched_faces": total_face_matches,
        "matched_speakers": total_voice_links,
        "linked_files": total_linked_files,
    }


def normalize_placeholder_maps(char_map: dict, voice_map: dict) -> tuple[int, int]:
    changed_faces = 0
    changed_voices = 0

    char_map["aliases"] = {}
    for cluster_id, payload in char_map.get("clusters", {}).items():
        if is_ignored_face_payload(payload):
            if payload.get("name") != "noface":
                payload["name"] = "noface"
                changed_faces += 1
            if not payload.get("ignored"):
                payload["ignored"] = True
                changed_faces += 1
            if payload.get("background_role"):
                payload["background_role"] = False
                changed_faces += 1
            if payload.get("priority"):
                payload["priority"] = False
                changed_faces += 1
            if payload.get("auto_named", False):
                payload["auto_named"] = False
                changed_faces += 1
            if payload.get("aliases"):
                payload["aliases"] = []
                changed_faces += 1
            continue

        normalized_name = display_person_name(str(payload.get("name", "")), cluster_id)
        background_role = is_background_person_name(normalized_name)
        if payload.get("name") != normalized_name:
            payload["name"] = normalized_name
            changed_faces += 1
        if payload.get("ignored"):
            payload["ignored"] = False
            changed_faces += 1
        if bool(payload.get("background_role")) != background_role:
            payload["background_role"] = background_role
            changed_faces += 1
        effective_priority = False if background_role else bool(payload.get("priority", False))
        if bool(payload.get("priority", False)) != effective_priority:
            payload["priority"] = effective_priority
            changed_faces += 1

        if has_manual_person_name(normalized_name):
            if payload.get("auto_named", True):
                payload["auto_named"] = False
                changed_faces += 1
            normalized_alias = normalize_alias_name(normalized_name)
            aliases = [normalized_alias] if normalized_alias and not background_role else []
            if payload.get("aliases") != aliases:
                payload["aliases"] = aliases
                changed_faces += 1
            if normalized_alias and not background_role:
                char_map["aliases"][normalized_alias] = cluster_id
        else:
            if not payload.get("auto_named", False):
                payload["auto_named"] = True
                changed_faces += 1
            if payload.get("background_role"):
                payload["background_role"] = False
                changed_faces += 1
            if payload.get("priority"):
                payload["priority"] = False
                changed_faces += 1
            if payload.get("aliases"):
                payload["aliases"] = []
                changed_faces += 1

    rebuild_character_map_identities(char_map)
    voice_map["aliases"] = {}
    for speaker_cluster, payload in voice_map.get("clusters", {}).items():
        linked_face = payload.get("linked_face_cluster")
        linked_face_payload = char_map.get("clusters", {}).get(linked_face, {})
        normalized_name = display_person_name(str(payload.get("name", "")), speaker_cluster)
        if linked_face and not is_ignored_face_payload(linked_face_payload):
            normalized_name = display_person_name(str(linked_face_payload.get("name", "")), speaker_cluster)
        if payload.get("name") != normalized_name:
            payload["name"] = normalized_name
            changed_voices += 1
        if payload.get("auto_named") is not True:
            payload["auto_named"] = True
            changed_voices += 1

    return changed_faces, changed_voices


def refresh_voice_map(char_map: dict, voice_map: dict) -> None:
    for speaker_cluster, payload in voice_map.get("clusters", {}).items():
        linked_face = payload.get("linked_face_cluster")
        if not linked_face:
            continue

        face_payload = char_map.get("clusters", {}).get(linked_face)
        if face_payload is None or is_ignored_face_payload(face_payload):
            payload.pop("linked_face_cluster", None)
            if payload.get("auto_named", True) or looks_auto_named(str(payload.get("name", ""))):
                payload["name"] = speaker_cluster
                payload["auto_named"] = True
            continue

        if payload.get("auto_named", True) or looks_auto_named(str(payload.get("name", ""))):
            payload["name"] = display_person_name(str(face_payload.get("name", "")), speaker_cluster)
            payload["auto_named"] = True


def refresh_linked_segments(cfg: dict, char_map: dict, voice_map: dict) -> int:
    linked_root = resolve_project_path(cfg["paths"]["linked_segments"])
    changed_files = 0
    for linked_file in sorted(linked_root.glob("*_linked_segments.json")):
        rows = read_json(linked_file, [])
        changed = False
        for row in rows:
            visible_clusters = [
                cluster_id
                for cluster_id in row.get("visible_face_clusters", [])
                if not is_ignored_face_payload(char_map.get("clusters", {}).get(cluster_id, {}))
            ]
            if visible_clusters != row.get("visible_face_clusters", []):
                row["visible_face_clusters"] = visible_clusters
                changed = True

            visible_names = [
                face_display_name(char_map.get("clusters", {}).get(cluster_id, {}), cluster_id)
                for cluster_id in visible_clusters
            ]
            if visible_names != row.get("visible_character_names", []):
                row["visible_character_names"] = visible_names
                changed = True

            voice_payload = voice_map.get("clusters", {}).get(row.get("speaker_cluster"), {})
            speaker_name = display_person_name(voice_payload.get("name", ""), row.get("speaker_cluster"))
            if speaker_name != row.get("speaker_name"):
                row["speaker_name"] = speaker_name
                changed = True

            speaker_face_cluster = row.get("speaker_face_cluster")
            if speaker_face_cluster and speaker_face_cluster not in visible_clusters:
                row["speaker_face_cluster"] = None
                changed = True

        if changed:
            write_json(linked_file, rows)
            changed_files += 1
    return changed_files


def rebuild_review_queue(cfg: dict) -> int:
    linked_root = resolve_project_path(cfg["paths"]["linked_segments"])
    items = []
    for linked_file in sorted(linked_root.glob("*_linked_segments.json")):
        rows = read_json(linked_file, [])
        for row in rows:
            speaker_name = str(row.get("speaker_name", ""))
            visible_names = [str(name) for name in row.get("visible_character_names", [])]
            if looks_auto_named(speaker_name) or any(looks_auto_named(name) for name in visible_names):
                items.append(row)
    write_json(resolve_project_path(cfg["paths"]["review_queue"]), {"items": items})
    return len(items)


def persist_updates(cfg: dict, char_map: dict, voice_map: dict) -> tuple[int, int]:
    rebuild_character_map_identities(char_map)
    refresh_voice_map(char_map, voice_map)
    write_json(resolve_project_path(cfg["paths"]["character_map"]), char_map)
    write_json(resolve_project_path(cfg["paths"]["voice_map"]), voice_map)
    changed_linked_files = refresh_linked_segments(cfg, char_map, voice_map)
    review_count = rebuild_review_queue(cfg)
    return changed_linked_files, review_count


def print_cluster(char_map: dict, cluster_id: str, payload: dict) -> None:
    status = "ignored" if is_ignored_face_payload(payload) else "aktiv"
    name = payload.get("name", cluster_id)
    preview_dir = payload.get("preview_dir", "-")
    samples = payload.get("samples", 0)
    scene_count = payload.get("scene_count", 0)
    detection_count = payload.get("detection_count", 0)
    auto_named = payload.get("auto_named", False)
    priority = bool(payload.get("priority", False))
    identity_count = identity_cluster_count(char_map, str(name)) if has_manual_person_name(str(name)) and not is_background_person_name(str(name)) else 1
    role_hint = suggested_face_role(payload)
    action_hint = suggested_face_action_hint(payload)
    print(f"{cluster_id}: {name}")
    print(f"  Status: {status}")
    print(f"  Samples: {samples}")
    print(f"  Scenes: {scene_count}")
    print(f"  Detections: {detection_count}")
    print(f"  Identity faces: {identity_count}")
    print(f"  Auto: {auto_named}")
    print(f"  Prioritized: {priority}")
    print(f"  Role hint: {role_hint}")
    print(f"  Review hint: {action_hint}")
    print(f"  Preview: {preview_dir}")


def list_faces(char_map: dict, limit: int, include_named: bool) -> None:
    clusters = sorted(char_map.get("clusters", {}).items(), key=cluster_sort_key)
    shown = 0
    for cluster_id, payload in clusters:
        if is_ignored_face_payload(payload):
            continue
        if looks_auto_named(str(payload.get("name", ""))):
            continue
        print_cluster(char_map, cluster_id, payload)
        shown += 1
        if limit > 0 and shown >= limit:
            break
    if shown == 0:
        info("No named characters were found.")


def created_face_names(char_map: dict) -> list[str]:
    names: set[str] = set()
    for _cluster_id, payload in sorted(char_map.get("clusters", {}).items(), key=cluster_sort_key):
        if is_ignored_face_payload(payload):
            continue
        name = canonical_person_name(str(payload.get("name", "")))
        if not has_manual_person_name(name):
            continue
        if is_background_person_name(name):
            continue
        names.add(name)
    return sorted(names, key=lambda value: value.lower())


def print_created_face_names(char_map: dict) -> None:
    names = created_face_names(char_map)
    if not names:
        info("No recognized named characters were found.")
        return
    for name in names:
        print(name)


def face_review_candidates(
    char_map: dict,
    include_named: bool,
    limit: int,
    skipped_clusters: set[str] | None = None,
) -> list[tuple[str, dict]]:
    candidates = []
    skipped = skipped_clusters or set()
    for cluster_id, payload in sorted(char_map.get("clusters", {}).items(), key=cluster_sort_key):
        if cluster_id in skipped:
            continue
        if is_ignored_face_payload(payload):
            continue
        if not include_named and not looks_auto_named(str(payload.get("name", ""))):
            continue
        candidates.append((cluster_id, payload))
    if limit > 0:
        candidates = candidates[:limit]
    return candidates


def session_case_budget_remaining(limit: int, handled_count: int) -> int:
    if limit <= 0:
        return 0
    return max(0, int(limit) - max(0, int(handled_count)))


def session_face_review_candidates(
    char_map: dict,
    include_named: bool,
    session_limit: int,
    handled_count: int,
    skipped_clusters: set[str] | None = None,
) -> list[tuple[str, dict]]:
    if session_limit > 0:
        remaining_limit = session_case_budget_remaining(session_limit, handled_count)
        if remaining_limit <= 0:
            return []
        effective_limit = remaining_limit
    else:
        effective_limit = 0
    return face_review_candidates(char_map, include_named, effective_limit, skipped_clusters)


def interactive_face_review(cfg: dict, char_map: dict, voice_map: dict, include_named: bool, limit: int, open_previews: bool) -> None:
    if not is_interactive_session():
        raise RuntimeError("--review-faces requires an interactive console.")

    skipped_clusters: set[str] = set()
    handled_count = 0
    candidates = session_face_review_candidates(char_map, include_named, limit, handled_count, skipped_clusters)
    if not candidates:
        remaining_queue_count = open_review_item_count(cfg)
        if remaining_queue_count > 0:
            info(
                f"No face clusters were found for review. "
                f"There are still {remaining_queue_count} speaker/segment review cases in review_queue.json; use --show-queue to inspect them."
            )
        else:
            info("No face clusters were found for review.")
        return

    changed = 0
    stop_review = False
    while True:
        auto_matched = auto_learn_remaining_reviews(cfg, char_map, voice_map)
        if auto_matched.get("matched_faces", 0) or auto_matched.get("matched_speakers", 0):
            changed_linked_files, review_count = persist_updates(cfg, char_map, voice_map)
            info(
                f"Before the next input, {auto_matched['matched_faces']} known face clusters "
                f"and {auto_matched['matched_speakers']} speaker assignments were applied automatically. "
                f"{changed_linked_files + int(auto_matched.get('linked_files', 0))} linked-segment files updated, "
                f"{review_count} open review cases."
            )
        candidates = session_face_review_candidates(char_map, include_named, limit, handled_count, skipped_clusters)
        if not candidates:
            break
        cluster_id, payload = candidates[0]
        session_remaining_count = session_case_budget_remaining(limit, handled_count)
        total_open_count = len(face_review_candidates(char_map, include_named, 0, set()))
        role_hint = suggested_face_role(payload)
        action_hint = suggested_face_action_hint(payload)
        answer = ""
        while True:
            montage = create_face_review_sheet(cluster_id, payload)
            print()
            print("-" * 72)
            print_cluster(char_map, cluster_id, payload)
            print(f"Remaining in this session including current case: {session_remaining_count}")
            print(f"Total actually still open: {total_open_count}")
            print(f"Automatic role hint: {role_hint}")
            print(f"Automatic review hint: {action_hint}")
            if montage:
                print(f"Contact sheet: {montage}")
                if open_previews:
                    info("Preview is open. You can enter the name directly in the window or in parallel in the terminal.")
            for context_path, crop_path in preview_pairs(payload):
                if context_path:
                    print(f"Szene: {context_path}")
                if crop_path:
                    print(f"Ausschnitt: {crop_path}")
            preview_name = ""
            preview_priority: bool | None = None
            if open_previews and montage:
                quick_assignments = known_identity_button_options(char_map, limit=16)
                preview_result = show_preview_assignment_window(
                    montage,
                    f"{cluster_id} Preview | Session: {session_remaining_count} | Open: {total_open_count}",
                    status_text=(
                        f"Remaining in this session including current case: {session_remaining_count} | "
                        f"Total actually still open: {total_open_count} | "
                        f"Role hint: {role_hint} | {action_hint}"
                    ),
                    initial_priority=identity_has_priority(char_map, str(payload.get("name", cluster_id))),
                    quick_assignments=quick_assignments,
                )
                if preview_result is not None:
                    preview_name = str(preview_result.get("value") or "").strip()
                    preview_priority = bool(preview_result.get("priority", False))
            try:
                if preview_name:
                    answer = preview_name
                    explicit_priority = preview_priority
                else:
                    answer, explicit_priority = prompt_terminal_assignment(char_map)
                answer = answer.strip()
            except EOFError:
                info("No further input is available. Review will stop after the current preview.")
                stop_review = True
                break
            if not answer:
                info("Please enter a name or use 'noface'. The current face cluster stays open.")
                continue
            if answer.lower() == "q" or answer == REVIEW_QUIT_TOKEN:
                break
            if answer == REVIEW_SKIP_TOKEN:
                info("Current face cluster was left unchanged.")
                skipped_clusters.add(cluster_id)
                handled_count += 1
                break
            assign_character_name(char_map, cluster_id, answer, priority=explicit_priority)
            payload = char_map["clusters"][cluster_id]
            auto_matched = auto_learn_remaining_reviews(cfg, char_map, voice_map)
            changed_linked_files, review_count = persist_updates(cfg, char_map, voice_map)
            changed += 1
            handled_count += 1
            merged_count = int(auto_matched.get("matched_faces", 0))
            speaker_count = int(auto_matched.get("matched_speakers", 0))
            sync_count = changed_linked_files + int(auto_matched.get("linked_files", 0))
            if merged_count or speaker_count:
                ok(
                    f"{cluster_id} saved. Afterwards, {merged_count} additional face clusters and "
                    f"{speaker_count} speaker assignments were applied automatically. "
                    f"{sync_count} linked-segment files synchronized, {review_count} open review cases."
                )
            else:
                ok(
                    f"{cluster_id} saved. "
                    f"{sync_count} linked-segment files synchronized, {review_count} open review cases."
                )
            break
        if stop_review or answer.lower() == "q" or answer == REVIEW_QUIT_TOKEN:
            break

    if changed:
        remaining_session_count = session_case_budget_remaining(limit, handled_count)
        ok(f"{changed} face clusters updated. Still open in this session: {remaining_session_count}.")
    else:
        info("No changes were made.")


def assign_single_face(cfg: dict, char_map: dict, voice_map: dict, cluster_id: str, assigned_name: str, priority: bool = False) -> None:
    payload = char_map.get("clusters", {}).get(cluster_id)
    if payload is None:
        raise FileNotFoundError(f"Face cluster not found: {cluster_id}")
    assign_character_name(char_map, cluster_id, assigned_name, priority=priority)
    auto_matched = auto_learn_remaining_reviews(cfg, char_map, voice_map)
    changed_linked_files, review_count = persist_updates(cfg, char_map, voice_map)
    final_name = char_map["clusters"][cluster_id]["name"]
    ok(
        f"{cluster_id} -> {final_name} saved. "
        f"{changed_linked_files + int(auto_matched.get('linked_files', 0))} linked-segment files synchronized, "
        f"{review_count} open review cases. "
        f"Auto-learn: {int(auto_matched.get('matched_faces', 0))} face clusters, {int(auto_matched.get('matched_speakers', 0))} speakers."
    )


def rename_face(cfg: dict, char_map: dict, voice_map: dict, reference: str, new_name: str, priority: bool = False) -> None:
    cluster_id = resolve_face_reference(char_map, reference)
    assign_character_name(char_map, cluster_id, new_name, priority=priority)
    auto_matched = auto_learn_remaining_reviews(cfg, char_map, voice_map)
    changed_linked_files, review_count = persist_updates(cfg, char_map, voice_map)
    final_name = char_map["clusters"][cluster_id]["name"]
    ok(
        f"{cluster_id} was renamed to {final_name}. "
        f"{changed_linked_files + int(auto_matched.get('linked_files', 0))} linked-segment files synchronized, "
        f"{review_count} open review cases. "
        f"Auto-learn: {int(auto_matched.get('matched_faces', 0))} face clusters, {int(auto_matched.get('matched_speakers', 0))} speakers."
    )


def set_character_priority(char_map: dict, reference: str, priority: bool) -> tuple[str, dict]:
    cluster_id = resolve_face_reference(char_map, reference)
    payload = char_map.get("clusters", {}).get(cluster_id)
    if payload is None:
        raise FileNotFoundError(f"Face cluster not found: {reference}")

    final_name = canonical_person_name(str(payload.get("name", cluster_id)))
    if is_ignored_face_payload(payload):
        raise ValueError(f"{cluster_id} is marked as noface/ignored and cannot be prioritized.")
    if is_background_person_name(final_name):
        raise ValueError(f"{cluster_id} is marked as statist/minor character and cannot be prioritized.")
    if not has_manual_person_name(final_name):
        raise ValueError(f"{cluster_id} does not have a manual character name yet and can only be prioritized afterwards.")

    payload["priority"] = bool(priority)
    payload["auto_named"] = False
    return cluster_id, payload


def update_face_priority(cfg: dict, char_map: dict, voice_map: dict, reference: str, priority: bool) -> None:
    cluster_id, payload = set_character_priority(char_map, reference, priority)
    changed_linked_files, review_count = persist_updates(cfg, char_map, voice_map)
    final_name = payload.get("name", cluster_id)
    state = "prioritized" if priority else "no longer prioritized"
    ok(
        f"{cluster_id} ({final_name}) is now {state}. "
        f"{changed_linked_files} linked-segment files synchronized, {review_count} open review cases."
    )


def show_review_queue(cfg: dict) -> None:
    queue = read_json(resolve_project_path(cfg["paths"]["review_queue"]), {"items": []})
    items = queue.get("items", [])
    if not items:
        info("No open review cases.")
        return
    for index, item in enumerate(items, start=1):
        print("-" * 72)
        print(f"Case {index}")
        print(f"Scene: {item.get('scene_id')}")
        print(f"Speaker: {item.get('speaker_name')}")
        print(f"Text: {str(item.get('text', ''))[:220]}")
        print(f"Visible characters: {', '.join(item.get('visible_character_names', []))}")
        frames = item.get("speaker_reference_frames", [])
        if frames:
            print(f"Frames: {', '.join(frames[:3])}")


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    effective_limit = 0 if args.all else max(0, args.limit)
    headline("Review Open Assignments")
    cfg = load_config()
    worker_id = shared_worker_id_for_args(args)
    shared_workers = shared_workers_enabled_for_args(cfg, args)
    mark_step_started("06_review_unknowns", "global")
    if shared_workers:
        info(f"Shared NAS workers: enabled ({worker_id})")
    lease_manager = distributed_item_lease(
        root=distributed_step_runtime_root("06_review_unknowns", "global"),
        lease_name="global",
        cfg=cfg,
        worker_id=worker_id,
        enabled=shared_workers,
        meta={"step": "06_review_unknowns", "scope": "global", "worker_id": worker_id},
    )
    acquired = lease_manager.__enter__()
    if not acquired:
        info("Review is already active on another worker.")
        lease_manager.__exit__(None, None, None)
        return
    char_map = read_json(resolve_project_path(cfg["paths"]["character_map"]), {"clusters": {}, "aliases": {}})
    voice_map = read_json(resolve_project_path(cfg["paths"]["voice_map"]), {"clusters": {}, "aliases": {}})
    action = "interactive_review"
    completion_payload: dict[str, object] = {}
    try:
        char_map.setdefault("clusters", {})
        char_map.setdefault("aliases", {})
        voice_map.setdefault("clusters", {})
        voice_map.setdefault("aliases", {})
        normalized_faces, normalized_voices = normalize_placeholder_maps(char_map, voice_map)
        if normalized_faces or normalized_voices:
            info(
                f"Normalized existing maps: {normalized_faces} face entries, "
                f"{normalized_voices} speaker entries."
            )
        hydrated = hydrate_face_clusters_from_previews(cfg, char_map)
        if hydrated:
            info(f"{hydrated} face clusters hydrated from existing preview folders.")
        auto_matched = auto_learn_remaining_reviews(cfg, char_map, voice_map)
        if auto_matched.get("matched_faces", 0) or auto_matched.get("matched_speakers", 0):
            info(
                f"Before review, {auto_matched['matched_faces']} unknown face clusters "
                f"and {auto_matched['matched_speakers']} speaker assignments were applied automatically."
            )
        face_cfg = cfg.get("character_detection", {}) if isinstance(cfg.get("character_detection"), dict) else {}
        auto_statists_enabled = (
            bool(args.auto_mark_statists)
            or (
                not bool(args.no_auto_mark_statists)
                and bool(face_cfg.get("auto_mark_statist_candidates", True))
            )
        )
        auto_statist_marked: list[dict[str, object]] = []
        if auto_statists_enabled:
            auto_statist_marked = mark_auto_statist_candidates(
                char_map,
                auto_statist_thresholds(cfg, args),
                effective_limit,
            )
            if auto_statist_marked:
                info(f"Marked {len(auto_statist_marked)} low-activity face clusters as 'statist'.")
        if (
            normalized_faces
            or normalized_voices
            or hydrated
            or auto_matched.get("matched_faces", 0)
            or auto_matched.get("matched_speakers", 0)
            or auto_statist_marked
        ):
            changed_linked_files, review_count = persist_updates(cfg, char_map, voice_map)
            info(
                f"Maps synchronized: {changed_linked_files + int(auto_matched.get('linked_files', 0))} linked-segment files updated, "
                f"{review_count} open review cases."
            )

        if args.assign_face:
            assigned_name = "noface" if args.ignore else (args.name or "").strip()
            if not assigned_name:
                raise ValueError("Please provide --name or use --ignore.")
            assign_single_face(cfg, char_map, voice_map, args.assign_face, assigned_name, priority=args.priority)
            action = "assign_face"
            completion_payload = {"cluster_id": args.assign_face, "assigned_name": assigned_name}
        elif args.rename_face:
            new_name = (args.rename_to or "").strip()
            if not new_name:
                raise ValueError("Please provide --rename-to.")
            rename_face(cfg, char_map, voice_map, args.rename_face, new_name, priority=args.priority)
            action = "rename_face"
            completion_payload = {"reference": args.rename_face, "new_name": new_name}
        elif args.set_priority:
            update_face_priority(cfg, char_map, voice_map, args.set_priority, True)
            action = "set_priority"
            completion_payload = {"reference": args.set_priority, "priority": True}
        elif args.clear_priority:
            update_face_priority(cfg, char_map, voice_map, args.clear_priority, False)
            action = "clear_priority"
            completion_payload = {"reference": args.clear_priority, "priority": False}
        elif args.auto_mark_statists:
            if auto_statist_marked:
                ok(f"{len(auto_statist_marked)} low-activity face clusters were marked as 'statist'.")
            else:
                info("No safe low-activity face clusters matched the automatic 'statist' thresholds.")
            action = "auto_mark_statists"
            completion_payload = {
                "marked": len(auto_statist_marked),
                "clusters": [item["cluster_id"] for item in auto_statist_marked],
            }
        elif args.review_faces:
            interactive_face_review(
                cfg,
                char_map,
                voice_map,
                include_named=args.include_named,
                limit=effective_limit,
                open_previews=args.open_previews,
            )
            action = "review_faces"
            completion_payload = {"include_named": bool(args.include_named), "open_previews": bool(args.open_previews)}
        elif args.list_faces:
            list_faces(char_map, effective_limit, args.include_named)
            action = "list_faces"
            completion_payload = {"limit": effective_limit, "include_named": bool(args.include_named)}
        elif args.created:
            print_created_face_names(char_map)
            action = "created_faces"
        elif args.show_queue:
            show_review_queue(cfg)
            action = "show_queue"
        elif not is_interactive_session():
            info("No interactive console detected. Showing unnamed face clusters with preview paths instead.")
            list_faces(char_map, effective_limit, args.include_named)
            action = "list_faces_noninteractive"
            completion_payload = {"limit": effective_limit, "include_named": bool(args.include_named)}
        else:
            interactive_face_review(
                cfg,
                char_map,
                voice_map,
                include_named=args.include_named,
                limit=effective_limit,
                open_previews=True,
            )
            action = "interactive_review"
            completion_payload = {"limit": effective_limit, "include_named": bool(args.include_named)}

        mark_step_completed(
            "06_review_unknowns",
            "global",
            {"action": action, **completion_payload},
        )
    except Exception as exc:
        mark_step_failed("06_review_unknowns", str(exc), "global")
        raise
    finally:
        lease_manager.__exit__(None, None, None)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise

