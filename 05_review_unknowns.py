#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "ai_series_project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import argparse
import base64
import ctypes
import html
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from support_scripts.pipeline_common import (
    add_shared_worker_arguments,
    canonical_person_name,
    character_appearance_embedding,
    cosine_similarity,
    current_os,
    display_person_name,
    distributed_item_lease,
    distributed_step_runtime_root,
    error,
    has_manual_person_name,
    headline,
    info,
    interactive_display_state,
    is_background_person_name,
    is_interactive_session,
    load_config,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    open_review_item_count,
    ok,
    print_interactive_display_diagnostics,
    normalize_portable_project_paths,
    portable_project_path,
    read_json,
    rerun_in_runtime,
    resolve_project_path,
    resolve_stored_project_path,
    shared_worker_id_for_args,
    shared_workers_enabled_for_args,
    terminal_clickable_path,
    warn,
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
    "statist = Statist/background role",
]
REVIEW_SKIP_TOKEN = "__skip__"
REVIEW_QUIT_TOKEN = "__quit__"
_TK_PREVIEW_ROOT = None
FACE_CLUSTER_ID_PATTERN = re.compile(r"^face_[0-9a-z]+$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review, assign, rename, or ignore face clusters")
    parser.add_argument("--list-faces", action="store_true", help="Show already named characters together with preview paths.")
    parser.add_argument("--created", action="store_true", help="Show only the names of already created recognized characters.")
    parser.add_argument("--show-queue", action="store_true", help="Show the open review_queue.json instead of the face review.")
    parser.add_argument("--queue-limit", type=int, default=50, help="Maximum review queue rows printed by --show-queue. Use 0 for all.")
    parser.add_argument("--edit-names", action="store_true", help="Open a Tk name editor for existing face and speaker names.")
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
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of clusters to process if --all is disabled internally. Default: 20")
    parser.add_argument("--all", action="store_true", default=True, help="Process every currently open face cluster. This is the default.")
    parser.add_argument(
        "--open-previews",
        action="store_true",
        default=True,
        help="Open the contact sheet during interactive review. This is enabled by default.",
    )
    parser.add_argument(
        "--no-open-previews",
        dest="open_previews",
        action="store_false",
        help="Do not open preview windows or the system image viewer during interactive review.",
    )
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
    parser.add_argument(
        "--no-internet-lookup",
        action="store_true",
        help="Disable public metadata/name lookup before manual review. Use --offline to skip every online lookup.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run the review fully offline. Public metadata and online face lookup are skipped and can be refreshed later.",
    )
    parser.add_argument(
        "--refresh-internet-lookup",
        action="store_true",
        help="Ignore cached public metadata lookup results for this run.",
    )
    parser.add_argument(
        "--deep-online-face-lookup",
        action="store_true",
        help="Compatibility flag. Deep built-in public-image face lookup is now enabled by default unless --offline is used.",
    )
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
            actual_count = identity_face_count(char_map, final_name) or int(
                payload.get("face_cluster_count", 0) or payload.get("cluster_count", 0) or 0
            )
            options.append((final_name, bool(payload.get("priority", False)), actual_count))
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


def preview_dir_path(payload: dict) -> Path | None:
    preview_dir_text = str(payload.get("preview_dir", "") or "").strip()
    if not preview_dir_text:
        return None
    return resolve_stored_project_path(preview_dir_text)


def preview_files(payload: dict) -> list[Path]:
    preview_dir = preview_dir_path(payload)
    if not preview_dir:
        return []
    if not preview_dir.is_dir():
        return []
    files = [
        path
        for path in sorted(preview_dir.glob("*.jpg"))
        if "_montage" not in path.name
    ]
    return files[:6]


def preview_pairs(payload: dict) -> list[tuple[Path | None, Path | None]]:
    preview_dir = preview_dir_path(payload)
    if not preview_dir:
        return []
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


def create_face_review_sheet(cluster_id: str, payload: dict, output_dir: Path | None = None) -> Path | None:
    preview_dir = preview_dir_path(payload)
    if not preview_dir:
        return None
    target_dir = output_dir or preview_dir
    pairs = preview_pairs(payload)
    if not pairs:
        files = preview_files(payload)
        if not files:
            return None
        return create_contact_sheet(files, target_dir / f"{cluster_id}_montage.jpg", title=f"{cluster_id} | Preview")

    try:
        from PIL import Image, ImageDraw, ImageOps
    except Exception:
        fallback_files = [path for pair in pairs for path in pair if path is not None]
        return create_contact_sheet(fallback_files, target_dir / f"{cluster_id}_montage.jpg", title=f"{cluster_id} | Preview")

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

    output_path = target_dir / f"{cluster_id}_montage.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def create_face_review_html(cluster_id: str, payload: dict, output_dir: Path | None = None) -> Path | None:
    preview_dir = preview_dir_path(payload)
    if not preview_dir:
        return None
    pairs = preview_pairs(payload)
    files = preview_files(payload)
    if not pairs and not files:
        return None

    review_dir = output_dir or resolve_project_path("characters/review")
    review_dir.mkdir(parents=True, exist_ok=True)
    output_path = review_dir / f"{cluster_id}_preview.html"

    cards: list[str] = []
    if pairs:
        for index, (context_path, crop_path) in enumerate(pairs, start=1):
            image_blocks: list[str] = []
            for label, image_path in (("Scene", context_path), ("Crop", crop_path)):
                if image_path is None or not image_path.exists():
                    continue
                mime = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
                encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
                image_blocks.append(
                    f"<figure><figcaption>{html.escape(label)}: {html.escape(image_path.name)}</figcaption>"
                    f"<img src=\"data:{mime};base64,{encoded}\" alt=\"{html.escape(image_path.name)}\"></figure>"
                )
            if image_blocks:
                cards.append(f"<section class=\"pair\"><h2>Sample {index}</h2>{''.join(image_blocks)}</section>")
    else:
        for image_path in files[:6]:
            if not image_path.exists():
                continue
            mime = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            cards.append(
                "<section class=\"single\">"
                f"<figure><figcaption>{html.escape(image_path.name)}</figcaption>"
                f"<img src=\"data:{mime};base64,{encoded}\" alt=\"{html.escape(image_path.name)}\"></figure>"
                "</section>"
            )
    if not cards:
        return None

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(cluster_id)} Preview</title>
  <style>
    body {{
      font-family: "Segoe UI", sans-serif;
      background: #111827;
      color: #f9fafb;
      margin: 0;
      padding: 24px;
    }}
    h1 {{
      margin-top: 0;
    }}
    .grid {{
      display: grid;
      gap: 20px;
    }}
    .pair, .single {{
      background: #1f2937;
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }}
    .pair {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }}
    figure {{
      margin: 0;
    }}
    figcaption {{
      font-weight: 600;
      margin-bottom: 8px;
    }}
    img {{
      width: 100%;
      height: auto;
      border-radius: 12px;
      background: #374151;
    }}
    .hint {{
      color: #cbd5e1;
      margin-bottom: 20px;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(cluster_id)} Preview</h1>
  <p class="hint">Review the samples here and enter the assignment back in the terminal or the optional Tk window.</p>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def gui_preview_available() -> bool:
    state = interactive_display_state()
    try:
        os_name = current_os()
    except Exception:
        return bool(state.get("gui_available"))
    if os_name == "windows":
        return bool(state.get("interactive_console"))
    if os_name == "linux":
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return bool(state.get("interactive_console"))


def windows_preview_commands(path: Path) -> list[list[str]]:
    path_text = str(path)
    return [
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "Start-Process -LiteralPath $args[0]",
            path_text,
        ],
        ["cmd", "/c", "start", "", path_text],
        ["explorer.exe", path_text],
    ]


def windows_shell_open(path: Path) -> bool:
    try:
        shell32 = ctypes.windll.shell32
    except Exception:
        return False
    try:
        result = shell32.ShellExecuteW(None, "open", str(path), None, None, 1)
    except Exception:
        return False
    return int(result) > 32


def open_preview_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        os_name = current_os()
    except Exception:
        return False
    try:
        if os_name == "windows":
            if windows_shell_open(path):
                return True
            for command in windows_preview_commands(path):
                try:
                    subprocess.Popen(
                        command,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return True
                except Exception:
                    continue
            try:
                os.startfile(str(path))  # type: ignore[attr-defined]
                return True
            except Exception:
                return False
        if os_name == "linux" and gui_preview_available():
            subprocess.Popen(
                ["xdg-open", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
    except Exception:
        return False
    return False


def selected_preview_images(payload: dict) -> list[Path]:
    targets: list[Path] = []
    for context_path, crop_path in preview_pairs(payload):
        for image_path in (crop_path, context_path):
            if image_path is None or not image_path.exists():
                continue
            if image_path not in targets:
                targets.append(image_path)
        if len(targets) >= 4:
            return targets[:4]

    for image_path in preview_files(payload):
        if image_path.exists() and image_path not in targets:
            targets.append(image_path)
        if len(targets) >= 4:
            break
    return targets[:4]


def preview_open_targets(cluster_id: str, payload: dict) -> list[Path]:
    try:
        montage = create_face_review_sheet(cluster_id, payload)
    except Exception:
        montage = None
    if montage and montage.exists():
        return [montage]
    exact_images = selected_preview_images(payload)
    if exact_images:
        return [exact_images[0]]
    return []


def local_preview_cache_dir(cluster_id: str) -> Path:
    return Path(tempfile.gettempdir()) / "ai_series_review_previews" / cluster_id


def materialize_local_preview_bundle(cluster_id: str, payload: dict) -> dict[str, object]:
    source_images = selected_preview_images(payload)
    cache_dir = local_preview_cache_dir(cluster_id)
    try:
        shutil.rmtree(cache_dir, ignore_errors=True)
    except Exception:
        pass
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_images: list[Path] = []
    for source_path in source_images:
        if not source_path.exists() or not source_path.is_file():
            continue
        target_path = cache_dir / source_path.name
        try:
            shutil.copy2(source_path, target_path)
        except Exception:
            continue
        local_images.append(target_path)

    if local_images:
        local_payload = {"preview_dir": str(cache_dir)}
        montage = create_face_review_sheet(cluster_id, local_payload, output_dir=cache_dir)
        launch_target = montage if montage and montage.exists() else local_images[0]
        open_targets: list[Path] = [launch_target]
        return {
            "source_images": source_images,
            "local_images": local_images,
            "open_targets": open_targets,
            "preview_window_image": launch_target,
            "montage": montage,
            "cache_dir": cache_dir,
        }

    fallback_targets = preview_open_targets(cluster_id, payload)
    local_targets: list[Path] = []
    for source_path in fallback_targets:
        if not source_path.exists() or not source_path.is_file():
            continue
        target_path = cache_dir / source_path.name
        try:
            shutil.copy2(source_path, target_path)
        except Exception:
            continue
        local_targets.append(target_path)
    return {
        "source_images": source_images,
        "local_images": [],
        "open_targets": local_targets,
        "preview_window_image": None,
        "montage": None,
        "cache_dir": cache_dir,
    }


def open_preview_targets(paths: list[Path]) -> int:
    if not paths:
        return 0
    return 1 if open_preview_file(paths[0]) else 0


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


def get_preview_tk_root(tk_module):
    global _TK_PREVIEW_ROOT
    root = _TK_PREVIEW_ROOT
    if root is not None:
        try:
            root.winfo_exists()
            return root
        except Exception:
            _TK_PREVIEW_ROOT = None
    try:
        root = tk_module.Tk()
        root.withdraw()
    except Exception:
        return None
    _TK_PREVIEW_ROOT = root
    return root


def show_preview_assignment_window(
    image_path: Path,
    title: str,
    status_text: str = "",
    initial_priority: bool = False,
    quick_assignments: list[tuple[str, bool, int]] | None = None,
) -> dict[str, object] | None:
    if not image_path.exists():
        return None
    if not gui_preview_available():
        return None
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except Exception:
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((1200, 900))
    except Exception:
        return None

    result: dict[str, object] = {"value": None, "priority": bool(initial_priority)}
    terminal_buffer: list[str] = []

    root = get_preview_tk_root(tk)
    if root is None:
        return None
    try:
        window = tk.Toplevel(root)
    except Exception:
        try:
            root.destroy()
        except Exception:
            pass
        global _TK_PREVIEW_ROOT
        _TK_PREVIEW_ROOT = None
        root = get_preview_tk_root(tk)
        if root is None:
            return None
        try:
            window = tk.Toplevel(root)
        except Exception:
            return None
    window.title(title)
    try:
        window.attributes("-topmost", True)
    except Exception:
        pass
    window.configure(bg="#1f2937")

    def finish(value: str | None, priority: bool | None = None) -> None:
        if result["value"] is not None:
            return
        result["value"] = None if value is None else value.strip()
        if priority is not None:
            result["priority"] = bool(priority)
        try:
            window.quit()
        except Exception:
            pass
        try:
            window.destroy()
        except Exception:
            pass

    window.protocol("WM_DELETE_WINDOW", lambda: finish(None))
    window.bind("<Escape>", lambda _event: finish(None))

    try:
        photo = ImageTk.PhotoImage(image)
    except Exception:
        try:
            window.quit()
        except Exception:
            pass
        try:
            window.destroy()
        except Exception:
            pass
        return None
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
        text="Type a name here, click a quick-assign button, or enter it in the terminal. Enter confirms immediately. 'noface' ignores the cluster, 'statist' marks it as a background/statist role, and '!Name' in the terminal marks a main character right away.",
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
    tk.Button(button_row, text="Statist", command=lambda: finish("statist", False)).pack(side="left", padx=(0, 8))
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

    try:
        window.update_idletasks()
        window.lift()
        window.focus_force()
    except Exception:
        pass

    try:
        window.mainloop()
    finally:
        try:
            if window.winfo_exists():
                window.destroy()
        except Exception:
            pass
    return result


def prompt_terminal_assignment(char_map: dict) -> tuple[str, bool | None]:
    print(f"Example names: {' | '.join(EXAMPLE_FACE_HINTS)}")
    quick_names = [name for name, _priority, _count in known_identity_button_options(char_map, limit=8)]
    if quick_names:
        print(f"Known characters: {' | '.join(quick_names)}")
    print("Enter a name. 'noface' ignores the match, 'statist' saves a background/statist role, '!Name' prioritizes as a main character, and 'q' quits the review.")
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
        return "Hint: already saved as Statist/background role"
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


def preview_crop_paths(payload: dict, limit: int = 4) -> list[Path]:
    paths: list[Path] = []
    for _context_path, crop_path in preview_pairs(payload):
        if crop_path is not None and crop_path.exists() and crop_path.is_file():
            paths.append(crop_path)
    preview_dir = preview_dir_path(payload)
    if preview_dir and preview_dir.is_dir():
        for crop_path in sorted(preview_dir.glob("*_crop.jpg")):
            if crop_path not in paths:
                paths.append(crop_path)
    return paths[: max(1, int(limit or 1))]


def preview_crop_internal_feature_count(crop_path: Path, cfg: dict) -> int | None:
    try:
        import cv2
    except Exception:
        return None

    image = cv2.imread(str(crop_path))
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if height <= 0 or width <= 0:
        return None

    face_cfg = cfg.get("character_detection", {}) if isinstance(cfg.get("character_detection"), dict) else {}
    upper_ratio = float(face_cfg.get("face_validation_upper_region_ratio", 0.68))
    upper = gray[: max(1, min(height, int(height * upper_ratio))), :]
    min_width = max(5, int(width * float(face_cfg.get("face_validation_min_eye_width_ratio", 0.07))))
    min_height = max(5, int(height * float(face_cfg.get("face_validation_min_eye_height_ratio", 0.05))))
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
    if eye_cascade.empty():
        return None
    boxes = eye_cascade.detectMultiScale(
        upper,
        scaleFactor=float(face_cfg.get("face_validation_eye_scale_factor", 1.08)),
        minNeighbors=int(face_cfg.get("face_validation_eye_min_neighbors", 3)),
        minSize=(min_width, min_height),
    )
    return len(boxes)


def false_positive_face_report(cluster_id: str, payload: dict, cfg: dict) -> dict[str, object]:
    face_cfg = cfg.get("character_detection", {}) if isinstance(cfg.get("character_detection"), dict) else {}
    max_crops = int(face_cfg.get("review_false_positive_max_crops", 4))
    min_checked = int(face_cfg.get("review_false_positive_min_checked_crops", 1))
    min_features = int(face_cfg.get("face_validation_min_internal_features", 1))
    checked = 0
    valid = 0
    unreadable = 0
    feature_counts: list[int] = []
    for crop_path in preview_crop_paths(payload, max_crops):
        feature_count = preview_crop_internal_feature_count(crop_path, cfg)
        if feature_count is None:
            unreadable += 1
            continue
        checked += 1
        feature_counts.append(int(feature_count))
        if feature_count >= min_features:
            valid += 1
    return {
        "cluster_id": cluster_id,
        "checked_crops": checked,
        "valid_crops": valid,
        "unreadable_crops": unreadable,
        "feature_counts": feature_counts,
        "auto_ignore": checked >= min_checked and valid == 0,
    }


def auto_ignore_false_positive_face_clusters(cfg: dict, char_map: dict, limit: int = 0) -> list[dict[str, object]]:
    face_cfg = cfg.get("character_detection", {}) if isinstance(cfg.get("character_detection"), dict) else {}
    if not bool(face_cfg.get("review_auto_ignore_false_positive_faces", True)):
        return []
    if not bool(face_cfg.get("strict_face_validation", True)):
        return []

    marked: list[dict[str, object]] = []
    for cluster_id, payload in sorted(char_map.get("clusters", {}).items(), key=cluster_sort_key):
        if is_ignored_face_payload(payload) or bool(payload.get("priority", False)):
            continue
        if not looks_auto_named(str(payload.get("name", cluster_id))):
            continue
        report = false_positive_face_report(cluster_id, payload, cfg)
        if not bool(report.get("auto_ignore", False)):
            continue
        assign_character_name(char_map, cluster_id, "noface", priority=False)
        updated_payload = char_map.get("clusters", {}).get(cluster_id, {})
        updated_payload["auto_ignored_reason"] = "no_internal_face_features"
        updated_payload["false_positive_validation"] = {
            "checked_crops": int(report.get("checked_crops", 0) or 0),
            "valid_crops": int(report.get("valid_crops", 0) or 0),
            "feature_counts": list(report.get("feature_counts", []) or []),
        }
        marked.append(report)
        if limit > 0 and len(marked) >= limit:
            break
    return marked


def hydrate_face_clusters_from_previews(cfg: dict, char_map: dict) -> int:
    previews_root_value = cfg.get("paths", {}).get("character_previews", "characters/previews")
    previews_root = resolve_project_path(previews_root_value)
    if not previews_root.exists():
        return 0

    added = 0
    clusters = char_map.setdefault("clusters", {})
    if clusters:
        for cluster_id, payload in sorted(clusters.items(), key=cluster_sort_key):
            if not str(cluster_id).startswith("face_"):
                continue
            preview_dir = previews_root / str(cluster_id)
            if not preview_dir.is_dir():
                continue
            payload.setdefault("preview_dir", portable_project_path(preview_dir))
            if not payload.get("samples"):
                try:
                    payload["samples"] = max(1, sum(1 for _path in preview_dir.glob("*_crop.jpg")))
                except Exception:
                    payload["samples"] = 1
        for cluster_id in sorted(referenced_face_cluster_ids(cfg) - set(clusters)):
            preview_dir = previews_root / str(cluster_id)
            if not preview_dir.is_dir():
                continue
            payload = clusters.setdefault(cluster_id, {})
            payload["name"] = cluster_id
            payload["preview_dir"] = portable_project_path(preview_dir)
            # Avoid a per-cluster glob over NAS here; the real preview files are loaded lazily during review.
            payload["samples"] = int(payload.get("samples", 0) or 1)
            payload["auto_named"] = True
            payload["ignored"] = False
            payload["aliases"] = []
            added += 1
        return added

    for preview_dir in sorted(previews_root.glob("face_*")):
        if not preview_dir.is_dir():
            continue
        cluster_id = preview_dir.name
        payload = clusters.setdefault(cluster_id, {})
        payload["name"] = cluster_id
        payload["preview_dir"] = portable_project_path(preview_dir)
        payload["samples"] = preview_sample_count(preview_dir)
        payload["auto_named"] = True
        payload["ignored"] = False
        payload["aliases"] = []
        added += 1
    return added


def preview_sample_count(preview_dir: Path) -> int:
    try:
        return max(1, sum(1 for _path in preview_dir.glob("*_crop.jpg")))
    except Exception:
        return 1


def collect_face_cluster_ids_from_value(value: object) -> set[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return {cleaned} if FACE_CLUSTER_ID_PATTERN.match(cleaned) else set()
    if isinstance(value, list):
        ids: set[str] = set()
        for item in value:
            ids.update(collect_face_cluster_ids_from_value(item))
        return ids
    return set()


def referenced_face_cluster_ids(cfg: dict) -> set[str]:
    ids: set[str] = set()
    review_queue_path = resolve_project_path(cfg.get("paths", {}).get("review_queue", "characters/review/review_queue.json"))
    queue = read_json(review_queue_path, {"items": []})
    queue_items = queue.get("items", []) if isinstance(queue, dict) else []
    for item in queue_items:
        if not isinstance(item, dict):
            continue
        ids.update(collect_face_cluster_ids_from_value(item.get("visible_face_clusters", [])))
        ids.update(collect_face_cluster_ids_from_value(item.get("visible_character_names", [])))
        ids.update(collect_face_cluster_ids_from_value(item.get("speaker_face_cluster", "")))
    if ids:
        return ids

    linked_root = resolve_project_path(cfg.get("paths", {}).get("linked_segments", "characters/linked_segments"))
    if not linked_root.exists():
        return ids
    for linked_file in sorted(linked_root.glob("*_linked_segments.json")):
        for row in read_json(linked_file, []):
            if not isinstance(row, dict):
                continue
            ids.update(collect_face_cluster_ids_from_value(row.get("visible_face_clusters", [])))
            ids.update(collect_face_cluster_ids_from_value(row.get("visible_character_names", [])))
            ids.update(collect_face_cluster_ids_from_value(row.get("speaker_face_cluster", "")))
    return ids


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


def face_ids_for_cluster(cluster_id: str, payload: dict) -> list[str]:
    face_ids: list[str] = []
    for value in [cluster_id, *(payload.get("merged_cluster_ids", []) or [])]:
        text = str(value or "").strip()
        if text and text not in face_ids:
            face_ids.append(text)
    return face_ids


def identity_face_ids(char_map: dict, identity_name: str) -> list[str]:
    face_ids: list[str] = []
    for cluster_id, payload in identity_clusters(char_map, identity_name):
        for face_id in face_ids_for_cluster(cluster_id, payload):
            if face_id not in face_ids:
                face_ids.append(face_id)
    return face_ids


def identity_face_count(char_map: dict, identity_name: str) -> int:
    return len(identity_face_ids(char_map, identity_name))


def identity_cluster_count(char_map: dict, identity_name: str) -> int:
    return identity_face_count(char_map, identity_name)


def rebuild_character_map_identities(char_map: dict) -> None:
    aliases: dict[str, str] = {}
    identities: dict[str, dict] = {}
    grouped: dict[str, list[tuple[str, dict]]] = {}

    for cluster_id, payload in char_map.get("clusters", {}).items():
        payload.pop("identity_name", None)
        payload.pop("identity_primary_cluster", None)
        payload.pop("identity_cluster_ids", None)
        payload.pop("identity_face_cluster_ids", None)
        payload.pop("identity_face_cluster_count", None)
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
        face_ids: list[str] = []
        for cluster_id, payload in clusters:
            for face_id in face_ids_for_cluster(cluster_id, payload):
                if face_id not in face_ids:
                    face_ids.append(face_id)
        cluster_count = len(face_ids)
        detection_count = sum(int(payload.get("detection_count", 0) or 0) for _cluster_id, payload in clusters)
        sample_count = sum(int(payload.get("samples", 0) or 0) for _cluster_id, payload in clusters)
        priority = any(bool(payload.get("priority", False)) for _cluster_id, payload in clusters)
        background_role = is_background_person_name(identity_name)
        identities[identity_name] = {
            "name": identity_name,
            "primary_cluster": primary_cluster,
            "cluster_ids": cluster_ids,
            "active_cluster_count": len(cluster_ids),
            "face_cluster_ids": face_ids,
            "face_cluster_count": cluster_count,
            "cluster_count": cluster_count,
            "detection_count": detection_count,
            "sample_count": sample_count,
            "priority": False if background_role else priority,
            "background_role": background_role,
        }
        normalized_alias = normalize_alias_name(identity_name)
        if normalized_alias and not background_role:
            aliases[normalized_alias] = primary_cluster
        if not background_role:
            for _cluster_id, payload in clusters:
                for alias in payload.get("aliases", []) or []:
                    normalized_payload_alias = normalize_alias_name(str(alias))
                    if normalized_payload_alias and not is_ignored_face_name(normalized_payload_alias):
                        aliases[normalized_payload_alias] = primary_cluster
        for cluster_id, payload in clusters:
            payload["identity_name"] = identity_name
            payload["identity_primary_cluster"] = primary_cluster
            payload["identity_cluster_ids"] = cluster_ids
            payload["identity_face_cluster_ids"] = face_ids
            payload["identity_face_cluster_count"] = cluster_count
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
            "cluster_count": max(1, identity_face_count(char_map, identity_name)),
            "embedding": embedding,
            "priority": identity_has_priority(char_map, identity_name),
        }
        identities[identity_name]["identity_strength"] = known_face_identity_strength(identities[identity_name])
    return identities


def unknown_face_candidates(char_map: dict, include_background: bool = False) -> list[tuple[str, dict]]:
    candidates: list[tuple[str, dict]] = []
    for cluster_id, payload in sorted(char_map.get("clusters", {}).items(), key=cluster_sort_key):
        if is_ignored_face_payload(payload):
            continue
        name = str(payload.get("name", cluster_id))
        if not looks_auto_named(name) and not (include_background and is_background_person_name(name)):
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


def plan_known_face_matches(cfg: dict, char_map: dict, include_background: bool = False) -> dict[str, dict]:
    identities = known_face_reference_identities(char_map, cfg)
    if not identities:
        return {}

    match_cfg = known_face_match_config(cfg)
    threshold = float(match_cfg.get("threshold", 0.72))
    margin = float(match_cfg.get("margin", 0.05))
    min_consensus = int(match_cfg.get("min_consensus", 2))
    strong_match_threshold = float(match_cfg.get("strong_match_threshold", 0.84))
    matches: dict[str, dict] = {}
    for cluster_id, payload in unknown_face_candidates(char_map, include_background=include_background):
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
        preview_dir = portable_project_path(payload.get("preview_dir", ""))
        if preview_dir and preview_dir not in preview_dirs:
            preview_dirs.append(preview_dir)
        for extra_dir in payload.get("merged_preview_dirs", []) or []:
            extra_value = portable_project_path(extra_dir)
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
            write_json(linked_file, normalize_portable_project_paths(rows))
            changed_files += 1
    return changed_files


def auto_match_known_faces(cfg: dict, char_map: dict, voice_map: dict, include_background: bool | None = None) -> dict[str, object]:
    if include_background is None:
        include_background = bool(internet_lookup_config(cfg).get("match_background_faces", True))
    planned_matches = plan_known_face_matches(cfg, char_map, include_background=include_background)
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


def speaker_transcript_rows(cfg: dict) -> list[dict]:
    transcript_root = resolve_project_path(cfg["paths"].get("speaker_transcripts", "data/processed/speaker_transcripts"))
    rows: list[dict] = []
    if not transcript_root.exists():
        return rows
    for transcript_file in sorted(transcript_root.glob("*_segments.json")):
        payload = read_json(transcript_file, [])
        if isinstance(payload, list):
            rows.extend(payload)
    return rows


def ensure_voice_clusters_from_project_speakers(cfg: dict, voice_map: dict) -> int:
    clusters = voice_map.setdefault("clusters", {})
    voice_map.setdefault("aliases", {})
    added = 0
    for row in [*speaker_transcript_rows(cfg), *linked_segment_rows(cfg)]:
        speaker_cluster = str(row.get("speaker_cluster", "")).strip()
        if not speaker_cluster or speaker_cluster == "speaker_unknown":
            continue
        payload = clusters.setdefault(speaker_cluster, {})
        if not payload:
            added += 1
        if not payload.get("name"):
            payload["name"] = speaker_cluster
            payload["auto_named"] = True
        elif looks_auto_named(str(payload.get("name", ""))):
            payload["auto_named"] = True
    return added


def auto_link_speakers_from_single_visible_faces(cfg: dict, char_map: dict, voice_map: dict) -> dict[str, int]:
    face_cfg = cfg.get("character_detection", {}) if isinstance(cfg.get("character_detection"), dict) else {}
    direct_weight = float(face_cfg.get("speaker_face_cluster_vote_weight", 4.0) or 4.0)
    single_visible_weight = float(face_cfg.get("speaker_single_visible_vote_weight", 1.0) or 1.0)
    min_votes = float(face_cfg.get("speaker_face_link_min_votes", 3.0) or 3.0)
    min_share = float(face_cfg.get("speaker_face_link_min_share", 0.70) or 0.70)
    min_margin = float(face_cfg.get("speaker_face_link_min_margin", 2.0) or 2.0)
    speaker_votes: dict[str, dict[str, float]] = {}
    direct_counts: dict[str, dict[str, int]] = {}
    for row in linked_segment_rows(cfg):
        speaker_cluster = str(row.get("speaker_cluster", "")).strip()
        if not speaker_cluster:
            continue
        if speaker_cluster == "speaker_unknown":
            continue

        speaker_face_cluster = str(row.get("speaker_face_cluster", "") or "").strip()
        if speaker_face_cluster:
            payload = char_map.get("clusters", {}).get(speaker_face_cluster, {})
            name = str(payload.get("name", speaker_face_cluster))
            if not is_ignored_face_payload(payload) and has_manual_person_name(name) and not is_background_person_name(name):
                speaker_votes.setdefault(speaker_cluster, {})
                direct_counts.setdefault(speaker_cluster, {})
                speaker_votes[speaker_cluster][speaker_face_cluster] = speaker_votes[speaker_cluster].get(speaker_face_cluster, 0.0) + direct_weight
                direct_counts[speaker_cluster][speaker_face_cluster] = direct_counts[speaker_cluster].get(speaker_face_cluster, 0) + 1

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
        speaker_votes[speaker_cluster][only_cluster] = speaker_votes[speaker_cluster].get(only_cluster, 0.0) + single_visible_weight

    linked = 0
    for speaker_cluster, votes in speaker_votes.items():
        ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        if not ranked:
            continue
        best_cluster, best_votes = ranked[0]
        second_votes = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        total_votes = sum(votes.values())
        vote_share = (float(best_votes) / total_votes) if total_votes else 0.0
        if float(best_votes) < min_votes or vote_share < min_share or (float(best_votes) - second_votes) < min_margin:
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
        speaker_payload["speaker_face_link_evidence"] = {
            "best_votes": round(float(best_votes), 3),
            "total_votes": round(float(total_votes), 3),
            "share": round(vote_share, 4),
            "margin": round(float(best_votes) - second_votes, 3),
            "direct_face_cluster_rows": int(direct_counts.get(speaker_cluster, {}).get(best_cluster, 0)),
        }
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


def internet_lookup_config(cfg: dict) -> dict[str, object]:
    face_cfg = cfg.get("character_detection", {}) if isinstance(cfg.get("character_detection"), dict) else {}
    face_lookup_url_env = str(face_cfg.get("internet_face_lookup_url_env", "SERIES_FACE_LOOKUP_URL"))
    face_lookup_token_env = str(face_cfg.get("internet_face_lookup_token_env", "SERIES_FACE_LOOKUP_TOKEN"))
    return {
        "enabled": bool(face_cfg.get("internet_name_lookup", True)),
        "timeout_seconds": max(1.0, float(face_cfg.get("internet_name_lookup_timeout_seconds", 2.0) or 2.0)),
        "cache_days": max(0.0, float(face_cfg.get("internet_name_lookup_cache_days", 30.0) or 30.0)),
        "min_confidence": max(
            0.0,
            min(1.0, float(face_cfg.get("internet_name_lookup_min_confidence", 0.95) or 0.95)),
        ),
        "max_updates": max(0, int(face_cfg.get("internet_name_lookup_max_updates", 50) or 50)),
        "languages": face_cfg.get("internet_name_lookup_languages", ["en", "de"]),
        "context_terms": face_cfg.get("internet_name_lookup_context_terms", []),
        "match_background_faces": bool(face_cfg.get("review_match_background_faces", True)),
        "face_lookup_enabled": bool(face_cfg.get("internet_face_lookup", True)),
        "face_lookup_command": face_cfg.get("internet_face_lookup_command", ""),
        "face_lookup_env": str(face_cfg.get("internet_face_lookup_env", "SERIES_FACE_LOOKUP_COMMAND")),
        "face_lookup_url": str(face_cfg.get("internet_face_lookup_url", "") or os.environ.get(face_lookup_url_env, "")),
        "face_lookup_url_env": face_lookup_url_env,
        "face_lookup_token_env": face_lookup_token_env,
        "face_lookup_builtin_public_images": bool(face_cfg.get("internet_face_lookup_builtin_public_images", True)),
        "face_lookup_public_image_min_similarity": max(
            0.0,
            min(1.0, float(face_cfg.get("internet_face_lookup_public_image_min_similarity", 0.72) or 0.72)),
        ),
        "face_lookup_public_image_min_margin": max(
            0.0,
            min(1.0, float(face_cfg.get("internet_face_lookup_public_image_min_margin", 0.05) or 0.05)),
        ),
        "face_lookup_public_image_max_names": max(0, int(face_cfg.get("internet_face_lookup_public_image_max_names", 24) or 24)),
        "face_lookup_public_image_max_images_per_name": max(
            1,
            int(face_cfg.get("internet_face_lookup_public_image_max_images_per_name", 2) or 2),
        ),
        "face_lookup_public_image_max_seconds": max(
            1.0,
            float(face_cfg.get("internet_face_lookup_public_image_max_seconds", 30.0) or 30.0),
        ),
        "face_lookup_public_image_allow_slow_torch": bool(
            face_cfg.get("internet_face_lookup_public_image_allow_slow_torch_import", True)
        ),
        "face_lookup_public_image_cache": str(
            face_cfg.get("internet_face_lookup_public_image_cache", "runtime/internet_face_lookup_public_images") or
            "runtime/internet_face_lookup_public_images"
        ),
        "face_lookup_min_confidence": max(
            0.0,
            min(1.0, float(face_cfg.get("internet_face_lookup_min_confidence", 0.95) or 0.95)),
        ),
        "face_lookup_max_clusters": max(0, int(face_cfg.get("internet_face_lookup_max_clusters", 80) or 80)),
        "face_lookup_max_images": max(1, int(face_cfg.get("internet_face_lookup_max_images", 2) or 2)),
    }


def lookup_cache_path(cfg: dict) -> Path:
    configured = cfg.get("paths", {}).get("internet_name_lookup_cache", "runtime/internet_name_lookup_cache.json")
    return resolve_project_path(configured)


def text_tokens(text: object) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[^\W\d_](?:[^\W\d_]|[-'](?=[^\W\d_]))*", str(text or ""), flags=re.UNICODE)
    ]


def useful_context_tokens(text: object) -> list[str]:
    blocked = {
        "720p",
        "1080p",
        "2160p",
        "web",
        "h264",
        "h265",
        "x264",
        "x265",
        "german",
        "english",
        "deutsch",
        "episode",
        "folge",
        "season",
        "staffel",
        "serie",
        "series",
        "training",
        "ai",
        "ki",
    }
    tokens = [token for token in text_tokens(text) if len(token) >= 3 and token not in blocked and not token.startswith(("s0", "e0"))]
    return list(dict.fromkeys(tokens))


def project_series_context_terms(cfg: dict, limit: int = 10) -> list[str]:
    lookup_cfg = internet_lookup_config(cfg)
    configured_terms = lookup_cfg.get("context_terms", [])
    context_terms: list[str] = []
    if isinstance(configured_terms, str):
        context_terms.extend(useful_context_tokens(configured_terms))
    elif isinstance(configured_terms, list):
        for item in configured_terms:
            context_terms.extend(useful_context_tokens(item))

    for key in ("series_title", "title", "name", "project_name"):
        if key in cfg:
            context_terms.extend(useful_context_tokens(cfg.get(key)))

    generation_cfg = cfg.get("generation", {}) if isinstance(cfg.get("generation"), dict) else {}
    context_terms.extend(useful_context_tokens(generation_cfg.get("active_series_input", "")))

    for path_key in ("episodes", "inbox_episodes", "metadata"):
        configured_path = cfg.get("paths", {}).get(path_key)
        if not configured_path:
            continue
        root = resolve_project_path(configured_path)
        if not root.exists():
            continue
        for path in sorted(root.iterdir())[:20]:
            context_terms.extend(useful_context_tokens(path.stem if path.is_file() else path.name))

    collapsed: list[str] = []
    for token in context_terms:
        if token not in collapsed:
            collapsed.append(token)
        if len(collapsed) >= limit:
            break
    return collapsed


def normalize_lookup_languages(value: object) -> list[str]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        items = [str(item).strip() for item in value]
    else:
        items = []
    languages = []
    for item in items:
        language = item.lower()
        if re.match(r"^[a-z]{2,3}$", language) and language not in languages:
            languages.append(language)
    return languages or ["en", "de"]


def lookup_cache_key(name: str, context_terms: list[str], languages: list[str]) -> str:
    context = ",".join(sorted(context_terms[:8]))
    language_key = ",".join(languages)
    return f"{normalize_alias_name(name)}|{context}|{language_key}"


def request_json(url: str, timeout_seconds: float) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AI-Series-Training/1.0 public-name-metadata-lookup",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        import json

        return json.loads(response.read().decode("utf-8", errors="replace"))


def strip_html_tags(text: object) -> str:
    return re.sub(r"<[^>]+>", " ", str(text or ""))


def candidate_record(source: str, label: object, description: object = "", url: object = "", aliases: object = None) -> dict[str, object]:
    return {
        "source": source,
        "label": canonical_person_name(str(label or "")),
        "description": " ".join(str(description or "").split()),
        "url": str(url or ""),
        "aliases": [str(alias) for alias in aliases or [] if str(alias).strip()],
    }


def fetch_wikidata_name_candidates(query: str, language: str, timeout_seconds: float) -> list[dict[str, object]]:
    params = urllib.parse.urlencode(
        {
            "action": "wbsearchentities",
            "search": query,
            "language": language,
            "uselang": language,
            "format": "json",
            "limit": 8,
        }
    )
    payload = request_json(f"https://www.wikidata.org/w/api.php?{params}", timeout_seconds)
    candidates: list[dict[str, object]] = []
    for item in payload.get("search", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        label = item.get("label", "")
        if not label:
            continue
        entity_id = str(item.get("id", ""))
        url = f"https://www.wikidata.org/wiki/{entity_id}" if entity_id else str(item.get("concepturi", ""))
        candidates.append(candidate_record("wikidata", label, item.get("description", ""), url, item.get("aliases", [])))
    return candidates


def fetch_wikipedia_name_candidates(query: str, language: str, timeout_seconds: float) -> list[dict[str, object]]:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 8,
        }
    )
    payload = request_json(f"https://{language}.wikipedia.org/w/api.php?{params}", timeout_seconds)
    candidates: list[dict[str, object]] = []
    for item in payload.get("query", {}).get("search", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        title = item.get("title", "")
        if not title:
            continue
        page_id = item.get("pageid", "")
        url = f"https://{language}.wikipedia.org/?curid={page_id}" if page_id else ""
        candidates.append(candidate_record("wikipedia", title, strip_html_tags(item.get("snippet", "")), url))
    return candidates


def fandom_wiki_slugs(context_terms: list[str]) -> list[str]:
    if not context_terms:
        return []
    joined = "".join(token for token in context_terms[:4] if token.isalnum())
    hyphenated = "-".join(token for token in context_terms[:4] if token.isalnum())
    compact_pair = "".join(token for token in context_terms[:2] if token.isalnum())
    hyphen_pair = "-".join(token for token in context_terms[:2] if token.isalnum())
    slugs: list[str] = []
    for slug in (compact_pair, hyphen_pair, joined, hyphenated):
        cleaned = re.sub(r"[^a-z0-9-]+", "", slug.lower()).strip("-")
        if cleaned and cleaned not in slugs:
            slugs.append(cleaned)
    return slugs[:6]


def fetch_fandom_name_candidates(name: str, context_terms: list[str], timeout_seconds: float) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for slug in fandom_wiki_slugs(context_terms):
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": name,
                "format": "json",
                "srlimit": 8,
            }
        )
        try:
            payload = request_json(f"https://{slug}.fandom.com/api.php?{params}", timeout_seconds)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            continue
        for item in payload.get("query", {}).get("search", []) if isinstance(payload, dict) else []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            page_url = f"https://{slug}.fandom.com/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            description = f"{slug} fandom wiki {strip_html_tags(item.get('snippet', ''))}"
            candidates.append(candidate_record("fandom", title, description, page_url))
    return candidates


def score_internet_name_candidate(partial_name: str, candidate: dict, context_terms: list[str]) -> float:
    label = canonical_person_name(str(candidate.get("label", "")))
    if not label:
        return 0.0
    partial_tokens = text_tokens(partial_name)
    label_tokens = text_tokens(label)
    if not partial_tokens or not label_tokens:
        return 0.0
    label_text = normalize_alias_name(label)
    partial_text = normalize_alias_name(partial_name)
    aliases = " ".join(str(alias) for alias in candidate.get("aliases", []) or [])
    search_blob = f"{label} {candidate.get('description', '')} {aliases}".lower()
    score = 0.0

    matched_tokens = sum(1 for token in partial_tokens if token in label_tokens or token in search_blob)
    score += 0.42 * (matched_tokens / max(1, len(partial_tokens)))
    if partial_text == label_text:
        score += 0.12
    elif partial_text and partial_text in label_text:
        score += 0.22
    elif any(token and any(label_token.startswith(token) for label_token in label_tokens) for token in partial_tokens):
        score += 0.12
    if len(label_tokens) > len(partial_tokens):
        score += 0.16
    if len(partial_tokens) == 1 and len(label_tokens) == 2 and label_tokens[0] == partial_tokens[0]:
        score += 0.12

    descriptor_terms = {
        "character",
        "fictional",
        "television",
        "tv",
        "sitcom",
        "series",
        "actor",
        "actress",
        "cast",
        "rolle",
        "figur",
        "fernsehserie",
        "schauspieler",
        "schauspielerin",
    }
    descriptor_hits = sum(1 for token in descriptor_terms if token in search_blob)
    score += min(0.18, descriptor_hits * 0.045)
    context_hits = sum(1 for token in context_terms if token.lower() in search_blob)
    score += min(0.22, context_hits * 0.08)
    if "(" in label or ")" in label:
        score -= 0.05
    if "/" in label:
        score -= 0.25
    if "&" in label:
        score -= 0.22
    if re.search(r"\b(gets|loves|fake|boys|relationships|gallery|quotes|appearances)\b", label.lower()):
        score -= 0.16
    return round(max(0.0, min(1.0, score)), 4)


def internet_name_candidates(name: str, cfg: dict, *, force_refresh: bool = False) -> list[dict[str, object]]:
    lookup_cfg = internet_lookup_config(cfg)
    if not bool(lookup_cfg.get("enabled", True)):
        return []
    normalized_name = canonical_person_name(name)
    if not normalized_name or not has_manual_person_name(normalized_name) or is_background_person_name(normalized_name):
        return []
    context_terms = project_series_context_terms(cfg)
    languages = normalize_lookup_languages(lookup_cfg.get("languages", ["en", "de"]))
    cache_path = lookup_cache_path(cfg)
    cache = read_json(cache_path, {"queries": {}})
    if not isinstance(cache, dict):
        cache = {"queries": {}}
    queries = cache.setdefault("queries", {})
    cache_key = lookup_cache_key(normalized_name, context_terms, languages)
    now = time.time()
    cached = queries.get(cache_key) if isinstance(queries, dict) else None
    max_age_seconds = float(lookup_cfg.get("cache_days", 30.0) or 0.0) * 86400.0
    if (
        isinstance(cached, dict)
        and not force_refresh
        and max_age_seconds > 0
        and now - float(cached.get("fetched_at", 0.0) or 0.0) <= max_age_seconds
    ):
        return list(cached.get("candidates", []) or [])

    timeout_seconds = float(lookup_cfg.get("timeout_seconds", 5.0) or 5.0)
    query_terms = " ".join([normalized_name, *context_terms[:4]]).strip()
    raw_candidates: list[dict[str, object]] = []
    errors: list[str] = []
    for language in languages:
        for fetcher in (fetch_wikidata_name_candidates, fetch_wikipedia_name_candidates):
            try:
                raw_candidates.extend(fetcher(query_terms, language, timeout_seconds))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                errors.append(f"{language}:{fetcher.__name__}:{exc}")
    try:
        raw_candidates.extend(fetch_fandom_name_candidates(normalized_name, context_terms, timeout_seconds))
    except Exception as exc:
        errors.append(f"fandom:{exc}")

    deduped: dict[str, dict[str, object]] = {}
    for candidate in raw_candidates:
        label = canonical_person_name(str(candidate.get("label", "")))
        if not label:
            continue
        key = normalize_alias_name(label)
        score = score_internet_name_candidate(normalized_name, candidate, context_terms)
        if score <= 0.0:
            continue
        enriched = {**candidate, "label": label, "confidence": score, "query": normalized_name, "context_terms": context_terms}
        previous = deduped.get(key)
        if previous is None or float(enriched.get("confidence", 0.0)) > float(previous.get("confidence", 0.0)):
            deduped[key] = enriched

    candidates = sorted(deduped.values(), key=lambda item: (-float(item.get("confidence", 0.0)), str(item.get("label", ""))))[:8]
    queries[cache_key] = {
        "fetched_at": now,
        "query": normalized_name,
        "context_terms": context_terms,
        "languages": languages,
        "candidates": candidates,
        "errors": errors[:5],
    }
    write_json(cache_path, cache)
    return candidates


def best_internet_name_completion(name: str, cfg: dict, *, force_refresh: bool = False) -> dict[str, object] | None:
    candidates = internet_name_candidates(name, cfg, force_refresh=force_refresh)
    if not candidates:
        return None
    lookup_cfg = internet_lookup_config(cfg)
    min_confidence = float(lookup_cfg.get("min_confidence", 0.72) or 0.72)
    old_tokens = set(text_tokens(name))
    for candidate in candidates:
        label = canonical_person_name(str(candidate.get("label", "")))
        candidate_tokens = set(text_tokens(label))
        if not label or normalize_alias_name(label) == normalize_alias_name(name):
            continue
        if not old_tokens or not old_tokens.issubset(candidate_tokens):
            continue
        if len(candidate_tokens) <= len(old_tokens):
            continue
        if float(candidate.get("confidence", 0.0) or 0.0) >= min_confidence:
            return candidate
    return None


def rollback_low_confidence_internet_names(cfg: dict, char_map: dict) -> dict[str, object]:
    lookup_cfg = internet_lookup_config(cfg)
    min_confidence = float(lookup_cfg.get("min_confidence", 0.95) or 0.95)
    restored: list[dict[str, object]] = []
    cleared_history: list[dict[str, object]] = []
    for cluster_id, payload in list(char_map.get("clusters", {}).items()):
        meta = payload.get("internet_name_lookup", {}) if isinstance(payload.get("internet_name_lookup", {}), dict) else {}
        confidence = float(meta.get("confidence", 0.0) or 0.0)
        previous_name = canonical_person_name(str(meta.get("previous_name", "")))
        resolved_name = canonical_person_name(str(meta.get("resolved_name", "")))
        current_name = canonical_person_name(str(payload.get("name", cluster_id)))
        if not previous_name or not resolved_name:
            continue
        if confidence >= min_confidence:
            continue
        if normalize_alias_name(current_name) == normalize_alias_name(previous_name):
            payload.setdefault("internet_name_lookup_rejected", []).append(
                {
                    "previous_name": previous_name,
                    "rejected_name": resolved_name,
                    "confidence": confidence,
                    "required_confidence": min_confidence,
                    "source": meta.get("source", ""),
                    "url": meta.get("url", ""),
                }
            )
            payload.pop("internet_name_lookup", None)
            cleared_history.append(
                {
                    "cluster_id": cluster_id,
                    "current_name": current_name,
                    "rejected_name": resolved_name,
                    "confidence": confidence,
                }
            )
            continue
        if normalize_alias_name(current_name) != normalize_alias_name(resolved_name):
            continue
        priority = bool(payload.get("priority", False))
        assign_character_name(char_map, cluster_id, previous_name, priority=priority)
        updated_payload = char_map.get("clusters", {}).get(cluster_id, {})
        updated_payload["internet_name_lookup_rollback"] = {
            "restored_name": previous_name,
            "rejected_name": resolved_name,
            "confidence": confidence,
            "required_confidence": min_confidence,
        }
        restored.append(
            {
                "cluster_id": cluster_id,
                "restored_name": previous_name,
                "rejected_name": resolved_name,
                "confidence": confidence,
            }
        )
    if restored or cleared_history:
        rebuild_character_map_identities(char_map)
    return {"restored": len(restored), "items": restored, "cleared_history": len(cleared_history), "cleared_items": cleared_history}


def add_name_alias(char_map: dict, cluster_id: str, alias_name: str) -> None:
    alias = normalize_alias_name(alias_name)
    if not alias or is_background_person_name(alias) or is_ignored_face_name(alias):
        return
    payload = char_map.get("clusters", {}).get(cluster_id)
    if payload is None:
        return
    aliases = list(payload.get("aliases", []) or [])
    if alias not in aliases:
        aliases.append(alias)
    payload["aliases"] = aliases
    char_map.setdefault("aliases", {})[alias] = cluster_id


def rename_identity_everywhere(char_map: dict, old_name: str, new_name: str, source_meta: dict | None = None) -> int:
    changed = 0
    for cluster_id, payload in list(char_map.get("clusters", {}).items()):
        current_name = canonical_person_name(str(payload.get("name", cluster_id)))
        if normalize_alias_name(current_name) != normalize_alias_name(old_name):
            continue
        priority = bool(payload.get("priority", False))
        assign_character_name(char_map, cluster_id, new_name, priority=priority)
        add_name_alias(char_map, cluster_id, old_name)
        if source_meta:
            payload = char_map.get("clusters", {}).get(cluster_id, {})
            payload["internet_name_lookup"] = {
                "previous_name": old_name,
                "resolved_name": new_name,
                "confidence": float(source_meta.get("confidence", 0.0) or 0.0),
                "source": source_meta.get("source", ""),
                "url": source_meta.get("url", ""),
            }
        changed += 1
    return changed


def enrich_existing_character_names_from_internet(cfg: dict, char_map: dict, *, force_refresh: bool = False) -> dict[str, object]:
    lookup_cfg = internet_lookup_config(cfg)
    if not bool(lookup_cfg.get("enabled", True)):
        return {"checked": 0, "renamed": 0, "updates": [], "disabled": True}
    names = []
    for _cluster_id, payload in sorted(char_map.get("clusters", {}).items(), key=cluster_sort_key):
        if is_ignored_face_payload(payload):
            continue
        name = canonical_person_name(str(payload.get("name", "")))
        if not name or is_background_person_name(name) or not has_manual_person_name(name):
            continue
        if normalize_alias_name(name) not in [normalize_alias_name(item) for item in names]:
            names.append(name)

    updates: list[dict[str, object]] = []
    max_updates = int(lookup_cfg.get("max_updates", 50) or 50)
    checked = 0
    for name in names:
        checked += 1
        candidate = best_internet_name_completion(name, cfg, force_refresh=force_refresh)
        if not candidate:
            continue
        new_name = canonical_person_name(str(candidate.get("label", "")))
        changed = rename_identity_everywhere(char_map, name, new_name, candidate)
        if changed:
            updates.append(
                {
                    "old_name": name,
                    "new_name": new_name,
                    "clusters": changed,
                    "confidence": float(candidate.get("confidence", 0.0) or 0.0),
                    "source": candidate.get("source", ""),
                    "url": candidate.get("url", ""),
                }
            )
        if max_updates > 0 and len(updates) >= max_updates:
            break
    if updates:
        rebuild_character_map_identities(char_map)
    return {"checked": checked, "renamed": len(updates), "updates": updates}


def face_lookup_command_template(cfg: dict) -> list[str]:
    lookup_cfg = internet_lookup_config(cfg)
    configured = lookup_cfg.get("face_lookup_command", "")
    env_name = str(lookup_cfg.get("face_lookup_env", "SERIES_FACE_LOOKUP_COMMAND"))
    if not configured and env_name:
        configured = os.environ.get(env_name, "")
    if isinstance(configured, list):
        return [str(item) for item in configured if str(item).strip()]
    if isinstance(configured, str) and configured.strip():
        return shlex.split(configured.strip(), posix=(current_os() != "windows"))
    return []


def render_face_lookup_command(template: list[str], image_path: Path, cluster_id: str) -> list[str]:
    replacements = {
        "image_path": str(image_path),
        "cluster_id": cluster_id,
        "project_root": str(PROJECT_DIR),
    }
    return [part.format(**replacements) for part in template]


def normalize_face_lookup_matches(matches: object, image_path: Path, default_source: str) -> list[dict[str, object]]:
    if not isinstance(matches, list):
        return []
    normalized: list[dict[str, object]] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        label = canonical_person_name(str(match.get("name") or match.get("label") or ""))
        if not label:
            continue
        normalized.append(
            {
                "label": label,
                "confidence": float(match.get("confidence", 0.0) or 0.0),
                "source": str(match.get("source", default_source)),
                "url": str(match.get("url", "")),
                "image": str(image_path),
            }
        )
    normalized.sort(key=lambda item: (-float(item.get("confidence", 0.0)), str(item.get("label", ""))))
    return normalized


def run_online_face_lookup_command(template: list[str], image_path: Path, cluster_id: str, cfg: dict) -> list[dict[str, object]]:
    lookup_cfg = internet_lookup_config(cfg)
    command = render_face_lookup_command(template, image_path, cluster_id)
    if not command:
        return []
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=float(lookup_cfg.get("timeout_seconds", 5.0) or 5.0),
            check=False,
        )
    except Exception as exc:
        warn(f"Online face lookup failed for {cluster_id}: {exc}. Continuing offline.")
        return []
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        warn(f"Online face lookup returned exit code {completed.returncode} for {cluster_id}. {detail[:220]}")
        return []
    try:
        import json

        payload = json.loads(completed.stdout or "{}")
    except Exception as exc:
        warn(f"Online face lookup returned invalid JSON for {cluster_id}: {exc}. Continuing offline.")
        return []
    if not isinstance(payload, dict):
        return []
    return normalize_face_lookup_matches(payload.get("matches", []), image_path, "online_face_lookup_command")


def image_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def run_online_face_lookup_http(url: str, image_path: Path, cluster_id: str, cfg: dict) -> list[dict[str, object]]:
    lookup_cfg = internet_lookup_config(cfg)
    if not url:
        return []
    try:
        import json

        image_bytes = image_path.read_bytes()
        request_body = {
            "cluster_id": cluster_id,
            "filename": image_path.name,
            "mime_type": image_mime_type(image_path),
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            "project_root": str(PROJECT_DIR),
        }
        data = json.dumps(request_body).encode("utf-8")
        headers = {
            "User-Agent": "AI-Series-Training/1.0 online-face-lookup",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        token_env = str(lookup_cfg.get("face_lookup_token_env", "SERIES_FACE_LOOKUP_TOKEN"))
        token = os.environ.get(token_env, "").strip() if token_env else ""
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=float(lookup_cfg.get("timeout_seconds", 5.0) or 5.0)) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        warn(f"Online face lookup API failed for {cluster_id}: {exc}. Continuing offline.")
        return []
    if not isinstance(payload, dict):
        return []
    return normalize_face_lookup_matches(payload.get("matches", []), image_path, "online_face_lookup_api")


def safe_cache_stem(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("._-")[:90] or "item"


def public_face_lookup_cache_root(cfg: dict) -> Path:
    lookup_cfg = internet_lookup_config(cfg)
    return resolve_project_path(str(lookup_cfg.get("face_lookup_public_image_cache", "runtime/internet_face_lookup_public_images")))


def project_facenet_checkpoint_path() -> Path:
    return resolve_project_path("runtime/models/torch/checkpoints/20180402-114759-vggface2.pt")


def ensure_project_facenet_checkpoint_available() -> bool:
    project_checkpoint = project_facenet_checkpoint_path()
    if project_checkpoint.exists() and project_checkpoint.stat().st_size > 0:
        return True
    user_checkpoint = Path.home() / ".cache" / "torch" / "checkpoints" / project_checkpoint.name
    if user_checkpoint.exists() and user_checkpoint.stat().st_size > 0:
        try:
            project_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(user_checkpoint, project_checkpoint)
            return True
        except Exception:
            return False
    return False


def public_face_lookup_reference_names(cfg: dict, char_map: dict) -> list[str]:
    lookup_cfg = internet_lookup_config(cfg)
    max_names = int(lookup_cfg.get("face_lookup_public_image_max_names", 24) or 24)
    counts: dict[str, int] = {}
    for _cluster_id, payload in sorted(char_map.get("clusters", {}).items(), key=cluster_sort_key):
        if is_ignored_face_payload(payload):
            continue
        name = canonical_person_name(str(payload.get("identity_name") or payload.get("name") or ""))
        if not name or looks_auto_named(name) or is_background_person_name(name) or not has_manual_person_name(name):
            continue
        counts[name] = counts.get(name, 0) + int(payload.get("samples", 1) or 1)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [name for name, _count in ranked[:max_names]]


def public_image_urls_from_candidate(candidate: dict, cfg: dict) -> list[str]:
    lookup_cfg = internet_lookup_config(cfg)
    timeout_seconds = float(lookup_cfg.get("timeout_seconds", 5.0) or 5.0)
    candidate_url = str(candidate.get("url", "") or "")
    label = canonical_person_name(str(candidate.get("label", "")))
    urls: list[str] = []
    try:
        parsed = urllib.parse.urlparse(candidate_url)
    except Exception:
        parsed = urllib.parse.urlparse("")

    try:
        if parsed.netloc.endswith(".fandom.com"):
            title = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1]).replace("_", " ").strip() or label
            params = urllib.parse.urlencode(
                {
                    "action": "query",
                    "titles": title,
                    "prop": "pageimages",
                    "pithumbsize": 900,
                    "format": "json",
                }
            )
            payload = request_json(f"{parsed.scheme or 'https'}://{parsed.netloc}/api.php?{params}", timeout_seconds)
            pages = payload.get("query", {}).get("pages", {}) if isinstance(payload, dict) else {}
            for page in pages.values() if isinstance(pages, dict) else []:
                thumb = page.get("thumbnail", {}) if isinstance(page, dict) else {}
                source = str(thumb.get("source", "") or "")
                if source:
                    urls.append(source)
        elif "wikipedia.org" in parsed.netloc:
            language = parsed.netloc.split(".", 1)[0] if "." in parsed.netloc else "en"
            query: dict[str, object] = {
                "action": "query",
                "prop": "pageimages",
                "pithumbsize": 900,
                "format": "json",
            }
            qs = urllib.parse.parse_qs(parsed.query)
            if qs.get("curid"):
                query["pageids"] = qs["curid"][0]
            else:
                title = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1]).replace("_", " ").strip() or label
                query["titles"] = title
            payload = request_json(f"https://{language}.wikipedia.org/w/api.php?{urllib.parse.urlencode(query)}", timeout_seconds)
            pages = payload.get("query", {}).get("pages", {}) if isinstance(payload, dict) else {}
            for page in pages.values() if isinstance(pages, dict) else []:
                thumb = page.get("thumbnail", {}) if isinstance(page, dict) else {}
                source = str(thumb.get("source", "") or "")
                if source:
                    urls.append(source)
        elif parsed.netloc == "www.wikidata.org":
            entity_id = parsed.path.rsplit("/", 1)[-1].strip()
            if entity_id:
                params = urllib.parse.urlencode(
                    {
                        "action": "wbgetentities",
                        "ids": entity_id,
                        "props": "claims",
                        "format": "json",
                    }
                )
                payload = request_json(f"https://www.wikidata.org/w/api.php?{params}", timeout_seconds)
                entity = payload.get("entities", {}).get(entity_id, {}) if isinstance(payload, dict) else {}
                claims = entity.get("claims", {}) if isinstance(entity, dict) else {}
                image_claims = claims.get("P18", []) if isinstance(claims, dict) else []
                for claim in image_claims[:2]:
                    value = (
                        claim.get("mainsnak", {})
                        .get("datavalue", {})
                        .get("value", "")
                        if isinstance(claim, dict)
                        else ""
                    )
                    if value:
                        urls.append(f"https://commons.wikimedia.org/wiki/Special:FilePath/{urllib.parse.quote(str(value))}")
    except Exception:
        return []
    deduped: list[str] = []
    for url in urls:
        if url and url not in deduped:
            deduped.append(url)
    return deduped


def download_public_face_image(url: str, target: Path, cfg: dict) -> Path | None:
    lookup_cfg = internet_lookup_config(cfg)
    if target.exists() and target.stat().st_size > 0:
        return target
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "AI-Series-Training/1.0 public-face-image-lookup",
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(request, timeout=float(lookup_cfg.get("timeout_seconds", 5.0) or 5.0)) as response:
            data = response.read(8 * 1024 * 1024)
        if not data:
            return None
        target.write_bytes(data)
        return target
    except Exception:
        return None


def public_image_embeddings_for_name(name: str, cfg: dict, *, deadline: float | None = None) -> list[dict[str, object]]:
    if deadline is not None and time.monotonic() >= deadline:
        return []
    lookup_cfg = internet_lookup_config(cfg)
    max_images = int(lookup_cfg.get("face_lookup_public_image_max_images_per_name", 2) or 2)
    cache_root = public_face_lookup_cache_root(cfg)
    image_root = cache_root / safe_cache_stem(normalize_alias_name(name))
    candidates = internet_name_candidates(name, cfg)
    image_urls: list[str] = []
    for candidate in candidates[:8]:
        label = canonical_person_name(str(candidate.get("label", "")))
        if label and normalize_alias_name(name) not in normalize_alias_name(label) and normalize_alias_name(label) not in normalize_alias_name(name):
            continue
        for url in public_image_urls_from_candidate(candidate, cfg):
            if url not in image_urls:
                image_urls.append(url)
            if len(image_urls) >= max_images:
                break
        if len(image_urls) >= max_images:
            break
    image_paths: list[Path] = []
    source_urls: list[str] = []
    for index, url in enumerate(image_urls[:max_images], start=1):
        if deadline is not None and time.monotonic() >= deadline:
            break
        suffix = Path(urllib.parse.urlparse(url).path).suffix
        if suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        target = image_root / f"public_{index:02d}{suffix}"
        downloaded = download_public_face_image(url, target, cfg)
        if downloaded:
            image_paths.append(downloaded)
            source_urls.append(url)
    if not image_paths:
        return []
    lookup_cfg = internet_lookup_config(cfg)
    if not bool(lookup_cfg.get("face_lookup_public_image_allow_slow_torch", False)) and "torch" not in sys.modules:
        warn(
            "Built-in public-image face lookup found public images but skipped FaceNet comparison because Torch "
            "is not already loaded and deep public-image matching was disabled in config."
        )
        return []
    if not ensure_project_facenet_checkpoint_available():
        warn(
            "Built-in public-image face lookup skipped embedding comparison because the FaceNet "
            "checkpoint is not project-local yet. Prepare/copy the FaceNet vggface2 checkpoint into "
            f"{project_facenet_checkpoint_path()}."
        )
        return []

    torch_home = str(resolve_project_path("runtime/models/torch"))
    previous_torch_home = os.environ.get("TORCH_HOME")
    os.environ.setdefault("TORCH_HOME", torch_home)
    try:
        embedding_payload = character_appearance_embedding(image_paths)
    finally:
        if previous_torch_home is None:
            os.environ.pop("TORCH_HOME", None)
        else:
            os.environ["TORCH_HOME"] = previous_torch_home
    embeddings = embedding_payload.get("embeddings", []) if isinstance(embedding_payload, dict) else []
    rows: list[dict[str, object]] = []
    for image_path, source_url, embedding in zip(image_paths, source_urls, embeddings):
        if embedding:
            rows.append({"label": name, "embedding": embedding, "image": str(image_path), "url": source_url})
    return rows


def public_face_lookup_bank(cfg: dict, char_map: dict) -> list[dict[str, object]]:
    lookup_cfg = internet_lookup_config(cfg)
    if not bool(lookup_cfg.get("face_lookup_builtin_public_images", True)):
        return []
    if not bool(lookup_cfg.get("face_lookup_public_image_allow_slow_torch", False)) and "torch" not in sys.modules:
        warn(
            "Built-in public-image face lookup is enabled, but Torch/FaceNet is not already loaded. "
            "Skipping the heavy local image comparison because deep public-image matching was disabled in config."
        )
        return []
    deadline = time.monotonic() + float(lookup_cfg.get("face_lookup_public_image_max_seconds", 30.0) or 30.0)
    bank: list[dict[str, object]] = []
    for name in public_face_lookup_reference_names(cfg, char_map):
        if time.monotonic() >= deadline:
            break
        bank.extend(public_image_embeddings_for_name(name, cfg, deadline=deadline))
    return bank


def run_builtin_public_image_face_lookup(payload: dict, public_bank: list[dict[str, object]], cfg: dict) -> list[dict[str, object]]:
    embedding = payload.get("embedding") or []
    if not embedding or not public_bank:
        return []
    lookup_cfg = internet_lookup_config(cfg)
    min_similarity = float(lookup_cfg.get("face_lookup_public_image_min_similarity", 0.72) or 0.72)
    min_margin = float(lookup_cfg.get("face_lookup_public_image_min_margin", 0.05) or 0.05)
    scored: list[dict[str, object]] = []
    for reference in public_bank:
        reference_embedding = reference.get("embedding") or []
        if not reference_embedding:
            continue
        similarity = cosine_similarity(embedding, reference_embedding)
        scored.append({**reference, "similarity": similarity})
    scored.sort(key=lambda item: (-float(item.get("similarity", 0.0)), str(item.get("label", ""))))
    if not scored:
        return []
    best = scored[0]
    second = float(scored[1].get("similarity", 0.0)) if len(scored) > 1 else 0.0
    best_similarity = float(best.get("similarity", 0.0))
    if best_similarity < min_similarity or (best_similarity - second) < min_margin:
        return []
    return [
        {
            "label": canonical_person_name(str(best.get("label", ""))),
            "confidence": 1.0,
            "source": "builtin_public_image_embedding",
            "url": str(best.get("url", "")),
            "image": str(best.get("image", "")),
            "similarity": round(best_similarity, 4),
            "margin": round(best_similarity - second, 4),
        }
    ]


def online_face_lookup_candidates(
    cfg: dict,
    cluster_id: str,
    payload: dict,
    *,
    offline: bool = False,
    public_bank: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    lookup_cfg = internet_lookup_config(cfg)
    if offline or not bool(lookup_cfg.get("face_lookup_enabled", True)):
        return []
    command_template = face_lookup_command_template(cfg)
    api_url = str(lookup_cfg.get("face_lookup_url", "") or "").strip()
    public_bank = public_bank or []
    if not command_template and not api_url and not public_bank:
        return []
    max_images = int(lookup_cfg.get("face_lookup_max_images", 2) or 2)
    candidates: list[dict[str, object]] = []
    for image_path in preview_crop_paths(payload, max_images):
        if command_template:
            candidates.extend(run_online_face_lookup_command(command_template, image_path, cluster_id, cfg))
        if api_url:
            candidates.extend(run_online_face_lookup_http(api_url, image_path, cluster_id, cfg))
    candidates.extend(run_builtin_public_image_face_lookup(payload, public_bank, cfg))
    deduped: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        label = canonical_person_name(str(candidate.get("label", "")))
        if not label or not has_manual_person_name(label) or is_background_person_name(label):
            continue
        key = normalize_alias_name(label)
        previous = deduped.get(key)
        if previous is None or float(candidate.get("confidence", 0.0) or 0.0) > float(previous.get("confidence", 0.0) or 0.0):
            deduped[key] = {**candidate, "label": label}
    return sorted(deduped.values(), key=lambda item: (-float(item.get("confidence", 0.0)), str(item.get("label", ""))))


def apply_online_face_lookup(cfg: dict, char_map: dict, *, offline: bool = False) -> dict[str, object]:
    lookup_cfg = internet_lookup_config(cfg)
    if offline or not bool(lookup_cfg.get("face_lookup_enabled", True)):
        return {"checked": 0, "assigned": 0, "updates": [], "disabled": True}
    command_template = face_lookup_command_template(cfg)
    api_url = str(lookup_cfg.get("face_lookup_url", "") or "").strip()
    builtin_enabled = bool(lookup_cfg.get("face_lookup_builtin_public_images", True))
    if not command_template and not api_url and not builtin_enabled:
        return {"checked": 0, "assigned": 0, "updates": [], "missing_lookup_backend": True}

    min_confidence = float(lookup_cfg.get("face_lookup_min_confidence", 0.95) or 0.95)
    max_clusters = int(lookup_cfg.get("face_lookup_max_clusters", 80) or 80)
    candidates = unknown_face_candidates(
        char_map,
        include_background=bool(lookup_cfg.get("match_background_faces", True)),
    )
    if max_clusters > 0:
        candidates = candidates[:max_clusters]
    public_bank: list[dict[str, object]] = []
    if builtin_enabled and candidates:
        try:
            public_bank = public_face_lookup_bank(cfg, char_map)
        except Exception as exc:
            warn(f"Built-in public-image face lookup failed to prepare reference images: {exc}. Continuing offline.")
            public_bank = []
    if not command_template and not api_url and not public_bank:
        return {
            "checked": 0,
            "assigned": 0,
            "updates": [],
            "builtin_public_images": bool(builtin_enabled),
            "builtin_reference_images": 0,
        }
    updates: list[dict[str, object]] = []
    checked = 0
    for cluster_id, payload in candidates:
        checked += 1
        matches = online_face_lookup_candidates(cfg, cluster_id, payload, offline=offline, public_bank=public_bank)
        if not matches:
            continue
        best = matches[0]
        if float(best.get("confidence", 0.0) or 0.0) < min_confidence:
            continue
        assigned_name = canonical_person_name(str(best.get("label", "")))
        assign_character_name(char_map, cluster_id, assigned_name, priority=None)
        char_map["clusters"][cluster_id]["internet_face_lookup"] = {
            "resolved_name": assigned_name,
            "confidence": float(best.get("confidence", 0.0) or 0.0),
            "source": best.get("source", ""),
            "url": best.get("url", ""),
            "image": best.get("image", ""),
        }
        updates.append(
            {
                "cluster_id": cluster_id,
                "name": assigned_name,
                "confidence": float(best.get("confidence", 0.0) or 0.0),
                "source": best.get("source", ""),
            }
        )
    if updates:
        rebuild_character_map_identities(char_map)
    return {
        "checked": checked,
        "assigned": len(updates),
        "updates": updates,
        "builtin_public_images": bool(builtin_enabled),
        "builtin_reference_images": len(public_bank),
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
            write_json(linked_file, normalize_portable_project_paths(rows))
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
    write_json(resolve_project_path(cfg["paths"]["review_queue"]), normalize_portable_project_paths({"items": items}))
    return len(items)


def persist_updates(cfg: dict, char_map: dict, voice_map: dict) -> tuple[int, int]:
    rebuild_character_map_identities(char_map)
    refresh_voice_map(char_map, voice_map)
    write_json(resolve_project_path(cfg["paths"]["character_map"]), normalize_portable_project_paths(char_map))
    write_json(resolve_project_path(cfg["paths"]["voice_map"]), normalize_portable_project_paths(voice_map))
    changed_linked_files = refresh_linked_segments(cfg, char_map, voice_map)
    review_count = rebuild_review_queue(cfg)
    return changed_linked_files, review_count


def persist_maps_only(cfg: dict, char_map: dict, voice_map: dict) -> int:
    rebuild_character_map_identities(char_map)
    refresh_voice_map(char_map, voice_map)
    write_json(resolve_project_path(cfg["paths"]["character_map"]), normalize_portable_project_paths(char_map))
    write_json(resolve_project_path(cfg["paths"]["voice_map"]), normalize_portable_project_paths(voice_map))
    return open_review_item_count(cfg)


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


def name_editor_rows(char_map: dict, voice_map: dict) -> list[dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for cluster_id, payload in char_map.get("clusters", {}).items():
        name = canonical_person_name(str(payload.get("name", cluster_id)))
        if not name or looks_auto_named(name) or is_ignored_face_payload(payload):
            continue
        row = rows.setdefault(name, {"name": name, "faces": 0, "speakers": 0, "priority": False})
        row["faces"] = int(row.get("faces", 0)) + 1
        row["priority"] = bool(row.get("priority")) or bool(payload.get("priority", False))
    for speaker_id, payload in voice_map.get("clusters", {}).items():
        name = canonical_person_name(str(payload.get("name", speaker_id)))
        if not name or looks_auto_named(name) or is_background_person_name(name):
            continue
        row = rows.setdefault(name, {"name": name, "faces": 0, "speakers": 0, "priority": False})
        row["speakers"] = int(row.get("speakers", 0)) + 1
    return sorted(rows.values(), key=lambda row: str(row.get("name", "")).lower())


def rename_name_everywhere(char_map: dict, voice_map: dict, old_name: str, new_name: str, *, priority: bool | None = None) -> dict[str, int]:
    old_final = canonical_person_name(old_name)
    new_final = canonical_person_name(new_name)
    if not old_final or not new_final:
        raise ValueError("Both old and new names are required.")
    old_norm = normalize_alias_name(old_final)
    faces = 0
    speakers = 0
    for cluster_id, payload in list(char_map.get("clusters", {}).items()):
        current = canonical_person_name(str(payload.get("name", cluster_id)))
        if normalize_alias_name(current) != old_norm:
            continue
        assign_character_name(
            char_map,
            cluster_id,
            new_final,
            priority=priority if priority is not None else bool(payload.get("priority", False)),
        )
        add_name_alias(char_map, cluster_id, old_final)
        payload = char_map.get("clusters", {}).get(cluster_id, {})
        payload["manual_name_edit"] = {"previous_name": old_final, "new_name": new_final}
        faces += 1
    for speaker_id, payload in list(voice_map.get("clusters", {}).items()):
        current = canonical_person_name(str(payload.get("name", speaker_id)))
        if normalize_alias_name(current) != old_norm:
            continue
        payload["name"] = new_final
        payload["auto_named"] = False
        payload["manual_name_edit"] = {"previous_name": old_final, "new_name": new_final}
        speakers += 1
    if faces:
        rebuild_character_map_identities(char_map)
    return {"faces": faces, "speakers": speakers}


def open_name_editor_gui(cfg: dict, char_map: dict, voice_map: dict) -> bool:
    if not gui_preview_available():
        warn("Name editor GUI is not available in this session. Use a local desktop session or --rename-face.")
        return False
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as exc:
        warn(f"Name editor GUI could not start: {exc}")
        return False

    changed = {"value": False}
    root = tk.Tk()
    root.title("AI Series Training - Name Editor")
    root.geometry("840x560")

    frame = ttk.Frame(root, padding=10)
    frame.pack(fill="both", expand=True)
    ttk.Label(
        frame,
        text="Select an existing face/speaker name, enter the corrected name, then save.",
        font=("Segoe UI", 10, "bold"),
    ).pack(anchor="w", pady=(0, 8))

    tree = ttk.Treeview(frame, columns=("name", "faces", "speakers", "priority"), show="headings", height=16)
    for column, label, width in (
        ("name", "Name", 360),
        ("faces", "Face clusters", 110),
        ("speakers", "Speaker entries", 120),
        ("priority", "Main", 70),
    ):
        tree.heading(column, text=label)
        tree.column(column, width=width, anchor="w")
    tree.pack(fill="both", expand=True)

    edit_frame = ttk.Frame(frame)
    edit_frame.pack(fill="x", pady=(10, 0))
    old_var = tk.StringVar()
    new_var = tk.StringVar()
    priority_var = tk.BooleanVar(value=False)
    ttk.Label(edit_frame, text="Selected").grid(row=0, column=0, sticky="w", padx=(0, 6))
    ttk.Entry(edit_frame, textvariable=old_var, state="readonly", width=34).grid(row=0, column=1, sticky="ew", padx=(0, 10))
    ttk.Label(edit_frame, text="New name").grid(row=0, column=2, sticky="w", padx=(0, 6))
    new_entry = ttk.Entry(edit_frame, textvariable=new_var, width=34)
    new_entry.grid(row=0, column=3, sticky="ew")
    ttk.Checkbutton(edit_frame, text="Main character", variable=priority_var).grid(row=1, column=3, sticky="w", pady=(8, 0))
    edit_frame.grid_columnconfigure(1, weight=1)
    edit_frame.grid_columnconfigure(3, weight=1)

    def refresh() -> None:
        tree.delete(*tree.get_children())
        for row in name_editor_rows(char_map, voice_map):
            tree.insert("", "end", values=(row.get("name", ""), row.get("faces", 0), row.get("speakers", 0), "yes" if row.get("priority") else ""))

    def selected_name() -> str:
        selection = tree.selection()
        if not selection:
            return ""
        values = tree.item(selection[0], "values")
        return str(values[0]) if values else ""

    def on_select(_event=None) -> None:
        name = selected_name()
        old_var.set(name)
        new_var.set(name)
        priority_var.set(any(row.get("name") == name and row.get("priority") for row in name_editor_rows(char_map, voice_map)))
        new_entry.focus_set()
        new_entry.selection_range(0, "end")

    def save_selected() -> None:
        old_name = old_var.get().strip()
        new_name = new_var.get().strip()
        if not old_name or not new_name:
            messagebox.showwarning("Missing name", "Select a row and enter a new name.")
            return
        if normalize_alias_name(old_name) == normalize_alias_name(new_name):
            messagebox.showinfo("No change", "The name is unchanged.")
            return
        try:
            summary = rename_name_everywhere(char_map, voice_map, old_name, new_name, priority=priority_var.get())
            changed_linked_files, review_count = persist_updates(cfg, char_map, voice_map)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        changed["value"] = True
        refresh()
        old_var.set(new_name)
        new_var.set(new_name)
        messagebox.showinfo(
            "Saved",
            f"{old_name} -> {new_name}\n"
            f"Faces: {summary['faces']} | Speakers: {summary['speakers']}\n"
            f"Linked files updated: {changed_linked_files}\n"
            f"Open review cases: {review_count}",
        )

    button_frame = ttk.Frame(frame)
    button_frame.pack(fill="x", pady=(10, 0))
    ttk.Button(button_frame, text="Save rename", command=save_selected).pack(side="left")
    ttk.Button(button_frame, text="Refresh", command=refresh).pack(side="left", padx=(8, 0))
    ttk.Button(button_frame, text="Close", command=root.destroy).pack(side="right")

    tree.bind("<<TreeviewSelect>>", on_select)
    new_entry.bind("<Return>", lambda _event: save_selected())
    refresh()
    root.mainloop()
    return bool(changed["value"])


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
    display_state = print_interactive_display_diagnostics(
        "05_review_unknowns.py",
        require_gui=open_previews,
    )

    skipped_clusters: set[str] = set()
    handled_count = 0
    candidates = session_face_review_candidates(char_map, include_named, limit, handled_count, skipped_clusters)
    if not candidates:
        remaining_queue_count = open_review_item_count(cfg)
        if remaining_queue_count > 0:
            info(
                f"No face clusters were found for review. "
                f"There are still {remaining_queue_count} speaker/segment review cases in review_queue.json; "
                "use --show-queue for a summary, assign the repeated speaker/face IDs first, "
                "or use --edit-names to correct existing names in a GUI."
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
            source_preview_targets = preview_open_targets(cluster_id, payload)
            local_preview_bundle = materialize_local_preview_bundle(cluster_id, payload)
            preview_targets = list(local_preview_bundle.get("open_targets", []) or [])
            preview_window_image = local_preview_bundle.get("preview_window_image")
            if not isinstance(preview_window_image, Path):
                preview_window_image = preview_targets[0] if preview_targets else None
            print()
            print("-" * 72)
            print_cluster(char_map, cluster_id, payload)
            print(f"Remaining in this session including current case: {session_remaining_count}")
            print(f"Total actually still open: {total_open_count}")
            print(f"Automatic role hint: {role_hint}")
            print(f"Automatic review hint: {action_hint}")
            if preview_targets:
                print("Local launch targets:")
                for preview_target in preview_targets:
                    print(f"Local preview target: {preview_target}")
                    print(f"Local open link: {terminal_clickable_path(preview_target)}")
            montage = local_preview_bundle.get("montage")
            if not isinstance(montage, Path):
                montage = create_face_review_sheet(cluster_id, payload)
            if montage and montage.exists():
                print(f"Contact sheet: {montage}")
                print(f"Open contact sheet link: {terminal_clickable_path(montage)}")
            preview_name = ""
            preview_priority: bool | None = None
            if open_previews and preview_targets:
                quick_assignments = known_identity_button_options(char_map, limit=16)
                preview_result = None
                if preview_window_image and display_state.get("gui_available"):
                    preview_result = show_preview_assignment_window(
                        preview_window_image,
                        f"{cluster_id} Preview | Session: {session_remaining_count} | Open: {total_open_count}",
                        status_text=(
                            f"Remaining in this session including current case: {session_remaining_count} | "
                            f"Total actually still open: {total_open_count} | "
                            f"Role hint: {role_hint} | {action_hint}"
                        ),
                        initial_priority=identity_has_priority(char_map, str(payload.get("name", cluster_id))),
                        quick_assignments=quick_assignments,
                    )
                opened_count = 0
                if preview_result is None and preview_targets:
                    opened_count = open_preview_targets(preview_targets)
                    if opened_count:
                        info(
                            "Tk preview is not available for this case. "
                            "Opened one local preview file in the system viewer instead."
                        )
                if preview_result is not None:
                    preview_name = str(preview_result.get("value") or "").strip()
                    preview_priority = bool(preview_result.get("priority", False))
                elif not opened_count:
                    warn(
                        "Automatic preview opening is not available in this session. "
                        "Open one of the local preview targets above manually or run 06 from a desktop session with a display."
                    )
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
        raise ValueError(f"{cluster_id} is marked as Statist/background role and cannot be prioritized.")
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


def review_queue_summary(items: list[dict]) -> dict[str, object]:
    speaker_open = 0
    visible_open = 0
    speaker_names: Counter[str] = Counter()
    visible_names: Counter[str] = Counter()
    scenes: Counter[str] = Counter()
    for item in items:
        scene_id = str(item.get("scene_id", "") or "-")
        scenes[scene_id] += 1
        speaker_name = str(item.get("speaker_name", "") or "")
        if looks_auto_named(speaker_name):
            speaker_open += 1
            speaker_names[speaker_name or "unknown"] += 1
        for name in [str(name) for name in item.get("visible_character_names", [])]:
            if looks_auto_named(name):
                visible_open += 1
                visible_names[name or "unknown"] += 1
    return {
        "total": len(items),
        "speaker_open": speaker_open,
        "visible_open": visible_open,
        "top_speakers": speaker_names.most_common(8),
        "top_visible": visible_names.most_common(8),
        "top_scenes": scenes.most_common(8),
    }


def format_counter_rows(rows: list[tuple[str, int]]) -> str:
    if not rows:
        return "-"
    return ", ".join(f"{name} ({count})" for name, count in rows)


def show_review_queue(cfg: dict, limit: int = 50) -> None:
    queue = read_json(resolve_project_path(cfg["paths"]["review_queue"]), {"items": []})
    items = queue.get("items", [])
    if not items:
        info("No open review cases.")
        return
    summary = review_queue_summary(items)
    print(f"Open review cases: {summary['total']}")
    print(f"Open speaker references: {summary['speaker_open']}")
    print(f"Open visible-face references: {summary['visible_open']}")
    print(f"Top unresolved speakers: {format_counter_rows(summary['top_speakers'])}")
    print(f"Top unresolved visible names: {format_counter_rows(summary['top_visible'])}")
    print(f"Scenes with most open cases: {format_counter_rows(summary['top_scenes'])}")
    print("Tip: assign or rename the top repeated speaker/face IDs first, then run 22_refresh_after_manual_review.py.")
    shown_items = items if limit <= 0 else items[:limit]
    if len(shown_items) < len(items):
        print(f"Showing first {len(shown_items)} cases. Use --queue-limit 0 to print all cases.")
    for index, item in enumerate(shown_items, start=1):
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
    cfg.setdefault("character_detection", {})["internet_face_lookup_public_image_allow_slow_torch_import"] = True
    if args.show_queue:
        show_review_queue(cfg, limit=args.queue_limit)
        return
    worker_id = shared_worker_id_for_args(args)
    shared_workers = shared_workers_enabled_for_args(cfg, args)
    mark_step_started("05_review_unknowns", "global")
    if shared_workers:
        info(f"Shared NAS workers: enabled ({worker_id})")
    lease_manager = distributed_item_lease(
        root=distributed_step_runtime_root("05_review_unknowns", "global"),
        lease_name="global",
        cfg=cfg,
        worker_id=worker_id,
        enabled=shared_workers,
        meta={"step": "05_review_unknowns", "scope": "global", "worker_id": worker_id},
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
        loaded_faces = len(char_map.get("clusters", {}))
        loaded_voices = len(voice_map.get("clusters", {}))
        if loaded_faces or loaded_voices:
            info(f"Loaded maps: {loaded_faces} face entries, {loaded_voices} speaker entries.")
        seeded_voices = ensure_voice_clusters_from_project_speakers(cfg, voice_map)
        if seeded_voices:
            info(f"Registered speaker entries from transcripts/linked segments: {seeded_voices}")
        normalized_faces, normalized_voices = normalize_placeholder_maps(char_map, voice_map)
        if normalized_faces or normalized_voices:
            info(
                f"Normalized existing maps: adjusted {normalized_faces} face entries, "
                f"adjusted {normalized_voices} speaker entries."
            )
        hydrated = hydrate_face_clusters_from_previews(cfg, char_map)
        if hydrated:
            info(f"{hydrated} face clusters hydrated from existing preview folders.")
        offline_mode = bool(args.offline)
        rollback_summary = rollback_low_confidence_internet_names(cfg, char_map)
        if int(rollback_summary.get("restored", 0) or 0):
            for item in list(rollback_summary.get("items", []) or [])[:8]:
                info(
                    "Restored low-confidence public metadata rename: "
                    f"{item.get('rejected_name')} -> {item.get('restored_name')} "
                    f"({float(item.get('confidence', 0.0) or 0.0):.0%})"
                )
            if int(rollback_summary.get("restored", 0) or 0) > 8:
                info(f"Restored {int(rollback_summary.get('restored', 0) or 0) - 8} additional low-confidence metadata renames.")
        if int(rollback_summary.get("cleared_history", 0) or 0):
            info(
                "Cleared rejected low-confidence public metadata history for "
                f"{rollback_summary.get('cleared_history')} already-correct face cluster(s)."
            )
        internet_summary: dict[str, object] = {"checked": 0, "renamed": 0, "updates": []}
        face_lookup_summary: dict[str, object] = {"checked": 0, "assigned": 0, "updates": []}
        if not offline_mode:
            info("Online-first mode: trying public metadata and configured face lookup before local review fallback.")
            if args.no_internet_lookup:
                info("Public metadata/name lookup skipped by --no-internet-lookup; configured online face lookup can still run.")
            else:
                try:
                    internet_summary = enrich_existing_character_names_from_internet(
                        cfg,
                        char_map,
                        force_refresh=bool(args.refresh_internet_lookup),
                    )
                except Exception as exc:
                    warn(f"Public character-name lookup failed: {exc}. Continuing with local/offline review data.")
                    internet_summary = {"checked": 0, "renamed": 0, "updates": [], "error": str(exc)}
                updates = internet_summary.get("updates", []) if isinstance(internet_summary, dict) else []
                if updates:
                    for update in updates[:8]:
                        info(
                            "Public metadata completed character name: "
                            f"{update.get('old_name')} -> {update.get('new_name')} "
                            f"({float(update.get('confidence', 0.0) or 0.0):.0%})"
                        )
                    if len(updates) > 8:
                        info(f"Public metadata completed {len(updates) - 8} additional character names.")
                elif int(internet_summary.get("checked", 0) or 0):
                    info(f"Public metadata checked {internet_summary.get('checked')} existing character name(s); no safe completion found.")
            try:
                face_lookup_summary = apply_online_face_lookup(cfg, char_map, offline=False)
            except Exception as exc:
                warn(f"Online face lookup failed: {exc}. Continuing with local/offline review data.")
                face_lookup_summary = {"checked": 0, "assigned": 0, "updates": [], "error": str(exc)}
            if int(face_lookup_summary.get("assigned", 0) or 0):
                info(
                    f"Online face lookup assigned {face_lookup_summary.get('assigned')} face cluster(s) "
                    "before manual review."
                )
            elif face_lookup_summary.get("builtin_public_images"):
                info(
                    "Built-in public-image face lookup is available without login/API credentials "
                    f"({face_lookup_summary.get('builtin_reference_images', 0)} public reference image embeddings); "
                    "no safe 95% match was found."
                )
            elif face_lookup_summary.get("missing_lookup_backend"):
                info(
                    "Online face lookup is available but no command or API endpoint is configured. "
                    "Set SERIES_FACE_LOOKUP_COMMAND, SERIES_FACE_LOOKUP_URL, "
                    "character_detection.internet_face_lookup_command, or "
                    "character_detection.internet_face_lookup_url to enable face-image upload lookup."
                )
            if not int(internet_summary.get("renamed", 0) or 0) and not int(face_lookup_summary.get("assigned", 0) or 0):
                info("No online assignment was applied; continuing with local/offline review checks.")
        else:
            info("Offline mode: public metadata and online face-image lookup are skipped for this run.")
        auto_ignored_false_faces = auto_ignore_false_positive_face_clusters(cfg, char_map)
        if auto_ignored_false_faces:
            info(
                f"Marked {len(auto_ignored_false_faces)} implausible face clusters as 'noface' "
                "before review."
            )
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
        needs_linked_sync = (
            normalized_faces
            or normalized_voices
            or int(rollback_summary.get("restored", 0) or 0)
            or int(rollback_summary.get("cleared_history", 0) or 0)
            or int(internet_summary.get("renamed", 0) or 0)
            or int(face_lookup_summary.get("assigned", 0) or 0)
            or auto_ignored_false_faces
            or auto_matched.get("matched_faces", 0)
            or auto_matched.get("matched_speakers", 0)
            or auto_statist_marked
        )
        if needs_linked_sync:
            changed_linked_files, review_count = persist_updates(cfg, char_map, voice_map)
            info(
                f"Maps synchronized: {changed_linked_files + int(auto_matched.get('linked_files', 0))} linked-segment files updated, "
                f"{review_count} open review cases."
            )
        elif seeded_voices or hydrated:
            review_count = persist_maps_only(cfg, char_map, voice_map)
            info(f"Maps saved without segment rewrite, {review_count} open review cases.")

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
        elif args.edit_names:
            changed = open_name_editor_gui(cfg, char_map, voice_map)
            action = "edit_names"
            completion_payload = {"changed": bool(changed)}
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
            show_review_queue(cfg, limit=args.queue_limit)
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
            "05_review_unknowns",
            "global",
            {"action": action, **completion_payload},
        )
    except Exception as exc:
        mark_step_failed("05_review_unknowns", str(exc), "global")
        raise
    finally:
        lease_manager.__exit__(None, None, None)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise


