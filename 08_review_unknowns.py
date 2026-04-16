#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_common import (
    error,
    headline,
    info,
    is_interactive_session,
    load_config,
    ok,
    read_json,
    rerun_in_runtime,
    resolve_project_path,
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
EXAMPLE_FACE_NAMES = [
    "Babe Carano",
    "Mr. Sammich",
    "Teague/Busboy",
    "Hudson",
    "Triple G",
    "noface",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review, Benennung und Ignorieren von Face-Clustern")
    parser.add_argument("--list-faces", action="store_true", help="Zeigt standardmaessig nur bereits benannte Figuren mit Preview-Pfaden.")
    parser.add_argument("--show-queue", action="store_true", help="Zeigt die offene review_queue.json statt der Face-Review.")
    parser.add_argument("--assign-face", help="Face-Cluster-ID wie face_001.")
    parser.add_argument("--name", help="Name fuer --assign-face, z. B. 'Babe Carano'.")
    parser.add_argument("--rename-face", help="Bereits benannten Face-Cluster per ID oder aktuellem Namen umbenennen.")
    parser.add_argument("--rename-to", help="Neuer Name fuer --rename-face.")
    parser.add_argument("--ignore", action="store_true", help="Setzt --assign-face auf 'noface' und ignoriert das Cluster kuenftig.")
    parser.add_argument(
        "--review-faces",
        action="store_true",
        help="Interaktive Benennung fuer automatisch benannte Face-Cluster.",
    )
    parser.add_argument(
        "--include-named",
        action="store_true",
        help="Nimmt bei --review-faces auch bereits benannte Cluster mit auf.",
    )
    parser.add_argument("--limit", type=int, default=25, help="Maximale Anzahl ausgegebener Cluster.")
    parser.add_argument("--open-previews", action="store_true", help="Oeffnet die Montage-Datei bei der interaktiven Review.")
    return parser.parse_args()


def looks_auto_named(name: str) -> bool:
    return (
        not name
        or name == "unbekannt"
        or name.startswith("figur_")
        or name.startswith("stimme_")
    )


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
        raise FileNotFoundError(f"Face-Cluster oder Name nicht gefunden: {reference}")
    if len(matches) > 1:
        raise ValueError(f"Face-Referenz ist nicht eindeutig: {reference} -> {', '.join(sorted(matches))}")
    return matches[0]


def assign_character_name(char_map: dict, cluster_id: str, assigned_name: str) -> dict:
    payload = char_map.setdefault("clusters", {}).setdefault(cluster_id, {})
    remove_cluster_aliases(char_map, cluster_id)

    final_name = (assigned_name or cluster_id).strip() or cluster_id
    ignored = is_ignored_face_name(final_name)
    if ignored:
        final_name = "noface"

    payload["name"] = final_name
    payload["ignored"] = ignored
    payload["auto_named"] = False

    normalized_alias = normalize_alias_name(final_name)
    if ignored or not normalized_alias:
        payload["aliases"] = []
    else:
        payload["aliases"] = [normalized_alias]
        char_map.setdefault("aliases", {})[normalized_alias] = cluster_id
    return payload


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
        return create_contact_sheet(files, preview_dir / f"{cluster_id}_montage.jpg", title=f"{cluster_id} | Vorschau")

    try:
        from PIL import Image, ImageDraw, ImageOps
    except Exception:
        fallback_files = [path for pair in pairs for path in pair if path is not None]
        return create_contact_sheet(fallback_files, preview_dir / f"{cluster_id}_montage.jpg", title=f"{cluster_id} | Vorschau")

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


def show_preview_window(image_path: Path, title: str) -> bool:
    if not image_path.exists():
        return False
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except Exception:
        return False

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((1200, 900))

    window = tk.Tk()
    window.title(title)
    window.attributes("-topmost", True)
    window.configure(bg="#1f2937")
    window.bind("<Escape>", lambda _event: window.destroy())
    window.bind("<Return>", lambda _event: window.destroy())

    photo = ImageTk.PhotoImage(image)
    label = tk.Label(window, image=photo, bg="#1f2937")
    label.image = photo
    label.pack(padx=12, pady=(12, 8))

    hint = tk.Label(
        window,
        text="Fenster schliessen oder Enter/Escape druecken, dann Namen eingeben.",
        fg="white",
        bg="#1f2937",
    )
    hint.pack(padx=12, pady=(0, 8))

    button = tk.Button(window, text="Weiter zur Eingabe", command=window.destroy)
    button.pack(padx=12, pady=(0, 12))
    button.focus_set()
    window.mainloop()
    return True


def cluster_sort_key(item: tuple[str, dict]) -> tuple[int, int, int, str]:
    cluster_id, payload = item
    ignored_rank = 1 if is_ignored_face_payload(payload) else 0
    named_rank = 1 if not looks_auto_named(str(payload.get("name", ""))) else 0
    scene_rank = -int(payload.get("scene_count", 0))
    detection_rank = -int(payload.get("detection_count", 0))
    samples_rank = -int(payload.get("samples", 0))
    return (ignored_rank, named_rank, scene_rank, detection_rank, samples_rank, cluster_id)


def hydrate_face_clusters_from_previews(cfg: dict, char_map: dict) -> int:
    previews_root_value = cfg.get("paths", {}).get("character_previews", "characters/previews")
    previews_root = resolve_project_path(previews_root_value)
    if not previews_root.exists():
        return 0

    added = 0
    for preview_dir in sorted(previews_root.glob("face_*")):
        if not preview_dir.is_dir():
            continue
        cluster_id = preview_dir.name
        payload = char_map.setdefault("clusters", {}).setdefault(cluster_id, {})
        if payload:
            payload.setdefault("preview_dir", str(preview_dir))
            payload.setdefault("samples", max(1, len(list(preview_dir.glob("*_crop.jpg")))))
            continue
        suffix = cluster_id.split("_", 1)[1] if "_" in cluster_id else cluster_id
        payload["name"] = f"figur_{suffix}"
        payload["preview_dir"] = str(preview_dir)
        payload["samples"] = max(1, len(list(preview_dir.glob("*_crop.jpg"))))
        payload["auto_named"] = True
        payload["ignored"] = False
        payload["aliases"] = []
        added += 1
    return added


def refresh_voice_map(char_map: dict, voice_map: dict) -> None:
    for speaker_cluster, payload in voice_map.get("clusters", {}).items():
        linked_face = payload.get("linked_face_cluster")
        if not linked_face:
            continue

        face_payload = char_map.get("clusters", {}).get(linked_face)
        if face_payload is None or is_ignored_face_payload(face_payload):
            payload.pop("linked_face_cluster", None)
            if payload.get("auto_named", True) or looks_auto_named(str(payload.get("name", ""))):
                number = speaker_cluster.split("_")[1] if "_" in speaker_cluster else speaker_cluster
                payload["name"] = f"stimme_{number}"
                payload["auto_named"] = True
            continue

        if payload.get("auto_named", True) or looks_auto_named(str(payload.get("name", ""))):
            payload["name"] = face_payload.get("name", linked_face)
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
                char_map.get("clusters", {}).get(cluster_id, {}).get("name", cluster_id)
                for cluster_id in visible_clusters
            ]
            if visible_names != row.get("visible_character_names", []):
                row["visible_character_names"] = visible_names
                changed = True

            voice_payload = voice_map.get("clusters", {}).get(row.get("speaker_cluster"), {})
            speaker_name = voice_payload.get("name", row.get("speaker_cluster"))
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
    refresh_voice_map(char_map, voice_map)
    write_json(resolve_project_path(cfg["paths"]["character_map"]), char_map)
    write_json(resolve_project_path(cfg["paths"]["voice_map"]), voice_map)
    changed_linked_files = refresh_linked_segments(cfg, char_map, voice_map)
    review_count = rebuild_review_queue(cfg)
    return changed_linked_files, review_count


def print_cluster(cluster_id: str, payload: dict) -> None:
    status = "ignored" if is_ignored_face_payload(payload) else "aktiv"
    name = payload.get("name", cluster_id)
    preview_dir = payload.get("preview_dir", "-")
    samples = payload.get("samples", 0)
    scene_count = payload.get("scene_count", 0)
    detection_count = payload.get("detection_count", 0)
    auto_named = payload.get("auto_named", False)
    print(f"{cluster_id}: {name}")
    print(f"  Status: {status}")
    print(f"  Samples: {samples}")
    print(f"  Szenen: {scene_count}")
    print(f"  Treffer: {detection_count}")
    print(f"  Auto: {auto_named}")
    print(f"  Preview: {preview_dir}")


def list_faces(char_map: dict, limit: int, include_named: bool) -> None:
    clusters = sorted(char_map.get("clusters", {}).items(), key=cluster_sort_key)
    shown = 0
    for cluster_id, payload in clusters:
        if is_ignored_face_payload(payload):
            continue
        if looks_auto_named(str(payload.get("name", ""))):
            continue
        print_cluster(cluster_id, payload)
        shown += 1
        if shown >= limit:
            break
    if shown == 0:
        info("Keine benannten Figuren gefunden.")


def interactive_face_review(cfg: dict, char_map: dict, voice_map: dict, include_named: bool, limit: int, open_previews: bool) -> None:
    if not is_interactive_session():
        raise RuntimeError("--review-faces benoetigt eine interaktive Konsole.")

    candidates = []
    for cluster_id, payload in sorted(char_map.get("clusters", {}).items(), key=cluster_sort_key):
        if is_ignored_face_payload(payload):
            continue
        if not include_named and not looks_auto_named(str(payload.get("name", ""))):
            continue
        candidates.append((cluster_id, payload))
    if limit > 0:
        candidates = candidates[:limit]

    if not candidates:
        info("Keine Face-Cluster fuer die Review gefunden.")
        return

    changed = 0
    stop_review = False
    for cluster_id, payload in candidates:
        answer = ""
        while True:
            montage = create_face_review_sheet(cluster_id, payload)
            print()
            print("-" * 72)
            print_cluster(cluster_id, payload)
            if montage:
                print(f"Montage: {montage}")
                if open_previews:
                    shown = show_preview_window(montage, f"{cluster_id} Vorschau")
                    if not shown:
                        info("Keine blockierende Vorschau verfuegbar. Verwende den Montage-Pfad oben.")
            for context_path, crop_path in preview_pairs(payload):
                if context_path:
                    print(f"Szene: {context_path}")
                if crop_path:
                    print(f"Ausschnitt: {crop_path}")
            print(f"Beispielnamen: {' | '.join(EXAMPLE_FACE_NAMES)}")
            print("Name eingeben. Weiter geht es erst nach Benennung oder 'noface'. 'q' beendet die Review.")
            try:
                answer = input("> ").strip()
            except EOFError:
                info("Keine weitere Eingabe verfuegbar. Review wird nach der aktuellen Vorschau beendet.")
                stop_review = True
                break
            if not answer:
                info("Bitte einen Namen eingeben oder 'noface' verwenden. Der aktuelle Face-Cluster bleibt geoeffnet.")
                continue
            if answer.lower() == "q":
                break
            assign_character_name(char_map, cluster_id, answer)
            payload = char_map["clusters"][cluster_id]
            changed += 1
            break
        if stop_review or answer.lower() == "q":
            break

    if changed:
        changed_linked_files, review_count = persist_updates(cfg, char_map, voice_map)
        ok(f"{changed} Face-Cluster aktualisiert, {changed_linked_files} Linked-Segment-Dateien synchronisiert, {review_count} offene Review-Faelle.")
    else:
        info("Keine Aenderungen vorgenommen.")


def assign_single_face(cfg: dict, char_map: dict, voice_map: dict, cluster_id: str, assigned_name: str) -> None:
    payload = char_map.get("clusters", {}).get(cluster_id)
    if payload is None:
        raise FileNotFoundError(f"Face-Cluster nicht gefunden: {cluster_id}")
    assign_character_name(char_map, cluster_id, assigned_name)
    changed_linked_files, review_count = persist_updates(cfg, char_map, voice_map)
    final_name = char_map["clusters"][cluster_id]["name"]
    ok(
        f"{cluster_id} -> {final_name} gespeichert. "
        f"{changed_linked_files} Linked-Segment-Dateien synchronisiert, {review_count} offene Review-Faelle."
    )


def rename_face(cfg: dict, char_map: dict, voice_map: dict, reference: str, new_name: str) -> None:
    cluster_id = resolve_face_reference(char_map, reference)
    assign_character_name(char_map, cluster_id, new_name)
    changed_linked_files, review_count = persist_updates(cfg, char_map, voice_map)
    final_name = char_map["clusters"][cluster_id]["name"]
    ok(
        f"{cluster_id} wurde umbenannt zu {final_name}. "
        f"{changed_linked_files} Linked-Segment-Dateien synchronisiert, {review_count} offene Review-Faelle."
    )


def show_review_queue(cfg: dict) -> None:
    queue = read_json(resolve_project_path(cfg["paths"]["review_queue"]), {"items": []})
    items = queue.get("items", [])
    if not items:
        info("Keine offenen Review-Fälle.")
        return
    for index, item in enumerate(items, start=1):
        print("-" * 72)
        print(f"Fall {index}")
        print(f"Szene: {item.get('scene_id')}")
        print(f"Sprecher: {item.get('speaker_name')}")
        print(f"Text: {str(item.get('text', ''))[:220]}")
        print(f"Sichtbare Figuren: {', '.join(item.get('visible_character_names', []))}")
        frames = item.get("speaker_reference_frames", [])
        if frames:
            print(f"Frames: {', '.join(frames[:3])}")


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Review offener Zuordnungen")
    cfg = load_config()
    char_map = read_json(resolve_project_path(cfg["paths"]["character_map"]), {"clusters": {}, "aliases": {}})
    voice_map = read_json(resolve_project_path(cfg["paths"]["voice_map"]), {"clusters": {}, "aliases": {}})
    char_map.setdefault("clusters", {})
    char_map.setdefault("aliases", {})
    voice_map.setdefault("clusters", {})
    voice_map.setdefault("aliases", {})
    hydrated = hydrate_face_clusters_from_previews(cfg, char_map)
    if hydrated:
        info(f"{hydrated} Face-Cluster aus vorhandenen Preview-Ordnern ergänzt.")

    if args.assign_face:
        assigned_name = "noface" if args.ignore else (args.name or "").strip()
        if not assigned_name:
            raise ValueError("Bitte --name angeben oder --ignore verwenden.")
        assign_single_face(cfg, char_map, voice_map, args.assign_face, assigned_name)
        return

    if args.rename_face:
        new_name = (args.rename_to or "").strip()
        if not new_name:
            raise ValueError("Bitte --rename-to angeben.")
        rename_face(cfg, char_map, voice_map, args.rename_face, new_name)
        return

    if args.review_faces:
        interactive_face_review(
            cfg,
            char_map,
            voice_map,
            include_named=args.include_named,
            limit=max(0, args.limit),
            open_previews=args.open_previews,
        )
        return

    if args.list_faces:
        list_faces(char_map, max(1, args.limit), args.include_named)
        return

    if args.show_queue:
        show_review_queue(cfg)
        return

    if not is_interactive_session():
        info("Keine interaktive Konsole erkannt. Zeige stattdessen unbenannte Face-Cluster mit Preview-Pfaden.")
        list_faces(char_map, max(1, args.limit), args.include_named)
        return

    interactive_face_review(
        cfg,
        char_map,
        voice_map,
        include_named=args.include_named,
        limit=max(0, args.limit),
        open_previews=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
