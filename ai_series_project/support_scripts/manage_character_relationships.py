#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from support_scripts.pipeline_common import (
    canonical_person_name,
    character_relationships_path,
    empty_character_relationships,
    has_manual_person_name,
    headline,
    is_background_person_name,
    load_character_relationships,
    load_config,
    ok,
    read_json,
    relationship_prompt_fragments,
    resolve_project_path,
    write_character_relationships,
)


def split_values(value: str) -> list[str]:
    rows: list[str] = []
    for part in str(value or "").replace(";", ",").split(","):
        text = part.strip()
        if text and text not in rows:
            rows.append(text)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage manual character groups, relationships, and series inputs.")
    parser.add_argument("--gui", action="store_true", help="Open a Tk GUI for selecting names and adding groups/relationships.")
    parser.add_argument("--init", action="store_true", help="Create or normalize characters/relationships.json.")
    parser.add_argument("--list", action="store_true", help="Print the current relationship overview.")
    parser.add_argument("--print-json", action="store_true", help="Print the normalized JSON payload.")
    parser.add_argument("--set-group", metavar="GROUP_ID", help="Create or update a character group.")
    parser.add_argument("--characters", default="", help="Comma-separated group characters.")
    parser.add_argument("--series-inputs", default="", help="Comma-separated series input ids for the group.")
    parser.add_argument("--label", default="", help="Display label for a group or series input.")
    parser.add_argument("--description", default="", help="Description for a group, relationship, or series input.")
    parser.add_argument("--notes", default="", help="Free-form notes for a group or series input.")
    parser.add_argument("--add-relationship", nargs=2, metavar=("SOURCE", "TARGET"), help="Add or update a relationship pair.")
    parser.add_argument("--type", default="related", help="Relationship type, e.g. friends, rivals, siblings, mentor.")
    parser.add_argument("--group", default="", help="Optional group id for the relationship.")
    parser.add_argument("--tone", default="", help="Relationship tone, e.g. teasing, protective, tense.")
    parser.add_argument("--status", default="active", help="Relationship status.")
    parser.add_argument("--story-rule", action="append", default=[], help="Story rule. Can be repeated.")
    parser.add_argument("--remove-relationship", nargs=2, metavar=("SOURCE", "TARGET"), help="Remove all relationship rows for a pair.")
    parser.add_argument("--series-input", metavar="INPUT_ID", help="Create or update a named source-series input.")
    parser.add_argument("--default-group", default="", help="Default character group for a series input.")
    parser.add_argument("--episode-glob", action="append", default=[], help="Episode glob for a series input. Can be repeated.")
    return parser.parse_args()


def known_character_rows(cfg: dict) -> list[dict]:
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    char_map = read_json(resolve_project_path(paths.get("character_map", "characters/maps/character_map.json")), {"clusters": {}})
    voice_map = read_json(resolve_project_path(paths.get("voice_map", "characters/maps/voice_map.json")), {"clusters": {}})
    rows_by_name: dict[str, dict] = {}

    def add_row(name: str, priority: bool = False, count: int = 0, source: str = "") -> None:
        final_name = canonical_person_name(name)
        if not final_name or is_background_person_name(final_name) or not has_manual_person_name(final_name):
            return
        row = rows_by_name.setdefault(final_name, {"name": final_name, "priority": False, "face_count": 0, "voice_count": 0, "sources": set()})
        row["priority"] = bool(row["priority"] or priority)
        if source == "face":
            row["face_count"] = max(int(row.get("face_count", 0)), int(count or 0))
        elif source == "voice":
            row["voice_count"] = max(int(row.get("voice_count", 0)), int(count or 0))
        if source:
            row["sources"].add(source)

    identities = char_map.get("identities", {}) if isinstance(char_map, dict) else {}
    if isinstance(identities, dict):
        for identity_name, payload in identities.items():
            payload = payload if isinstance(payload, dict) else {}
            count = int(payload.get("face_cluster_count", 0) or payload.get("cluster_count", 0) or 0)
            add_row(identity_name, bool(payload.get("priority", False)), count, "face")

    clusters = char_map.get("clusters", {}) if isinstance(char_map, dict) else {}
    if isinstance(clusters, dict):
        counts: dict[str, int] = {}
        priorities: dict[str, bool] = {}
        for _cluster_id, payload in clusters.items():
            if not isinstance(payload, dict) or bool(payload.get("ignored", False)):
                continue
            final_name = canonical_person_name(str(payload.get("name", "")))
            if not final_name or is_background_person_name(final_name) or not has_manual_person_name(final_name):
                continue
            counts[final_name] = counts.get(final_name, 0) + 1
            priorities[final_name] = bool(priorities.get(final_name, False) or payload.get("priority", False))
        for name, count in counts.items():
            add_row(name, priorities.get(name, False), count, "face")

    voice_clusters = voice_map.get("clusters", {}) if isinstance(voice_map, dict) else {}
    if isinstance(voice_clusters, dict):
        voice_counts: dict[str, int] = {}
        for _speaker_id, payload in voice_clusters.items():
            if not isinstance(payload, dict):
                continue
            final_name = canonical_person_name(str(payload.get("name", "")))
            if not final_name or is_background_person_name(final_name) or not has_manual_person_name(final_name):
                continue
            voice_counts[final_name] = voice_counts.get(final_name, 0) + 1
        for name, count in voice_counts.items():
            add_row(name, False, count, "voice")

    rows = []
    for row in rows_by_name.values():
        rows.append(
            {
                "name": row["name"],
                "priority": bool(row.get("priority", False)),
                "face_count": int(row.get("face_count", 0) or 0),
                "voice_count": int(row.get("voice_count", 0) or 0),
                "sources": sorted(row.get("sources", set())),
            }
        )
    rows.sort(key=lambda item: (0 if item["priority"] else 1, -item["face_count"], -item["voice_count"], item["name"].lower()))
    return rows


def upsert_relationship(payload: dict, row: dict) -> None:
    source = row["source"]
    target = row["target"]
    relation_type = row["type"]
    group = row.get("group", "")
    pair_key = tuple(sorted([source, target]))
    for existing in payload["relationships"]:
        if tuple(sorted([existing.get("source", ""), existing.get("target", "")])) == pair_key:
            if existing.get("type") == relation_type and existing.get("group", "") == group:
                existing.update(row)
                return
    payload["relationships"].append(row)


def remove_relationship(payload: dict, source: str, target: str) -> int:
    pair_key = tuple(sorted([source.strip(), target.strip()]))
    before = len(payload.get("relationships", []))
    payload["relationships"] = [
        row
        for row in payload.get("relationships", [])
        if tuple(sorted([row.get("source", ""), row.get("target", "")])) != pair_key
    ]
    return before - len(payload["relationships"])


def print_overview(payload: dict) -> None:
    print(f"Relationship file version: {payload.get('version', 1)}")
    print("")
    print("Groups:")
    if payload.get("groups"):
        for group_id, group in payload["groups"].items():
            print(f"- {group.get('label') or group_id}: {', '.join(group.get('characters', []) or []) or '-'}")
    else:
        print("- none")
    print("")
    print("Relationships:")
    fragments = relationship_prompt_fragments(payload, limit=999)
    if fragments:
        for fragment in fragments:
            print(f"- {fragment}")
    else:
        print("- none")
    print("")
    print("Series inputs:")
    if payload.get("series_inputs"):
        for input_id, series_input in payload["series_inputs"].items():
            globs = ", ".join(series_input.get("episode_globs", []) or []) or "-"
            print(f"- {series_input.get('label') or input_id}: default group {series_input.get('default_group') or '-'} | {globs}")
    else:
        print("- none")


RELATIONSHIP_TYPES = [
    "best friends",
    "friends",
    "family",
    "siblings",
    "rivals",
    "mentor",
    "authority",
    "romantic",
    "team",
    "related",
]


def slug_from_label(label: str, fallback: str = "group") -> str:
    text = "".join(char.lower() if char.isalnum() else "_" for char in str(label or "").strip())
    text = "_".join(part for part in text.split("_") if part)
    return text or fallback


def relationship_display_text(row: dict) -> str:
    source = str(row.get("source", "")).strip()
    target = str(row.get("target", "")).strip()
    relation_type = str(row.get("type", "related")).strip() or "related"
    group = str(row.get("group", "")).strip()
    tone = str(row.get("tone", "")).strip()
    suffix = []
    if group:
        suffix.append(f"group={group}")
    if tone:
        suffix.append(tone)
    suffix_text = f" ({'; '.join(suffix)})" if suffix else ""
    return f"{source} <{relation_type}> {target}{suffix_text}"


def launch_relationship_gui(cfg: dict, payload: dict) -> dict:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as exc:
        raise RuntimeError("Tk GUI is not available on this system.") from exc

    character_rows = known_character_rows(cfg)
    root = tk.Tk()
    root.title("Character Relationship Manager")
    root.geometry("1180x780")
    try:
        root.minsize(980, 620)
    except Exception:
        pass

    selected_vars: dict[str, object] = {}
    status_var = tk.StringVar(value=f"Loaded {len(character_rows)} known character(s).")
    selected_count_var = tk.StringVar(value="Selected: 0")
    group_id_var = tk.StringVar()
    group_label_var = tk.StringVar()
    group_description_var = tk.StringVar()
    relation_group_var = tk.StringVar()
    relation_type_var = tk.StringVar(value=RELATIONSHIP_TYPES[0])
    relation_tone_var = tk.StringVar()
    relation_description_var = tk.StringVar()
    relation_rule_var = tk.StringVar()
    search_var = tk.StringVar()
    active_relationship_rows: list[dict] = []

    def selected_names() -> list[str]:
        names: list[str] = []
        for name, variable in selected_vars.items():
            try:
                if bool(variable.get()):
                    names.append(name)
            except Exception:
                pass
        return names

    def update_selected_count() -> None:
        selected_count_var.set(f"Selected: {len(selected_names())}")

    def save_payload(message: str) -> None:
        nonlocal payload
        payload = write_character_relationships(cfg, payload)
        status_var.set(message)

    def group_ids() -> list[str]:
        return sorted(payload.get("groups", {}).keys())

    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(1, weight=1)

    header = ttk.Frame(root, padding=10)
    header.grid(row=0, column=0, columnspan=2, sticky="ew")
    ttk.Label(header, text="Character Relationships", font=("Segoe UI", 15, "bold")).pack(side="left")
    ttk.Label(header, textvariable=status_var).pack(side="right")

    left = ttk.LabelFrame(root, text="Known Characters - check names", padding=10)
    left.grid(row=1, column=0, sticky="nsew", padx=(10, 6), pady=(0, 10))
    left.columnconfigure(0, weight=1)
    left.rowconfigure(2, weight=1)

    ttk.Entry(left, textvariable=search_var).grid(row=0, column=0, sticky="ew", pady=(0, 8))
    ttk.Label(left, textvariable=selected_count_var).grid(row=1, column=0, sticky="w", pady=(0, 8))
    canvas = tk.Canvas(left, highlightthickness=0)
    scrollbar = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
    check_frame = ttk.Frame(canvas)
    check_window = canvas.create_window((0, 0), window=check_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.grid(row=2, column=0, sticky="nsew")
    scrollbar.grid(row=2, column=1, sticky="ns")

    def on_check_frame_configure(_event=None) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def on_canvas_configure(event) -> None:
        canvas.itemconfigure(check_window, width=event.width)

    check_frame.bind("<Configure>", on_check_frame_configure)
    canvas.bind("<Configure>", on_canvas_configure)

    def render_character_checks() -> None:
        for child in check_frame.winfo_children():
            child.destroy()
        filter_text = search_var.get().strip().lower()
        visible_rows = [row for row in character_rows if not filter_text or filter_text in row["name"].lower()]
        if not visible_rows:
            ttk.Label(check_frame, text="No known character names found. Run 05_review_unknowns.py first or add names manually in the maps.").pack(anchor="w", pady=6)
            return
        for row in visible_rows:
            name = row["name"]
            variable = selected_vars.setdefault(name, tk.BooleanVar(value=False))
            label = f"{name}"
            details = []
            if row.get("priority"):
                details.append("main")
            if row.get("face_count"):
                details.append(f"faces {row['face_count']}")
            if row.get("voice_count"):
                details.append(f"voices {row['voice_count']}")
            if details:
                label += f" ({', '.join(details)})"
            check = ttk.Checkbutton(check_frame, text=label, variable=variable, command=update_selected_count)
            check.pack(anchor="w", fill="x", pady=2)

    search_var.trace_add("write", lambda *_args: render_character_checks())
    render_character_checks()

    action_row = ttk.Frame(left)
    action_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def clear_selection() -> None:
        for variable in selected_vars.values():
            variable.set(False)
        update_selected_count()

    ttk.Button(action_row, text="Clear selection", command=clear_selection).pack(side="left")

    right = ttk.Frame(root, padding=(6, 0, 10, 10))
    right.grid(row=1, column=1, sticky="nsew")
    right.columnconfigure(0, weight=1)
    right.rowconfigure(2, weight=1)

    group_box = ttk.LabelFrame(right, text="Create/update group from checked names", padding=10)
    group_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    for column in range(2):
        group_box.columnconfigure(column, weight=1)
    ttk.Label(group_box, text="Group id").grid(row=0, column=0, sticky="w")
    ttk.Entry(group_box, textvariable=group_id_var).grid(row=1, column=0, sticky="ew", padx=(0, 6))
    ttk.Label(group_box, text="Label").grid(row=0, column=1, sticky="w")
    ttk.Entry(group_box, textvariable=group_label_var).grid(row=1, column=1, sticky="ew")
    ttk.Label(group_box, text="Description").grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
    ttk.Entry(group_box, textvariable=group_description_var).grid(row=3, column=0, columnspan=2, sticky="ew")

    relation_box = ttk.LabelFrame(right, text="Add relationship for checked names", padding=10)
    relation_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
    for column in range(2):
        relation_box.columnconfigure(column, weight=1)
    ttk.Label(relation_box, text="Type").grid(row=0, column=0, sticky="w")
    type_combo = ttk.Combobox(relation_box, textvariable=relation_type_var, values=RELATIONSHIP_TYPES)
    type_combo.grid(row=1, column=0, sticky="ew", padx=(0, 6))
    ttk.Label(relation_box, text="Group").grid(row=0, column=1, sticky="w")
    group_combo = ttk.Combobox(relation_box, textvariable=relation_group_var, values=group_ids())
    group_combo.grid(row=1, column=1, sticky="ew")
    ttk.Label(relation_box, text="Tone").grid(row=2, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(relation_box, textvariable=relation_tone_var).grid(row=3, column=0, sticky="ew", padx=(0, 6))
    ttk.Label(relation_box, text="Description").grid(row=2, column=1, sticky="w", pady=(8, 0))
    ttk.Entry(relation_box, textvariable=relation_description_var).grid(row=3, column=1, sticky="ew")
    ttk.Label(relation_box, text="Story rules, comma separated").grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
    ttk.Entry(relation_box, textvariable=relation_rule_var).grid(row=5, column=0, columnspan=2, sticky="ew")

    list_box = ttk.LabelFrame(right, text="Existing relationships", padding=10)
    list_box.grid(row=2, column=0, sticky="nsew")
    list_box.columnconfigure(0, weight=1)
    list_box.rowconfigure(0, weight=1)
    relationship_listbox = tk.Listbox(list_box, height=12)
    relationship_scrollbar = ttk.Scrollbar(list_box, orient="vertical", command=relationship_listbox.yview)
    relationship_listbox.configure(yscrollcommand=relationship_scrollbar.set)
    relationship_listbox.grid(row=0, column=0, sticky="nsew")
    relationship_scrollbar.grid(row=0, column=1, sticky="ns")

    def refresh_groups() -> None:
        ids = group_ids()
        group_combo.configure(values=ids)
        if not relation_group_var.get() and ids:
            relation_group_var.set(ids[0])

    def refresh_relationship_list() -> None:
        active_relationship_rows.clear()
        relationship_listbox.delete(0, tk.END)
        for row in payload.get("relationships", []) or []:
            if not isinstance(row, dict):
                continue
            active_relationship_rows.append(row)
            relationship_listbox.insert(tk.END, relationship_display_text(row))

    def create_group() -> None:
        names = selected_names()
        if not names:
            messagebox.showwarning("No names selected", "Check at least one character name first.")
            return
        group_id = group_id_var.get().strip() or slug_from_label(group_label_var.get() or names[0], "group")
        label = group_label_var.get().strip() or group_id
        existing = payload.setdefault("groups", {}).get(group_id, {})
        payload["groups"][group_id] = {
            "id": group_id,
            "label": label,
            "description": group_description_var.get().strip() or existing.get("description", ""),
            "characters": names,
            "series_inputs": existing.get("series_inputs", []),
            "notes": existing.get("notes", ""),
        }
        relation_group_var.set(group_id)
        save_payload(f"Saved group '{label}' with {len(names)} character(s).")
        refresh_groups()

    def add_relationships() -> None:
        names = selected_names()
        if len(names) < 2:
            messagebox.showwarning("Need at least two names", "Check two or more character names first.")
            return
        relation_type = relation_type_var.get().strip() or "related"
        group_id = relation_group_var.get().strip()
        added_pairs = 0
        for source, target in itertools.combinations(names, 2):
            upsert_relationship(
                payload,
                {
                    "source": source,
                    "target": target,
                    "type": relation_type,
                    "group": group_id,
                    "description": relation_description_var.get().strip(),
                    "tone": relation_tone_var.get().strip(),
                    "status": "active",
                    "story_rules": split_values(relation_rule_var.get()),
                    "first_seen": "",
                    "last_seen": "",
                },
            )
            added_pairs += 1
        save_payload(f"Saved {added_pairs} relationship pair(s).")
        refresh_relationship_list()

    def remove_selected_relationships() -> None:
        selected_indices = list(relationship_listbox.curselection())
        if not selected_indices:
            messagebox.showwarning("No relationship selected", "Select one or more relationship rows first.")
            return
        remove_ids = {id(active_relationship_rows[index]) for index in selected_indices if index < len(active_relationship_rows)}
        payload["relationships"] = [row for row in payload.get("relationships", []) if id(row) not in remove_ids]
        save_payload(f"Removed {len(remove_ids)} relationship row(s).")
        refresh_relationship_list()

    button_row = ttk.Frame(right)
    button_row.grid(row=3, column=0, sticky="ew", pady=(8, 0))
    ttk.Button(button_row, text="Save group from checks", command=create_group).pack(side="left", padx=(0, 6))
    ttk.Button(button_row, text="Add relationship(s)", command=add_relationships).pack(side="left", padx=(0, 6))
    ttk.Button(button_row, text="Remove selected", command=remove_selected_relationships).pack(side="left", padx=(0, 6))
    ttk.Button(button_row, text="Close", command=root.destroy).pack(side="right")

    refresh_groups()
    refresh_relationship_list()
    update_selected_count()
    try:
        root.attributes("-topmost", True)
        root.after(700, lambda: root.attributes("-topmost", False))
    except Exception:
        pass
    root.mainloop()
    return payload


def main() -> None:
    args = parse_args()
    headline("Manage Character Relationships")
    cfg = load_config()
    path = character_relationships_path(cfg)
    payload = load_character_relationships(cfg) if path.exists() else empty_character_relationships()
    changed = False

    if args.gui:
        launch_relationship_gui(cfg, payload)
        payload = load_character_relationships(cfg)
        print_overview(payload)
        return

    if args.init:
        changed = True

    if args.set_group:
        group_id = args.set_group.strip()
        if not group_id:
            raise ValueError("--set-group needs a non-empty group id.")
        existing = payload["groups"].get(group_id, {})
        payload["groups"][group_id] = {
            "id": group_id,
            "label": args.label.strip() or existing.get("label") or group_id,
            "description": args.description.strip() or existing.get("description", ""),
            "characters": split_values(args.characters) or existing.get("characters", []),
            "series_inputs": split_values(args.series_inputs) or existing.get("series_inputs", []),
            "notes": args.notes.strip() or existing.get("notes", ""),
        }
        changed = True

    if args.add_relationship:
        source, target = [value.strip() for value in args.add_relationship]
        if not source or not target:
            raise ValueError("--add-relationship needs two non-empty character names.")
        upsert_relationship(
            payload,
            {
                "source": source,
                "target": target,
                "type": args.type.strip() or "related",
                "group": args.group.strip(),
                "description": args.description.strip(),
                "tone": args.tone.strip(),
                "status": args.status.strip() or "active",
                "story_rules": [value.strip() for value in args.story_rule if value.strip()],
                "first_seen": "",
                "last_seen": "",
            },
        )
        changed = True

    if args.remove_relationship:
        removed = remove_relationship(payload, args.remove_relationship[0], args.remove_relationship[1])
        ok(f"Removed {removed} relationship row(s).")
        changed = True

    if args.series_input:
        input_id = args.series_input.strip()
        if not input_id:
            raise ValueError("--series-input needs a non-empty id.")
        existing = payload["series_inputs"].get(input_id, {})
        payload["series_inputs"][input_id] = {
            "id": input_id,
            "label": args.label.strip() or existing.get("label") or input_id,
            "description": args.description.strip() or existing.get("description", ""),
            "default_group": args.default_group.strip() or existing.get("default_group", ""),
            "episode_globs": [value.strip() for value in args.episode_glob if value.strip()] or existing.get("episode_globs", []),
            "notes": args.notes.strip() or existing.get("notes", ""),
        }
        changed = True

    if changed:
        payload = write_character_relationships(cfg, payload)
        ok(f"Relationship config updated: {path}")

    if args.print_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.list or not changed and not args.print_json:
        print_overview(payload)


if __name__ == "__main__":
    main()
