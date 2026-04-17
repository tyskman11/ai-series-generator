#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pipeline_common import (
    LiveProgressReporter,
    PROJECT_ROOT,
    canonical_person_name,
    completed_step_state,
    coalesce_text,
    cosine_similarity,
    display_person_name,
    error,
    first_dir,
    has_manual_person_name,
    headline,
    info,
    is_background_person_name,
    is_interactive_session,
    limited_items,
    load_config,
    ok,
    open_file_default,
    preferred_compute_label,
    preferred_execution_label,
    preferred_torch_device,
    progress,
    read_json,
    rerun_in_runtime,
    resolve_project_path,
    save_step_autosave,
    mark_step_started,
    mark_step_completed,
    mark_step_failed,
    write_json,
)

PROCESS_VERSION = 6
EMPTY_CLUSTER_MAP = {"clusters": {}, "aliases": {}}
IGNORED_FACE_NAMES = {
    "noface",
    "no face",
    "ignore",
    "ignored",
    "falsepositive",
    "false positive",
    "kein gesicht",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesichter erkennen und mit Stimmen verknüpfen")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Loescht Cache, Previews, Maps und Linked-Segments von Schritt 05 vor dem Lauf.",
    )
    parser.add_argument(
        "--episode",
        help="Name des Szenenordners unter data/processed/scene_clips. Standard: erster verfuegbarer Ordner.",
    )
    return parser.parse_args()


def remove_output_path(path: Path) -> bool:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        return True
    if path.exists():
        path.unlink()
        return True
    return False


def reset_step_outputs(cfg: dict, episode_dir: Path, faces_episode_dir: Path, previews_root: Path) -> None:
    linked_root = resolve_project_path(cfg["paths"]["linked_segments"])
    removed = 0
    targets = [
        faces_episode_dir,
        linked_root / f"{episode_dir.name}_linked_segments.json",
        resolve_project_path(cfg["paths"]["character_map"]),
        resolve_project_path(cfg["paths"]["voice_map"]),
        resolve_project_path(cfg["paths"]["review_queue"]),
    ]
    for target in targets:
        removed += int(remove_output_path(target))
    if previews_root.exists():
        for preview_dir in previews_root.iterdir():
            if preview_dir.is_dir() and (
                preview_dir.name.startswith("face_") or preview_dir.name.startswith("speaker_refs_")
            ):
                removed += int(remove_output_path(preview_dir))
    info(f"Fresh-Reset abgeschlossen: {removed} Artefakte entfernt.")


def resolve_episode_dir(scene_root: Path, episode_name: str | None) -> Path | None:
    if episode_name:
        candidate = scene_root / episode_name
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"Szenenordner nicht gefunden: {candidate}")
    return first_dir(scene_root)


def face_linking_marker_path(faces_episode_dir: Path) -> Path:
    return faces_episode_dir / "_face_linking_success.json"


def episode_face_linking_completed(episode_dir: Path, cfg: dict) -> bool:
    linked_file = resolve_project_path(cfg["paths"]["linked_segments"]) / f"{episode_dir.name}_linked_segments.json"
    faces_episode_dir = resolve_project_path(cfg["paths"]["faces"]) / episode_dir.name
    face_summary_file = faces_episode_dir / f"{episode_dir.name}_face_summary.json"
    marker_file = face_linking_marker_path(faces_episode_dir)
    if not linked_file.exists() or not face_summary_file.exists():
        return False
    try:
        linked_rows = read_json(linked_file, [])
        face_summary = read_json(face_summary_file, [])
    except Exception:
        return False
    if not (isinstance(linked_rows, list) and len(linked_rows) > 0 and isinstance(face_summary, list) and len(face_summary) > 0):
        return False
    marker = read_json(marker_file, {}) if marker_file.exists() else {}
    autosave_state = completed_step_state("05_link_faces_and_speakers", episode_dir.name, PROCESS_VERSION)
    if marker:
        if int(marker.get("process_version", 0) or 0) != PROCESS_VERSION:
            return False
        linked_count = int(marker.get("linked_row_count", 0) or 0)
        face_scene_count = int(marker.get("face_scene_count", 0) or 0)
        if linked_count > len(linked_rows) or face_scene_count > len(face_summary):
            return False
    if autosave_state:
        linked_count = int(autosave_state.get("linked_row_count", 0) or 0)
        if linked_count > len(linked_rows):
            return False
    return True


def next_unlinked_episode_dir(scene_root: Path, cfg: dict) -> Path | None:
    for folder in sorted(scene_root.glob("*")):
        if not folder.is_dir():
            continue
        if not episode_face_linking_completed(folder, cfg):
            return folder
    return first_dir(scene_root)


def pending_unlinked_episode_dirs(scene_root: Path, cfg: dict) -> list[Path]:
    pending = []
    for folder in sorted(scene_root.glob("*")):
        if not folder.is_dir():
            continue
        if not episode_face_linking_completed(folder, cfg):
            pending.append(folder)
    return pending


def resolve_episode_dir_for_processing(scene_root: Path, episode_name: str | None, cfg: dict) -> Path | None:
    if episode_name:
        return resolve_episode_dir(scene_root, episode_name)
    return next_unlinked_episode_dir(scene_root, cfg)


def resolve_episode_dirs_for_processing(scene_root: Path, episode_name: str | None, cfg: dict) -> list[Path]:
    if episode_name:
        episode_dir = resolve_episode_dir(scene_root, episode_name)
        return [episode_dir] if episode_dir is not None else []
    return pending_unlinked_episode_dirs(scene_root, cfg)


def create_contact_sheet(image_paths: list[Path], output_path: Path) -> Path | None:
    if not image_paths:
        return None
    try:
        from PIL import Image, ImageDraw, ImageOps
    except Exception:
        return None

    cards = []
    for image_path in image_paths[:6]:
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.contain(image, (260, 260))
        canvas = Image.new("RGB", (280, 300), "white")
        canvas.paste(image, ((280 - image.width) // 2, 10))
        ImageDraw.Draw(canvas).text((10, 270), image_path.name[:28], fill="black")
        cards.append(canvas)
    sheet = Image.new("RGB", (560, 300 * ((len(cards) + 1) // 2)), "#dddddd")
    for index, card in enumerate(cards):
        sheet.paste(card, ((index % 2) * 280, (index // 2) * 300))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def ask_name(kind: str, cluster_id: str, preview_files: list[Path], auto_open: bool) -> str:
    montage = create_contact_sheet(preview_files, preview_files[0].parent / f"{cluster_id}_montage.jpg") if preview_files else None
    print()
    print("-" * 72)
    print(f"Neue Zuordnung für {kind}: {cluster_id}")
    if montage:
        print(f"Montage: {montage}")
        if auto_open:
            open_file_default(montage)
    print("Name eingeben, 'noface' zum Ignorieren, leer = automatische Bezeichnung")
    return input("> Name: ").strip()


def next_cluster_id(prefix: str, existing: dict[str, dict]) -> str:
    numbers = []
    for cluster_id in existing:
        if cluster_id.startswith(prefix):
            try:
                numbers.append(int(cluster_id.split("_")[1]))
            except Exception:
                pass
    return f"{prefix}_{(max(numbers) + 1) if numbers else 1:03d}"


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


def assign_character_name(
    char_map: dict,
    cluster_id: str,
    assigned_name: str,
    *,
    auto_named: bool | None = None,
    priority: bool | None = None,
) -> dict:
    payload = char_map.setdefault("clusters", {}).setdefault(cluster_id, {})
    remove_cluster_aliases(char_map, cluster_id)

    final_name = canonical_person_name((assigned_name or cluster_id).strip() or cluster_id) or cluster_id
    ignored = is_ignored_face_name(final_name)
    if ignored:
        final_name = "noface"
    background_role = is_background_person_name(final_name)
    effective_priority = False if ignored or background_role else bool(priority) if priority is not None else bool(payload.get("priority", False))

    payload["name"] = final_name
    payload["ignored"] = ignored
    payload["background_role"] = background_role
    payload["priority"] = effective_priority
    payload["auto_named"] = looks_auto_named(final_name) if auto_named is None else bool(auto_named)

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


def normalize_loaded_maps(char_map: dict, voice_map: dict) -> tuple[int, int]:
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


def normalize_embedding(vector: list[float]) -> list[float]:
    if not vector:
        return []
    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    if not np.isfinite(norm) or norm == 0:
        return []
    return (array / norm).round(6).tolist()


def create_fallback_face_embedding(crop: np.ndarray) -> list[float]:
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
    histogram = cv2.calcHist([resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    return normalize_embedding(histogram.tolist())


def box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = box_area((inter_x1, inter_y1, inter_x2, inter_y2))
    if inter_area <= 0:
        return 0.0
    union = box_area(box_a) + box_area(box_b) - inter_area
    return inter_area / union if union > 0 else 0.0


def dedupe_frame_detections(
    detections: list[tuple[tuple[int, int, int, int], list[float]]],
    iou_threshold: float = 0.45,
) -> list[tuple[tuple[int, int, int, int], list[float]]]:
    kept: list[tuple[tuple[int, int, int, int], list[float]]] = []
    for box, embedding in detections:
        if any(box_iou(box, kept_box) >= iou_threshold for kept_box, _ in kept):
            continue
        kept.append((box, embedding))
    return kept


def extract_opencv_faces(
    frame: np.ndarray,
    cascade,
    min_face_size: int,
) -> list[tuple[tuple[int, int, int, int], list[float]]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_face_size, min_face_size),
    )
    results = []
    for x, y, w, h in boxes:
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        embedding = create_fallback_face_embedding(crop)
        if embedding:
            results.append(((x1, y1, x2, y2), embedding))
    results.sort(key=lambda item: box_area(item[0]), reverse=True)
    return dedupe_frame_detections(results)


def create_face_engine(cfg: dict):
    min_face_size = int(cfg["character_detection"].get("min_face_size", 32))
    confidence_threshold = float(cfg["character_detection"].get("detection_confidence_threshold", 0.80))
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    try:
        import torch
        from PIL import Image
        from facenet_pytorch import InceptionResnetV1, MTCNN
    except Exception:
        def extract(frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], list[float]]]:
            return extract_opencv_faces(frame, cascade, min_face_size)

        return {"mode": "opencv", "extract": extract}

    device_name = preferred_torch_device(cfg)
    device = torch.device(device_name)
    mtcnn = MTCNN(
        keep_all=True,
        device=device,
        min_face_size=min_face_size,
        thresholds=[0.45, 0.55, 0.65],
        factor=0.709,
        post_process=True,
    )
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    def extract(frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], list[float]]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        boxes, probabilities = mtcnn.detect(image)
        faces = mtcnn(image)
        if boxes is None or faces is None:
            return extract_opencv_faces(frame, cascade, min_face_size)
        if faces.ndim == 3:
            faces = faces.unsqueeze(0)
        result = []
        for index, (box, face_tensor) in enumerate(zip(boxes, faces)):
            probability = float(probabilities[index]) if probabilities is not None and probabilities[index] is not None else 1.0
            if probability < confidence_threshold:
                continue
            x1, y1, x2, y2 = [max(0, int(value)) for value in box]
            if x2 <= x1 or y2 <= y1 or (x2 - x1) < min_face_size or (y2 - y1) < min_face_size:
                continue
            face_tensor = face_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                vector = resnet(face_tensor).cpu().numpy()[0].tolist()
            embedding = normalize_embedding(vector)
            if embedding:
                area = box_area((x1, y1, x2, y2))
                result.append(((x1, y1, x2, y2), embedding, area, probability))
        if result:
            result.sort(key=lambda item: (item[2], item[3]), reverse=True)
            return dedupe_frame_detections([(box, embedding) for box, embedding, _, _ in result])
        return extract_opencv_faces(frame, cascade, min_face_size)

    mode_label = f"facenet+opencv ({'hybrid cpu+gpu' if device_name == 'cuda' else 'cpu'})"
    return {"mode": mode_label, "extract": extract}


def save_speaker_reference_frames(scene_path: Path, start_sec: float, end_sec: float, output_dir: Path) -> list[Path]:
    capture = cv2.VideoCapture(str(scene_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps > 0 else 0.0
    output_dir.mkdir(parents=True, exist_ok=True)
    points = [start_sec, (start_sec + end_sec) / 2.0, max(start_sec, end_sec - 0.05)]
    saved = []
    for index, second in enumerate(points, start=1):
        second = max(0.0, min(second, max(0.0, duration - 0.05)))
        capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000.0)
        ok_frame, frame = capture.read()
        if not ok_frame:
            continue
        output_file = output_dir / f"{scene_path.stem}_speaker_frame_{index:02d}_{second:.2f}s.jpg"
        cv2.imwrite(str(output_file), frame)
        saved.append(output_file)
    capture.release()
    return saved


def best_cluster_match(embedding: list[float], clusters: dict[str, dict]) -> tuple[str | None, float]:
    best_cluster = None
    best_score = -1.0
    for cluster_id, payload in clusters.items():
        reference = payload.get("embedding") or []
        if not reference:
            continue
        score = cosine_similarity(embedding, reference)
        if score > best_score:
            best_score = score
            best_cluster = cluster_id
    return best_cluster, best_score


def match_face_cluster(embedding: list[float], char_map: dict, threshold: float) -> str | None:
    best_cluster, best_score = best_cluster_match(embedding, char_map.get("clusters", {}))
    if best_cluster and best_score >= threshold:
        return best_cluster
    return None


def register_new_face(
    char_map: dict,
    cluster_id: str,
    embedding: list[float],
    preview_dir: Path,
    preview_files: list[Path],
    interactive: bool,
    auto_open: bool,
) -> str:
    default_name = cluster_id
    assigned_name = default_name
    if interactive:
        user_name = ask_name("Figur", cluster_id, preview_files, auto_open)
        if user_name:
            assigned_name = user_name
    payload = assign_character_name(
        char_map,
        cluster_id,
        assigned_name,
        auto_named=assigned_name == default_name,
    )
    payload["embedding"] = embedding
    payload["preview_dir"] = str(preview_dir)
    payload["samples"] = 1
    return cluster_id


def update_embedding_payload(payload: dict, embedding: list[float]) -> None:
    existing = payload.get("embedding") or []
    samples = int(payload.get("samples", 1))
    if existing and len(existing) == len(embedding):
        averaged = [
            round(((existing[i] * samples) + embedding[i]) / (samples + 1), 6)
            for i in range(len(embedding))
        ]
        payload["embedding"] = normalize_embedding(averaged)
        payload["samples"] = samples + 1
        return
    payload["embedding"] = embedding
    payload["samples"] = max(samples, 1)


def update_cluster_embedding(char_map: dict, cluster_id: str, embedding: list[float]) -> None:
    payload = char_map["clusters"][cluster_id]
    update_embedding_payload(payload, embedding)


def write_preview_images(
    preview_dir: Path,
    scene_file: Path,
    frame_no: int,
    det_index: int,
    crop: np.ndarray,
    context: np.ndarray,
) -> list[Path]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    crop_path = preview_dir / f"{scene_file.stem}_f{frame_no}_{det_index}_crop.jpg"
    context_path = preview_dir / f"{scene_file.stem}_f{frame_no}_{det_index}_context.jpg"
    cv2.imwrite(str(crop_path), crop)
    cv2.imwrite(str(context_path), context)
    return [crop_path, context_path]


def collect_face_cluster_stats(face_by_scene: dict[str, dict]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int | set[str]]] = {}
    for scene_id, payload in face_by_scene.items():
        face_clusters = payload.get("face_clusters", []) or []
        detections = payload.get("detections", []) or []
        for cluster_id in face_clusters:
            cluster_stats = stats.setdefault(cluster_id, {"detections": 0, "scenes": set()})
            cluster_stats["scenes"].add(scene_id)
        for detection in detections:
            cluster_id = detection.get("face_cluster")
            if not cluster_id:
                continue
            cluster_stats = stats.setdefault(cluster_id, {"detections": 0, "scenes": set()})
            cluster_stats["detections"] = int(cluster_stats.get("detections", 0)) + 1
            cluster_stats["scenes"].add(scene_id)
    return {
        cluster_id: {
            "detections": int(payload.get("detections", 0)),
            "scene_count": len(payload.get("scenes", set())),
        }
        for cluster_id, payload in stats.items()
    }


def prune_face_clusters(char_map: dict, face_by_scene: dict[str, dict], cfg: dict) -> set[str]:
    min_scenes = int(cfg["character_detection"].get("face_cluster_min_scenes", 2))
    min_detections = int(cfg["character_detection"].get("face_cluster_min_detections", 3))
    stats = collect_face_cluster_stats(face_by_scene)
    kept_clusters: set[str] = set()

    for cluster_id, payload in list(char_map.get("clusters", {}).items()):
        cluster_stats = stats.get(cluster_id, {})
        payload["detection_count"] = int(cluster_stats.get("detections", 0))
        payload["scene_count"] = int(cluster_stats.get("scene_count", 0))

        keep_cluster = (
            is_ignored_face_payload(payload)
            or (payload.get("name") and not looks_auto_named(str(payload.get("name", ""))))
            or payload["scene_count"] >= min_scenes
            or payload["detection_count"] >= min_detections
        )
        if keep_cluster:
            kept_clusters.add(cluster_id)
            continue
        remove_cluster_aliases(char_map, cluster_id)
        char_map["clusters"].pop(cluster_id, None)

    for payload in face_by_scene.values():
        payload["face_clusters"] = [cluster_id for cluster_id in payload.get("face_clusters", []) if cluster_id in kept_clusters]
        payload["detections"] = [
            detection
            for detection in payload.get("detections", [])
            if detection.get("face_cluster") in kept_clusters
        ]
    return kept_clusters


def visible_faces_for_segment(
    row: dict,
    scene_payload: dict,
    char_map: dict,
    padding_seconds: float,
    max_visible_faces: int,
) -> list[str]:
    detections = scene_payload.get("detections") or []
    if not detections:
        fallback_clusters = [
            cluster_id
            for cluster_id in scene_payload.get("face_clusters", [])
            if not is_ignored_face_payload(char_map.get("clusters", {}).get(cluster_id, {}))
        ]
        return fallback_clusters[:max_visible_faces]

    start_sec = max(0.0, float(row.get("start", 0.0)) - padding_seconds)
    end_sec = float(row.get("end", 0.0)) + padding_seconds
    cluster_votes: dict[str, float] = defaultdict(float)
    for detection in detections:
        time_sec = float(detection.get("time_seconds", 0.0))
        if start_sec <= time_sec <= end_sec:
            cluster_id = detection["face_cluster"]
            if is_ignored_face_payload(char_map.get("clusters", {}).get(cluster_id, {})):
                continue
            cluster_votes[cluster_id] += 1.0

    if not cluster_votes:
        midpoint = (start_sec + end_sec) / 2.0
        nearest = sorted(
            detections,
            key=lambda item: abs(float(item.get("time_seconds", 0.0)) - midpoint),
        )
        for detection in nearest[:max_visible_faces]:
            cluster_id = detection["face_cluster"]
            if is_ignored_face_payload(char_map.get("clusters", {}).get(cluster_id, {})):
                continue
            cluster_votes[cluster_id] += 1.0

    sorted_clusters = sorted(cluster_votes.items(), key=lambda item: (-item[1], item[0]))
    return [cluster_id for cluster_id, _ in sorted_clusters[:max_visible_faces]]


def process_scene_faces(
    scene_file: Path,
    engine: dict,
    char_map: dict,
    cfg: dict,
    faces_episode_dir: Path,
    previews_root: Path,
    sample_every: int,
    max_faces_per_frame: int,
    threshold: float,
    interactive: bool,
    auto_open: bool,
) -> dict:
    cache_file = faces_episode_dir / f"{scene_file.stem}.json"
    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        if payload.get("process_version") == PROCESS_VERSION:
            return payload

    capture = cv2.VideoCapture(str(scene_file))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_no = 0
    visible_clusters: set[str] = set()
    detections = 0
    extract = engine["extract"]
    scene_threshold = float(cfg["character_detection"].get("scene_embedding_threshold", max(0.68, threshold - 0.04)))
    max_scene_clusters = int(cfg["character_detection"].get("max_scene_clusters", max(4, max_faces_per_frame * 3)))
    local_clusters: dict[str, dict] = {}
    scene_detections: list[dict] = []
    while True:
        ok_frame, frame = capture.read()
        if not ok_frame:
            break
        frame_no += 1
        if frame_no % sample_every != 0:
            continue
        extracted_faces = dedupe_frame_detections(extract(frame))
        if not extracted_faces:
            continue
        for det_index, ((x1, y1, x2, y2), embedding) in enumerate(
            extracted_faces[:max_faces_per_frame],
            start=1,
        ):
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            best_local_id, best_local_score = best_cluster_match(embedding, local_clusters)
            cluster_id = None
            relaxed_threshold = max(0.60, scene_threshold - 0.08)
            if best_local_id and best_local_score >= scene_threshold:
                cluster_id = best_local_id
            elif len(local_clusters) >= max_scene_clusters and best_local_id and best_local_score >= relaxed_threshold:
                cluster_id = best_local_id
            elif len(local_clusters) < max_scene_clusters:
                cluster_id = f"local_{len(local_clusters) + 1:03d}"
                context = frame.copy()
                cv2.rectangle(context, (x1, y1), (x2, y2), (0, 255, 0), 2)
                local_clusters[cluster_id] = {
                    "embedding": embedding,
                    "samples": 1,
                    "best_area": box_area((x1, y1, x2, y2)),
                    "preview_crop": crop.copy(),
                    "preview_context": context,
                    "preview_frame_no": frame_no,
                    "preview_det_index": det_index,
                }
            else:
                continue

            payload = local_clusters[cluster_id]
            if payload.get("embedding") != embedding:
                update_embedding_payload(payload, embedding)
            area = box_area((x1, y1, x2, y2))
            if area >= int(payload.get("best_area", 0)):
                context = frame.copy()
                cv2.rectangle(context, (x1, y1), (x2, y2), (0, 255, 0), 2)
                payload["best_area"] = area
                payload["preview_crop"] = crop.copy()
                payload["preview_context"] = context
                payload["preview_frame_no"] = frame_no
                payload["preview_det_index"] = det_index
            scene_detections.append(
                {
                    "time_seconds": round((frame_no - 1) / fps, 3),
                    "local_cluster": cluster_id,
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "area": area,
                }
            )
            detections += 1
        if detections >= 24:
            break
    capture.release()

    local_to_global: dict[str, str] = {}
    for local_cluster_id, payload in sorted(
        local_clusters.items(),
        key=lambda item: (-int(item[1].get("best_area", 0)), item[0]),
    ):
        cluster_id = match_face_cluster(payload.get("embedding") or [], char_map, threshold)
        if cluster_id is None:
            cluster_id = next_cluster_id("face", char_map.get("clusters", {}))
            preview_dir = previews_root / cluster_id
            preview_files = write_preview_images(
                preview_dir,
                scene_file,
                int(payload.get("preview_frame_no", 0)),
                int(payload.get("preview_det_index", 0)),
                payload["preview_crop"],
                payload["preview_context"],
            )
            register_new_face(
                char_map,
                cluster_id,
                payload.get("embedding") or [],
                preview_dir,
                preview_files,
                interactive,
                auto_open,
            )
        else:
            update_cluster_embedding(char_map, cluster_id, payload.get("embedding") or [])
        local_to_global[local_cluster_id] = cluster_id

    detection_rows = []
    for detection in scene_detections:
        global_cluster = local_to_global.get(detection["local_cluster"])
        if not global_cluster:
            continue
        if is_ignored_face_payload(char_map.get("clusters", {}).get(global_cluster, {})):
            continue
        visible_clusters.add(global_cluster)
        detection_rows.append(
            {
                "time_seconds": detection["time_seconds"],
                "face_cluster": global_cluster,
                "box": detection["box"],
                "area": detection["area"],
            }
        )

    payload = {
        "process_version": PROCESS_VERSION,
        "scene_id": scene_file.stem,
        "face_clusters": sorted(visible_clusters),
        "detections": detection_rows,
    }
    write_json(cache_file, payload)
    return payload


def resolve_voice_names(voice_map: dict, speaker_votes: dict[str, dict[str, float]], char_map: dict) -> None:
    for speaker_cluster, votes in speaker_votes.items():
        if speaker_cluster == "speaker_unknown":
            continue
        payload = voice_map.setdefault("clusters", {}).setdefault(speaker_cluster, {})
        if payload.get("name") and not looks_auto_named(payload["name"]):
            continue
        sorted_votes = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        linked_face = None
        if sorted_votes:
            best_face, best_score = sorted_votes[0]
            second_score = sorted_votes[1][1] if len(sorted_votes) > 1 else 0.0
            if best_score >= 2.5 and (best_score >= second_score + 1.0 or best_score >= second_score * 1.2):
                linked_face = best_face
        if linked_face:
            face_payload = char_map["clusters"].get(linked_face, {})
            if is_ignored_face_payload(face_payload):
                linked_face = None
            else:
                payload["linked_face_cluster"] = linked_face
                payload["name"] = display_person_name(str(face_payload.get("name", "")), speaker_cluster)
                payload["auto_named"] = True
        if linked_face is None:
            payload.pop("linked_face_cluster", None)
            payload["name"] = speaker_cluster
            payload["auto_named"] = True


def process_episode_dir(
    episode_dir: Path,
    cfg: dict,
    *,
    fresh_run: bool,
    faces_root: Path,
    linked_root: Path,
    previews_root: Path,
    char_map: dict,
    voice_map: dict,
    engine,
    interactive: bool,
    auto_open: bool,
    sample_every: int,
    max_faces_per_frame: int,
    max_visible_faces_per_segment: int,
    segment_padding_seconds: float,
    threshold: float,
    live_reporter: LiveProgressReporter | None = None,
    episode_index: int = 1,
    episode_total: int = 1,
) -> bool:
    autosave_target = episode_dir.name
    if not fresh_run and episode_face_linking_completed(episode_dir, cfg):
        mark_step_completed(
            "05_link_faces_and_speakers",
            autosave_target,
            {
                "episode_id": episode_dir.name,
                "process_version": PROCESS_VERSION,
                "linked_file": str(resolve_project_path(cfg["paths"]["linked_segments"]) / f"{episode_dir.name}_linked_segments.json"),
                "face_summary_file": str(resolve_project_path(cfg["paths"]["faces"]) / episode_dir.name / f"{episode_dir.name}_face_summary.json"),
            },
        )
        ok(f"Gesichts-/Stimmen-Verknüpfung bereits vorhanden: {episode_dir.name}")
        return False

    faces_episode_dir = faces_root / episode_dir.name
    if fresh_run:
        reset_step_outputs(cfg, episode_dir, faces_episode_dir, previews_root)

    transcript_rows = read_json(
        resolve_project_path(cfg["paths"]["speaker_transcripts"]) / f"{episode_dir.name}_segments.json",
        [],
    )
    if not transcript_rows:
        info("Keine Sprecher-Transkripte gefunden.")
        return False

    scenes = {scene.stem: scene for scene in limited_items(sorted(episode_dir.glob("*.mp4")))}
    faces_episode_dir.mkdir(parents=True, exist_ok=True)
    linked_root.mkdir(parents=True, exist_ok=True)
    previews_root.mkdir(parents=True, exist_ok=True)

    face_by_scene: dict[str, dict] = {}
    scene_files = list(scenes.values())
    completed_scene_ids: list[str] = []
    mark_step_started(
        "05_link_faces_and_speakers",
        autosave_target,
            {
                "episode_id": episode_dir.name,
                "process_version": PROCESS_VERSION,
                "scene_count": len(scene_files),
                "transcript_count": len(transcript_rows),
                "fresh_run": bool(fresh_run),
            },
        )
    try:
        episode_started_at = time.time()
        episode_total_units = max(1, len(scene_files) + len(transcript_rows))
        for index, scene_file in enumerate(scene_files, start=1):
            payload = process_scene_faces(
                scene_file,
                engine,
                char_map,
                cfg,
                faces_episode_dir,
                previews_root,
                sample_every,
                max_faces_per_frame,
                threshold,
                interactive,
                auto_open,
            )
            face_by_scene[payload["scene_id"]] = payload
            completed_scene_ids.append(payload["scene_id"])
            save_step_autosave(
                "05_link_faces_and_speakers",
                autosave_target,
                {
                    "status": "in_progress",
                    "episode_id": episode_dir.name,
                    "process_version": PROCESS_VERSION,
                    "scene_count": len(scene_files),
                    "transcript_count": len(transcript_rows),
                    "completed_scene_ids": completed_scene_ids,
                    "last_scene_id": payload["scene_id"],
                },
            )
            if live_reporter is not None:
                live_reporter.update(
                    (episode_index - 1) + (index / episode_total_units),
                    current_label=scene_file.name,
                    parent_label=episode_dir.name,
                    extra_label=f"Face-Cluster bisher: {len(face_clusters)}",
                    scope_current=index,
                    scope_total=episode_total_units,
                    scope_started_at=episode_started_at,
                    scope_label=f"Folge {episode_index}/{episode_total}",
                )

        kept_face_clusters = prune_face_clusters(char_map, face_by_scene, cfg)
        info(f"Face-Cluster nach Filter: {len(kept_face_clusters)}")

        speaker_votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for row in transcript_rows:
            scene_payload = face_by_scene.get(row["scene_id"], {})
            visible_faces = visible_faces_for_segment(
                row,
                scene_payload,
                char_map,
                segment_padding_seconds,
                max_visible_faces_per_segment,
            )
            if not visible_faces:
                continue
            if row.get("speaker_cluster") == "speaker_unknown":
                continue
            base_weight = 3.0 if len(visible_faces) == 1 else 2.0 if len(visible_faces) == 2 else 1.0
            for rank, face_cluster in enumerate(visible_faces, start=1):
                speaker_votes[row["speaker_cluster"]][face_cluster] += base_weight / rank

        resolve_voice_names(voice_map, speaker_votes, char_map)

        review_items = []
        linked_rows = []
        link_reporter = LiveProgressReporter(
            script_name="05_link_faces_and_speakers.py",
            total=len(transcript_rows),
            phase_label="Stimmen mit Figuren verknuepfen",
            parent_label=episode_dir.name,
        )
        for index, row in enumerate(transcript_rows, start=1):
            scene_payload = face_by_scene.get(row["scene_id"], {})
            visible_faces = visible_faces_for_segment(
                row,
                scene_payload,
                char_map,
                segment_padding_seconds,
                max_visible_faces_per_segment,
            )
            visible_names = [
                face_display_name(char_map["clusters"].get(face_cluster, {}), face_cluster)
                for face_cluster in visible_faces
            ]
            voice_payload = voice_map.get("clusters", {}).get(row["speaker_cluster"], {})
            speaker_name = display_person_name(voice_payload.get("name", ""), row["speaker_cluster"])
            linked_face = voice_payload.get("linked_face_cluster")
            segment_linked_face = linked_face if linked_face in visible_faces else None
            needs_preview = looks_auto_named(speaker_name)
            speaker_reference_frames: list[str] = []
            if needs_preview:
                scene_file = scenes.get(row["scene_id"])
                if scene_file is not None:
                    preview_dir = previews_root / f"speaker_refs_{row['speaker_cluster']}"
                    frames = save_speaker_reference_frames(
                        scene_file,
                        float(row.get("start", 0.0)),
                        float(row.get("end", 0.0)),
                        preview_dir,
                    )
                    speaker_reference_frames = [str(path) for path in frames]

            linked_row = {
                "scene_id": row["scene_id"],
                "segment_id": row["segment_id"],
                "start": row["start"],
                "end": row["end"],
                "speaker_cluster": row["speaker_cluster"],
                "speaker_name": speaker_name,
                "speaker_face_cluster": segment_linked_face,
                "text": coalesce_text(row.get("text", "")),
                "visible_face_clusters": visible_faces,
                "visible_character_names": visible_names,
                "speaker_reference_frames": speaker_reference_frames,
            }
            linked_rows.append(linked_row)
            if looks_auto_named(speaker_name) or any(looks_auto_named(name) for name in visible_names):
                review_items.append(linked_row)
            if live_reporter is not None:
                live_reporter.update(
                    (episode_index - 1) + ((len(scene_files) + index) / episode_total_units),
                    current_label=str(row.get("segment_id", "")),
                    parent_label=episode_dir.name,
                    extra_label=f"Sprecher: {str(row.get('speaker_cluster', '')) or 'unbekannt'}",
                    scope_current=len(scene_files) + index,
                    scope_total=episode_total_units,
                    scope_started_at=episode_started_at,
                    scope_label=f"Folge {episode_index}/{episode_total}",
                )

        face_summary = [
            {
                "scene_id": scene_id,
                "face_clusters": payload.get("face_clusters", []),
                "detections": payload.get("detections", []),
            }
            for scene_id, payload in sorted(face_by_scene.items())
        ]
        face_summary_file = faces_episode_dir / f"{episode_dir.name}_face_summary.json"
        linked_file = linked_root / f"{episode_dir.name}_linked_segments.json"
        write_json(face_summary_file, face_summary)
        write_json(
            face_linking_marker_path(faces_episode_dir),
            {
                "episode_id": episode_dir.name,
                "process_version": PROCESS_VERSION,
                "scene_count": len(scene_files),
                "face_scene_count": len(face_summary),
                "linked_row_count": len(linked_rows),
                "review_count": len(review_items),
            },
        )
        write_json(resolve_project_path(cfg["paths"]["character_map"]), char_map)
        write_json(resolve_project_path(cfg["paths"]["voice_map"]), voice_map)
        write_json(linked_file, linked_rows)
        write_json(resolve_project_path(cfg["paths"]["review_queue"]), {"items": review_items})
        mark_step_completed(
            "05_link_faces_and_speakers",
            autosave_target,
            {
                "episode_id": episode_dir.name,
                "process_version": PROCESS_VERSION,
                "scene_count": len(scene_files),
                "transcript_count": len(transcript_rows),
                "completed_scene_ids": completed_scene_ids,
                "linked_row_count": len(linked_rows),
                "review_count": len(review_items),
                "face_summary_file": str(face_summary_file),
                "linked_file": str(linked_file),
            },
        )
        ok(f"Verknüpfung abgeschlossen: {len(linked_rows)} Segmente")
        return True
    except Exception as exc:
        mark_step_failed(
            "05_link_faces_and_speakers",
            str(exc),
            autosave_target,
            {
                "episode_id": episode_dir.name,
                "process_version": PROCESS_VERSION,
                "scene_count": len(scene_files),
                "transcript_count": len(transcript_rows),
                "completed_scene_ids": completed_scene_ids,
            },
        )
        raise


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Gesichter erkennen und mit Stimmen verknüpfen")
    cfg = load_config()
    scene_root = resolve_project_path(cfg["paths"]["scene_clips"])
    episode_dirs = resolve_episode_dirs_for_processing(scene_root, args.episode, cfg)
    if not episode_dirs:
        if args.episode:
            info("Keine passenden Szenenordner gefunden.")
        else:
            info("Keine offenen Folgen für Schritt 05 gefunden.")
        return
    if args.fresh and not args.episode:
        raise ValueError("--fresh erfordert --episode, damit bestehende globale Maps nicht versehentlich für einen Batchlauf geleert werden.")

    faces_root = resolve_project_path(cfg["paths"]["faces"])
    linked_root = resolve_project_path(cfg["paths"]["linked_segments"])
    previews_root = PROJECT_ROOT / "characters" / "previews"
    if args.fresh:
        char_map = json.loads(json.dumps(EMPTY_CLUSTER_MAP))
        voice_map = json.loads(json.dumps(EMPTY_CLUSTER_MAP))
    else:
        char_map = read_json(resolve_project_path(cfg["paths"]["character_map"]), EMPTY_CLUSTER_MAP)
        voice_map = read_json(resolve_project_path(cfg["paths"]["voice_map"]), EMPTY_CLUSTER_MAP)
    char_map.setdefault("clusters", {})
    char_map.setdefault("aliases", {})
    voice_map.setdefault("clusters", {})
    voice_map.setdefault("aliases", {})
    normalized_faces, normalized_voices = normalize_loaded_maps(char_map, voice_map)
    if normalized_faces or normalized_voices:
        info(
            f"Bestehende Maps normalisiert: {normalized_faces} Face-Eintraege, "
            f"{normalized_voices} Sprecher-Eintraege."
        )

    interactive = bool(cfg["character_detection"].get("interactive_assignment", False)) and is_interactive_session()
    auto_open = bool(cfg.get("preview_open_automatically", False))
    sample_every = int(cfg["character_detection"].get("sample_every_n_frames", 8))
    max_faces_per_frame = int(cfg["character_detection"].get("max_faces_per_frame", 3))
    max_visible_faces_per_segment = int(cfg["character_detection"].get("max_visible_faces_per_segment", 3))
    segment_padding_seconds = float(cfg["character_detection"].get("segment_visibility_padding_seconds", 0.35))
    threshold = float(cfg["character_detection"].get("embedding_threshold", 0.72))
    engine = create_face_engine(cfg)
    info(f"Ausführungsmodus: {preferred_execution_label(cfg)}")
    info(f"Rechengerät: {preferred_compute_label(cfg)}")
    info(f"Gesichtserkennung: {engine['mode']}")

    processed_count = 0
    total = len(episode_dirs)
    live_reporter = LiveProgressReporter(
        script_name="05_link_faces_and_speakers.py",
        total=max(1, total),
        phase_label="Gesichter und Stimmen verknuepfen",
        parent_label="Batch",
    )
    for index, episode_dir in enumerate(episode_dirs, start=1):
        if process_episode_dir(
            episode_dir,
            cfg,
            fresh_run=bool(args.fresh),
            faces_root=faces_root,
            linked_root=linked_root,
            previews_root=previews_root,
            char_map=char_map,
            voice_map=voice_map,
            engine=engine,
            interactive=interactive,
            auto_open=auto_open,
            sample_every=sample_every,
            max_faces_per_frame=max_faces_per_frame,
            max_visible_faces_per_segment=max_visible_faces_per_segment,
            segment_padding_seconds=segment_padding_seconds,
            threshold=threshold,
            live_reporter=live_reporter,
            episode_index=index,
            episode_total=total,
        ):
            processed_count += 1
            live_reporter.update(
                index,
                current_label=episode_dir.name,
                parent_label=episode_dir.name,
                extra_label=f"Folge abgeschlossen: {episode_dir.name}",
                scope_current=1,
                scope_total=1,
                scope_started_at=time.time(),
                scope_label=f"Folge {index}/{total}",
            )
    live_reporter.finish(current_label="Batch", extra_label=f"Folgen verarbeitet: {processed_count}")

    ok(f"Batch abgeschlossen: {processed_count} Folgen in 05 verarbeitet.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
