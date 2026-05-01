#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

from support_scripts.pipeline_common import (
    headline,
    ok,
    info,
    load_config,
    resolve_project_path,
    read_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect voice clones across episodes."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Similarity threshold (0-1)."
    )
    parser.add_argument(
        "--compare-episode",
        help="Compare specific episode with others."
    )
    return parser.parse_args()


FEATURE_KEYWORDS = {
    "deep": ["low", "deep", "rich", "grave"],
    "high": ["high", "pitch", " shrill", "reedy"],
    "fast": ["fast", "quick", "rapid"],
    "slow": ["slow", "deliberate", "measured"],
    "raspy": ["raspy", "rough", "gravel"],
    "smooth": ["smooth", "velvet", "silky"],
}


def extract_voice_features(speaker_segments: list[dict]) -> dict:
    features = defaultdict(int)
    
    for seg in speaker_segments:
        text = " ".join(seg.get("words", [])).lower()
        
        for feature, keywords in FEATURE_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                features[feature] += 1
    
    return dict(features)


def calculate_similarity(features1: dict, features2: dict) -> float:
    if not features1 or not features2:
        return 0.0
    
    all_keys = set(features1.keys()) | set(features2.keys())
    
    matches = sum(1 for k in all_keys if features1.get(k, 0) > 0 and features2.get(k, 0) > 0)
    
    return matches / len(all_keys) if all_keys else 0.0


def detect_clones(
    packages_dir: Path,
    threshold: float
) -> list[dict]:
    all_speakers = {}
    
    for ep_dir in packages_dir.iterdir():
        if not ep_dir.is_dir():
            continue
        
        segments_file = ep_dir / "speaker_segments.json"
        if not segments_file.exists():
            continue
        
        try:
            segments = read_json(segments_file, [])
            
            for speaker in segments:
                spk_id = speaker.get("speaker_id", "")
                if spk_id not in all_speakers:
                    all_speakers[spk_id] = []
                
                all_speakers[spk_id].append({
                    "episode": ep_dir.name,
                    "features": extract_voice_features([speaker]),
                })
        except Exception:
            continue
    
    clones = []
    speaker_ids = list(all_speakers.keys())
    
    for i, spk1 in enumerate(speaker_ids):
        for spk2 in speaker_ids[i+1:]:
            features1 = all_speakers[spk1][0]["features"] if all_speakers[spk1] else {}
            features2 = all_speakers[spk2][0]["features"] if all_speakers[spk2] else {}
            
            similarity = calculate_similarity(features1, features2)
            
            if similarity >= threshold:
                clones.append({
                    "speaker_1": spk1,
                    "speaker_2": spk2,
                    "similarity": similarity,
                    "episodes": [
                        e["episode"] for e in all_speakers[spk1]
                    ] + [
                        e["episode"] for e in all_speakers[spk2]
                    ],
                })
    
    return clones


def main() -> None:
    args = parse_args()
    headline("Voice Clone Detector")
    cfg = load_config()
    
    packages_dir = resolve_project_path("generation/final_episode_packages")
    
    clones = detect_clones(packages_dir, args.threshold)
    
    output = {
        "threshold": args.threshold,
        "clone_pairs": len(clones),
        "clones": clones,
    }
    
    output_path = packages_dir / "voice_clone_report.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    
    ok(f"Found {len(clones)} potential voice clones")


if __name__ == "__main__":
    main()
