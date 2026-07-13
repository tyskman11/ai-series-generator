"""Offline local-LLM dialogue rewriting for the trained series planner."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from support_scripts.pipeline_common import PROJECT_ROOT, active_local_generation_profile, coalesce_text

_RUNTIME: tuple[Any, Any] | None = None


def screenwriter_config(cfg: dict[str, Any]) -> dict[str, Any]:
    local_cfg = cfg.get("local_generation", {}) if isinstance(cfg.get("local_generation"), dict) else {}
    value = local_cfg.get("scriptwriter", {}) if isinstance(local_cfg.get("scriptwriter"), dict) else {}
    return dict(value)


def model_dir(cfg: dict[str, Any]) -> Path:
    configured = coalesce_text(screenwriter_config(cfg).get("model_dir", ""))
    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_ready(cfg: dict[str, Any]) -> Path:
    settings = screenwriter_config(cfg)
    if not bool(settings.get("enabled", False)):
        raise RuntimeError("Local screenwriter is disabled. Set local_generation.scriptwriter.enabled=true.")
    if str(settings.get("engine", "") or "").strip().lower() != "transformers":
        raise RuntimeError("Local screenwriter must use the transformers engine in local-models-only mode.")
    if not bool(settings.get("local_files_only", False)):
        raise RuntimeError("Local screenwriter must set local_files_only=true; runtime downloads are not allowed.")
    path = model_dir(cfg)
    if not (path / "config.json").is_file() or not any(path.rglob("*.safetensors")):
        raise RuntimeError(
            f"Project-local screenwriter model is incomplete: {path}. Run 00_prepare_runtime.py without --skip-downloads."
        )
    return path


def load_runtime(cfg: dict[str, Any]) -> tuple[Any, Any]:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    path = ensure_ready(cfg)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Local Qwen screenwriter requires transformers and torch. Run 00_prepare_runtime.py.") from exc
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device = "cuda" if bool(torch.cuda.is_available()) else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    _RUNTIME = (tokenizer, model)
    return _RUNTIME


def _json_payload(text: str) -> dict[str, Any]:
    clean = text.strip()
    if "```" in clean:
        chunks = [part.strip() for part in clean.split("```") if "{" in part and "}" in part]
        if chunks:
            clean = chunks[0].removeprefix("json").strip()
    start, end = clean.find("{"), clean.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("Local screenwriter did not return a JSON object.")
    payload = json.loads(clean[start : end + 1])
    return payload if isinstance(payload, dict) else {}


def prepare_generation_inputs(tokenized: Any, device: Any) -> tuple[dict[str, Any], int]:
    """Normalize tensor and BatchEncoding chat-template outputs for ``generate``."""
    if isinstance(tokenized, dict) or hasattr(tokenized, "items"):
        raw_inputs = dict(tokenized)
    else:
        raw_inputs = {"input_ids": tokenized}
    inputs = {
        str(name): value.to(device)
        for name, value in raw_inputs.items()
        if value is not None and hasattr(value, "to")
    }
    input_ids = inputs.get("input_ids")
    if input_ids is None or not hasattr(input_ids, "shape"):
        raise RuntimeError("Local screenwriter tokenizer did not return input_ids tensors for generation.")
    return inputs, int(input_ids.shape[-1])


def rewrite_scene_dialogue(
    cfg: dict[str, Any],
    *,
    scene: dict[str, Any],
    series_language: str,
    target_lines: int,
) -> list[dict[str, str]]:
    tokenizer, model = load_runtime(cfg)
    characters = [coalesce_text(value) for value in scene.get("characters", []) if coalesce_text(value)]
    if not characters:
        raise RuntimeError(f"{coalesce_text(scene.get('scene_id', 'scene'))}: local screenwriter has no named characters.")
    plan = scene.get("writer_room_plan", {}) if isinstance(scene.get("writer_room_plan"), dict) else {}
    profile = active_local_generation_profile(cfg)
    behavior = scene.get("behavior_constraints", []) if isinstance(scene.get("behavior_constraints"), list) else []
    style = scene.get("dialogue_style_constraints", []) if isinstance(scene.get("dialogue_style_constraints"), list) else []
    prompt = {
        "task": "Write original dialogue for one new episode scene. Return JSON only.",
        "language": series_language or "source language",
        "allowed_speakers": characters,
        "target_line_count": max(3, target_lines),
        "scene_function": coalesce_text(plan.get("scene_function", scene.get("scene_function", ""))),
        "conflict": coalesce_text(scene.get("conflict", "")),
        "relationship_context": scene.get("relationship_context", []),
        "behavior_constraints": behavior[:5],
        "dialogue_style_constraints": style[:5],
        "style_profile": coalesce_text(profile.get("style_prompt", "")),
        "rules": [
            "Use only allowed_speakers.",
            "Write new dialogue; do not quote source episodes.",
            "Use short natural turns, reactions, interruptions, and one scene payoff where appropriate.",
            "Return exactly {\"lines\":[{\"speaker\":\"...\",\"text\":\"...\"}]}.",
        ],
    }
    messages = [
        {"role": "system", "content": "You are an offline writer-room model. Produce only valid JSON."},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ]
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("Local screenwriter runtime lost torch during inference.") from exc
    device = next(model.parameters()).device
    tokenized = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    generation_inputs, prompt_token_count = prepare_generation_inputs(tokenized, device)
    settings = screenwriter_config(cfg)
    with torch.inference_mode():
        generated = model.generate(
            **generation_inputs,
            max_new_tokens=max(128, int(settings.get("max_new_tokens", 768) or 768)),
            do_sample=True,
            temperature=max(0.1, float(settings.get("temperature", 0.75) or 0.75)),
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(generated[0][prompt_token_count:], skip_special_tokens=True)
    payload = _json_payload(text)
    lines = payload.get("lines", []) if isinstance(payload.get("lines"), list) else []
    validated: list[dict[str, str]] = []
    for row in lines[: max(3, target_lines + 2)]:
        if not isinstance(row, dict):
            continue
        speaker = coalesce_text(row.get("speaker", ""))
        text = coalesce_text(row.get("text", ""))
        if speaker in characters and text and len(text) <= 420:
            validated.append({"speaker": speaker, "text": text})
    if len(validated) < 3:
        raise RuntimeError(
            f"{coalesce_text(scene.get('scene_id', 'scene'))}: local screenwriter returned too few valid lines for the approved cast."
        )
    return validated
