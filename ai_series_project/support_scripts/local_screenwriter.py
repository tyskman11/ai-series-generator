"""Offline local-LLM dialogue rewriting for the trained series planner."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from support_scripts.pipeline_common import PROJECT_ROOT, active_local_generation_profile, coalesce_text, warn

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
    model_kwargs = {"local_files_only": True, "dtype": dtype, "low_cpu_mem_usage": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(str(path), **model_kwargs).to(device)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        # Older Transformers releases used the deprecated torch_dtype keyword.
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(str(path), **model_kwargs).to(device)
    model.eval()
    _RUNTIME = (tokenizer, model)
    return _RUNTIME


_JSON_STRING = r'"(?:\\.|[^"\\])*"'
_DIALOGUE_PAIR = re.compile(
    rf'"speaker"\s*:\s*(?P<speaker>{_JSON_STRING})\s*,\s*"text"\s*:\s*(?P<text>{_JSON_STRING})',
    re.DOTALL,
)


def _clean_json_response(text: str) -> str:
    clean = text.strip()
    if "```" in clean:
        chunks = [part.strip() for part in clean.split("```") if "{" in part and "}" in part]
        if chunks:
            clean = chunks[0].removeprefix("json").strip()
    return clean


def _recover_dialogue_lines(clean: str) -> list[dict[str, str]]:
    """Keep complete model-produced dialogue pairs when one later JSON row is malformed."""
    recovered: list[dict[str, str]] = []
    for match in _DIALOGUE_PAIR.finditer(clean):
        try:
            speaker = json.loads(match.group("speaker"))
            dialogue = json.loads(match.group("text"))
        except json.JSONDecodeError:
            continue
        if isinstance(speaker, str) and isinstance(dialogue, str):
            recovered.append({"speaker": speaker, "text": dialogue})
    return recovered


def _json_payload(text: str) -> dict[str, Any]:
    clean = _clean_json_response(text)
    decoder = json.JSONDecoder()
    parse_error: json.JSONDecodeError | None = None
    for start in (index for index, char in enumerate(clean) if char == "{"):
        try:
            payload, _end = decoder.raw_decode(clean[start:])
        except json.JSONDecodeError as exc:
            parse_error = exc
            continue
        if isinstance(payload, dict) and isinstance(payload.get("lines"), list):
            return payload
    recovered = _recover_dialogue_lines(clean)
    if recovered:
        return {"lines": recovered, "format_recovered": True}
    if parse_error is not None:
        raise RuntimeError(f"Local screenwriter returned malformed JSON: {parse_error.msg}.") from parse_error
    raise RuntimeError("Local screenwriter did not return a JSON object with dialogue lines.")


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
            "Use RFC 8259 JSON only: no Markdown, code fences, comments, or trailing commas.",
            "Escape any double quote inside text and keep each text value on one line.",
        ],
    }
    messages = [
        {"role": "system", "content": "You are an offline writer-room model. Produce only valid RFC 8259 JSON."},
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
    if payload.get("format_recovered"):
        warn("Local screenwriter returned partially malformed JSON; recovered complete generated dialogue rows.")
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
