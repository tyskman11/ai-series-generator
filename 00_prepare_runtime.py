#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import shutil
import sys
import time
from pathlib import Path

from pipeline_common import (
    SCRIPT_DIR,
    ensure_project_structure,
    error,
    headline,
    info,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    nvidia_gpu_available,
    ok,
    runtime_settings,
    warn,
    write_json,
)


def venv_python() -> Path:
    return SCRIPT_DIR / "runtime" / "venv" / "Scripts" / "python.exe"


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def ensure_venv() -> Path:
    py = venv_python()
    headline("Python-Umgebung erstellen")
    venv_dir = py.parent.parent
    cfg_file = venv_dir / "pyvenv.cfg"
    if py.exists() and cfg_file.exists():
        cfg_text = cfg_file.read_text(encoding="utf-8", errors="ignore").lower()
        if "include-system-site-packages = true" not in cfg_text:
            return py
        warn("Vorhandene venv nutzt systemweite Site-Packages und wird fuer einen sauberen GPU-Stack neu erstellt.")
        shutil.rmtree(venv_dir, ignore_errors=True)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        check=True,
    )
    return py


def module_available(py: Path, module_name: str) -> bool:
    result = run([str(py), "-c", f"import {module_name}"], check=False)
    return result.returncode == 0


def install_group(
    py: Path,
    name: str,
    modules: list[str],
    packages: list[str],
    required: bool = True,
    pip_extra_args: list[str] | None = None,
) -> bool:
    if all(module_available(py, module) for module in modules):
        ok(f"{name} bereits vorhanden.")
        return True
    info(f"Installiere {name} ...")
    result = run(
        [str(py), "-m", "pip", "install", "--upgrade", *(pip_extra_args or []), *packages],
        check=False,
    )
    log_dir = SCRIPT_DIR / "runtime" / "install_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}_{int(time.time())}.log"
    log_file.write_text(result.stdout or "", encoding="utf-8")
    if result.returncode == 0 and all(module_available(py, module) for module in modules):
        ok(f"{name} erfolgreich installiert.")
        return True
    if required:
        error(f"{name} konnte nicht installiert werden. Siehe {log_file}")
        raise RuntimeError(f"{name} konnte nicht installiert werden.")
    warn(f"{name} konnte nicht installiert werden. Siehe {log_file}")
    return False


def torch_status(py: Path) -> dict:
    result = run(
        [
            str(py),
            "-c",
            (
                "import json, torch; "
                "print(json.dumps({"
                "'available': True, "
                "'torch_version': getattr(torch, '__version__', ''), "
                "'cuda_available': bool(torch.cuda.is_available()), "
                "'cuda_version': str(getattr(getattr(torch, 'version', None), 'cuda', '') or ''), "
                "'device_count': int(torch.cuda.device_count()) if torch.cuda.is_available() else 0, "
                "'device_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []"
                "}))"
            ),
        ],
        check=False,
    )
    if result.returncode != 0:
        return {
            "available": False,
            "torch_version": "",
            "cuda_available": False,
            "cuda_version": "",
            "device_count": 0,
            "device_names": [],
            "error": (result.stdout or "").strip(),
        }
    try:
        data = json.loads((result.stdout or "").strip())
    except Exception:
        return {
            "available": False,
            "torch_version": "",
            "cuda_available": False,
            "cuda_version": "",
            "device_count": 0,
            "device_names": [],
            "error": (result.stdout or "").strip(),
        }
    data.setdefault("error", "")
    return data


def install_torch_stack(py: Path, cfg: dict) -> tuple[bool, dict]:
    runtime_cfg = runtime_settings(cfg)
    wants_gpu = bool(runtime_cfg.get("prefer_gpu", True)) and nvidia_gpu_available()
    cuda_index_url = str(runtime_cfg.get("torch_cuda_index_url", "") or "").strip()
    current_status = torch_status(py)
    if current_status.get("available") and (not wants_gpu or current_status.get("cuda_available")):
        ok("torch bereits vorhanden.")
        return True, current_status

    if wants_gpu and not current_status.get("cuda_available"):
        info("NVIDIA-GPU erkannt. Versuche CUDA-faehigen Torch-Stack zu installieren ...")
    else:
        info("Installiere torch ...")

    log_dir = SCRIPT_DIR / "runtime" / "install_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run([str(py), "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], check=False)
    attempts: list[tuple[str, list[str]]] = []
    if wants_gpu and cuda_index_url:
        attempts.append(
            (
                "torch_cuda",
                [
                    str(py),
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--force-reinstall",
                    "--no-cache-dir",
                    "--index-url",
                    cuda_index_url,
                    "torch",
                    "torchvision",
                    "torchaudio",
                ],
            )
        )
    attempts.append(
        (
            "torch_default",
            [
                str(py),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                "--no-cache-dir",
                "torch",
                "torchvision",
                "torchaudio",
            ],
        )
    )

    last_output = ""
    for attempt_name, command in attempts:
        result = run(command, check=False)
        last_output = result.stdout or ""
        log_file = log_dir / f"{attempt_name}_{int(time.time())}.log"
        log_file.write_text(last_output, encoding="utf-8")
        current_status = torch_status(py)
        if current_status.get("available") and (not wants_gpu or current_status.get("cuda_available")):
            ok("torch erfolgreich installiert.")
            return True, current_status
        if result.returncode == 0 and current_status.get("available") and not wants_gpu:
            ok("torch erfolgreich installiert.")
            return True, current_status

    warn("CUDA-Torch konnte nicht bestaetigt werden. Nutze notfalls CPU-Fallback.")
    return bool(current_status.get("available")), current_status


def main() -> None:
    headline("Laufzeit vorbereiten")
    mark_step_started("00_prepare_runtime", "global")
    cfg = ensure_project_structure(write_config_file=True)
    py = ensure_venv()
    info(f"Verwende Python: {py}")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools<81", "wheel"], check=False)
    clone_cfg = cfg.get("cloning", {}) if isinstance(cfg.get("cloning"), dict) else {}
    requested_voice_engine = str(clone_cfg.get("voice_clone_engine", "pyttsx3") or "pyttsx3").strip().lower()
    optional_tts_requested = requested_voice_engine in {"auto", "xtts"} or str(
        os.environ.get("SERIES_ENABLE_OPTIONAL_TTS", "")
    ).strip().lower() in {"1", "true", "yes", "y"}

    core_ok = install_group(
        py,
        "core_ai",
        ["numpy", "PIL", "cv2", "librosa", "whisper"],
        ["numpy", "Pillow", "opencv-python", "librosa", "openai-whisper"],
    )
    scene_ok = install_group(py, "scene_detection", ["scenedetect"], ["scenedetect[opencv]"])
    facenet_ok = install_group(
        py,
        "face_recognition",
        ["facenet_pytorch"],
        ["facenet-pytorch"],
        required=False,
        pip_extra_args=["--no-deps"],
    )
    speaker_embeddings_ok = install_group(
        py,
        "speaker_embeddings",
        ["speechbrain"],
        ["speechbrain"],
        required=False,
    )
    tts_ok = install_group(py, "render_tts", ["pyttsx3"], ["pyttsx3"])
    voice_clone_ok = False
    if optional_tts_requested:
        voice_clone_ok = install_group(
            py,
            "voice_cloning",
            ["pkg_resources", "TTS.api"],
            ["setuptools<81", "TTS"],
            required=False,
        )
    else:
        info("Optionales XTTS/Coqui-Paket wird im lizenzfreien Standardpfad nicht automatisch installiert.")
    torch_ok, torch_info = install_torch_stack(py, cfg)

    write_json(
        SCRIPT_DIR / "runtime" / "package_status.json",
        {
            "python": str(py),
            "torch": torch_ok,
            "torch_info": torch_info,
            "nvidia_gpu_detected": nvidia_gpu_available(),
            "core_ai": core_ok,
            "scene_detection": scene_ok,
            "facenet_pytorch": facenet_ok,
            "speaker_embeddings": speaker_embeddings_ok,
            "render_tts": tts_ok,
            "voice_cloning": voice_clone_ok,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    if torch_info.get("cuda_available"):
        device_names = ", ".join(torch_info.get("device_names") or [])
        ok(f"GPU bereit fuer Torch: {device_names}")
    elif nvidia_gpu_available():
        warn("NVIDIA-GPU gefunden, aber Torch meldet noch kein CUDA. CPU-Fallback bleibt aktiv.")
    mark_step_completed(
        "00_prepare_runtime",
        "global",
        {
            "python": str(py),
            "torch": bool(torch_ok),
            "cuda_available": bool(torch_info.get("cuda_available", False)),
            "voice_cloning": bool(voice_clone_ok),
        },
    )
    ok("Laufzeit ist bereit.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        mark_step_failed("00_prepare_runtime", str(exc), "global")
        error(str(exc))
        raise
