#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import shutil
import sys
import time
from pathlib import Path

from support_scripts.pipeline_common import (
    SCRIPT_DIR,
    add_shared_worker_arguments,
    distributed_item_lease,
    distributed_step_runtime_root,
    ensure_project_structure,
    error,
    headline,
    info,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    nvidia_gpu_available,
    ok,
    pip_install_command,
    runtime_environment_tag,
    runtime_python,
    runtime_settings,
    runtime_venv_dir,
    shared_worker_id_for_args,
    shared_workers_enabled_for_args,
    warn,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare runtime dependencies and tools")
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Skip backend/model downloads and only validate the current local setup state.",
    )
    add_shared_worker_arguments(parser)
    parsed_args, _unknown_args = parser.parse_known_args(argv)
    return parsed_args


def venv_python() -> Path:
    return runtime_python()


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def ensure_venv() -> Path:
    headline("Create Python Environment")
    if sys.platform != "win32":
        info("Linux runtime uses the active python3 interpreter with --break-system-packages.")
        return Path(sys.executable).resolve()
    py = venv_python()
    venv_dir = runtime_venv_dir()
    cfg_file = venv_dir / "pyvenv.cfg"
    if py.exists() and cfg_file.exists():
        cfg_text = cfg_file.read_text(encoding="utf-8", errors="ignore").lower()
        if "include-system-site-packages = true" not in cfg_text:
            return py
        warn("Existing venv uses system site-packages and will be recreated for a clean GPU stack.")
        shutil.rmtree(venv_dir, ignore_errors=True)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        check=True,
    )
    return runtime_python()


def module_available(py: Path, module_name: str) -> bool:
    result = run([str(py), "-c", f"import {module_name}"], check=False)
    return result.returncode == 0


def runtime_pip_install_command(py: Path, *args: str) -> list[str]:
    return list(pip_install_command(py, *args))


def runtime_ffmpeg_path(py: Path) -> Path:
    result = run(
        [
            str(py),
            "-c",
            "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())",
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stdout or "").strip() or "imageio-ffmpeg could not resolve ffmpeg.")
    resolved = Path((result.stdout or "").strip().splitlines()[-1]).resolve()
    if not resolved.exists():
        raise RuntimeError(f"Runtime ffmpeg executable was reported but not found: {resolved}")
    return resolved


def install_ffmpeg_binaries(py: Path) -> dict[str, str]:
    install_group(py, "runtime_ffmpeg", ["imageio_ffmpeg"], ["imageio-ffmpeg"])
    ffmpeg_path = runtime_ffmpeg_path(py)
    ok(f"FFmpeg ready from python runtime: {ffmpeg_path}")
    return {"ready": "true", "ffmpeg": str(ffmpeg_path), "provider": "imageio-ffmpeg"}


def run_setup_script(py: Path, script_name: str, *extra_args: str) -> None:
    command = [str(py), str(SCRIPT_DIR / script_name), *extra_args]
    result = run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError((result.stdout or "").strip() or f"{script_name} failed.")
    ok(f"{script_name} completed.")


def install_group(
    py: Path,
    name: str,
    modules: list[str],
    packages: list[str],
    required: bool = True,
    pip_extra_args: list[str] | None = None,
) -> bool:
    if all(module_available(py, module) for module in modules):
        ok(f"{name} already present.")
        return True
    info(f"Installing {name} ...")
    result = run(
        runtime_pip_install_command(py, "--upgrade", *(pip_extra_args or []), *packages),
        check=False,
    )
    log_dir = SCRIPT_DIR / "runtime" / "install_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}_{int(time.time())}.log"
    log_file.write_text(result.stdout or "", encoding="utf-8")
    if result.returncode == 0 and all(module_available(py, module) for module in modules):
        ok(f"{name} installed successfully.")
        return True
    if required:
        error(f"{name} could not be installed. See {log_file}")
        raise RuntimeError(f"{name} could not be installed.")
    warn(f"{name} could not be installed. See {log_file}")
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
        ok("torch is already installed.")
        return True, current_status

    if wants_gpu and not current_status.get("cuda_available"):
        info("NVIDIA GPU detected. Trying to install a CUDA-enabled torch stack ...")
    else:
        info("Installing torch ...")

    log_dir = SCRIPT_DIR / "runtime" / "install_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run([str(py), "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], check=False)
    attempts: list[tuple[str, list[str]]] = []
    if wants_gpu and cuda_index_url:
        attempts.append(
            (
                "torch_cuda",
                [
                    *runtime_pip_install_command(
                        py,
                        "--upgrade",
                        "--force-reinstall",
                        "--no-cache-dir",
                        "--index-url",
                        cuda_index_url,
                        "torch",
                        "torchvision",
                        "torchaudio",
                    ),
                ],
            )
        )
    attempts.append(
        (
            "torch_default",
                [
                    *runtime_pip_install_command(
                        py,
                        "--upgrade",
                        "--force-reinstall",
                        "--no-cache-dir",
                        "torch",
                        "torchvision",
                    "torchaudio",
                ),
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
            ok("torch installed successfully.")
            return True, current_status
        if result.returncode == 0 and current_status.get("available") and not wants_gpu:
            ok("torch installed successfully.")
            return True, current_status

    warn("CUDA torch could not be confirmed. Falling back to CPU if needed.")
    return bool(current_status.get("available")), current_status


def main() -> None:
    args = parse_args()
    headline("Prepare Runtime")
    cfg = ensure_project_structure(write_config_file=True)
    worker_id = shared_worker_id_for_args(args)
    shared_workers = shared_workers_enabled_for_args(cfg, args)
    mark_step_started("00_prepare_runtime", "global")
    if shared_workers:
        info(f"Shared NAS workers: enabled ({worker_id})")
    lease_manager = distributed_item_lease(
        root=distributed_step_runtime_root("00_prepare_runtime", "global"),
        lease_name="global",
        cfg=cfg,
        worker_id=worker_id,
        enabled=shared_workers,
        meta={"step": "00_prepare_runtime", "scope": "global", "worker_id": worker_id},
    )
    acquired = lease_manager.__enter__()
    if not acquired:
        info("Runtime preparation is already running on another worker.")
        lease_manager.__exit__(None, None, None)
        return
    try:
        py = ensure_venv()
        info(f"Runtime-Tag: {runtime_environment_tag()}")
        info(f"Using Python: {py}")
        run(runtime_pip_install_command(py, "--upgrade", "pip", "setuptools<81", "wheel"), check=False)
        ffmpeg_info = install_ffmpeg_binaries(py)
        clone_cfg = cfg.get("cloning", {}) if isinstance(cfg.get("cloning"), dict) else {}
        requested_voice_engine = str(clone_cfg.get("voice_clone_engine", "pyttsx3") or "pyttsx3").strip().lower()
        optional_tts_requested = requested_voice_engine in {"auto", "xtts"} or str(
            os.environ.get("SERIES_ENABLE_OPTIONAL_TTS", "")
        ).strip().lower() in {"1", "true", "yes", "y"}

        core_ok = install_group(
            py,
            "core_ai",
            ["numpy", "PIL", "cv2", "librosa"],
            ["numpy", "Pillow", "opencv-python", "librosa"],
        )
        scene_ok = install_group(py, "scene_detection", ["scenedetect"], ["scenedetect[opencv]"])
        tts_ok = install_group(py, "render_tts", ["pyttsx3"], ["pyttsx3"])
        torch_ok, torch_info = install_torch_stack(py, cfg)
        whisper_ok = install_group(
            py,
            "speech_to_text",
            ["whisper"],
            ["openai-whisper"],
        )
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
            info("Optional XTTS/Coqui packages are not installed automatically in the license-free default path.")

        write_json(
            SCRIPT_DIR / "runtime" / "package_status.json",
            {
                "python": str(py),
                "runtime_tag": runtime_environment_tag(),
                "runtime_dir": str(runtime_venv_dir()),
                "ffmpeg": ffmpeg_info,
                "torch": torch_ok,
                "torch_info": torch_info,
                "nvidia_gpu_detected": nvidia_gpu_available(),
                "core_ai": core_ok,
                "scene_detection": scene_ok,
                "speech_to_text": whisper_ok,
                "facenet_pytorch": facenet_ok,
                "speaker_embeddings": speaker_embeddings_ok,
                "render_tts": tts_ok,
                "voice_cloning": voice_clone_ok,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        info("Running full project setup tasks ...")
        run_setup_script(py, "58_configure_quality_backends.py")
        backend_prepare_args: list[str] = ["--skip-downloads"] if bool(args.skip_downloads) else []
        run_setup_script(py, "59_prepare_quality_backends.py", *backend_prepare_args)
        if torch_info.get("cuda_available"):
            device_names = ", ".join(torch_info.get("device_names") or [])
            ok(f"GPU ready for torch: {device_names}")
        elif nvidia_gpu_available():
            warn("NVIDIA GPU detected, but torch does not report CUDA yet. CPU fallback remains active.")
        mark_step_completed(
            "00_prepare_runtime",
            "global",
            {
                "python": str(py),
                "runtime_tag": runtime_environment_tag(),
                "runtime_dir": str(runtime_venv_dir()),
                "ffmpeg": ffmpeg_info.get("ffmpeg", ""),
                "torch": bool(torch_ok),
                "cuda_available": bool(torch_info.get("cuda_available", False)),
                "voice_cloning": bool(voice_clone_ok),
                "backend_setup_skip_downloads": bool(args.skip_downloads),
            },
        )
        ok("Runtime and full project setup are ready.")
    finally:
        lease_manager.__exit__(None, None, None)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        mark_step_failed("00_prepare_runtime", str(exc), "global")
        error(str(exc))
        raise


