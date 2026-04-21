#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

from pipeline_common import (
    PROJECT_ROOT,
    SCRIPT_DIR,
    add_shared_worker_arguments,
    current_architecture,
    current_os,
    detect_tool,
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
    platform_tool_filenames,
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


def ffmpeg_asset_spec() -> dict[str, str]:
    os_name = current_os()
    architecture = current_architecture()
    specs = {
        ("windows", "x86_64"): {
            "url": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
            "archive_name": "ffmpeg-win64.zip",
        },
        ("windows", "arm64"): {
            "url": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-winarm64-gpl.zip",
            "archive_name": "ffmpeg-winarm64.zip",
        },
        ("linux", "x86_64"): {
            "url": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz",
            "archive_name": "ffmpeg-linux64.tar.xz",
        },
        ("linux", "arm64"): {
            "url": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linuxarm64-gpl.tar.xz",
            "archive_name": "ffmpeg-linuxarm64.tar.xz",
        },
    }
    spec = specs.get((os_name, architecture))
    if spec is None:
        raise RuntimeError(f"FFmpeg auto-download is not configured for {os_name}/{architecture}.")
    return spec


def ffmpeg_bin_dir() -> Path:
    return PROJECT_ROOT / "tools" / "ffmpeg" / "bin"


def ffmpeg_archive_path() -> Path:
    return PROJECT_ROOT / "tools" / "ffmpeg" / ffmpeg_asset_spec()["archive_name"]


def ffmpeg_extract_members(archive_path: Path) -> dict[str, bytes]:
    expected_names = set(platform_tool_filenames("ffmpeg") + platform_tool_filenames("ffprobe"))
    extracted: dict[str, bytes] = {}
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                member_name = Path(member.filename).name
                if member_name not in expected_names:
                    continue
                extracted[member_name] = archive.read(member)
    else:
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                member_name = Path(member.name).name
                if member_name not in expected_names:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                extracted[member_name] = handle.read()
    return extracted


def install_ffmpeg_binaries() -> dict[str, str]:
    bin_dir = ffmpeg_bin_dir()
    try:
        ffmpeg_path = detect_tool(bin_dir, "ffmpeg")
        return {"ready": "true", "ffmpeg": str(ffmpeg_path)}
    except FileNotFoundError:
        pass

    spec = ffmpeg_asset_spec()
    archive_path = ffmpeg_archive_path()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    info(f"Downloading FFmpeg for {current_os()} {current_architecture()} ...")
    urllib.request.urlretrieve(spec["url"], archive_path)
    extracted = ffmpeg_extract_members(archive_path)
    if not extracted:
        raise RuntimeError(f"Downloaded FFmpeg archive does not contain platform binaries: {archive_path.name}")

    expected_names = set(platform_tool_filenames("ffmpeg") + platform_tool_filenames("ffprobe"))
    for stale_path in bin_dir.iterdir():
        if stale_path.is_file() and stale_path.name in expected_names:
            stale_path.unlink()

    for filename, payload in extracted.items():
        target = bin_dir / filename
        target.write_bytes(payload)
        if current_os() != "windows":
            target.chmod(0o755)

    ffmpeg_path = detect_tool(bin_dir, "ffmpeg")
    ok(f"FFmpeg ready: {ffmpeg_path}")
    return {"ready": "true", "ffmpeg": str(ffmpeg_path), "archive": str(archive_path)}


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
        ffmpeg_info = install_ffmpeg_binaries()
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
            },
        )
        ok("Runtime is ready.")
    finally:
        lease_manager.__exit__(None, None, None)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        mark_step_failed("00_prepare_runtime", str(exc), "global")
        error(str(exc))
        raise

