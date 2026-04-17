#!/usr/bin/env python3
from __future__ import annotations
import platform
RESET="\033[0m"; BOLD="\033[1m"; YELLOW="\033[33m"; RED="\033[31m"; GREEN="\033[32m"; BLUE="\033[34m"
def enable_ansi():
    if platform.system().lower()!="windows": return
    try:
        import ctypes
        kernel32=ctypes.windll.kernel32; handle=kernel32.GetStdHandle(-11); mode=ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)): kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception: pass
def info(text): print(f"{BLUE}[INFO]{RESET} {text}")
def ok(text): print(f"{GREEN}[OK]{RESET}   {text}")
def warn(text): print(f"{YELLOW}[WARN]{RESET} {text}")
def error(text): print(f"{RED}[ERROR]{RESET} {text}")
def headline(text):
    print(); print(f"{BOLD}{'='*72}{RESET}"); print(f"{BOLD}{text}{RESET}"); print(f"{BOLD}{'='*72}{RESET}")

