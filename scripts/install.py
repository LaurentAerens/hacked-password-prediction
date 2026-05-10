#!/usr/bin/env python3
"""Auto-detect hardware and install appropriate requirements.

Usage:
  python scripts/install.py --auto        # auto-detect and install
  python scripts/install.py --intel       # force Intel build
  python scripts/install.py --cpu         # force generic CPU
  python scripts/install.py --gpu         # force GPU build
"""
import argparse
import shutil
import subprocess
import sys
import platform
import os
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONSTRAINTS = ROOT / "constraints.txt"


def detect_gpu():
    return shutil.which("nvidia-smi") is not None


def detect_intel():
    proc = platform.processor() or ""
    return "intel" in proc.lower()


def install(req_file: Path, dry_run: bool = False):
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-c", str(CONSTRAINTS)]
    print("Running:", " ".join(cmd))
    if dry_run:
        print("Dry-run: not executing pip install")
        return
    # Run pip from the repository root so relative -r includes resolve correctly
    subprocess.check_call(cmd, cwd=str(ROOT))


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group()
    group.add_argument("--auto", action="store_true", help="Auto-detect hardware (default)")
    group.add_argument("--intel", action="store_true", help="Force Intel-specific requirements")
    group.add_argument("--cpu", action="store_true", help="Force generic CPU requirements")
    group.add_argument("--gpu", action="store_true", help="Force GPU requirements")
    p.add_argument("--dry-run", action="store_true", help="Print command but do not execute installs")
    p.add_argument("--with-ui", action="store_true", help="Also create a .venv-ui and install the Streamlit UI there")
    p.add_argument("--upgrade-pip", action="store_true", help="Upgrade pip in the current Python before installing packages")
    args = p.parse_args()

    if not any((args.auto, args.intel, args.cpu, args.gpu)):
        args.auto = True

    if args.auto:
        if detect_gpu():
            target = "requirements-gpu.txt"
        elif detect_intel():
            target = "requirements-intel.txt"
        else:
            target = "requirements-cpu.txt"
    elif args.intel:
        target = "requirements-intel.txt"
    elif args.gpu:
        target = "requirements-gpu.txt"
    else:
        target = "requirements-cpu.txt"

    req_path = ROOT / target
    if not req_path.exists():
        print(f"Requirements file not found: {req_path}")
        sys.exit(2)

    print(f"Installing using {target} (constraints: {CONSTRAINTS.name})")
    if args.upgrade_pip:
        upg_cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
        print("Running:", " ".join(upg_cmd))
        if not args.dry_run:
            subprocess.check_call(upg_cmd)
        else:
            print("Dry-run: not upgrading pip in current Python")
    install(req_path, dry_run=args.dry_run)

    if args.with_ui:
        venv_dir = ROOT / ".venv-ui"
        ui_reqs = ROOT / "requirements-ui.txt"
        if not ui_reqs.exists():
            print(f"UI requirements not found: {ui_reqs}")
            sys.exit(2)
        print(f"Setting up UI venv at {venv_dir}")
        if args.dry_run:
            print(f"Dry-run: would create venv and install UI into {venv_dir}")
            return
        # create venv (retry once if creation fails due to leftover files)
        def on_rm_error(func, path, exc_info):
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                pass

        try:
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        except subprocess.CalledProcessError:
            if venv_dir.exists():
                print("Initial venv creation failed; removing existing .venv-ui and retrying")
                try:
                    shutil.rmtree(venv_dir, onerror=on_rm_error)
                except Exception as e:
                    print(f"Failed to remove {venv_dir}: {e}")
                    raise
                subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        # determine python executable in venv
        if (venv_dir / "Scripts" / "python.exe").exists():
            venv_python = venv_dir / "Scripts" / "python.exe"
        else:
            venv_python = venv_dir / "bin" / "python"
        # install UI requirements into the venv
        if args.upgrade_pip:
            upg_cmd = [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"]
            print("Running:", " ".join(upg_cmd))
            subprocess.check_call(upg_cmd, cwd=str(ROOT))
        cmd = [str(venv_python), "-m", "pip", "install", "-r", str(ui_reqs)]
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(ROOT))
        # Install minimal UI runtime requirements (joblib, numpy, pandas, plotly)
        ui_runtime = ROOT / "requirements-ui-runtime.txt"
        if ui_runtime.exists():
            cmd2 = [str(venv_python), "-m", "pip", "install", "-r", str(ui_runtime)]
            print("Running:", " ".join(cmd2))
            subprocess.check_call(cmd2, cwd=str(ROOT))


if __name__ == "__main__":
    main()
