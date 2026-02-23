#!/usr/bin/env python
"""
Frontend composite runner for React/Vite app.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from loguru import logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frontend composite runner")
    parser.add_argument(
        "--mode",
        choices=("dev", "build", "preview"),
        default="dev",
        help="npm script to run",
    )
    parser.add_argument(
        "--frontend-dir",
        default=str(Path(__file__).resolve().parents[2] / "frontend"),
        help="path to React/Vite frontend directory",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    frontend_dir = Path(args.frontend_dir).resolve()

    if not frontend_dir.exists():
        logger.error("Frontend directory not found: {}", frontend_dir)
        return 2
    if not (frontend_dir / "package.json").exists():
        logger.error("package.json not found in frontend directory: {}", frontend_dir)
        return 2
    if shutil.which("npm") is None:
        logger.error("npm is not installed or not available in PATH")
        return 2

    logger.info("=" * 80)
    logger.info("Trading frontend runner")
    logger.info("Directory: {}", frontend_dir)
    logger.info("Mode: {}", args.mode)
    logger.info("=" * 80)

    command = ["npm", "run", args.mode]
    logger.info("Executing: {}", " ".join(command))
    completed = subprocess.run(command, cwd=str(frontend_dir))
    return int(completed.returncode)


if __name__ == "__main__":
    sys.exit(main())
