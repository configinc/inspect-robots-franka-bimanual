#!/usr/bin/env python3
"""Continuously turn completed eval logs into MP4s and static HTML reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def render_completed(log_dir: Path) -> None:
    """Render completed top-level logs whose HTML is missing or stale."""
    for log in sorted(log_dir.glob("*.json")):
        if log.name.endswith(".live.json"):
            continue
        try:
            if json.loads(log.read_text()).get("status") == "started":
                continue
        except (OSError, json.JSONDecodeError):
            continue
        html = log_dir / "html" / f"{log.stem}.html"
        if html.exists() and html.stat().st_mtime >= log.stat().st_mtime:
            continue
        videos = log_dir / "videos" / log.stem
        subprocess.run(
            ["inspect-robots", "video", str(log), "--out", str(videos)],
            check=False,
        )
        subprocess.run(
            ["inspect-robots", "view", str(log), "--out", str(html), "--force"],
            check=True,
        )


def main() -> None:
    """Watch a logs directory, or process it once for setup verification."""
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", nargs="?", type=Path, default=Path("logs"))
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            render_completed(args.log_dir)
        except subprocess.CalledProcessError as exc:
            print(f"render failed: {exc}", flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
