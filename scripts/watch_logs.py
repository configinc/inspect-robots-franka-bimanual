#!/usr/bin/env python3
"""Continuously turn completed eval logs into MP4s and static HTML reports."""

from __future__ import annotations

import argparse
import base64
import html as html_lib
import json
import subprocess
import time
from pathlib import Path


def embed_video_gallery(report: Path, videos: list[Path]) -> None:
    """Add camera videos when the upstream report has no transcript-led composite."""
    if not report.exists() or not videos:
        return
    document = report.read_text()
    if "data:video/mp4" in document:
        return
    players = "".join(
        f"<h3>{html_lib.escape(video.stem)}</h3><video controls muted "
        f'style="width:100%;max-width:960px" src="data:video/mp4;base64,'
        f'{base64.b64encode(video.read_bytes()).decode()}"></video>'
        for video in videos
    )
    gallery = f'<section class="scene"><h2>Run videos</h2>{players}</section>'
    report.write_text(document.replace("</main>", gallery + "</main>"))


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
        videos = log_dir / "videos" / log.stem
        if html.exists() and html.stat().st_mtime >= log.stat().st_mtime:
            embed_video_gallery(html, sorted(videos.glob("*.mp4")))
            continue
        subprocess.run(
            ["inspect-robots", "video", str(log), "--out", str(videos)],
            check=False,
        )
        subprocess.run(
            ["inspect-robots", "view", str(log), "--out", str(html), "--force"],
            check=True,
        )
        embed_video_gallery(html, sorted(videos.glob("*.mp4")))


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
