#!/usr/bin/env python3
"""Continuously turn completed eval logs into MP4s and static HTML reports."""

from __future__ import annotations

import argparse
import base64
import html as html_lib
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

CAMERAS = (
    ("left_wrist_cam", "LEFT"),
    ("exterior_cam", "CENTER"),
    ("right_wrist_cam", "RIGHT"),
)


def _video_identity(video: Path) -> tuple[str, int, str]:
    for index, (camera, label) in enumerate(CAMERAS):
        suffix = f"_{camera}"
        if video.stem.endswith(suffix):
            return video.stem[: -len(suffix)], index, label
    return video.stem, len(CAMERAS), video.stem


def embed_video_gallery(report: Path, videos: list[Path]) -> None:
    """Add synchronized LEFT-CENTER-RIGHT camera rows to a rendered report."""
    if not report.exists() or not videos:
        return
    document = report.read_text()
    if 'data-sync-gallery="true"' in document:
        return
    groups: dict[str, list[tuple[int, str, Path]]] = defaultdict(list)
    for video in videos:
        trial, index, label = _video_identity(video)
        groups[trial].append((index, label, video))
    galleries = []
    for trial, streams in sorted(groups.items()):
        players = "".join(
            f"<figure><figcaption>{html_lib.escape(label)}</figcaption>"
            f'<video muted playsinline preload="metadata" '
            f'src="data:video/mp4;base64,{base64.b64encode(video.read_bytes()).decode()}">'
            "</video></figure>"
            for _index, label, video in sorted(streams)
        )
        galleries.append(
            f'<section class="scene sync-video-gallery" data-sync-gallery="true">'
            f"<h2>Run videos · {html_lib.escape(trial)}</h2>"
            f'<div class="sync-video-row">{players}</div>'
            '<div class="sync-video-controls"><button type="button">Play</button>'
            '<input type="range" min="0" max="1000" value="0" aria-label="Playback">'
            "<span>0:00 / 0:00</span></div></section>"
        )
    assets = """
<style>
.sync-video-row { display:flex; gap:12px; width:100%; }
.sync-video-row figure { flex:1 1 0; min-width:0; margin:0; }
.sync-video-row figcaption { font-weight:700; margin:0 0 6px; text-align:center; }
.sync-video-row video { display:block; width:100%; background:#000; }
.sync-video-controls { display:flex; gap:10px; align-items:center; margin-top:12px; }
.sync-video-controls input { flex:1; }
</style>
<script>
document.addEventListener('DOMContentLoaded', () => {
document.querySelectorAll('[data-sync-gallery="true"]').forEach((gallery) => {
  const videos = Array.from(gallery.querySelectorAll('video'));
  const button = gallery.querySelector('button');
  const seek = gallery.querySelector('input');
  const clock = gallery.querySelector('span');
  const duration = () => Math.min(...videos.map((v) => v.duration).filter(Number.isFinite));
  const stamp = (seconds) =>
    `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
  const update = () => {
    const total = duration();
    const current = videos[0]?.currentTime || 0;
    if (Number.isFinite(total)) seek.value = String(Math.round(current / total * 1000));
    clock.textContent = `${stamp(current)} / ${stamp(Number.isFinite(total) ? total : 0)}`;
  };
  button.addEventListener('click', () => {
    if (videos.some((video) => video.paused)) {
      const current = videos[0]?.currentTime || 0;
      videos.forEach((video) => { video.currentTime = current; void video.play(); });
      button.textContent = 'Pause';
    } else {
      videos.forEach((video) => video.pause());
      button.textContent = 'Play';
    }
  });
  seek.addEventListener('input', () => {
    const total = duration();
    if (!Number.isFinite(total)) return;
    videos.forEach((video) => {
      video.pause();
      video.currentTime = total * Number(seek.value) / 1000;
    });
    button.textContent = 'Play';
    update();
  });
  videos[0]?.addEventListener('timeupdate', update);
  videos.forEach((video) => video.addEventListener('loadedmetadata', update));
});
});
</script>
"""
    report.write_text(
        document.replace("</head>", assets + "</head>").replace(
            "</main>", "".join(galleries) + "</main>"
        )
    )


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
            [
                "inspect-robots",
                "view",
                str(log),
                "--out",
                str(html),
                "--force",
                "--no-video",
            ],
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
