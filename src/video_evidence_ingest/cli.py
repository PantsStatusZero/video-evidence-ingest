from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


def run_capture(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_vtt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cues: list[dict] = []
    i = 0
    ts = re.compile(r"^(?P<start>\d{2}:\d{2}(?::\d{2})?[.,]\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}(?::\d{2})?[.,]\d{3})")
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == "WEBVTT" or line.startswith(("NOTE", "STYLE", "REGION")):
            i += 1
            continue
        m = ts.match(line)
        if not m and i + 1 < len(lines):
            m = ts.match(lines[i + 1].strip())
            if m:
                i += 1
        if not m:
            i += 1
            continue
        start, end = m.group("start"), m.group("end")
        i += 1
        parts: list[str] = []
        while i < len(lines) and lines[i].strip():
            s = re.sub(r"<[^>]+>", "", lines[i]).strip()
            if s:
                parts.append(s)
            i += 1
        cue_text = re.sub(r"\s+", " ", " ".join(parts)).strip()
        if cue_text:
            if cues and cues[-1]["text"] == cue_text:
                cues[-1]["end"] = end
            else:
                cues.append({"start": start, "end": end, "text": cue_text})
        i += 1
    return cues


def write_transcript(cues: list[dict], out_dir: Path) -> None:
    with (out_dir / "transcript.jsonl").open("w", encoding="utf-8") as f:
        for cue in cues:
            f.write(json.dumps(cue, ensure_ascii=False) + "\n")
    with (out_dir / "transcript.md").open("w", encoding="utf-8") as f:
        f.write("# Transcript\n\n")
        for cue in cues:
            f.write(f"- `{cue['start']}–{cue['end']}` {cue['text']}\n")


def seconds_to_timestamp(value: float) -> str:
    total_ms = max(0, int(round(value * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def select_media_file(work_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for ext in ("*.mp4", "*.mkv", "*.webm", "*.mov", "*.m4v"):
        candidates.extend(work_dir.glob(ext))
    candidates = [p for p in candidates if not p.name.endswith(".part")]
    return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


def run_asr(media: Path, out_dir: Path, model_name: str) -> tuple[list[dict], dict]:
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(media), beam_size=5, vad_filter=True, condition_on_previous_text=False)
    cues: list[dict] = []
    for seg in segments:
        text = re.sub(r"\s+", " ", (seg.text or "")).strip()
        if text:
            cues.append({"start": seconds_to_timestamp(float(seg.start)), "end": seconds_to_timestamp(float(seg.end)), "text": text})
    if cues:
        write_transcript(cues, out_dir)
    return cues, {"model": model_name, "language": getattr(info, "language", None), "language_probability": getattr(info, "language_probability", None)}


def extract_frames(media: Path, out_dir: Path) -> list[Path]:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    periodic = frames_dir / "periodic_%04d.jpg"
    p = run_capture(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(media), "-vf", "fps=1/60,scale='min(1280,iw)':-2", "-q:v", "5", str(periodic)])
    (out_dir / "ffmpeg-frames.log").write_text(p.stdout or "", encoding="utf-8", errors="replace")
    return sorted(frames_dir.glob("periodic_*.jpg"))


def extract_visual_text(out_dir: Path) -> list[dict]:
    tesseract = shutil.which("tesseract")
    if not tesseract:
        return []
    rows: list[dict] = []
    for path in sorted((out_dir / "frames").glob("periodic_*.jpg"))[:120]:
        m = re.search(r"_(\d+)\.jpg$", path.name)
        index = int(m.group(1)) if m else len(rows) + 1
        p = run_capture([tesseract, str(path), "stdout", "--psm", "6"])
        text = re.sub(r"\s+", " ", p.stdout or "").strip()
        if text:
            rows.append({"timestamp_seconds": max(0, (index - 1) * 60), "text": text})
    if rows:
        with (out_dir / "visual_text.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def safe_info(info: dict) -> dict:
    keys = ["id", "title", "channel", "channel_id", "uploader", "duration", "upload_date", "timestamp", "webpage_url", "extractor", "extractor_key", "availability", "language", "license", "age_limit", "categories", "tags", "view_count", "like_count"]
    return {k: info.get(k) for k in keys if info.get(k) is not None}


def ingest(source_url: str, out_dir: Path, max_height: int, retain_media: bool, asr_model: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_work"
    work_dir.mkdir(exist_ok=True)
    started = time.time()
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp or not shutil.which("ffmpeg"):
        (out_dir / "diagnostic.txt").write_text("required runtime dependency missing\n", encoding="utf-8")
        return 2
    version = run_capture([ytdlp, "--version"]).stdout.strip()
    ffmpeg_version = (run_capture(["ffmpeg", "-version"]).stdout or "").splitlines()[0]
    node_version = run_capture(["node", "--version"]).stdout.strip() if shutil.which("node") else None
    cmd = [ytdlp, "--quiet", "--no-warnings", "--no-progress", "--no-playlist", "--js-runtimes", "node", "--write-info-json", "--write-description", "--write-subs", "--write-auto-subs", "--sub-langs", "en.*,en", "--sub-format", "vtt", "--merge-output-format", "mp4", "--extractor-args", "youtube:player_client=mweb,default", "-f", f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b", "-o", str(work_dir / "source.%(ext)s")]
    provider_home = os.environ.get("POT_PROVIDER_HOME", "").strip()
    if provider_home:
        cmd.extend(["--extractor-args", f"youtubepot-bgutilscript:server_home={provider_home}"])
    cmd.append(source_url)
    result = run_capture(cmd)
    (out_dir / "provider.log").write_text(result.stdout or "", encoding="utf-8", errors="replace")
    if result.returncode != 0:
        (out_dir / "STATUS").write_text("ACQUISITION_FAILED\n", encoding="utf-8")
        return result.returncode or 1
    info_files = list(work_dir.glob("*.info.json"))
    info = json.loads(info_files[0].read_text(encoding="utf-8")) if info_files else {}
    (out_dir / "source.info.json").write_text(json.dumps(safe_info(info), indent=2, ensure_ascii=False), encoding="utf-8")
    media = select_media_file(work_dir)
    frames = extract_frames(media, out_dir) if media else []
    visual_text = extract_visual_text(out_dir)
    cues: list[dict] = []
    transcript_source = "NONE"
    vtts = sorted(work_dir.glob("*.vtt"))
    if vtts:
        chosen = max(vtts, key=lambda p: p.stat().st_size)
        cues = parse_vtt(chosen)
        if cues:
            transcript_source = "PROVIDER_CAPTION"
            write_transcript(cues, out_dir)
    asr_meta: dict = {}
    if not cues and media:
        cues, asr_meta = run_asr(media, out_dir, asr_model)
        if cues:
            transcript_source = f"LOCAL_ASR:{asr_model}"
    if media and retain_media:
        shutil.copy2(media, out_dir / f"source{media.suffix.lower()}")
    for path in work_dir.glob("*.description"):
        shutil.copy2(path, out_dir / "description.txt")
        break
    provenance = {"schema": "VIDEO_EVIDENCE_PROVENANCE/1", "source": safe_info(info), "runtime": {"repository": os.environ.get("GITHUB_REPOSITORY"), "run_id": os.environ.get("GITHUB_RUN_ID"), "sha": os.environ.get("GITHUB_SHA"), "runner_os": os.environ.get("RUNNER_OS"), "runner_arch": os.environ.get("RUNNER_ARCH")}, "tooling": {"yt_dlp": version, "ffmpeg": ffmpeg_version, "node": node_version, "python": platform.python_version()}, "max_height": max_height, "transcript_source": transcript_source, "transcript_cues": len(cues), "asr": asr_meta or None, "frame_count": len(frames), "visual_ocr_samples": len(visual_text), "source_media_sha256": sha256_file(media) if media else None, "elapsed_seconds": round(time.time() - started, 3)}
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    checksum_lines = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and "_work" not in path.parts and path.name != "checksums.sha256":
            checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(out_dir).as_posix()}")
    (out_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    (out_dir / "STATUS").write_text("PASS\n", encoding="utf-8")
    shutil.rmtree(work_dir, ignore_errors=True)
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Video evidence ingestion")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-height", type=int, default=720)
    parser.add_argument("--retain-media", action="store_true")
    parser.add_argument("--asr-model", default="base")
    args = parser.parse_args(argv)
    source_url = os.environ.get("SOURCE_URL", "").strip()
    if not source_url:
        print("source fixture is not configured", file=sys.stderr)
        return 2
    try:
        return ingest(source_url, Path(args.out), args.max_height, args.retain_media, args.asr_model)
    except Exception as exc:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "diagnostic.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8", errors="replace")
        (out_dir / "STATUS").write_text("FAILED\n", encoding="utf-8")
        print("ingest failed; diagnostic retained in encrypted evidence", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
