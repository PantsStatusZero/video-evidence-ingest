from pathlib import Path

from video_evidence_ingest.cli import parse_vtt


def test_parse_vtt(tmp_path: Path):
    p = tmp_path / "sample.vtt"
    p.write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.000\nHello <c>world</c>\n\n"
        "00:00:02.000 --> 00:00:04.000\nHello world\n\n"
        "00:00:04.000 --> 00:00:06.000\nSecond line\n",
        encoding="utf-8",
    )
    assert parse_vtt(p) == [
        {"start": "00:00:00.000", "end": "00:00:04.000", "text": "Hello world"},
        {"start": "00:00:04.000", "end": "00:00:06.000", "text": "Second line"},
    ]
