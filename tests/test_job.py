import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from video_evidence_ingest.job import (
    JobValidationError,
    decrypt_result_chunks,
    encrypt_result_chunks,
    validate_job,
)


JOB = "0123456789abcdef0123456789abcdef"


def envelope(**extra):
    data = {
        "schema": "VIDEO_EVIDENCE_JOB/1",
        "job_id": JOB,
        "source_url": "https://example.test/video",
        "capture": {"max_height": 480, "retain_media": True, "asr_model": "base"},
        "result_key_b64": base64.b64encode(b"x" * 32).decode(),
        "created_at": 1000,
        "expires_at": 2000,
    }
    data.update(extra)
    return data


def test_valid_job():
    job = validate_job(envelope(), JOB, now=1500)
    assert job["job_id"] == JOB
    assert job["max_height"] == 480


def test_identity_metadata_is_rejected():
    with pytest.raises(JobValidationError):
        validate_job(envelope(requester="internal-team"), JOB, now=1500)


def test_job_reference_must_match():
    with pytest.raises(JobValidationError):
        validate_job(envelope(), "f" * 32, now=1500)


def test_expired_job_is_rejected():
    with pytest.raises(JobValidationError):
        validate_job(envelope(), JOB, now=2000)


def test_invalid_url_scheme_is_rejected():
    with pytest.raises(JobValidationError):
        validate_job(envelope(source_url="file:///tmp/input"), JOB, now=1500)


def test_result_encryption_round_trip(tmp_path: Path):
    source = tmp_path / "source.bin"
    restored = tmp_path / "restored.bin"
    result = tmp_path / "result"
    source.write_bytes((b"synthetic evidence\x00" * 13) + b"end")
    key = b"k" * 32
    encrypt_result_chunks(source, result, key, JOB, chunk_size=31)
    decrypt_result_chunks(result, restored, key, JOB)
    assert restored.read_bytes() == source.read_bytes()


def test_two_jobs_are_filesystem_isolated(tmp_path: Path):
    jobs = [("1" * 32, b"alpha" * 37, b"a" * 32), ("2" * 32, b"beta" * 41, b"b" * 32)]

    def run(item):
        job_id, content, key = item
        root = tmp_path / job_id
        root.mkdir()
        source = root / "source.bin"
        restored = root / "restored.bin"
        result = root / "result"
        source.write_bytes(content)
        encrypt_result_chunks(source, result, key, job_id, chunk_size=29)
        decrypt_result_chunks(result, restored, key, job_id)
        return restored.read_bytes(), {p.relative_to(root).as_posix() for p in root.rglob("*")}

    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(run, jobs))
    assert outputs[0][0] == jobs[0][1]
    assert outputs[1][0] == jobs[1][1]
    assert all(not any(other in path for path in paths) for (_, paths), other in zip(outputs, (jobs[1][0], jobs[0][0])))
