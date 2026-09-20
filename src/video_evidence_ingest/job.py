from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tarfile
import time
from pathlib import Path
from urllib.parse import urlparse

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .cli import ingest

JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")
ALLOWED_TOP_LEVEL = {
    "schema",
    "job_id",
    "source_url",
    "capture",
    "result_key_b64",
    "created_at",
    "expires_at",
}
ALLOWED_CAPTURE = {"max_height", "retain_media", "asr_model"}
ALLOWED_ASR_MODELS = {"tiny", "base", "small"}


class JobValidationError(ValueError):
    pass


def validate_job(data: dict, expected_job_id: str, now: int | None = None) -> dict:
    if not isinstance(data, dict):
        raise JobValidationError("job envelope must be a JSON object")
    unknown = set(data) - ALLOWED_TOP_LEVEL
    if unknown:
        raise JobValidationError("job envelope contains unsupported fields")
    if data.get("schema") != "VIDEO_EVIDENCE_JOB/1":
        raise JobValidationError("unsupported job schema")
    job_id = data.get("job_id")
    if job_id != expected_job_id or not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
        raise JobValidationError("job reference mismatch")
    source_url = data.get("source_url")
    if not isinstance(source_url, str) or not source_url:
        raise JobValidationError("source URL missing")
    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise JobValidationError("source URL scheme is not allowed")
    capture = data.get("capture", {})
    if not isinstance(capture, dict) or (set(capture) - ALLOWED_CAPTURE):
        raise JobValidationError("capture profile contains unsupported fields")
    max_height = capture.get("max_height", 720)
    if not isinstance(max_height, int) or not 144 <= max_height <= 1080:
        raise JobValidationError("max_height outside bounded range")
    retain_media = capture.get("retain_media", False)
    if not isinstance(retain_media, bool):
        raise JobValidationError("retain_media must be boolean")
    asr_model = capture.get("asr_model", "base")
    if asr_model not in ALLOWED_ASR_MODELS:
        raise JobValidationError("ASR model outside bounded set")
    created_at = data.get("created_at")
    expires_at = data.get("expires_at")
    if not isinstance(created_at, int) or not isinstance(expires_at, int):
        raise JobValidationError("job timestamps missing")
    current = int(time.time()) if now is None else now
    if created_at > current + 300:
        raise JobValidationError("job creation time is in the future")
    if expires_at <= current:
        raise JobValidationError("job expired")
    if expires_at - created_at > 86400:
        raise JobValidationError("job validity window is too large")
    key_text = data.get("result_key_b64")
    if not isinstance(key_text, str):
        raise JobValidationError("result key missing")
    try:
        result_key = base64.b64decode(key_text, validate=True)
    except Exception as exc:
        raise JobValidationError("result key encoding invalid") from exc
    if len(result_key) != 32:
        raise JobValidationError("result key must be 256 bits")
    return {
        "job_id": job_id,
        "source_url": source_url,
        "max_height": max_height,
        "retain_media": retain_media,
        "asr_model": asr_model,
        "result_key": result_key,
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encrypt_result_chunks(
    source_tar: Path,
    destination: Path,
    key: bytes,
    job_id: str,
    chunk_size: int = 32 * 1024 * 1024,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    aes = AESGCM(key)
    chunks = []
    index = 0
    with source_tar.open("rb") as handle:
        while True:
            plain = handle.read(chunk_size)
            if not plain:
                break
            nonce = os.urandom(12)
            aad = f"{job_id}:{index}".encode("ascii")
            cipher = aes.encrypt(nonce, plain, aad)
            name = f"chunk-{index:05d}.bin"
            (destination / name).write_bytes(cipher)
            chunks.append(
                {
                    "index": index,
                    "name": name,
                    "nonce_b64": base64.b64encode(nonce).decode("ascii"),
                    "aad": aad.decode("ascii"),
                    "plaintext_bytes": len(plain),
                    "ciphertext_bytes": len(cipher),
                    "ciphertext_sha256": _sha256(cipher),
                }
            )
            index += 1
    manifest = {
        "schema": "VIDEO_EVIDENCE_RESULT/1",
        "job_id": job_id,
        "cipher": "AES-256-GCM",
        "chunk_size": chunk_size,
        "chunks": chunks,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def decrypt_result_chunks(source: Path, destination: Path, key: bytes, job_id: str) -> None:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "VIDEO_EVIDENCE_RESULT/1":
        raise JobValidationError("unsupported result schema")
    if manifest.get("job_id") != job_id:
        raise JobValidationError("result job reference mismatch")
    if manifest.get("cipher") != "AES-256-GCM":
        raise JobValidationError("unsupported result cipher")
    aes = AESGCM(key)
    with destination.open("wb") as output:
        for expected_index, chunk in enumerate(manifest.get("chunks", [])):
            if chunk.get("index") != expected_index:
                raise JobValidationError("result chunk order invalid")
            name = chunk.get("name")
            if not isinstance(name, str) or not re.fullmatch(r"chunk-[0-9]{5}\.bin", name):
                raise JobValidationError("result chunk name invalid")
            cipher = (source / name).read_bytes()
            if _sha256(cipher) != chunk.get("ciphertext_sha256"):
                raise JobValidationError("result chunk digest mismatch")
            nonce = base64.b64decode(chunk["nonce_b64"], validate=True)
            aad = chunk["aad"].encode("ascii")
            if aad != f"{job_id}:{expected_index}".encode("ascii"):
                raise JobValidationError("result chunk binding invalid")
            output.write(aes.decrypt(nonce, cipher, aad))


def execute_job(request_path: Path, expected_job_id: str, workspace: Path, result_dir: Path) -> int:
    data = json.loads(request_path.read_text(encoding="utf-8"))
    job = validate_job(data, expected_job_id)
    evidence = workspace / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    try:
        code = ingest(
            job["source_url"],
            evidence,
            job["max_height"],
            job["retain_media"],
            job["asr_model"],
        )
    except Exception as exc:
        (evidence / "STATUS").write_text("FAILED\n", encoding="utf-8")
        (evidence / "diagnostic.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8", errors="replace"
        )
        code = 1
    tar_path = workspace / "evidence.tar"
    with tarfile.open(tar_path, "w") as archive:
        archive.add(evidence, arcname="evidence")
    encrypt_result_chunks(tar_path, result_dir, job["result_key"], expected_job_id)
    return code
