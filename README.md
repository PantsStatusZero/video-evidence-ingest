# video-evidence-ingest

Open-source toolkit for reproducible video ingestion, transcription, frame extraction, OCR, metadata capture, and provenance-aware evidence packaging.

This repository contains a bounded integration-test backend for encrypted, per-job video ingestion. Each run accepts only an opaque job reference, validates a strict technical envelope, isolates processing in an ephemeral workspace, and encrypts chunked evidence before returning it to a private result store.

The public workflow is intended for development, provider compatibility checks, regression testing, and release qualification. It is not a general-purpose production processing service.
