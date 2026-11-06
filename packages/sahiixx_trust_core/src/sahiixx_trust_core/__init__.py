"""Standalone, ML-independent signing and audit primitives."""
from .core import (TrustCoreError, canonical_bytes, content_sha256, sign_envelope,
                   verify_envelope, append_event, verify_event_chain)
__all__ = ["TrustCoreError", "canonical_bytes", "content_sha256", "sign_envelope",
           "verify_envelope", "append_event", "verify_event_chain"]
