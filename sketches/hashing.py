"""Deterministic 64-bit hashing shared by the sketches.

Python's built-in hash() is salted per process, so the same value would land in a
different counter on every run. BLAKE2b with a per-row salt gives stable, independent
hash functions, which keeps experiments reproducible.
"""
import hashlib


def hash64(value, seed: int = 0) -> int:
    data = str(value).encode("utf-8")
    digest = hashlib.blake2b(data, digest_size=8, salt=seed.to_bytes(8, "little")).digest()
    return int.from_bytes(digest, "little")
