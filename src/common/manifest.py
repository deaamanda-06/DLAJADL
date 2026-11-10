import hashlib
import json
from pathlib import Path

AKAR = Path(__file__).resolve().parents[2]

def data_hash(data_bytes: bytes = b"") -> str:
    """Menghitung hash data simulasi/dummy tanpa mengecek data/latih.csv."""
    hasher = hashlib.sha256()
    hasher.update(data_bytes)
    return hasher.hexdigest()[:16]

def catat_manifest(nim: str, model: str, seed: int):
    """Fungsi pembantu manifest opsional."""
    return {"nim": nim, "model": model, "seed": seed}