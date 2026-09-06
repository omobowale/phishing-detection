"""Refuse checkpoint reuse across datasets, code, or training configurations."""
import json
from hashlib import sha256
from pathlib import Path


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_experiment(run_dir: Path, identity: dict) -> bool:
    """Return whether checkpoints exist; never adopt an unverified old run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "experiment.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("Experiment identity changed; use a new --run-dir")
    else:
        if any(run_dir.iterdir()):
            raise RuntimeError("Unidentified experiment files exist; use a new --run-dir")
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump(identity, stream, indent=2)
    return any((run_dir / "checkpoints").glob("checkpoint-*"))
