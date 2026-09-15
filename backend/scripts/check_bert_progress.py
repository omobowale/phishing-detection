"""One-shot, human-readable status check for the BERT fine-tuning run.

Usage:
    python -m scripts.check_bert_progress
"""
import re
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = BACKEND_ROOT / "ml" / "saved_models" / "bert_runs" / "full.log"
CHECKPOINT_DIR = BACKEND_ROOT / "ml" / "saved_models" / "bert_runs" / "full" / "checkpoints"
METRICS_PATH = BACKEND_ROOT / "ml" / "saved_models" / "bert_runs" / "full" / "metrics.json"

_STEP_RE = re.compile(r"(\d+)/(\d+) \[([^<]+)<([^,]+),\s*([^\]]+)\]")


def main() -> None:
    if METRICS_PATH.exists():
        print(f"COMPLETE -- results at {METRICS_PATH}")
        return

    if not LOG_PATH.exists():
        print(f"No log found at {LOG_PATH} -- has the run ever been started?")
        return

    raw = LOG_PATH.read_bytes()[-8000:]
    # PowerShell's `*>>` redirect writes UTF-16LE; UTF-8 decoding of that
    # mangles every line into space-separated characters and silently breaks
    # the progress regex below. An arbitrary byte-offset slice can also land
    # mid-codeunit, so try both alignments before falling back to UTF-8.
    tail = None
    for offset in (0, 1):
        try:
            tail = raw[offset:].decode("utf-16-le")
            break
        except UnicodeDecodeError:
            continue
    if tail is None:
        tail = raw.decode("utf-8", errors="replace")
    matches = list(_STEP_RE.finditer(tail))
    if not matches:
        print("Log exists but no progress line found yet (still loading model/data, or just crashed).")
        last_progress_pos = -1
    else:
        last = matches[-1]
        step, total, elapsed, remaining, rate = last.groups()
        pct = 100 * int(step) / int(total)
        print(f"Step {step}/{total} ({pct:.1f}%) -- elapsed {elapsed}, ~{remaining} remaining, {rate}")
        last_progress_pos = last.end()

    # Only flag a crash if the error appears AFTER the last progress line --
    # otherwise a resumed run's tail still contains the previous crash's
    # traceback and would falsely report itself as currently crashed.
    error_tail = tail[last_progress_pos + 1:]
    if "RuntimeError" in error_tail or "Traceback" in error_tail:
        print("WARNING: an error appears after the last progress line -- it may have crashed. Check the log directly.")

    if CHECKPOINT_DIR.exists():
        checkpoints = sorted(int(p.name.split("-")[1]) for p in CHECKPOINT_DIR.glob("checkpoint-*"))
        if checkpoints:
            print(f"Latest saved checkpoint: step {checkpoints[-1]} (safe resume point if it crashed)")

    idle_seconds = time.time() - LOG_PATH.stat().st_mtime
    if idle_seconds > 300:
        print(
            f"WARNING: log hasn't been written to in {idle_seconds / 60:.1f} minutes -- "
            "likely stopped (crashed or interrupted). Re-run: python -u -m ml.training.train_email_bert"
        )
    else:
        print(f"Log last updated {idle_seconds:.0f}s ago -- still alive.")


if __name__ == "__main__":
    main()
