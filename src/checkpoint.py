"""
Resumable runs: remember finished work so a dead run is not a lost run.

Why this exists: generation and judging are the two slow, paid stages, and
both are rate limited. Before this module, a run that hit the daily token wall
at chunk 90 of 100 - or that you stopped with Ctrl-C - threw away every call
it had already paid for. Now each finished unit is appended to a JSONL file as
it completes, and the next run skips anything already there.

The safety mechanism is the fingerprint. A checkpoint record is only reused
when it was produced by the *same* run configuration: same model, same
temperature, same prompt, same settings. Change the prompt or the model and
every old record is ignored, because reusing them would silently mix two
different generators' output into one dataset and no number in the report
would mean anything.

Records are appended, never rewritten, so a crash mid-write can only ever
damage the last line - which `load` discards.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def fingerprint(**settings) -> str:
    """
    A short hash of everything that would change the output.

    Pass whatever identifies the run - model id, temperature, prompt version,
    thresholds. Keys are sorted so the hash does not depend on argument order.
    """
    blob = json.dumps(settings, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def load(path: Path, fp: str) -> dict[str, dict]:
    """
    Finished records from `path` that belong to this exact run, keyed by `key`.

    Records written under a different fingerprint are ignored rather than
    deleted: switching a setting back restores the matching checkpoint.
    """
    records: dict[str, dict] = {}
    if not Path(path).is_file():
        return records

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A line cut in half by a crash mid-write. Nothing after it in
                # this file is trustworthy either, so stop reading.
                break
            if record.get("fingerprint") == fp and "key" in record:
                records[record["key"]] = record
    return records


def append(path: Path, fp: str, key: str, payload) -> None:
    """
    Record one finished unit of work.

    Flushed and fsynced because the whole point is surviving a process that
    dies without warning - an entry still sitting in the OS buffer when the
    quota error kills the run would not have been worth writing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {"fingerprint": fp, "key": key, "payload": payload}, ensure_ascii=False
    )
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def clear(*paths: Path) -> None:
    """Delete checkpoint files. Used by --fresh and after a successful export."""
    for path in paths:
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass


def summarise(path: Path, fp: str) -> str:
    """One line describing what a resume would skip."""
    usable = len(load(path, fp))
    if not Path(path).is_file():
        return "no checkpoint"
    if not usable:
        return "checkpoint exists but is for a different configuration - ignoring it"
    return f"{usable} unit(s) already done"


if __name__ == "__main__":
    # Round-trip check against a temporary file.
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "cp.jsonl"
        fp_a = fingerprint(model="m1", temperature=0.4)
        fp_b = fingerprint(model="m2", temperature=0.4)

        append(f, fp_a, "chunk-1", [{"question": "Q1"}])
        append(f, fp_a, "chunk-2", [{"question": "Q2"}])
        append(f, fp_b, "chunk-1", [{"question": "WRONG MODEL"}])

        same = load(f, fp_a)
        other = load(f, fp_b)
        print(f"fingerprint A sees {len(same)} record(s): {sorted(same)}")
        print(f"fingerprint B sees {len(other)} record(s): {sorted(other)}")
        assert len(same) == 2 and len(other) == 1
        assert same["chunk-1"]["payload"][0]["question"] == "Q1"

        # A half-written final line must not break the load.
        with open(f, "a", encoding="utf-8") as fh:
            fh.write('{"fingerprint": "' + fp_a + '", "key": "chunk-3", "payl')
        assert len(load(f, fp_a)) == 2
        print("truncated final line ignored, earlier records intact")

        print("summary:", summarise(f, fp_a))
        clear(f)
        assert not f.exists()
        print("OK")
