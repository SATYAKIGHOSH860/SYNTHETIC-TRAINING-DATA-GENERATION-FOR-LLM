"""Load config.yaml and .env into a single validated config object.

Why a validated object instead of a raw dict: a typo such as
``cfg["validate"]["min_qualty_score"]`` would otherwise surface as a bare
KeyError twenty minutes into a run. Here every section and key is checked once,
at import time, with a message that says exactly what is wrong and where.

Why the API key is checked lazily (``require_api_key``) rather than at import:
the test suite must run offline with no key, and the dashboard must be able to
display finished results without one. Every code path that calls the API calls
``require_api_key`` first, so a missing key still fails before any work starts.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


class ConfigError(RuntimeError):
    """Raised for a missing/invalid config.yaml, a bad override, or a missing API key."""


def configure_console() -> None:
    """Force UTF-8 console output.

    Why: the default Windows console encoding is cp1252, and guideline PDFs
    contain characters (curly quotes, en dashes, micro signs) it cannot encode.
    Without this a harmless progress print crashes the run with
    UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


class Config:
    """Read-only attribute access over the nested config dict: ``cfg.model.id``."""

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(
                f"config has no key '{name}'. Available: {sorted(self._data)}"
            ) from None
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Config is read-only; use with_overrides() to vary a value")

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Config) and self._data == other._data

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)


# ---------------------------------------------------------------------------
# Schema: every key the code relies on, with its expected type.
# ---------------------------------------------------------------------------
_NUM = (int, float)
_SCHEMA: dict[str, dict[str, Any]] = {
    "model": {
        "provider": str, "id": str, "judge_id": str,
        "generation_temperature": _NUM, "judge_temperature": _NUM,
        "reasoning_effort": (str, type(None)), "generation_max_tokens": int, "judge_max_tokens": int,
        "max_retries": int, "request_delay_seconds": _NUM,
        "request_timeout_seconds": _NUM, "requests_per_minute": int,
        "tokens_per_minute": int,
    },
    "domain": {"name": str, "system_role": str},
    "paths": {"raw_dir": str, "cache_dir": str, "output_dir": str, "uploads_dir": str},
    "ingest": {
        "chunk_size": int, "chunk_overlap": int, "min_chunk_chars": int,
        "min_pdf_chars": int, "filter_boilerplate": bool,
        "max_citation_markers": int, "max_author_initials": int, "max_legal_markers": int,
        "max_toc_entries": int, "max_capitalised_ratio": _NUM, "max_numeric_ratio": _NUM,
    },
    "generate": {
        "questions_per_chunk": int, "question_types": list,
        "unanswerable_ratio": _NUM, "abstention_answer": str, "source_reference": str,
        "min_question_chars": int, "min_answer_chars": int,
    },
    "validate": {"min_quality_score": int, "min_criterion_scores": dict, "passage_chars": int,
                 "group_by_chunk": bool},
    "deduplicate": {"embedding_model": str, "embed": str, "similarity_threshold": _NUM,
                    "require_same_numbers": bool, "batch_size": int},
    "export": {"chatml_include_context": bool},
    "agreement": {"sample_size": int, "seed": int},
    "run": {"max_chunks": (int, type(None)), "checkpoint_every": int, "seed": int},
    "web": {"max_chunks_per_upload": int, "max_upload_mb": _NUM, "max_files_per_job": int,
            "show_all_upload_jobs": bool, "public_mode": (bool, str)},
}


def _validate(data: dict[str, Any]) -> None:
    """Check presence, types and sane ranges; raise ConfigError listing every problem."""
    problems: list[str] = []
    for section, keys in _SCHEMA.items():
        block = data.get(section)
        if not isinstance(block, dict):
            problems.append(f"missing section '{section}:'")
            continue
        for key, expected in keys.items():
            allowed = expected if isinstance(expected, tuple) else (expected,)
            names = " or ".join("null" if t is type(None) else t.__name__ for t in allowed)
            if key not in block:
                problems.append(f"missing key '{section}.{key}'")
                continue
            value = block[key]
            # bool is a subclass of int in Python, so "true" would pass an int check.
            if (isinstance(value, bool) and bool not in allowed) or not isinstance(value, allowed):
                problems.append(f"'{section}.{key}' is {type(value).__name__}, expected {names}")
    if problems:
        raise ConfigError("Invalid config.yaml:\n  - " + "\n  - ".join(problems))

    def check(cond: bool, msg: str) -> None:
        if not cond:
            problems.append(msg)

    ing, gen, val, ded = data["ingest"], data["generate"], data["validate"], data["deduplicate"]
    check(ing["chunk_size"] > 0, "ingest.chunk_size must be > 0")
    check(0 <= ing["chunk_overlap"] < ing["chunk_size"], "ingest.chunk_overlap must be in [0, chunk_size)")
    check(gen["questions_per_chunk"] >= 1, "generate.questions_per_chunk must be >= 1")
    check(len(gen["question_types"]) >= 1, "generate.question_types must not be empty")
    check(0.0 <= gen["unanswerable_ratio"] <= 1.0, "generate.unanswerable_ratio must be in [0, 1]")
    check(0 <= val["min_quality_score"] <= 6, "validate.min_quality_score must be in [0, 6]")
    floors = val["min_criterion_scores"]
    check(set(floors) <= {"groundedness", "specificity", "completeness"},
          "validate.min_criterion_scores may only name groundedness, specificity, completeness")
    check(all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 2 for v in floors.values()),
          "validate.min_criterion_scores values must be integers 0-2")
    check(0.0 < ded["similarity_threshold"] <= 1.0, "deduplicate.similarity_threshold must be in (0, 1]")
    check(ded["embed"] in ("question", "question_answer"), "deduplicate.embed must be 'question' or 'question_answer'")
    check(data["run"]["max_chunks"] is None or data["run"]["max_chunks"] > 0, "run.max_chunks must be > 0 or null")
    check(data["run"]["checkpoint_every"] >= 1, "run.checkpoint_every must be >= 1")
    check(data["model"]["max_retries"] >= 1, "model.max_retries must be >= 1")
    check(data["model"]["tokens_per_minute"] > 0, "model.tokens_per_minute must be > 0")
    check(data["model"]["requests_per_minute"] > 0, "model.requests_per_minute must be > 0")
    check(data["web"]["max_chunks_per_upload"] > 0, "web.max_chunks_per_upload must be > 0")
    check(data["web"]["public_mode"] in (True, False, "auto"), "web.public_mode must be auto, true or false")
    if problems:
        raise ConfigError("Invalid config.yaml values:\n  - " + "\n  - ".join(problems))


def _set_dotted(data: dict[str, Any], dotted_key: str, raw_value: str) -> None:
    """Apply one ``section.key=value`` override; the key must already exist.

    Why must it exist: an ablation run with a misspelt key would otherwise
    silently use the default and report a meaningless comparison.
    """
    parts = dotted_key.strip().split(".")
    node: Any = data
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"Unknown config key in override: '{dotted_key}'")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise ConfigError(f"Unknown config key in override: '{dotted_key}'")
    # yaml.safe_load gives "5" -> 5, "0.9" -> 0.9, "null" -> None, "true" -> True
    node[parts[-1]] = yaml.safe_load(raw_value) if isinstance(raw_value, str) else raw_value


def parse_overrides(overrides: Iterable[str] | dict[str, Any] | None) -> list[tuple[str, Any]]:
    if not overrides:
        return []
    if isinstance(overrides, dict):
        return list(overrides.items())
    pairs = []
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override must look like 'section.key=value', got '{item}'")
        key, value = item.split("=", 1)
        pairs.append((key, value))
    return pairs


def load_config(
    path: str | Path = CONFIG_PATH,
    overrides: Iterable[str] | dict[str, Any] | None = None,
) -> Config:
    """Load config.yaml (+ .env), apply dotted overrides, validate, return a Config."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(
            f"config.yaml not found at {path}. It holds every tunable setting; "
            "restore it from the repository before running."
        )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"config.yaml is not valid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigError("config.yaml must contain a mapping of sections at the top level")
    for key, value in parse_overrides(overrides):
        _set_dotted(data, key, value)
    _validate(data)
    return Config(data)


def with_overrides(cfg: Config, overrides: Iterable[str] | dict[str, Any]) -> Config:
    """Return a new Config with overrides applied; ``cfg`` is left unchanged."""
    data = cfg.to_dict()
    for key, value in parse_overrides(overrides):
        _set_dotted(data, key, value)
    _validate(data)
    return Config(data)


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path against the project root, so runs work from any cwd."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


LOCAL_ADDRESSES = ("localhost", "127.0.0.1", "::1")


def _hostname(host: str | None) -> str:
    """'localhost:8501' -> 'localhost', '[::1]:8501' -> '::1'."""
    host = (host or "").strip().lower()
    if host.startswith("["):
        return host[1:host.index("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def is_public(setting: bool | str, server_address: str | None, request_host: str | None) -> bool:
    """Resolve ``web.public_mode`` for one browser session.

    "auto" means public unless the server listens only on this computer AND
    the page was opened from this computer (the request's Host header). Why
    both: a deployment that forgets a flag must fail safe (read-only, no
    server key), and a hosting platform may bind the app to localhost behind
    a proxy, which forwards its public host name. The desktop launcher binds
    to localhost, so local use keeps every control.
    """
    if isinstance(setting, bool):
        return setting
    local_server = (server_address or "").strip().lower() in LOCAL_ADDRESSES
    return not (local_server and _hostname(request_host) in LOCAL_ADDRESSES)


def has_api_key() -> bool:
    return bool(os.environ.get("GROQ_API_KEY", "").strip())


def require_api_key(api_key: str | None = None) -> str:
    """Return the Groq key (explicit argument first, then environment) or fail clearly.

    The key is never included in any message or log.
    """
    key = (api_key or os.environ.get("GROQ_API_KEY", "")).strip()
    if not key:
        raise ConfigError(
            "GROQ_API_KEY is not set. Add a line 'GROQ_API_KEY=gsk_...' to the .env file "
            f"in the project root ({ENV_PATH}). Create a key at https://console.groq.com/keys."
        )
    return key


# ---------------------------------------------------------------------------
# Import-time loading: fail loudly on a missing or malformed config.yaml.
# ---------------------------------------------------------------------------
configure_console()
load_dotenv(ENV_PATH)
CONFIG = load_config()


if __name__ == "__main__":
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Config file  : {CONFIG_PATH}")
    print(f"Generator    : {CONFIG.model.id} (temperature {CONFIG.model.generation_temperature})")
    print(f"Judge        : {CONFIG.model.judge_id} (temperature {CONFIG.model.judge_temperature})")
    print(f"Raw PDFs     : {resolve_path(CONFIG.paths.raw_dir)}")
    print(f"GROQ_API_KEY : {'set' if has_api_key() else 'MISSING'}")
    print(f"HF_TOKEN     : {'set' if os.environ.get('HF_TOKEN') else 'not set (only needed for --push-to-hub)'}")
    demo = with_overrides(CONFIG, ["validate.min_quality_score=5"])
    print(f"Override demo: validate.min_quality_score {CONFIG.validate.min_quality_score} -> "
          f"{demo.validate.min_quality_score}")
    require_api_key()
    print("Config OK.")
