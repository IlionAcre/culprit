"""Consolidates every scattered threshold, model name, and connection default
into one place.

Precedence is CLI flag > culprit.toml > hardcoded fallback below.
Config-file resolution is deliberately a CLI-layer concern only - core
analysis modules (detectors, contrast, adjudicate, cluster, ...) keep their
own defaults in their own function signatures and never import this module;
only cli.py and the plugin registries read from here (see CLAUDE.md).

culprit.toml is a standalone file, not a [tool.culprit] section in
pyproject.toml - mirrors the same choice Litmus made, so the file reads as
culprit's own self-contained documentation with no wrapper section needed,
since the whole file already belongs to culprit."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

_CONFIG_FILENAME = "culprit.toml"


class ConfigError(Exception):
    """Raised when culprit.toml has an invalid value. Deliberately validated
    here rather than left to Typer's own min=/max= option constraints: those
    apply to a *default* value the same as a user-supplied one, but the
    resulting error message blames the CLI flag (e.g. "Invalid value for
    '--min-confidence'") even when the user never touched that flag and the
    bad value came from the config file - this exception names culprit.toml
    and the offending key directly instead."""


@dataclass(frozen=True)
class CulpritConfig:
    database_url: str = "postgresql://localhost:5432/culprit"
    redis_url: str = "redis://localhost:6379/0"
    model: str = "gemini/gemini-2.5-flash-lite"
    max_candidates: int = 5
    # 0.15: matches confidence.py's _DEFAULT_MIN_CONFIDENCE, chosen from the
    # 2026-08-17 richer-feature refit's precision-at-threshold table (see
    # confidence.py's module docstring and CLAUDE.md's "L3 adjudication"
    # section for the numbers this floor was picked from).
    min_confidence: float = 0.15
    ambiguity_margin: float = 0.08
    max_workers: int = 4
    min_reference_runs: int = 3
    max_reference_runs: int = 25
    reference_cosine_max: float = 0.18
    context_token_budget: int = 12_000
    plateau_eps: float = 0.05
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    min_cluster_size: int = 5
    retention_days: int = 90
    log_file: Path = Path("logs/culprit.jsonl")
    log_level: str = "INFO"
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 3


_UNIT_INTERVAL_FIELDS = ("min_confidence", "ambiguity_margin", "plateau_eps")
_NON_NEGATIVE_INT_FIELDS = (
    "max_candidates",
    "max_workers",
    "min_reference_runs",
    "max_reference_runs",
    "context_token_budget",
    "min_cluster_size",
    "retention_days",
    "log_max_bytes",
    "log_backup_count",
)
_NON_NEGATIVE_FLOAT_FIELDS = ("reference_cosine_max",)


def _validate(config: CulpritConfig) -> None:
    for field_name in _UNIT_INTERVAL_FIELDS:
        value = getattr(config, field_name)
        if not 0.0 <= value <= 1.0:
            raise ConfigError(
                f"culprit.toml has {field_name} = {value!r}, "
                "which must be between 0.0 and 1.0"
            )
    for field_name in _NON_NEGATIVE_INT_FIELDS:
        value = getattr(config, field_name)
        if value < 0:
            raise ConfigError(
                f"culprit.toml has {field_name} = {value!r}, "
                "which must not be negative"
            )
    for field_name in _NON_NEGATIVE_FLOAT_FIELDS:
        value = getattr(config, field_name)
        if value < 0.0:
            raise ConfigError(
                f"culprit.toml has {field_name} = {value!r}, "
                "which must not be negative"
            )
    if config.min_reference_runs > config.max_reference_runs:
        raise ConfigError(
            "culprit.toml has min_reference_runs = "
            f"{config.min_reference_runs!r} greater than max_reference_runs "
            f"= {config.max_reference_runs!r}"
        )


def load_config(start_dir: Path | None = None) -> CulpritConfig:
    """Load culprit.toml from `start_dir` (default: the current working
    directory - culprit is documented as run from the project root, so this
    does not walk up parent directories). A missing file, missing key, or
    empty file all fall back to CulpritConfig's own hardcoded default for
    that field - no override is ever required."""
    config_path = (start_dir or Path.cwd()) / _CONFIG_FILENAME
    values: dict = {}

    if config_path.exists():
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        values.update(data)

    if "log_file" in values:
        values["log_file"] = Path(values["log_file"])

    try:
        config = CulpritConfig(**values)
    except TypeError as e:
        raise ConfigError(
            f"culprit.toml has an unrecognized key or value: {e}"
        ) from e

    _validate(config)
    return config
