from pathlib import Path

import pytest

from culprit.config import ConfigError, CulpritConfig, load_config


def test_load_config_falls_back_to_defaults_with_no_culprit_toml(tmp_path):
    config = load_config(start_dir=tmp_path)

    assert config == CulpritConfig()


def test_load_config_falls_back_to_defaults_with_empty_culprit_toml(tmp_path):
    (tmp_path / "culprit.toml").write_text("")

    config = load_config(start_dir=tmp_path)

    assert config == CulpritConfig()


def test_load_config_overrides_only_the_keys_present_in_the_file(tmp_path):
    (tmp_path / "culprit.toml").write_text(
        "min_confidence = 0.7\nmax_candidates = 3\n"
    )

    config = load_config(start_dir=tmp_path)

    assert config.min_confidence == 0.7
    assert config.max_candidates == 3
    # everything else stays default
    assert config.ambiguity_margin == CulpritConfig().ambiguity_margin
    assert config.max_workers == CulpritConfig().max_workers


def test_load_config_converts_the_log_file_path_field(tmp_path):
    (tmp_path / "culprit.toml").write_text('log_file = "custom/log.jsonl"\n')

    config = load_config(start_dir=tmp_path)

    assert config.log_file == Path("custom/log.jsonl")


def test_load_config_reads_connection_and_model_strings(tmp_path):
    (tmp_path / "culprit.toml").write_text(
        'database_url = "postgresql://db/test"\n'
        'redis_url = "redis://cache/1"\n'
        'model = "gemini/custom-model"\n'
    )

    config = load_config(start_dir=tmp_path)

    assert config.database_url == "postgresql://db/test"
    assert config.redis_url == "redis://cache/1"
    assert config.model == "gemini/custom-model"


@pytest.mark.parametrize(
    "field_name,value",
    [("min_confidence", 1.7), ("ambiguity_margin", -0.1), ("plateau_eps", 2.0)],
)
def test_load_config_rejects_out_of_range_unit_interval_fields(tmp_path, field_name, value):
    (tmp_path / "culprit.toml").write_text(f"{field_name} = {value}\n")

    with pytest.raises(ConfigError, match=f"{field_name} = {value!r}"):
        load_config(start_dir=tmp_path)


def test_load_config_rejects_negative_int_fields(tmp_path):
    (tmp_path / "culprit.toml").write_text("max_workers = -1\n")

    with pytest.raises(ConfigError, match="max_workers"):
        load_config(start_dir=tmp_path)


def test_load_config_rejects_negative_reference_cosine_max(tmp_path):
    (tmp_path / "culprit.toml").write_text("reference_cosine_max = -0.5\n")

    with pytest.raises(ConfigError, match="reference_cosine_max"):
        load_config(start_dir=tmp_path)


def test_load_config_rejects_min_reference_runs_above_max(tmp_path):
    (tmp_path / "culprit.toml").write_text(
        "min_reference_runs = 30\nmax_reference_runs = 25\n"
    )

    with pytest.raises(ConfigError, match="min_reference_runs"):
        load_config(start_dir=tmp_path)


def test_config_error_names_culprit_toml_not_a_cli_flag(tmp_path):
    """Typer's own min=/max= validation would blame a flag the user never
    passed. ConfigError exists so the message points at the real source."""
    (tmp_path / "culprit.toml").write_text("min_confidence = 1.7\n")

    with pytest.raises(ConfigError, match="culprit.toml"):
        load_config(start_dir=tmp_path)
