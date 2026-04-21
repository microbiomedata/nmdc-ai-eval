"""Tests for the direct per-field env-triad scorer in run_suite.py."""

import json

from nmdc_ai_eval.run_suite import _env_triad_score, _try_parse_env_triad


def _make_ideal(broad: str, local: str, medium: str) -> str:
    return json.dumps(
        {
            "metadata_fields": [
                {"field_name": "env_broad_scale", "reason": "", "value": broad},
                {"field_name": "env_local_scale", "reason": "", "value": local},
                {"field_name": "env_medium", "reason": "", "value": medium},
            ]
        }
    )


def _make_fenced(broad: str, local: str, medium: str) -> str:
    return "```json\n" + _make_ideal(broad, local, medium) + "\n```"


IDEAL = _make_ideal(
    "terrestrial biome [ENVO:00000446]",
    "rhizosphere [ENVO:00005801]",
    "soil [ENVO:00001998]",
)


class TestTryParseEnvTriad:
    def test_raw_json(self) -> None:
        result = _try_parse_env_triad(IDEAL)
        assert result["broad"] == "terrestrial biome [ENVO:00000446]"
        assert result["local"] == "rhizosphere [ENVO:00005801]"
        assert result["medium"] == "soil [ENVO:00001998]"

    def test_fenced_json(self) -> None:
        fenced = _make_fenced(
            "terrestrial biome [ENVO:00000446]",
            "rhizosphere [ENVO:00005801]",
            "soil [ENVO:00001998]",
        )
        result = _try_parse_env_triad(fenced)
        assert result["broad"] == "terrestrial biome [ENVO:00000446]"

    def test_none_input(self) -> None:
        result = _try_parse_env_triad(None)
        assert result == {"broad": None, "local": None, "medium": None}

    def test_non_json(self) -> None:
        result = _try_parse_env_triad("I cannot determine the env triad values.")
        assert result == {"broad": None, "local": None, "medium": None}

    def test_json_embedded_in_prose(self) -> None:
        prose = (
            "Based on the biosample metadata, here are my suggestions:\n"
            + _make_ideal(
                "terrestrial biome [ENVO:00000446]",
                "rhizosphere [ENVO:00005801]",
                "soil [ENVO:00001998]",
            )
            + "\nPlease use these values."
        )
        result = _try_parse_env_triad(prose)
        assert result["broad"] == "terrestrial biome [ENVO:00000446]"
        assert result["medium"] == "soil [ENVO:00001998]"

    def test_malformed_json(self) -> None:
        result = _try_parse_env_triad("{not valid json}")
        assert result == {"broad": None, "local": None, "medium": None}

    def test_list_instead_of_dict(self) -> None:
        result = _try_parse_env_triad("[1, 2, 3]")
        assert result == {"broad": None, "local": None, "medium": None}

    def test_nan_input(self) -> None:
        # pandas passes NaN (float) for missing cells — the parser must
        # not crash when a scorer error left case_ideal unpopulated.
        import math

        result = _try_parse_env_triad(math.nan)  # type: ignore[arg-type]
        assert result == {"broad": None, "local": None, "medium": None}


class TestShortLabel:
    def test_nan_input_does_not_crash(self) -> None:
        # Mirrors _try_parse_env_triad regression: NaN rows (lost results)
        # must return empty string, not raise on slicing.
        import math

        from nmdc_ai_eval.run_suite import _short_label

        assert _short_label(math.nan) == ""  # type: ignore[arg-type]
        assert _short_label(None) == ""
        assert _short_label("") == ""


class TestEnvTriadScore:
    def test_perfect_match(self) -> None:
        assert _env_triad_score(IDEAL, IDEAL) == 1.0

    def test_fenced_response_perfect_match(self) -> None:
        fenced = _make_fenced(
            "terrestrial biome [ENVO:00000446]",
            "rhizosphere [ENVO:00005801]",
            "soil [ENVO:00001998]",
        )
        assert _env_triad_score(IDEAL, fenced) == 1.0

    def test_two_of_three_match(self) -> None:
        response = _make_ideal(
            "host-associated biome [ENVO:00000800]",  # wrong broad
            "rhizosphere [ENVO:00005801]",  # correct local
            "soil [ENVO:00001998]",  # correct medium
        )
        score = _env_triad_score(IDEAL, response)
        assert score == 0.67  # fixed bucket, not 2/3 ≈ 0.6667

    def test_one_of_three_match(self) -> None:
        response = _make_ideal(
            "host-associated biome [ENVO:00000800]",
            "garden [ENVO:00000011]",
            "soil [ENVO:00001998]",  # correct medium only
        )
        score = _env_triad_score(IDEAL, response)
        assert score == 0.33  # fixed bucket, not 1/3 ≈ 0.3333

    def test_no_match(self) -> None:
        response = _make_ideal(
            "host-associated biome [ENVO:00000800]",
            "garden [ENVO:00000011]",
            "bulk soil [ENVO:00005802]",
        )
        assert _env_triad_score(IDEAL, response) == 0.0

    def test_none_ideal_returns_none(self) -> None:
        assert _env_triad_score(None, IDEAL) is None

    def test_non_env_triad_ideal_returns_none(self) -> None:
        assert _env_triad_score("What is the capital of France?", "Paris") is None

    def test_unparsable_response_scores_zero(self) -> None:
        # Ideal parses but response doesn't — all fields are None, no matches
        score = _env_triad_score(IDEAL, "I cannot determine the values.")
        assert score == 0.0
