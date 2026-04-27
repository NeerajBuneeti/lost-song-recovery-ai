"""
6 tests for src/extractor.py using mocked OpenAI calls.
"""
import json
from unittest.mock import MagicMock, patch
import pytest

from src.extractor import extract_features, ExtractionResult, ConfidenceDict


def _mock_client(json_payload: dict):
    """Build a mock OpenAI client that returns json_payload as a chat completion."""
    client = MagicMock()
    message = MagicMock()
    message.content = json.dumps(json_payload)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    return client


def test_acoustic_preference_detection():
    payload = {
        "genre_hint": "folk",
        "mood_hint": "peaceful",
        "energy_level": 0.3,
        "acoustic_preference": True,
        "is_music_query": True,
        "confidence": {"genre_hint": 0.8, "mood_hint": 0.7, "energy_level": 0.75, "acoustic_preference": 0.9},
    }
    result = extract_features("gentle acoustic folk", client=_mock_client(payload))
    assert result.acoustic_preference is True
    assert result.confidence.acoustic_preference == pytest.approx(0.9)


def test_energy_from_adjectives():
    payload = {
        "genre_hint": "metal",
        "mood_hint": "angry",
        "energy_level": 0.95,
        "acoustic_preference": False,
        "is_music_query": True,
        "confidence": {"genre_hint": 0.85, "mood_hint": 0.9, "energy_level": 0.88, "acoustic_preference": 0.8},
    }
    result = extract_features("heavy fast aggressive workout track", client=_mock_client(payload))
    assert result.energy_level is not None
    assert result.energy_level >= 0.8
    assert result.confidence.energy_level >= 0.8


def test_genre_from_instrument_cues():
    payload = {
        "genre_hint": "jazz",
        "mood_hint": "relaxed",
        "energy_level": 0.35,
        "acoustic_preference": True,
        "is_music_query": True,
        "confidence": {"genre_hint": 0.75, "mood_hint": 0.7, "energy_level": 0.65, "acoustic_preference": 0.8},
    }
    result = extract_features("smooth saxophone and upright bass late night", client=_mock_client(payload))
    assert result.genre_hint == "jazz"
    assert result.confidence.genre_hint >= 0.7


def test_off_topic_detection():
    payload = {
        "genre_hint": None,
        "mood_hint": None,
        "energy_level": None,
        "acoustic_preference": None,
        "is_music_query": False,
        "confidence": {"genre_hint": 0.0, "mood_hint": 0.0, "energy_level": 0.0, "acoustic_preference": 0.0},
    }
    result = extract_features("what is the boiling point of water", client=_mock_client(payload))
    assert result.is_music_query is False
    assert result.genre_hint is None


def test_schema_validation_on_bad_response():
    """When the LLM returns malformed JSON on both attempts, return zeroed-confidence fallback."""
    client = MagicMock()
    message = MagicMock()
    message.content = "not json at all {{{"
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response

    result = extract_features("something chill", client=client)

    assert isinstance(result, ExtractionResult)
    assert result.confidence.genre_hint == 0.0
    assert result.confidence.mood_hint == 0.0
    assert result.genre_hint is None


def test_confidence_accumulates_across_turns():
    """
    Second extraction with higher confidence should overwrite the first.
    Simulates ConversationState.update_features merging behaviour.
    """
    from src.conversation import ConversationState

    first_payload = {
        "genre_hint": "lofi",
        "mood_hint": "chill",
        "energy_level": 0.4,
        "acoustic_preference": None,
        "is_music_query": True,
        "confidence": {"genre_hint": 0.6, "mood_hint": 0.7, "energy_level": 0.5, "acoustic_preference": 0.0},
    }
    second_payload = {
        "genre_hint": "lofi",
        "mood_hint": "chill",
        "energy_level": 0.4,
        "acoustic_preference": True,
        "is_music_query": True,
        "confidence": {"genre_hint": 0.85, "mood_hint": 0.9, "energy_level": 0.8, "acoustic_preference": 0.95},
    }

    state = ConversationState()

    r1 = extract_features("study music", client=_mock_client(first_payload))
    state.update_features({
        "genre_hint": r1.genre_hint,
        "mood_hint": r1.mood_hint,
        "energy_level": r1.energy_level,
        "acoustic_preference": r1.acoustic_preference,
        "confidence": r1.confidence,
    })

    r2 = extract_features("guitar acoustic lofi beats", client=_mock_client(second_payload))
    state.update_features({
        "genre_hint": r2.genre_hint,
        "mood_hint": r2.mood_hint,
        "energy_level": r2.energy_level,
        "acoustic_preference": r2.acoustic_preference,
        "confidence": r2.confidence,
    })

    assert state.features.get("acoustic_preference") is True
    conf = state.features.get("confidence", {})
    assert conf.get("acoustic_preference", 0.0) >= 0.9
