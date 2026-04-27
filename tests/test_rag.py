"""
5 tests for src/rag.py — all OpenAI calls are mocked.
"""
import numpy as np
from unittest.mock import MagicMock, patch
import pytest

import src.rag as rag
from src.recommender import load_songs, score_song


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_embedding_client(dim: int = 16):
    """
    Return a mock OpenAI client whose embeddings.create() returns a random unit vector
    per input string so cosine similarity tests are structurally correct.
    """
    def _fake_embed(model, input):
        rng = np.random.default_rng(42)
        items = []
        for _ in input:
            vec = rng.random(dim).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-10
            item = MagicMock()
            item.embedding = vec.tolist()
            items.append(item)
        resp = MagicMock()
        resp.data = items
        return resp

    client = MagicMock()
    client.embeddings.create.side_effect = _fake_embed
    return client


def _get_structured_scores():
    """Score all 20 songs against a generic lofi profile."""
    songs = load_songs("data/songs.csv")
    prefs = {"genre": "lofi", "mood": "chill", "energy": 0.38, "likes_acoustic": True}
    return {s["id"]: score_song(prefs, s)[0] for s in songs}


def _reset_rag():
    """Clear module-level cache so tests start fresh."""
    rag._song_dicts = []
    rag._song_embeddings = None
    rag._rag_available = True


# ── Tests ────────────────────────────────────────────────────────────────────

def test_lofi_description_retrieves_lofi_songs():
    """Structured scoring alone (mock embeddings) should put lofi songs in top results."""
    _reset_rag()
    client = _make_embedding_client()
    rag.initialize(client)

    structured = _get_structured_scores()
    results = rag.retrieve("chill lofi study music", structured, client=client)

    genres = [r["genre"] for r in results]
    assert "lofi" in genres, f"Expected lofi in top results, got: {genres}"


def test_folk_description_retrieves_folk_songs():
    """Structured scoring for folk profile should surface folk/country songs."""
    _reset_rag()
    client = _make_embedding_client()
    rag.initialize(client)

    songs = load_songs("data/songs.csv")
    prefs = {"genre": "folk", "mood": "melancholic", "energy": 0.32, "likes_acoustic": True}
    structured = {s["id"]: score_song(prefs, s)[0] for s in songs}

    results = rag.retrieve("acoustic folk sad guitar", structured, client=client)
    genres = [r["genre"] for r in results]
    assert any(g in ("folk", "country", "blues") for g in genres), (
        f"Expected folk/country/blues in top results, got: {genres}"
    )


def test_fallback_when_openai_unavailable():
    """When embeddings fail, rag_available should be False and retrieve falls back to structured."""
    _reset_rag()

    client = MagicMock()
    client.embeddings.create.side_effect = Exception("connection error")

    rag.initialize(client)
    assert rag._rag_available is False

    structured = _get_structured_scores()
    results = rag.retrieve("test query", structured, client=client)

    # Should still return results using structured scores only
    assert len(results) > 0
    # Top result should be the highest-structured-score song
    top_id = max(structured, key=lambda k: structured[k])
    assert results[0]["id"] == top_id


def test_score_fusion_formula():
    """Verify fused score = 0.4 * semantic + 0.6 * structured when RAG is available."""
    _reset_rag()
    client = _make_embedding_client(dim=8)
    rag.initialize(client)

    structured = {s["id"]: 0.5 for s in rag._song_dicts}  # uniform structured scores

    results = rag.retrieve("test", structured, client=client, top_k=1)
    song = results[0]

    sem = song["_semantic"]
    struct = song["_structured"]
    expected = 0.4 * sem + 0.6 * struct
    assert abs(song["_score"] - expected) < 1e-5, (
        f"Expected fusion {expected:.6f}, got {song['_score']:.6f}"
    )


def test_rejected_songs_excluded():
    """Songs in rejected_ids must not appear in retrieve results."""
    _reset_rag()
    client = _make_embedding_client()
    rag.initialize(client)

    structured = _get_structured_scores()
    # Reject the top-3 highest-scoring songs
    top_ids = sorted(structured, key=lambda k: structured[k], reverse=True)[:3]
    rejected = set(top_ids)

    results = rag.retrieve("chill lofi", structured, rejected_ids=rejected, client=client)
    returned_ids = {r["id"] for r in results}
    overlap = returned_ids & rejected
    assert len(overlap) == 0, f"Rejected songs appeared in results: {overlap}"
