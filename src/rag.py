import os
import csv
from typing import List, Dict, Optional, Set
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_EMBEDDING_MODEL = "text-embedding-3-small"
_CATALOG_PATH = "data/songs.csv"

# Module-level cache populated at first use
_song_dicts: List[Dict] = []
_song_embeddings: Optional[np.ndarray] = None
_rag_available: bool = True


def _song_to_rich_text(song: Dict) -> str:
    """Convert a song dict to a descriptive string for embedding."""
    acoustic_label = "acoustic" if float(song.get("acousticness", 0)) > 0.5 else "produced"
    energy_label = "high energy" if float(song.get("energy", 0)) > 0.6 else "low energy"
    return (
        f"{song['title']} by {song['artist']}. "
        f"Genre: {song['genre']}. Mood: {song['mood']}. "
        f"{energy_label}, {acoustic_label} sound. "
        f"Tempo {song.get('tempo_bpm', '?')} BPM, "
        f"danceability {song.get('danceability', '?')}, "
        f"valence {song.get('valence', '?')}."
    )


def _load_catalog() -> List[Dict]:
    float_fields = {"energy", "tempo_bpm", "valence", "danceability", "acousticness"}
    songs = []
    with open(_CATALOG_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            song = {}
            for key, value in row.items():
                if key == "id":
                    song[key] = int(value)
                elif key in float_fields:
                    song[key] = float(value)
                else:
                    song[key] = value
            songs.append(song)
    return songs


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b_norm @ a_norm


def _build_embeddings(client: OpenAI, texts: List[str]) -> np.ndarray:
    response = client.embeddings.create(model=_EMBEDDING_MODEL, input=texts)
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def initialize(client: Optional[OpenAI] = None) -> bool:
    """Embed all catalog songs. Returns True if RAG is available, False on failure."""
    global _song_dicts, _song_embeddings, _rag_available

    if _song_embeddings is not None:
        return _rag_available

    _song_dicts = _load_catalog()

    if client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        texts = [_song_to_rich_text(s) for s in _song_dicts]
        _song_embeddings = _build_embeddings(client, texts)
        _rag_available = True
    except Exception:
        _song_embeddings = np.zeros((len(_song_dicts), 1), dtype=np.float32)
        _rag_available = False

    return _rag_available


def retrieve(
    query: str,
    structured_scores: Dict[int, float],
    rejected_ids: Optional[Set[int]] = None,
    client: Optional[OpenAI] = None,
    top_k: int = 5,
) -> List[Dict]:
    """
    Retrieve top-k songs by fusing semantic similarity and structured scores.

    final = 0.4 * semantic + 0.6 * structured

    Falls back to structured-only if RAG is unavailable.
    """
    global _rag_available

    if client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if _song_embeddings is None:
        initialize(client)

    rejected = rejected_ids or set()

    if _rag_available:
        try:
            q_response = client.embeddings.create(model=_EMBEDDING_MODEL, input=[query])
            q_vec = np.array(q_response.data[0].embedding, dtype=np.float32)
            semantic_scores = _cosine_similarity(q_vec, _song_embeddings)
        except Exception:
            _rag_available = False
            semantic_scores = np.zeros(len(_song_dicts), dtype=np.float32)
    else:
        semantic_scores = np.zeros(len(_song_dicts), dtype=np.float32)

    results = []
    for i, song in enumerate(_song_dicts):
        if song["id"] in rejected:
            continue
        sem = float(semantic_scores[i]) if _rag_available else 0.0
        struct = structured_scores.get(song["id"], 0.0)
        if _rag_available:
            final = 0.4 * sem + 0.6 * struct
        else:
            final = struct
        results.append({**song, "_semantic": sem, "_structured": struct, "_score": final})

    results.sort(key=lambda x: x["_score"], reverse=True)
    return results[:top_k]
