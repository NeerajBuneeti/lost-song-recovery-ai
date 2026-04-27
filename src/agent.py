"""
Agentic loop for VibeFinder 2.0.

Yields step-event dicts throughout the conversation. Each dict has at minimum:
  {"type": str, "step": Step, ...extra payload}

Hard caps:
  - max 4 clarifying questions
  - max 3 refinement cycles
  - max 20 total OpenAI API calls per session
  - max 500 chars per user input
"""

import os
from typing import Generator, Dict, Any, Optional, List
from dotenv import load_dotenv
from openai import OpenAI

from src.conversation import ConversationState, Step
from src.extractor import extract_features
from src.recommender import load_songs, score_song
import src.rag as rag
from src.prompts import QUESTION_GENERATION_PROMPT, OFF_TOPIC_DETECTION_PROMPT, FINAL_EXPLANATION_PROMPT

load_dotenv()

_MAX_QUESTIONS = 4
_MAX_REFINEMENTS = 3
_MAX_API_CALLS = 20
_MAX_INPUT_CHARS = 500

_songs: Optional[List[Dict]] = None


def _get_songs() -> List[Dict]:
    global _songs
    if _songs is None:
        _songs = load_songs("data/songs.csv")
    return _songs


def _make_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _is_off_topic(user_input: str, client: OpenAI, state: ConversationState) -> bool:
    """Quick off-topic check using the OFF_TOPIC_DETECTION_PROMPT."""
    if state.api_calls_used >= _MAX_API_CALLS:
        return False
    try:
        state.api_calls_used += 1
        prompt = OFF_TOPIC_DETECTION_PROMPT.format(user_message=user_input)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer == "offtopic"
    except Exception:
        return False


def _ask_question(state: ConversationState, client: OpenAI) -> str:
    """Generate a targeted clarifying question for the lowest-confidence feature."""
    feature = state.lowest_confidence_feature() or "genre_hint"
    conf = state.features.get("confidence", {})
    conf_val = conf.get(feature, 0.0) if isinstance(conf, dict) else getattr(conf, feature, 0.0)

    known_lines = []
    for k in ("genre_hint", "mood_hint", "energy_level", "acoustic_preference"):
        v = state.features.get(k)
        if v is not None:
            known_lines.append(f"  {k}: {v}")
    known_str = "\n".join(known_lines) if known_lines else "  (nothing known yet)"

    prompt = QUESTION_GENERATION_PROMPT.format(
        known_features=known_str,
        lowest_confidence_feature=feature,
        lowest_confidence_value=conf_val,
    )

    if state.api_calls_used >= _MAX_API_CALLS:
        return "Can you describe the energy level — was it calm and quiet or loud and driving?"

    try:
        state.api_calls_used += 1
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=80,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Could you tell me more about the mood or energy of the song?"


def _score_all_songs(state: ConversationState) -> Dict[int, float]:
    prefs = state.to_prefs_dict()
    songs = _get_songs()
    return {s["id"]: score_song(prefs, s)[0] for s in songs}


def _generate_explanations(
    candidates: List[Dict],
    user_description: str,
    features: Dict[str, Any],
    state: ConversationState,
    client: OpenAI,
    max_candidates: int = 3,
) -> None:
    """
    Call gpt-4o-mini once with FINAL_EXPLANATION_PROMPT to write a 2-sentence
    explanation for each of the top candidates.  Results are stored in-place
    as candidate["_explanation"].  Silent no-op on any failure.
    """
    import json as _json

    if state.api_calls_used >= _MAX_API_CALLS or not candidates:
        return

    top = candidates[:max_candidates]
    candidates_text = "\n".join(
        f"ID {s['id']}: {s['title']} by {s['artist']}. "
        f"Genre: {s['genre']}. Mood: {s['mood']}. Energy: {s.get('energy', '?')}."
        for s in top
    )
    features_text = ", ".join(
        f"{k}: {v}"
        for k, v in features.items()
        if k != "confidence" and v is not None
    )
    prompt = FINAL_EXPLANATION_PROMPT.format(
        user_description=user_description,
        features=features_text or "unknown",
        candidates=candidates_text,
    )

    try:
        state.api_calls_used += 1
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=400,
        )
        mapping = _json.loads(response.choices[0].message.content)
        for song in top:
            song["_explanation"] = mapping.get(str(song["id"]), "")
    except Exception:
        pass  # explanations are optional; cards render without them if this fails


def run_session(client: Optional[OpenAI] = None) -> Generator[Dict[str, Any], str, None]:
    """
    Agentic generator loop.

    Usage:
        gen = run_session()
        event = next(gen)        # start — yields greeting event
        event = gen.send(user_input)  # drive with user messages
    """
    if client is None:
        client = _make_client()

    state = ConversationState()
    state.step = Step.INIT

    # --- INIT: greet the user ---
    greeting = (
        "Hi! I'm VibeFinder 2.0. Describe a song you half-remember or tell me the kind of music "
        "you're looking for right now, and I'll dig it up from our catalog."
    )
    state.add_message("assistant", greeting)
    user_input: str = yield {"type": "greeting", "step": state.step, "text": greeting}

    # Initialize RAG in background (errors silently handled inside rag.initialize)
    try:
        state.rag_available = rag.initialize(client)
    except Exception:
        state.rag_available = False

    while True:
        # --- Input length guard ---
        if len(user_input) > _MAX_INPUT_CHARS:
            msg = f"Please keep your description under {_MAX_INPUT_CHARS} characters."
            state.add_message("assistant", msg)
            user_input = yield {"type": "input_too_long", "step": state.step, "text": msg}
            continue

        state.add_message("user", user_input)

        # --- API call cap ---
        if state.api_calls_used >= _MAX_API_CALLS:
            msg = "We've hit the session limit. Here are the best matches found so far."
            state.step = Step.DONE
            structured = _score_all_songs(state)
            candidates = rag.retrieve(
                user_input, structured, state.rejected_candidates, client
            )
            yield {"type": "done", "step": state.step, "text": msg, "candidates": candidates}
            return

        # --- Off-topic detection ---
        if state.step in (Step.INIT, Step.EXTRACTING, Step.QUESTIONING):
            if _is_off_topic(user_input, client, state):
                msg = "I'm only able to help find songs! Try describing a track you remember or a mood you're in."
                state.add_message("assistant", msg)
                user_input = yield {"type": "off_topic", "step": state.step, "text": msg}
                continue

        # --- EXTRACTING ---
        state.step = Step.EXTRACTING
        yield {"type": "step_change", "step": state.step}

        if state.api_calls_used < _MAX_API_CALLS:
            state.api_calls_used += 1
            extraction = extract_features(user_input, client)
        else:
            from src.extractor import _ZEROED_FALLBACK
            extraction = _ZEROED_FALLBACK

        # Accumulate features
        state.update_features({
            "genre_hint": extraction.genre_hint,
            "mood_hint": extraction.mood_hint,
            "energy_level": extraction.energy_level,
            "acoustic_preference": extraction.acoustic_preference,
            "confidence": extraction.confidence,
        })

        yield {"type": "features_updated", "step": state.step, "features": dict(state.features)}

        # --- QUESTIONING: ask up to _MAX_QUESTIONS times if not yet confident ---
        if (
            not state.all_features_confident()
            and state.questions_asked < _MAX_QUESTIONS
            and state.step != Step.REFINING
        ):
            state.step = Step.QUESTIONING
            question = _ask_question(state, client)
            state.questions_asked += 1
            state.add_message("assistant", question)
            user_input = yield {"type": "question", "step": state.step, "text": question}
            continue

        # --- MATCHING ---
        state.step = Step.MATCHING
        yield {"type": "step_change", "step": state.step}

        structured = _score_all_songs(state)
        user_description = " ".join(
            m["content"] for m in state.messages if m["role"] == "user"
        )

        candidates = rag.retrieve(
            user_description, structured, state.rejected_candidates, client
        )

        if not candidates:
            msg = "I couldn't find any matches. Could you describe it differently?"
            state.add_message("assistant", msg)
            user_input = yield {"type": "no_results", "step": state.step, "text": msg}
            continue

        _generate_explanations(candidates, user_description, state.features, state, client)

        state.step = Step.CONFIRMING
        user_input = yield {
            "type": "candidates",
            "step": state.step,
            "candidates": candidates,
            "rag_available": state.rag_available,
        }

        # --- CONFIRMING: user says "yes" (confirmed) or "no" (rejection) ---
        # The app layer sends "CONFIRMED:<song_id>" or "REJECTED:<song_id>" signals
        if isinstance(user_input, str) and user_input.startswith("CONFIRMED:"):
            song_id = int(user_input.split(":")[1])
            confirmed = next((c for c in candidates if c["id"] == song_id), candidates[0])
            msg = f"Found it! {confirmed['title']} by {confirmed['artist']}. Enjoy! 🎵"
            state.add_message("assistant", msg)
            state.step = Step.DONE
            yield {"type": "done", "step": state.step, "text": msg, "song": confirmed}
            return

        if isinstance(user_input, str) and user_input.startswith("REJECTED:"):
            song_id = int(user_input.split(":")[1])
            state.rejected_candidates.add(song_id)
            state.refinement_cycles += 1

            if state.refinement_cycles >= _MAX_REFINEMENTS:
                msg = "I've tried several options. Here are all remaining matches."
                state.step = Step.DONE
                final = rag.retrieve(
                    user_description, structured, state.rejected_candidates, client
                )
                _generate_explanations(final, user_description, state.features, state, client)
                yield {"type": "done", "step": state.step, "text": msg, "candidates": final}
                return

            state.step = Step.REFINING
            msg = "Got it — let me refine. What else can you tell me about the song?"
            state.add_message("assistant", msg)
            user_input = yield {"type": "refine_prompt", "step": state.step, "text": msg}
            continue

        # Plain text response during CONFIRMING — treat as new description
        state.step = Step.REFINING
        state.refinement_cycles += 1
        continue
