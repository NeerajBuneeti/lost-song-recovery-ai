"""
VibeFinder 2.0 — Streamlit UI

Layout:
  Left sidebar  — live step events / activity log
  Center        — chat interface
  Right panel   — candidate song cards with score bars and confirm/reject buttons
"""

import streamlit as st
from src.agent import run_session

st.set_page_config(
    page_title="VibeFinder 2.0",
    page_icon="🎵",
    layout="wide",
)

# ── Session state init ──────────────────────────────────────────────────────

def _init_state():
    if "generator" not in st.session_state:
        st.session_state.generator = run_session()
        st.session_state.chat_history = []      # list of {"role", "content"}
        st.session_state.event_log = []          # sidebar step events
        st.session_state.candidates = []         # current candidate songs
        st.session_state.done = False
        st.session_state.confirmed_song = None
        # Kick off the generator to get the greeting
        event = next(st.session_state.generator)
        _handle_event(event)


def _handle_event(event: dict):
    """Process a yielded event dict and update session state."""
    st.session_state.event_log.append(event)
    etype = event.get("type", "")

    if etype in ("greeting", "question", "off_topic", "input_too_long",
                 "no_results", "refine_prompt"):
        st.session_state.chat_history.append(
            {"role": "assistant", "content": event["text"]}
        )

    if etype == "candidates":
        st.session_state.candidates = event.get("candidates", [])

    if etype == "done":
        st.session_state.done = True
        if "song" in event:
            st.session_state.confirmed_song = event["song"]
        if "candidates" in event:
            st.session_state.candidates = event["candidates"]
        st.session_state.chat_history.append(
            {"role": "assistant", "content": event["text"]}
        )

    if etype == "features_updated":
        features = event.get("features", {})
        summary_parts = []
        for k in ("genre_hint", "mood_hint", "energy_level", "acoustic_preference"):
            v = features.get(k)
            if v is not None:
                summary_parts.append(f"{k.replace('_', ' ')}: **{v}**")
        if summary_parts:
            st.session_state.event_log[-1]["_summary"] = ", ".join(summary_parts)


def _send_to_agent(user_text: str):
    """Drive the generator with user input and handle events until next yield."""
    st.session_state.chat_history.append({"role": "user", "content": user_text})
    try:
        event = st.session_state.generator.send(user_text)
        _handle_event(event)
        # Drain any intermediate events until we need user input
        while event.get("type") not in (
            "greeting", "question", "candidates", "done",
            "off_topic", "input_too_long", "no_results", "refine_prompt"
        ):
            event = st.session_state.generator.send(None)
            _handle_event(event)
    except StopIteration:
        st.session_state.done = True


# ── UI render ───────────────────────────────────────────────────────────────

_init_state()

col_sidebar, col_chat, col_cards = st.columns([1, 2, 1.5])

# LEFT: Step event log
with col_sidebar:
    st.markdown("### 🔍 Activity")
    for ev in reversed(st.session_state.event_log):
        step_label = ev.get("step", "")
        etype = ev.get("type", "")
        icon = {
            "greeting": "👋",
            "step_change": "⚙️",
            "features_updated": "📋",
            "question": "❓",
            "candidates": "🎶",
            "done": "✅",
            "off_topic": "🚫",
            "refine_prompt": "🔄",
            "no_results": "🔇",
        }.get(etype, "•")
        label = f"{icon} **{etype}** — {step_label}"
        st.markdown(label)
        if "_summary" in ev:
            st.caption(ev["_summary"])

# CENTER: Chat
with col_chat:
    st.markdown("## 🎵 VibeFinder 2.0")
    st.caption("Describe a song you half-remember and I'll find it.")

    # Chat bubbles
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box (disabled when done)
    if not st.session_state.done:
        user_input = st.chat_input("Describe your song...")
        if user_input:
            _send_to_agent(user_input)
            st.rerun()
    else:
        st.success("Session complete! Start a new conversation to find another song.")
        if st.button("🔄 Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# RIGHT: Candidate cards
with col_cards:
    st.markdown("### 🎧 Candidates")
    candidates = st.session_state.candidates

    if not candidates:
        st.caption("Candidates will appear here once matching begins.")
    else:
        for song in candidates:
            score = song.get("_score", 0.0)
            with st.container(border=True):
                st.markdown(f"**{song['title']}** — *{song['artist']}*")
                st.caption(f"{song['genre']} · {song['mood']} · energy {song.get('energy', '?'):.2f}")
                st.progress(min(float(score), 1.0), text=f"Match: {score * 100:.1f}%")

                if not st.session_state.done:
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✓ This is it", key=f"confirm_{song['id']}"):
                            _send_to_agent(f"CONFIRMED:{song['id']}")
                            st.rerun()
                    with c2:
                        if st.button("✗ Not quite", key=f"reject_{song['id']}"):
                            _send_to_agent(f"REJECTED:{song['id']}")
                            st.rerun()
