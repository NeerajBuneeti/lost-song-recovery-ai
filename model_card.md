# 🎧 Model Card: VibeFinder 2.0

## 1. Model Name

**VibeFinder 2.0** (built on VibeFinder 1.0 — Module 3 Music Recommender Simulation)

---

## 2. Intended Use

VibeFinder 2.0 is a classroom demonstration of a conversational AI agent layered on top of a transparent scoring recommender. It is designed for students and developers exploring how LLM-driven feature extraction, RAG (retrieval-augmented generation), and structured scoring can be combined into a practical search interface. The system targets a 20-song catalog and is not intended for production use. It requires an OpenAI API key and does not store user data beyond a single session.

---

## 3. How the Model Works

VibeFinder 2.0 operates as a stateful generator loop (`src/agent.py`) with six phases:

1. **EXTRACTING** — `gpt-4o-mini` in JSON mode parses the user's free-text description into four structured features: `genre_hint`, `mood_hint`, `energy_level`, and `acoustic_preference`, each with a per-field confidence score (0–1). The Pydantic `ExtractionResult` model validates the output; malformed responses are retried once, then fall back to a zeroed-confidence result.

2. **QUESTIONING** — If any feature confidence is below 0.75 and fewer than 4 questions have been asked, `gpt-4o-mini` generates one targeted clarifying question aimed at the lowest-confidence field.

3. **MATCHING** — Structured scores are computed with the VibeFinder 1.0 weighted formula (genre × 3.0, mood × 2.5, energy × 2.0, valence × 1.5, acousticness × 1.0). These are fused with semantic scores from `text-embedding-3-small` cosine similarity: `final = 0.4 × semantic + 0.6 × structured`. Rejected songs are filtered before the top-5 is returned.

4. **CONFIRMING / REFINING** — The user confirms or rejects candidates. Rejected IDs are excluded from future retrievals. After 3 rejections or 4 questions the session terminates.

**Guardrails:** off-topic redirect, malformed JSON retry, OpenAI failure fallback to structured-only, empty-results fallback message, 500-character input limit, 20 total API calls per session cap.

---

## 4. Data

The catalog contains 20 songs across 17 genres and 15 moods (same as VibeFinder 1.0). No user data is persisted. The catalog is English-language only; artists and titles are fictional. Catalog coverage is uneven — lofi has 3 songs, most genres have exactly 1.

---

## 5. AI Collaboration — What Helped and What I Had to Fix

**Helpful suggestion:** When designing the score fusion formula, an AI-generated first draft used `0.5 × semantic + 0.5 × structured`. Testing showed this over-weighted the semantic channel, which returned near-random results against a 20-song catalog when embeddings were computed from short rich-text strings. Shifting to 60/40 in favour of structured scores preserved the original system's precision while still letting the semantic channel catch natural-language descriptions.

**Flawed suggestion I had to fix:** An early AI-generated version of `ConversationState.update_features()` always overwrote existing features with new extraction results, regardless of confidence. This broke the accumulation requirement: a low-confidence second turn would silently erase high-confidence features from the first. The fix was to compare confidence values and only overwrite when the incoming confidence is equal to or higher than the stored one.

---

## 6. System Limitations

- **20-song catalog.** The system can only recommend songs that exist in `data/songs.csv`. Users looking for niche or specific artists will always be disappointed.
- **No audio input.** Features are extracted from text descriptions only. Humming, audio clips, or lyrics are not supported.
- **English only.** The extraction prompts and catalog are English-language. Descriptions in other languages may produce unreliable extractions.
- **Single-session, no learning.** Preferences are not stored between sessions. Accepting a song has no effect on future cold-start behaviour.
- **Catalog sparsity is not fixed.** The same adversarial failures documented in VibeFinder 1.0 (reggae fan shown electronic, classical fan shown electronic) still occur when the catalog has no semantically adjacent songs to the user's preference.

---

## 7. Responsible Use

This system is a teaching tool. It must not be used to:

- Infer sensitive personal attributes from music preferences (emotional state, mental health).
- Make recommendations that affect real purchasing, streaming, or curation decisions.
- Collect or store user descriptions without explicit consent.

Because the system uses `gpt-4o-mini`, any user input is transmitted to OpenAI's API. Users should not enter personally identifying information in their song descriptions.

---

## 8. Bias Documentation

Three biases are inherited from VibeFinder 1.0 and persist in VibeFinder 2.0:

1. **Silent valence default (0.5)** advantages mid-valence songs (Overdrive Protocol, valence 0.66) in every session where the user does not explicitly mention emotional brightness. The LLM extractor does not currently infer valence from descriptions.

2. **Orphaned mood nodes** (`confident`, `nostalgic`, `romantic`) have no adjacency connections in the structured scoring tier map. Songs carrying these moods (Block Party Anthem, Dust Road Memory, Velvet Nights) score 0.0 on mood unless the user explicitly uses those words. The semantic channel partially compensates but cannot fully override a 0.0 weighted mood score.

3. **Catalog free-rider (Overdrive Protocol)** accumulates points through energy, acousticness, and the valence default for nearly every non-acoustic user profile, regardless of whether the user's genre or mood has any connection to electronic music.

---

## 9. Evaluation

All 25 automated tests pass (7 recommender, 6 extractor, 5 RAG, 7 agent). No live API calls are required to run the test suite — all OpenAI interactions are mocked. Manual end-to-end testing was performed for the three sample interactions documented in the README.
