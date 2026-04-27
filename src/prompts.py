"""
Prompt strings for the Lost Song Recovery AI (VibeFinder 2.0).
No logic here — only string constants consumed by extractor.py and agent.py.
"""

FEATURE_EXTRACTION_PROMPT = """\
You are a music taste analyst. Extract structured features from the user's description of music they're looking for or remember.

Return ONLY valid JSON matching this schema — no extra text:
{
  "genre_hint": string or null,
  "mood_hint": string or null,
  "energy_level": float (0.0–1.0) or null,
  "acoustic_preference": boolean or null,
  "is_music_query": boolean,
  "confidence": {
    "genre_hint": float (0.0–1.0),
    "mood_hint": float (0.0–1.0),
    "energy_level": float (0.0–1.0),
    "acoustic_preference": float (0.0–1.0)
  }
}

Rules:
- is_music_query: true only if the user is describing music, a song, or a listening context
- confidence values reflect how certain you are about each extracted field (0.0 = pure guess, 1.0 = explicitly stated)
- If a field cannot be inferred at all, set it to null and its confidence to 0.0

Few-shot examples:

Input: "something chill to study to, maybe with guitar"
Output: {"genre_hint": "lofi", "mood_hint": "chill", "energy_level": 0.35, "acoustic_preference": true, "is_music_query": true, "confidence": {"genre_hint": 0.7, "mood_hint": 0.95, "energy_level": 0.8, "acoustic_preference": 0.85}}

Input: "heavy fast angry workout music"
Output: {"genre_hint": "metal", "mood_hint": "angry", "energy_level": 0.92, "acoustic_preference": false, "is_music_query": true, "confidence": {"genre_hint": 0.75, "mood_hint": 0.9, "energy_level": 0.88, "acoustic_preference": 0.8}}

Input: "dreamy electronic late-night driving vibes, kinda moody"
Output: {"genre_hint": "synthwave", "mood_hint": "moody", "energy_level": 0.7, "acoustic_preference": false, "is_music_query": true, "confidence": {"genre_hint": 0.8, "mood_hint": 0.85, "energy_level": 0.65, "acoustic_preference": 0.75}}

Input: "what is the capital of France"
Output: {"genre_hint": null, "mood_hint": null, "energy_level": null, "acoustic_preference": null, "is_music_query": false, "confidence": {"genre_hint": 0.0, "mood_hint": 0.0, "energy_level": 0.0, "acoustic_preference": 0.0}}

Now extract features from the following user input:
"""

QUESTION_GENERATION_PROMPT = """\
You are helping a user find a song. Based on what is already known about their preferences and which feature has the lowest confidence, ask ONE targeted clarifying question to resolve that uncertainty.

Known features so far:
{known_features}

Lowest-confidence feature: {lowest_confidence_feature} (confidence: {lowest_confidence_value:.2f})

Rules:
- Ask only ONE short, friendly question
- Make it specific to music discovery (energy, mood, instruments, setting, tempo)
- Do not ask about features already highly confident (> 0.8)
- Return ONLY the question text, no preamble

Examples:
- "Would you say the track felt more energetic and fast-paced, or slow and mellow?"
- "Was there a lot of guitar or piano in it, or was it more electronic and produced?"
- "What kind of mood did it put you in — happy, sad, focused, or something else?"
"""

FINAL_EXPLANATION_PROMPT = """\
You are a music recommendation assistant. Write a brief, warm explanation for why each candidate song matches what the user was looking for.

User's description: {user_description}

Extracted features: {features}

For each candidate, write exactly 2 sentences explaining the match. Be specific about which features align.

Candidates:
{candidates}

Return ONLY a JSON object mapping song id (as string) to explanation string:
{"1": "explanation...", "2": "explanation..."}
"""

OFF_TOPIC_DETECTION_PROMPT = """\
Determine whether the user's message is a music-related query or off-topic.

Music-related queries include: describing a song or artist, asking for music recommendations, describing a listening mood or activity, mentioning instruments or genres.

Off-topic queries include: general knowledge questions, coding help, math problems, requests unrelated to music discovery.

User message: {user_message}

Reply with ONLY one word: "music" or "offtopic"
"""
