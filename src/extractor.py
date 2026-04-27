import os
import json
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

from src.prompts import FEATURE_EXTRACTION_PROMPT

load_dotenv()


class ConfidenceDict(BaseModel):
    genre_hint: float = Field(0.0, ge=0.0, le=1.0)
    mood_hint: float = Field(0.0, ge=0.0, le=1.0)
    energy_level: float = Field(0.0, ge=0.0, le=1.0)
    acoustic_preference: float = Field(0.0, ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    genre_hint: Optional[str] = None
    mood_hint: Optional[str] = None
    energy_level: Optional[float] = Field(None, ge=0.0, le=1.0)
    acoustic_preference: Optional[bool] = None
    is_music_query: bool = True
    confidence: ConfidenceDict = Field(default_factory=ConfidenceDict)


_ZEROED_FALLBACK = ExtractionResult(
    genre_hint=None,
    mood_hint=None,
    energy_level=None,
    acoustic_preference=None,
    is_music_query=True,
    confidence=ConfidenceDict(),
)


def _parse_response(content: str) -> ExtractionResult:
    data = json.loads(content)
    conf_raw = data.get("confidence", {})
    confidence = ConfidenceDict(
        genre_hint=float(conf_raw.get("genre_hint", 0.0)),
        mood_hint=float(conf_raw.get("mood_hint", 0.0)),
        energy_level=float(conf_raw.get("energy_level", 0.0)),
        acoustic_preference=float(conf_raw.get("acoustic_preference", 0.0)),
    )
    return ExtractionResult(
        genre_hint=data.get("genre_hint"),
        mood_hint=data.get("mood_hint"),
        energy_level=data.get("energy_level"),
        acoustic_preference=data.get("acoustic_preference"),
        is_music_query=bool(data.get("is_music_query", True)),
        confidence=confidence,
    )


def extract_features(user_input: str, client: Optional[OpenAI] = None) -> ExtractionResult:
    """Call gpt-4o-mini in JSON mode to extract music features. Retries once on failure."""
    if client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = FEATURE_EXTRACTION_PROMPT + user_input

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=400,
            )
            return _parse_response(response.choices[0].message.content)
        except Exception:
            if attempt == 1:
                return _ZEROED_FALLBACK

    return _ZEROED_FALLBACK
