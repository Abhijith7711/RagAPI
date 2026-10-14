"""Language-model interface for grounded question answering."""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

from pydantic import ValidationError

from app.exceptions import LLMUnavailableError
from app.prompts import SYSTEM_PROMPT, build_user_prompt
from app.schemas import LLMStructuredResponse, RetrievedChunk

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def complete(self, question: str, evidence: list[RetrievedChunk]) -> LLMStructuredResponse: ...



def parse_llm_json(raw: str) -> LLMStructuredResponse:
    candidate = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = candidate[start : end + 1]
    try:
        payload = json.loads(candidate)
        return LLMStructuredResponse.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError("The model did not return valid structured output.") from exc


class GroqLLM:
    def __init__(self, api_key: str, model: str, temperature: float):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def complete(self, question: str, evidence: list[RetrievedChunk]) -> LLMStructuredResponse:
        if not self.api_key:
            raise LLMUnavailableError(
                "GROQ_API_KEY is not configured. Please set your GROQ_API_KEY in the .env file."
            )
        try:
            from groq import Groq

            client = Groq(api_key=self.api_key)
            completion = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(question, evidence)},
                ],
            )
            content = completion.choices[0].message.content or ""
        except LLMUnavailableError:
            raise
        except Exception as exc:
            logger.exception("Groq request failed")
            raise LLMUnavailableError("The language model provider request failed.") from exc
        try:
            return parse_llm_json(content)
        except ValueError as exc:
            logger.warning("Groq returned malformed JSON")
            raise ValueError(str(exc)) from exc


