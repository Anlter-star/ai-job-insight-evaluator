"""Provider-neutral structured extractor for OpenAI-compatible chat APIs."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.data_loader import PROJECT_ROOT, compose_source_text


DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "job_extraction_v1.txt"
EXPECTED_FIELDS = {
    "job_category",
    "employment_type",
    "location",
    "education_min",
    "student_status",
    "graduation_year",
    "work_days_per_week",
    "internship_months",
    "required_hard_skills",
    "required_tools",
    "preferred_hard_skills",
    "preferred_experience",
}
LIST_FIELDS = {
    "required_hard_skills",
    "required_tools",
    "preferred_hard_skills",
    "preferred_experience",
}


class LLMConfigurationError(ValueError):
    """Raised when the local model configuration is incomplete."""


class LLMResponseError(RuntimeError):
    """Raised when the provider response cannot be used as a prediction."""


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Load simple KEY=VALUE entries without overriding shell variables."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int = 90
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "LLMConfig":
        load_dotenv()
        values = {
            "api_key": os.getenv("LLM_API_KEY", "").strip(),
            "base_url": os.getenv("LLM_BASE_URL", "").strip(),
            "model": os.getenv("LLM_MODEL", "").strip(),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            env_names = {"api_key": "LLM_API_KEY", "base_url": "LLM_BASE_URL", "model": "LLM_MODEL"}
            raise LLMConfigurationError(
                "Missing configuration: " + ", ".join(env_names[name] for name in missing)
                + ". Copy .env.example to .env and fill in the values."
            )
        if values["api_key"] == "replace_with_your_api_key":
            raise LLMConfigurationError("Replace the placeholder LLM_API_KEY in .env")
        return cls(**values)


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _strip_code_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def validate_prediction(prediction: Any) -> dict[str, Any]:
    """Validate and lightly normalize a model prediction."""

    if not isinstance(prediction, dict):
        raise LLMResponseError("Model output must be a JSON object")
    missing = EXPECTED_FIELDS - prediction.keys()
    extra = prediction.keys() - EXPECTED_FIELDS
    if missing or extra:
        raise LLMResponseError(
            f"Unexpected schema; missing={sorted(missing)}, extra={sorted(extra)}"
        )

    normalized = dict(prediction)
    for field in LIST_FIELDS:
        value = normalized[field]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise LLMResponseError(f"{field} must be a list of strings")
        normalized[field] = list(dict.fromkeys(item.strip() for item in value if item.strip()))

    for field in ("work_days_per_week", "internship_months"):
        value = normalized[field]
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
            raise LLMResponseError(f"{field} must be an integer or null")

    year = normalized["graduation_year"]
    if year is not None and (
        not isinstance(year, dict)
        or set(year) != {"min"}
        or not isinstance(year["min"], int)
    ):
        raise LLMResponseError("graduation_year must be {\"min\": year} or null")
    return normalized


class OpenAICompatibleExtractor:
    """Call an OpenAI-compatible /chat/completions endpoint."""

    prompt_version = "job_extraction_v1"

    def __init__(self, config: LLMConfig, prompt_path: Path = DEFAULT_PROMPT_PATH):
        self.config = config
        self.system_prompt = prompt_path.read_text(encoding="utf-8")
        self.method_name = f"llm:{config.model}:{self.prompt_version}"

    def extract(self, record: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": compose_source_text(record)},
            ],
            "temperature": 0,
            "max_tokens": 1600,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        last_error: LLMResponseError | None = None
        for attempt in range(self.config.max_retries + 1):
            response = self._post(payload)
            try:
                return self._parse_response(response)
            except LLMResponseError as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    payload["messages"][-1]["content"] = (
                        compose_source_text(record)
                        + "\n\n重要：请立即返回一个非空的合法 JSON 对象。"
                    )
                    time.sleep(2**attempt)
        raise LLMResponseError(
            f"Model output remained unusable after retries: {last_error}"
        ) from last_error

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> dict[str, Any]:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("Provider response has no assistant content") from exc
        if not content or not str(content).strip():
            raise LLMResponseError("Provider returned empty assistant content")
        try:
            prediction = json.loads(_strip_code_fence(str(content)))
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"Assistant content is not valid JSON: {exc.msg}") from exc
        return validate_prediction(prediction)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            _chat_completions_url(self.config.base_url),
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with urlopen(request, timeout=self.config.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = LLMResponseError(f"HTTP {exc.code}: {detail[:500]}")
                if exc.code not in {429, 500, 502, 503, 504}:
                    break
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
            if attempt < self.config.max_retries:
                time.sleep(2**attempt)
        raise LLMResponseError(f"API request failed after retries: {last_error}") from last_error
