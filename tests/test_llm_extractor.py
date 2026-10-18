"""Unit tests for the provider-neutral LLM extractor."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from src.data_loader import load_jsonl
from src.llm_extractor import (
    LLMConfig,
    LLMResponseError,
    OpenAICompatibleExtractor,
    _chat_completions_url,
    validate_prediction,
)


VALID_PREDICTION = {
    "job_category": "产品经理",
    "employment_type": "实习",
    "location": "上海",
    "salary_yuan_per_day": {"min": 200, "max": 300},
    "education_min": "本科",
    "student_status": "在校生",
    "graduation_year": None,
    "work_days_per_week": 5,
    "internship_months": 6,
    "required_hard_skills": [],
    "required_tools": ["Office", "Office"],
    "preferred_hard_skills": ["Figma"],
    "preferred_experience": [],
}


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


class LLMExtractorTests(unittest.TestCase):
    def test_builds_chat_completion_url(self) -> None:
        self.assertEqual(
            _chat_completions_url("https://api.deepseek.com/"),
            "https://api.deepseek.com/chat/completions",
        )

    def test_validation_deduplicates_list_values(self) -> None:
        prediction = validate_prediction(VALID_PREDICTION)
        self.assertEqual(prediction["required_tools"], ["Office"])

    def test_validation_rejects_missing_fields(self) -> None:
        with self.assertRaises(LLMResponseError):
            validate_prediction({"job_category": "产品经理"})

    def test_validation_accepts_exact_graduation_year(self) -> None:
        prediction = dict(VALID_PREDICTION)
        prediction["graduation_year"] = {"exact": 2027}
        self.assertEqual(
            validate_prediction(prediction)["graduation_year"], {"exact": 2027}
        )

    @patch("src.llm_extractor.urlopen")
    def test_extracts_json_from_compatible_api(self, mocked_urlopen) -> None:
        mocked_urlopen.return_value = _FakeResponse(
            {"choices": [{"message": {"content": json.dumps(VALID_PREDICTION, ensure_ascii=False)}}]}
        )
        config = LLMConfig("test-key", "https://example.com/v1", "test-model", max_retries=0)
        extractor = OpenAICompatibleExtractor(config)
        prediction = extractor.extract(load_jsonl()[0])
        self.assertEqual(prediction["job_category"], "产品经理")
        request = mocked_urlopen.call_args.args[0]
        sent = json.loads(request.data.decode("utf-8"))
        self.assertEqual(sent["response_format"], {"type": "json_object"})
        self.assertEqual(sent["model"], "test-model")
        self.assertIn("用工类型=实习", sent["messages"][1]["content"])
        self.assertEqual(extractor.prompt_version, "job_extraction_v3")
        self.assertEqual(extractor.method_name, "llm:test-model:job_extraction_v3")
        self.assertNotIn("thinking", sent)

    @patch("src.llm_extractor.urlopen")
    def test_disables_thinking_for_deepseek(self, mocked_urlopen) -> None:
        mocked_urlopen.return_value = _FakeResponse(
            {"choices": [{"message": {"content": json.dumps(VALID_PREDICTION, ensure_ascii=False)}}]}
        )
        config = LLMConfig("test-key", "https://api.deepseek.com", "deepseek-flash", max_retries=0)
        OpenAICompatibleExtractor(config).extract(load_jsonl()[0])
        request = mocked_urlopen.call_args.args[0]
        sent = json.loads(request.data.decode("utf-8"))
        self.assertEqual(sent["thinking"], {"type": "disabled"})

    @patch("src.llm_extractor.time.sleep")
    @patch("src.llm_extractor.urlopen")
    def test_retries_an_empty_model_response(self, mocked_urlopen, _mocked_sleep) -> None:
        mocked_urlopen.side_effect = [
            _FakeResponse({"choices": [{"message": {"content": ""}}]}),
            _FakeResponse(
                {"choices": [{"message": {"content": json.dumps(VALID_PREDICTION, ensure_ascii=False)}}]}
            ),
        ]
        config = LLMConfig("test-key", "https://example.com/v1", "test-model", max_retries=1)
        extractor = OpenAICompatibleExtractor(config)
        prediction = extractor.extract(load_jsonl()[0])
        self.assertEqual(prediction["job_category"], "产品经理")
        self.assertEqual(mocked_urlopen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
