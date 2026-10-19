import unittest
from pathlib import Path

from app.demo_utils import build_demo_record, read_json, render_metric


class DemoUtilsTests(unittest.TestCase):
    def test_build_demo_record_normalizes_user_input(self) -> None:
        record = build_demo_record(
            title="  AI 产品实习生 ",
            company=" 示例公司 ",
            city=" 上海 ",
            responsibilities=" 设计评测体系 ",
            requirements=" 熟悉 Python ",
            preferred=" Agent 项目经验 ",
            employment_type="实习",
            salary_min=200,
            salary_max=300,
            education="本科",
            work_days=4,
            internship_months=3,
        )

        payload = record["input"]
        self.assertEqual(payload["job_title"], "AI 产品实习生")
        self.assertEqual(payload["page_metadata"]["location"], "上海")
        self.assertEqual(
            payload["page_metadata"]["salary_yuan_per_day"],
            {"min": 200, "max": 300},
        )
        self.assertEqual(payload["dataset_context"]["employment_type"], "实习")

    def test_json_and_metric_helpers(self) -> None:
        metrics_path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "test_v1_final"
            / "evaluation_v2_metrics.json"
        )
        metrics = read_json(metrics_path)
        self.assertIn("llm:deepseek-flash:job_extraction_v3", metrics)

        self.assertEqual(render_metric(0.9778), "97.78%")


if __name__ == "__main__":
    unittest.main()
