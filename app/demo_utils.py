"""Pure helpers shared by the Streamlit demo and its tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_demo_record(
    *,
    title: str,
    company: str,
    city: str,
    responsibilities: str,
    requirements: str,
    preferred: str,
    employment_type: str,
    salary_min: int,
    salary_max: int,
    education: str,
    work_days: int,
    internship_months: int,
) -> dict[str, Any]:
    return {
        "job_id": "DEMO",
        "input": {
            "job_title": title.strip(),
            "company": company.strip(),
            "city": city.strip(),
            "responsibilities": responsibilities.strip(),
            "requirements": requirements.strip(),
            "preferred_qualifications": preferred.strip(),
            "page_metadata": {
                "location": city.strip() or None,
                "salary_yuan_per_day": {
                    "min": int(salary_min),
                    "max": int(salary_max),
                },
                "education": education,
                "work_days_per_week": int(work_days),
                "internship_months": int(internship_months),
            },
            "dataset_context": {"employment_type": employment_type},
        },
    }


def render_metric(value: float) -> str:
    return f"{value:.2%}"
