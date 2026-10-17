"""Transparent keyword and regular-expression baseline extractor."""

from __future__ import annotations

import re
from typing import Any


CHINESE_NUMBERS = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
}

TOOL_PATTERNS = {
    "Office": r"Office",
    "MySQL": r"MySQL",
    "MongoDB": r"MongoDB",
    "Axure": r"Axure",
    "墨刀": r"墨刀",
    "Figma": r"Figma",
    "Cursor": r"Cursor",
    "Claude Code": r"Claude\s+Code",
    "VibeCoding": r"VibeCoding",
    "PR": r"(?<![A-Za-z])PR(?![A-Za-z])|Premiere",
    "剪映": r"剪映",
    "飞书": r"飞书",
}

HARD_SKILL_PATTERNS = {
    "SQL": r"SQL",
    "API与Webhook接口认证及调用": r"API\s*/?\s*Webhook.*接口认证",
    "请求参数与返回结果解析": r"请求参数.*返回结果解析",
    "数据表关系设计": r"表与表之间关系",
    "事件触发与流程机制": r"事件触发.*流程机制",
    "规则制定": r"规则制定",
    "项目管理": r"项目管理",
    "产品生命周期管理": r"产品生命周期管理",
    "逻辑拆解": r"逻辑拆解",
    "数理分析": r"数理分析",
    "识别AI逻辑漏洞与语言问题": r"识别AI的逻辑漏洞和语言能力问题",
    "视听语言基础": r"视听语言基础",
    "审美与视频内容判断": r"审美与视频内容判断",
    "生成式AIGC模型使用经验": r"生成式AIGC模型使用经验",
    "数据处理": r"数据处理能力",
    "AI Agent开发": r"开发AI\s*Agent",
    "脚本开发与数据校验": r"开发脚本.*校验数据",
    "基础数据分析": r"基础的数据分析能力",
    "数据异常模式识别": r"识别数据异常模式",
}

PREFERRED_SKILL_PATTERNS = {
    "数据分析能力": r"数据分析能力",
    "英语流利": r"英语流利",
    "熟悉大模型MaaS平台、训练平台、推理平台或公有云GPU产品": (
        r"熟悉任意一款大模型MaaS平台、训练平台、推理平台或公有云GPU产品"
    ),
}

PREFERRED_EXPERIENCE_PATTERNS = {
    "产品实习经历": r"产品实习经历",
    "户外领队经验": r"户外领队经验",
    "相关活动或比赛经验": r"相关活动或比赛经验",
    "软件开发经验": r"软件开发",
    "低代码系统实施搭建经验": r"(?:实施搭建)?低代码系统",
    "互联网行业经验": r"互联网(?:行业)?经验",
    "新能源汽车行业经验": r"新能源汽车.*行业经验",
    "汽车智能网联行业经验": r"汽车智能网联.*行业经验",
    "数据标注经验": r"数据标注",
    "内容审核经验": r"内容审核",
    "质量评估经验": r"质量评估",
    "独立视频创作经验": r"独立视频创作",
    "动画制作经验": r"动画制作",
    "数字艺术作品": r"数字艺术作品",
    "产品经理实习经验": r"产品经理.*实习经验",
    "产品运营实习经验": r"产品运营.*实习经验",
    "数据运营实习经验": r"数据运营.*实习经验",
    "云计算相关实习经验": r"云计算.*实习经验",
    "AI平台相关实习经验": r"AI平台.*实习经验",
    "商品中心产品经验": r"商品中心产品.*经验",
    "B端产品经验": r"B端.*产品经验",
    "策略产品经验": r"策略产品经验",
    "AI行业从业经验": r"AI行业从业经验",
    "海外留学经历": r"海外留学经验",
    "数据分析类实习经验": r"数据分析类实习经验",
    "互联网行业实习经历": r"互联网.*实习经历",
    "信贷行业实习经历": r"信贷行业实习经历",
}


def _to_int(value: str) -> int:
    return int(value) if value.isdigit() else CHINESE_NUMBERS[value]


def _split_clauses(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"[\n。；;]", text) if item.strip()]


def _matches(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL))


class RuleBasedExtractor:
    """Extract a small, auditable subset of the annotation schema."""

    method_name = "keyword_rules_v1"

    def extract(self, record: dict[str, Any]) -> dict[str, Any]:
        input_data = record["input"]
        title = input_data["job_title"]
        responsibilities = input_data["responsibilities"]
        requirements = input_data["requirements"]
        preferred = input_data["preferred_qualifications"]
        full_text = "\n".join([title, responsibilities, requirements, preferred])

        required_tools, preferred_tools = self._extract_tools(requirements, preferred)
        preferred_skills = self._extract_by_patterns(
            requirements + "\n" + preferred, PREFERRED_SKILL_PATTERNS
        )
        preferred_skills = sorted(set(preferred_skills + preferred_tools))

        return {
            "job_category": self._extract_category(title, full_text),
            "employment_type": "实习" if "实习" in full_text else None,
            "location": input_data["city"] or None,
            "education_min": self._extract_education(requirements),
            "student_status": self._extract_student_status(requirements),
            "graduation_year": self._extract_graduation_year(requirements),
            "work_days_per_week": self._extract_work_days(requirements),
            "internship_months": self._extract_internship_months(requirements),
            "required_hard_skills": self._extract_by_patterns(
                requirements, HARD_SKILL_PATTERNS
            ),
            "required_tools": required_tools,
            "preferred_hard_skills": preferred_skills,
            "preferred_experience": self._extract_by_patterns(
                requirements + "\n" + preferred, PREFERRED_EXPERIENCE_PATTERNS
            ),
        }

    @staticmethod
    def _extract_category(title: str, full_text: str) -> str | None:
        if re.search(r"大模型|评测|生成式AI|AIGC", title, flags=re.IGNORECASE):
            return "大模型评测"
        if "数据分析" in title:
            return "数据分析"
        if re.search(r"产品|Product", title, flags=re.IGNORECASE):
            return "产品经理"
        if re.search(r"模型评测|训练数据", full_text):
            return "大模型评测"
        return None

    @staticmethod
    def _extract_education(requirements: str) -> str | None:
        if "学历不限" in requirements:
            return None
        if "硕士" in requirements:
            return "硕士"
        if "本科" in requirements:
            return "本科"
        return None

    @staticmethod
    def _extract_student_status(requirements: str) -> str | None:
        if re.search(r"在校生|本科在读|在读大[一二三四]", requirements):
            return "在校生"
        return None

    @staticmethod
    def _extract_graduation_year(requirements: str) -> dict[str, int] | None:
        match = re.search(r"(20\d{2})届及以后", requirements)
        return {"min": int(match.group(1))} if match else None

    @staticmethod
    def _extract_work_days(requirements: str) -> int | None:
        patterns = [
            r"([1-7一二三四五六七])\s*天\s*/\s*周",
            r"每周(?:出勤|到岗)?\s*([1-7一二三四五六七])\s*天",
        ]
        for pattern in patterns:
            match = re.search(pattern, requirements)
            if match:
                return _to_int(match.group(1))
        return None

    @staticmethod
    def _extract_internship_months(requirements: str) -> int | None:
        values = [
            int(value)
            for value in re.findall(r"(\d+)\s*(?:个)?月", requirements)
        ]
        return min(values) if values else None

    @staticmethod
    def _extract_tools(requirements: str, preferred: str) -> tuple[list[str], list[str]]:
        required_tools: set[str] = set()
        preferred_tools: set[str] = set()

        for clause in _split_clauses(requirements):
            for tool, pattern in TOOL_PATTERNS.items():
                if _matches(clause, pattern):
                    target = preferred_tools if "优先" in clause else required_tools
                    target.add(tool)

        for tool, pattern in TOOL_PATTERNS.items():
            if _matches(preferred, pattern):
                preferred_tools.add(tool)

        return sorted(required_tools), sorted(preferred_tools)

    @staticmethod
    def _extract_by_patterns(text: str, patterns: dict[str, str]) -> list[str]:
        return sorted(
            label for label, pattern in patterns.items() if _matches(text, pattern)
        )
