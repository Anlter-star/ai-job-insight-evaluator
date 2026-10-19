"""Streamlit demo for interactive extraction and frozen-result review."""

from __future__ import annotations

import sys
from hashlib import sha256
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_extractor import RuleBasedExtractor  # noqa: E402
from src.data_loader import load_jsonl  # noqa: E402
from src.llm_extractor import (  # noqa: E402
    LLMConfigurationError,
    LLMResponseError,
    LLMConfig,
    OpenAICompatibleExtractor,
)
from app.demo_utils import build_demo_record, read_json, render_metric  # noqa: E402


TEST_DATASET = PROJECT_ROOT / "data" / "test" / "test_dataset_v1.jsonl"
TEST_CHECKSUM = PROJECT_ROOT / "data" / "test" / "test_dataset_v1.sha256"
FINAL_METRICS = PROJECT_ROOT / "results" / "test_v1_final" / "evaluation_v2_metrics.json"
LLM_METRICS = PROJECT_ROOT / "results" / "test_v1_final" / "llm_metrics.json"


st.set_page_config(
    page_title="AI Job Insight Evaluator",
    page_icon="🧭",
    layout="wide",
)

st.title("AI Job Insight Evaluator")
st.caption("招聘信息结构化提取 · Prompt 评测 · 可复现实验")

with st.sidebar:
    st.subheader("v1.0.0")
    st.write("DeepSeek Flash · Prompt v3")
    st.success("冻结测试集：15 条")
    st.caption("产品经理 / 数据分析 / 大模型评测各 5 条")
    st.divider()
    st.markdown(
        "[查看项目文档](https://github.com/Anlter-star/"
        "ai-job-insight-evaluator/blob/main/docs/test_v1_final_report.md)"
    )

extract_tab, metrics_tab, reproduce_tab = st.tabs(
    ["在线解析", "最终评测", "数据与复现"]
)

with extract_tab:
    st.subheader("解析一条招聘信息")
    st.write("规则模式可离线运行；Prompt v3 模式需要本地 `.env` 中的模型配置。")

    test_records = load_jsonl(TEST_DATASET)
    sample_options = {"空白输入": None}
    sample_options.update(
        {
            f"示例 · {item['input']['job_title']}": item
            for item in test_records[:3]
        }
    )
    selected_label = st.selectbox("快速载入示例", list(sample_options))
    sample = sample_options[selected_label]
    sample_input = sample["input"] if sample else {}
    sample_metadata = sample_input.get("page_metadata", {})
    sample_salary = sample_metadata.get("salary_yuan_per_day") or {}

    left, right = st.columns([3, 2])
    with left:
        title = st.text_input("职位名称", value=sample_input.get("job_title", ""))
        company = st.text_input("公司", value=sample_input.get("company", ""))
        responsibilities = st.text_area(
            "岗位职责", value=sample_input.get("responsibilities", ""), height=170
        )
        requirements = st.text_area(
            "任职要求", value=sample_input.get("requirements", ""), height=170
        )
        preferred = st.text_area(
            "加分项", value=sample_input.get("preferred_qualifications", ""), height=100
        )

    with right:
        city = st.text_input("城市", value=sample_input.get("city", ""))
        employment_type = st.selectbox("用工类型", ["实习", "全职", "兼职"])
        education = st.selectbox(
            "最低学历", ["本科", "硕士", "博士", "学历不限"],
            index=["本科", "硕士", "博士", "学历不限"].index(
                sample_metadata.get("education", "本科")
            ) if sample_metadata.get("education", "本科") in ["本科", "硕士", "博士", "学历不限"] else 0,
        )
        salary_min = st.number_input(
            "最低日薪（元）", min_value=0, value=int(sample_salary.get("min", 150))
        )
        salary_max = st.number_input(
            "最高日薪（元）", min_value=0, value=int(sample_salary.get("max", 250))
        )
        work_days = st.number_input(
            "每周到岗天数", min_value=1, max_value=7,
            value=int(sample_metadata.get("work_days_per_week", 4)),
        )
        internship_months = st.number_input(
            "实习期限（月）", min_value=1, max_value=24,
            value=int(sample_metadata.get("internship_months", 3)),
        )
        method = st.radio("解析方法", ["关键词规则", "DeepSeek + Prompt v3"])

    if st.button("开始解析", type="primary", use_container_width=True):
        if not title or not responsibilities or not requirements:
            st.error("请至少填写职位名称、岗位职责和任职要求。")
        elif salary_max < salary_min:
            st.error("最高日薪不能低于最低日薪。")
        else:
            demo_record = build_demo_record(
                title=title,
                company=company,
                city=city,
                responsibilities=responsibilities,
                requirements=requirements,
                preferred=preferred,
                employment_type=employment_type,
                salary_min=salary_min,
                salary_max=salary_max,
                education=education,
                work_days=work_days,
                internship_months=internship_months,
            )
            try:
                if method == "关键词规则":
                    prediction = RuleBasedExtractor().extract(demo_record)
                    method_name = RuleBasedExtractor.method_name
                else:
                    with st.spinner("正在调用模型并校验结构化输出…"):
                        extractor = OpenAICompatibleExtractor(LLMConfig.from_env())
                        prediction = extractor.extract(demo_record)
                        method_name = extractor.method_name
                st.success(f"解析完成 · {method_name}")
                st.json(prediction)
            except (LLMConfigurationError, LLMResponseError) as exc:
                st.error(str(exc))

with metrics_tab:
    st.subheader("冻结测试集 v1 · 最终结果")
    evaluation = read_json(FINAL_METRICS)
    llm = evaluation["llm:deepseek-flash:job_extraction_v3"]
    raw_llm = read_json(LLM_METRICS)

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "单值字段准确率", render_metric(llm["strict"]["scalar_macro_accuracy"])
    )
    metric_columns[1].metric(
        "严格列表 F1", render_metric(llm["strict"]["list_micro"]["f1"])
    )
    metric_columns[2].metric(
        "归一化列表 F1", render_metric(llm["normalized"]["list_micro"]["f1"])
    )
    metric_columns[3].metric(
        "潜在无依据率",
        render_metric(llm["groundedness"]["potential_unsupported_rate"]),
    )

    st.markdown("#### 单值字段准确率")
    scalar_rows = [
        {"字段": field, "准确率": render_metric(values["accuracy"])}
        for field, values in raw_llm["scalar_metrics"].items()
    ]
    st.dataframe(scalar_rows, use_container_width=True, hide_index=True)

    st.markdown("#### 列表字段严格匹配")
    list_rows = [
        {
            "字段": field,
            "Precision": render_metric(values["precision"]),
            "Recall": render_metric(values["recall"]),
            "F1": render_metric(values["f1"]),
        }
        for field, values in llm["strict"]["list_metrics"].items()
    ]
    st.dataframe(list_rows, use_container_width=True, hide_index=True)
    st.info(
        "严格额外项不等于幻觉：156 个模型列表项中，109 个额外项能在原文找到直接支持；"
        "真正需要重点复核的是 17 个潜在无原文支持项。"
    )

with reproduce_tab:
    st.subheader("冻结数据与一键复现")
    records = load_jsonl(TEST_DATASET)
    expected_checksum = TEST_CHECKSUM.read_text(encoding="utf-8").split()[0]
    actual_checksum = sha256(TEST_DATASET.read_bytes()).hexdigest()

    status_columns = st.columns(3)
    status_columns[0].metric("冻结记录", len(records))
    status_columns[1].metric("类别", "3 × 5")
    status_columns[2].metric("校验状态", "通过" if expected_checksum == actual_checksum else "失败")

    st.code(
        "python -m unittest discover -s tests -v\n"
        "python -m src.run_baseline --dataset data/test/test_dataset_v1.jsonl "
        "--results-dir results/test_v1_final\n"
        "python -m src.run_evaluation_v2 --dataset data/test/test_dataset_v1.jsonl "
        "--results-dir results/test_v1_final",
        language="bash",
    )
    st.caption(f"SHA-256 · {expected_checksum}")
    st.warning("冻结测试集仅用于最终评测和回归检查，不得用于继续调优 Prompt 或规则。")
