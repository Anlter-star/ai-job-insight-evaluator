# AI Job Insight Evaluator

[![tests](https://github.com/Anlter-star/ai-job-insight-evaluator/actions/workflows/tests.yml/badge.svg)](https://github.com/Anlter-star/ai-job-insight-evaluator/actions/workflows/tests.yml)
![release](https://img.shields.io/badge/release-v1.0.0-2563EB)
![python](https://img.shields.io/badge/python-3.10%2B-0F172A)

一个可复现的中文招聘信息结构化提取与大模型评测项目。项目覆盖产品经理、数据分析和大模型评测三类实习岗位，对比透明的关键词规则与 `DeepSeek Flash + Prompt v3`，并提供冻结测试集、双轨评测器、错误分析和 Streamlit 交互演示。

> v1.0.0 已完成：15 条未见测试数据正式冻结，Prompt v3 最终评测只运行一次，结果完整保存在仓库中。

## 项目亮点

- **真实数据链路**：从公开 JD 采集、人工标准答案、证据定位，到模型预测与逐岗位错误分析。
- **防数据泄漏**：开发集、调优集和最终测试集隔离；冻结测试集使用 SHA-256 校验。
- **结构化上下文**：页面顶部的地点、日薪、学历、到岗天数和实习期限会进入模型输入。
- **可信评测**：同时报告严格匹配、归一化匹配和原文依据检查，避免把所有标签差异误称为幻觉。
- **可交互演示**：支持离线规则解析、Prompt v3 在线解析、最终指标浏览和复现检查。

## v1.0.0 最终结果

测试集包含 15 条未见岗位，产品经理、数据分析和大模型评测各 5 条。

| 指标 | 关键词规则 | DeepSeek + Prompt v3 |
| --- | ---: | ---: |
| 单值字段宏平均准确率 | 98.52% | **97.78%** |
| 严格列表微平均 F1 | 2.13% | **22.22%** |
| 归一化列表微平均 F1 | 2.13% | **24.69%** |
| 潜在无原文支持项率 | 0.00% | **10.90%** |

Prompt v3 在用工类型、地点、日薪、学历、每周到岗天数和实习期限六项上全部达到 **100%**。列表字段仍受标签粒度、合并/拆分方式和工具字段归属影响，具体边界样本见 [最终评测报告](docs/test_v1_final_report.md)。

## 演示功能

Streamlit 页面包含三个页签：

1. **在线解析**：输入职位正文和页面顶部元数据，选择规则或 Prompt v3，查看结构化 JSON。
2. **最终评测**：直接浏览冻结测试集的单值字段和列表字段指标，无需 API 密钥。
3. **数据与复现**：检查测试集条数、类别平衡和 SHA-256 冻结状态。

```text
┌──────────────── 在线解析 ────────────────┐
│ JD 正文 + 顶部元数据 → 规则 / Prompt v3 │
│                       → 结构化 JSON      │
├──────────────── 最终评测 ────────────────┤
│ 单值准确率 97.78% · 归一化 F1 24.69%    │
├──────────────── 数据与复现 ──────────────┤
│ 15 条 · 3 类 × 5 条 · SHA-256 通过      │
└──────────────────────────────────────────┘
```

## 快速开始

要求 Python 3.10 或更高版本。

```bash
git clone https://github.com/Anlter-star/ai-job-insight-evaluator.git
cd ai-job-insight-evaluator
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app/app.py
```

macOS / Linux：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app/app.py
```

打开 `http://localhost:8501`。浏览最终评测和运行规则解析不需要模型密钥。

## 配置 Prompt v3 在线解析

复制配置模板：

```powershell
Copy-Item .env.example .env
```

在本地 `.env` 中填写 OpenAI 兼容服务：

```text
LLM_API_KEY=你的密钥
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
```

`.env` 已被 Git 忽略。不要在日志、截图、Issue 或提交中暴露真实密钥。

## 复现实验

运行全部自动化测试：

```bash
python -m unittest discover -s tests -v
```

运行开发集规则基线：

```bash
python -m src.run_baseline
```

运行一条模型冒烟测试：

```bash
python -m src.run_llm --limit 1 --output-prefix llm_smoke
```

复核已保存的冻结测试预测，不会再次请求模型：

```bash
python -m src.run_evaluation_v2 --dataset data/test/test_dataset_v1.jsonl --results-dir results/test_v1_final
python -m src.compare_models --results-dir results/test_v1_final
```

冻结测试集校验值保存在 `data/test/test_dataset_v1.sha256`，自动化测试会验证文件未被修改。该测试集不得用于继续调优 Prompt、规则或别名。

## 评测设计

评测对象包括 9 个单值字段与 4 个列表字段：

- 单值：岗位类别、用工类型、地点、日薪、学历、在校状态、毕业年份、到岗天数、实习期限。
- 列表：必需硬技能、必需工具、加分硬技能、加分经验。

评测器 v2 提供三层结果：

- **严格匹配**：预测标签必须与标准答案完全一致。
- **归一化匹配**：使用经过审核的字段级别名处理表达差异。
- **原文依据检查**：将额外项区分为原文支持项和潜在无依据项。

完整定义见 [评测方法](docs/evaluation_methodology.md)。

## 数据分割

| 分割 | 数量 | 用途 |
| --- | ---: | --- |
| 开发集 | 9 | 规则、Prompt 和评测流程开发 |
| 调优集 | 9 | 首轮盲测后用于 Prompt v2/v3 改进 |
| 冻结测试集 v1 | 15 | 唯一一次最终评测与后续不可变回归基准 |

所有数据均来自公开招聘信息，不包含求职者姓名、手机号等个人敏感信息。标注规则见 [标注规范](docs/annotation_guideline.md)。

## 仓库结构

```text
ai-job-insight-evaluator/
├── app/             # Streamlit 交互演示
├── data/            # 开发、调优、冻结测试数据与别名
├── docs/            # 标注规范、评测方法和实验报告
├── prompts/         # Prompt v1-v3
├── results/         # 各轮预测、指标与错误分析
├── scripts/         # 可审计的数据集构建脚本
├── src/             # 提取器、数据加载和评测器
├── tests/           # 自动化测试
└── requirements.txt
```

## 关键文档

- [冻结测试集最终报告](docs/test_v1_final_report.md)
- [Prompt v3 调优报告](docs/tuning_v3_report.md)
- [顶部元数据输入 v2 报告](docs/metadata_input_v2_report.md)
- [首轮冻结验证报告](docs/validation_report_v1.md)
- [评测方法](docs/evaluation_methodology.md)
- [标注规范](docs/annotation_guideline.md)

## 版本

`v1.0.0` 对应冻结测试集与 Prompt v3 最终结果。后续研发必须使用新的开发/验证数据，不得回看测试集进行针对性调优。
