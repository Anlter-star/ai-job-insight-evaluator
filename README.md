# AI Job Insight Evaluator

面向大模型评测、产品经理和数据分析岗位的招聘信息分析与大模型评测项目。

## 项目目标

本项目计划收集并标注公开招聘岗位描述（JD），评测不同方法对岗位信息的结构化提取能力，并进一步分析目标岗位的技能需求与差异。

首期覆盖三类实习岗位：

- 大模型评测
- 产品经理
- 数据分析

## MVP 范围

第一阶段将完成：

1. 收集并整理 30 条公开岗位描述。
2. 建立统一的数据字段和人工标注规范。
3. 实现关键词规则基线。
4. 实现大模型结构化提取方案。
5. 使用准确率、Precision、Recall、F1 和幻觉率进行评测。
6. 使用 Streamlit 展示岗位解析、模型评测和岗位洞察。

## 当前进度

- [x] 创建项目仓库
- [x] 定义项目目标与 MVP
- [x] 建立岗位数据模板
- [ ] 收集第一批 30 条岗位数据
- [x] 完成首批 9 条岗位数据与人工标准答案
- [x] 建立额外9条冻结验证集（3类岗位各3条）
- [x] 编写标注规范
- [x] 实现规则基线
- [x] 完成兼容多种模型的大模型提取器
- [x] 填写 API 配置并运行首轮大模型评测
- [x] 实现严格匹配、归一化匹配和原文依据检查的评测器 v2
- [x] 完成9条冻结验证集的首轮规则与DeepSeek盲测
- [ ] 开发可视化页面

## 运行规则基线

项目当前只使用 Python 标准库，不需要安装第三方依赖。建议使用 Python 3.10 或更高版本。

```bash
python -m src.run_baseline
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

程序读取 `data/annotated/gold_dataset.jsonl`，并生成：

- `results/baseline_predictions.jsonl`：每个岗位的规则提取结果。
- `results/baseline_metrics.json`：准确率、Precision、Recall、F1、幻觉率和逐岗位错误。

## 运行大模型提取器

提取器使用 OpenAI 兼容的 `/chat/completions` 协议，默认示例配置为 DeepSeek，也可以通过修改三项配置切换到其他兼容服务。程序只使用 Python 标准库，不需要安装 SDK。

首先复制 `.env.example` 为 `.env`，填写：

```text
LLM_API_KEY=你的密钥
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
```

`.env` 已被 Git 忽略，严禁把真实密钥提交到仓库。建议先调用一条数据检查配置和余额：

```bash
python -m src.run_llm --limit 1 --output-prefix llm_smoke
```

测试成功后运行全部 9 条：

```bash
python -m src.run_llm
python -m src.compare_models
```

完整运行会生成 `results/llm_predictions.jsonl` 和 `results/llm_metrics.json`。仓库保留首轮正式实验结果；使用 `llm_smoke` 等其他前缀产生的临时结果不会提交。

## 运行评测器 v2

评测器 v2 复用保存的规则预测和大模型预测，不会再次请求模型：

```bash
python -m src.run_evaluation_v2
```

它同时报告严格匹配与人工审核同义词后的归一化匹配，并把额外预测拆分为“原文支持、需要复核”和“潜在无依据”两类。详细定义见 `docs/evaluation_methodology.md`，输出文件为：

- `results/evaluation_v2_metrics.json`
- `results/error_analysis.jsonl`

## 首轮大模型评测结果

| 方法 | 单值字段准确率 | 严格技能 F1 | 归一化技能 F1 | 潜在无依据率 |
| --- | ---: | ---: | ---: | ---: |
| 关键词规则 v1 | 79.17% | 85.31% | 85.31% | 0.00% |
| DeepSeek Flash + Prompt v1 | 75.00% | 42.16% | 70.97% | 0.00% |

严格匹配会把“开发AI Agent”和“AI Agent开发”等同义表达判为错误。评测器 v2 使用字段级、人工审核的映射后，DeepSeek 技能 F1 提升至70.97%。第一版62.86%的所谓“幻觉率”实际是严格额外项率，其中包含同义表达、标签粒度和字段归属差异；详细分析见 `docs/evaluation_report_v2.md`。

## 首轮基线结果

当前数据集包含 9 条岗位，产品经理、大模型评测、数据分析各 3 条。

| 指标 | 结果 |
| --- | ---: |
| 单值字段宏平均准确率 | 79.17% |
| 技能字段微平均 Precision | 96.83% |
| 技能字段微平均 Recall | 76.25% |
| 技能字段微平均 F1 | 85.31% |
| 幻觉率 | 3.17% |

规则方法对工具名称和明确关键词表现较好，但容易遗漏需要语义理解的能力要求。到岗天数和实习期限的准确率均为 33.33%，主要因为部分招聘页面顶部的结构化信息没有进入职位正文。当前词典根据首批样本设计，后续需要用新增岗位检验泛化能力，不能将本轮结果视为最终效果。

## 冻结验证集

新增的9条岗位保存在 `data/validation/validation_dataset.jsonl`，不并入规则和提示词开发所用的 `gold_dataset.jsonl`。验证集覆盖产品经理、大模型评测和数据分析各3条，用于下一轮未见数据测试。首轮盲测完成前不得根据验证集修改规则、提示词或归一化别名；数据边界和冲突处理见 `docs/validation_dataset_notes.md`。

冻结当前代码后，使用独立目录运行验证集，避免覆盖开发集结果：

```bash
python -m src.run_baseline --dataset data/validation/validation_dataset.jsonl --results-dir results/validation
python -m src.run_evaluation_v2 --dataset data/validation/validation_dataset.jsonl --results-dir results/validation --rules-only
python -m src.run_llm --dataset data/validation/validation_dataset.jsonl --results-dir results/validation
python -m src.run_evaluation_v2 --dataset data/validation/validation_dataset.jsonl --results-dir results/validation
python -m src.compare_models --results-dir results/validation
```

大模型提取支持断点续跑；若中途网络失败，重新执行同一条 `run_llm` 命令即可跳过已完成岗位。

首轮冻结验证结果：规则技能F1为2.56%，DeepSeek技能F1为19.00%；单值字段准确率分别为66.67%和70.83%。本轮暴露出顶部元数据未进入模型输入、标签粒度不统一和开发集别名表泛化不足等问题。完整分析见 `docs/validation_report_v1.md`。

## 仓库结构

```text
ai-job-insight-evaluator/
├── data/          # 原始数据与人工标注数据
├── docs/          # 标注规范、评测方案和产品文档
├── src/           # 数据加载、规则提取与评测代码
├── app/           # Streamlit 应用
├── results/       # 预测结果、评测指标和图表
├── tests/         # 自动化测试
└── README.md
```

## 数据与隐私

项目仅使用公开招聘信息，不收集求职者的姓名、手机号或其他个人敏感信息。所有数据都将记录来源和采集日期，并遵守来源网站的使用条款。

## License

本项目暂未选择开源许可证。
