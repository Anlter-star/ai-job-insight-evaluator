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
- [x] 编写标注规范
- [x] 实现规则基线
- [x] 完成兼容多种模型的大模型提取器
- [ ] 填写 API 配置并运行大模型评测
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

完整运行会在本地生成 `results/llm_predictions.jsonl` 和 `results/llm_metrics.json`。这两个文件默认不提交，以免不同模型或重复实验的输出互相覆盖；正式实验结果确认后可另行选择性纳入版本管理。

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
