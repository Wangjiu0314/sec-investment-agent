# SEC Investment Research Agent

面向 SEC 10-K 长文档的智能投研 Agent，基于 LangChain、DeepSeek 和 Streamlit 实现财报获取、结构解析、分阶段分析、语义复核与研究报告生成。

项目同时保留通用 Tool Calling 示例，用于演示模型如何选择和执行 Python 工具。

它演示完整流程：

```text
用户输入
  -> DeepSeek 判断是否需要工具
  -> LangChain 执行 Python 工具函数
  -> 工具结果返回给模型
  -> 模型生成最终回答
```

## 环境

```powershell
git clone https://github.com/Wangjiu0314/sec-investment-agent.git
cd sec-investment-agent
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果 PowerShell 阻止激活脚本：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 配置 DeepSeek

打开 `.env`，确认至少有：

```text
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

注意：做工具调用 agent 时建议使用 `deepseek-chat`。

## 运行

```powershell
python .\run_agent.py
```

图形工作台：

```powershell
streamlit run .\streamlit_app.py
```

## 自动化测试

测试不调用 DeepSeek API，也不需要真实 SEC 下载数据：

```powershell
python -m unittest discover -s tests -q
```

工作台提供公司选择、同步分析、阶段进度、Markdown 研究报告、SEC 证据检查和状态下载。

财报工具对话示例：

```text
查看 AAPL 已有的财报分析状态，不新增调用
继续分析 AAPL 最新的 10-K
分析 MSFT 最新的 10-K
```

`analyze_sec_filing` 会在一次工具调用中自动完成全部待处理的 Heading Router、Map、Reduce 和 Semantic Review 任务，并保留缓存断点。只查看状态时不会新增分析任务。

可测试：

```text
现在几点？
帮我计算 12 * 8 + 5
把“今天学习了 LangChain 工具调用”保存成笔记
```

本地命令：

```text
help   显示帮助
tools  显示工具
reset  清空对话上下文
exit   退出
```

## 当前工具

- `get_current_time`: 获取当前本地时间
- `calculate`: 安全计算基础数学表达式
- `save_note`: 把笔记保存到 `notes.txt`
- `analyze_sec_filing`: 运行或继续 SEC 10-K 财报分析 Pipeline

## 核心代码位置

```text
src/agent_playground/main.py
src/agent_playground/filing_tool.py
```

## 财报分析 Pipeline

Notebook `02` 到 `08` 保留为学习和调试记录。日常运行使用自动化入口：

```powershell
python .\run_filing_pipeline.py --ticker AAPL --form 10-K
```

默认每轮最多新增 3 次 DeepSeek 调用，并自动完成：

```text
查找或下载 SEC 财报
  -> 保留 HTML 标题、段落、表格和样式信息
  -> Python 生成通用标题候选
  -> DeepSeek 判断正文标题、目录、页眉、表格标题和交叉引用
  -> Python 校验标题 ID，并根据标题层级构造语义区间
  -> 在 Business / Risk Factors / MD&A 区间内部切分析块
  -> 按标题路由选择分析结构
  -> 校验引用、风险语气和财务数字
  -> 缓存结果并记录断点状态
  -> Business / Risk / MD&A 三路 Reduce
  -> Business / Risk / MD&A 三路语义审查
  -> 修正风险语气、翻译和证据不充分的结论
  -> 校验全局引用并生成研究备忘录
```

只查看本地进度，不访问 SEC、不新增模型调用：

```powershell
python .\run_filing_pipeline.py --ticker AAPL --offline --max-new-calls 0
```

只处理风险章节，本轮最多新增 3 次调用：

```powershell
python .\run_filing_pipeline.py --ticker AAPL --sections risk_factors --max-new-calls 3
```

处理全部待分析 chunks：

```powershell
python .\run_filing_pipeline.py --ticker AAPL --all
```

新股票首次下载前，需要在 `.env` 配置：

```text
SEC_USER_AGENT=你的名字 你的邮箱
```

Pipeline 状态保存在：

```text
data/sec/<TICKER>/pipeline_status.json
```

新语义路由层会额外保存：

```text
data/sec/<TICKER>/*_structural_blocks_v1.json
data/sec/<TICKER>/heading_routes_v1.json
data/sec/<TICKER>/*_heading_chunks_v1.json
```

完整财报原文不会因为当前研究报告只分析三个目标而被删除。Router 会为标题候选保留 `role`、`heading_level`、`sec_section`、`topics`、`target_analyzers` 和 `confidence`；非目标章节仍保存在标题缓存中，后续可以接入新的分析器。

Map 全部完成后会继续生成：

```text
data/sec/<TICKER>/reduce_v1.json
data/sec/<TICKER>/semantic_review_v1.json
data/sec/<TICKER>/research_memo_draft.md
data/sec/<TICKER>/research_memo.md
```

`research_memo_draft.md` 是未经第二层语义审查的 Reduce 草稿。
`research_memo.md` 只保留 Reviewer 支持或修正后的内容，并带 SEC 原文位置和证据索引；仍建议由专业人员完成最终复核。
