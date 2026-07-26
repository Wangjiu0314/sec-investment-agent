# Presentation Assets

这些图按 1600 x 900（16:9）制作，可直接插入 PowerPoint。SVG 是矢量格式，放大不会失真。

本地预览入口：`index.html`。

## 图稿

1. `01_agent_architecture.svg`
   - 用途：项目总览、技术架构介绍
   - 核心信息：DeepSeek 做语义判断，Python 做验证与执行

2. `02_architecture_evolution.svg`
   - 用途：讲项目难点、泛化问题和架构迭代
   - 核心信息：AAPL -> GOOGL -> JPM 如何推动三次架构变化

3. `03_reliability_guardrails.svg`
   - 用途：回答“如何降低幻觉、如何保证可信”
   - 核心信息：Schema、边界、证据、财务数字、Review/Cache 五层护栏

4. `04_project_evidence.svg`
   - 用途：展示测试、跨公司结果和当前进度
   - 注意：JPM 明确标记为进行中，不应被标注为已完成报告

5. `05_product_mockup.svg`
   - 用途：展示未来面向投行/研究人员的产品形态
   - 注意：这是效果图，不是当前已经实现的网页界面

## 推荐 PPT 顺序

```text
1. 问题：10-K 很长，人工研究慢且难以追溯
2. 总体架构：01_agent_architecture.svg
3. 技术难点与迭代：02_architecture_evolution.svg
4. 可信性设计：03_reliability_guardrails.svg
5. 实际结果：04_project_evidence.svg
6. 产品愿景：05_product_mockup.svg
```

## 使用建议

- PowerPoint 中优先插入 SVG。
- 不要把 `05_product_mockup.svg` 描述为已经上线的产品。
- 数据状态变化后，需要同步更新 `04_project_evidence.svg` 和效果图中的 JPM 进度。
