# 教育心理情绪分析系统（可运行骨架）

**主题**：教育与心理 —— 分析课堂或访谈视频中的情绪状态，辅助教师关注学生心理。
基于「多模态数据融合与分析系统（multimodal_system）」与「Multimodalv2.0-edu」扩展构建。

## 功能
- **访谈模式（单人）**：逐句情绪时间线 + 整体心理状态摘要 + 关注判定与建议。
  优先复用主系统注意力融合模型（torch 环境）；无 torch 时自动降级为中文情感词典基线，骨架零依赖可跑。
- **课堂模式（多人）**：逐秒 per-face 表情时间线 + 班级聚合指标 + **心理关注预警名单**。
  有 opencv 时复用 edu 系统的 YuNet 人脸检测 + FERplus 表情 ONNX；否则输出带标注的模拟数据供联调。
- **预警规则**（core/alerts.py，可调阈值）：消极占比 >40%、情绪波动度 >0.35、情绪骤降 >0.8。
- **建议层**（core/advice.py）：规则模板兜底；`llm_advice()` 为 LLM 个性化建议预留接口。
- **SQLite 档案**：分析会话与预警记录留痕，支持跨次趋势观察。

## 运行（零依赖，Python≥3.8 标准库即可）
```
python app.py        # → http://localhost:5100
```
课堂视频真实分析需 opencv-python 与 edu 系统的 ONNX 模型（config.EDU_MODELS_DIR）；
访谈模式接主系统模型需在 conda pytorch 环境运行。

## 目录
```
app.py            零依赖 HTTP 服务（ThreadingHTTPServer + REST）
config.py         路径 / 阈值 / 复用系统位置
core/interview.py 访谈单人分析（注意力融合 → 词典基线降级）
core/classroom.py 课堂多人脸表情时间线（YuNet+FERplus → 模拟降级）
core/alerts.py    心理关注预警规则
core/advice.py    规则建议 + LLM 预留接口
core/db.py        SQLite 会话 / 预警存储
web/index.html    ECharts 单页前端（访谈 / 课堂双模式）
```

## 隐私与伦理约定
人脸与情绪属敏感数据：本地处理、匿名 track id、不留存原始视频（uploads 可定期清理）；
识别结果仅供教师参考，不构成心理评估结论。
