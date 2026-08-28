# -*- coding: utf-8 -*-
"""教育心理情绪分析系统 —— 全局配置。

基于「多模态数据融合与分析系统」与 Multimodalv2.0-edu 扩展：
- 访谈模式：单人三模态情感分析（复用 multimodal_system 的注意力融合模型，缺环境时降级为词典基线）。
- 课堂模式：多人脸表情时间线（复用 Multimodalv2.0-edu 的 YuNet+FERplus ONNX，缺环境时输出模拟数据）。
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "db", "app.db")
WEB_DIR = os.path.join(BASE_DIR, "web")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

for _d in (os.path.dirname(DB_PATH), UPLOAD_DIR):
    os.makedirs(_d, exist_ok=True)

HOST, PORT = "0.0.0.0", 5100

# 复用路径：主系统（注意力融合模型）与 edu 系统（人脸/表情 ONNX 模型）
MMS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "02_项目开发阶段", "multimodal_system"))
EDU_MODELS_DIR = os.path.normpath(os.path.join(
    BASE_DIR, "..", "..", "Multimodalv2.0-edu", "Multimodalv2.0-edu", "webapp", "models"))

# 情绪类别（与 CH-SIMS 三分类对齐）
CLASS_NAMES_CN = ["消极", "中性", "积极"]

# 心理关注预警规则阈值
ALERT_NEG_RATIO = 0.40      # 消极情绪时间占比阈值
ALERT_VOLATILITY = 0.35     # 情绪波动度（相邻时刻价值变化均值）阈值
ALERT_DROP = 0.8            # 情绪骤降幅度阈值（价值 -1~1）
