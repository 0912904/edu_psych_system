# -*- coding: utf-8 -*-
"""访谈模式：单人情绪分析。

优先复用主系统 multimodal_system 的注意力融合模型（文本现场 BERT 特征 → 三分类
+ 模态权重）；若当前环境缺少 torch/模型，则降级为轻量中文情感词典基线，保证
骨架在任何机器上可运行。返回统一结构：
    {label, probs[3], valence, modal_weights|None, engine}
"""
import os
import re
import sys

import config

_PREDICTOR = None
_ENGINE = None


def _try_load_predictor():
    """尝试加载主系统的注意力融合 Predictor（torch 环境）。"""
    global _PREDICTOR, _ENGINE
    if _ENGINE is not None:
        return _PREDICTOR
    try:
        if config.MMS_DIR not in sys.path:
            sys.path.insert(0, config.MMS_DIR)
        from core.predictor import Predictor          # noqa: 主系统模块
        _PREDICTOR = Predictor("attention")
        _ENGINE = "attention-fusion"
    except Exception:
        _PREDICTOR = None
        _ENGINE = "lexicon-baseline"
    return _PREDICTOR


# ---- 降级基线：中文情感词典（骨架自带，离线可用） -------------------------
_POS = ("开心 高兴 喜欢 满意 顺利 进步 自信 期待 感谢 放松 有趣 认真 积极 "
        "好 棒 优秀 加油 掌握 收获 清楚 明白 支持 鼓励 温暖").split()
_NEG = ("难过 伤心 讨厌 失望 焦虑 紧张 害怕 担心 压力 烦 累 崩溃 孤独 委屈 "
        "生气 差 糟糕 听不懂 跟不上 没意思 放弃 无聊 走神 自卑 哭").split()
_DENY = ("不", "没", "别", "无")


def _lexicon_score(text):
    score = 0.0
    for w in _POS:
        for m in re.finditer(re.escape(w), text):
            neg = any(text[max(0, m.start() - 2):m.start()].endswith(d) for d in _DENY)
            score += -0.8 if neg else 1.0
    for w in _NEG:
        for m in re.finditer(re.escape(w), text):
            neg = any(text[max(0, m.start() - 2):m.start()].endswith(d) for d in _DENY)
            score += 0.8 if neg else -1.0
    return max(-1.0, min(1.0, score / 3.0))


def analyze_text(text):
    """分析一句访谈转写文本，返回统一结构。"""
    pred = _try_load_predictor()
    if pred is not None:
        try:
            r = pred.predict_text(text)
            return {
                "label": r["pred_label"],
                "probs": r["probs"],
                "valence": float(r.get("regression", 0.0) or 0.0),
                "modal_weights": r.get("weights"),
                "engine": _ENGINE,
            }
        except Exception:
            pass
    v = _lexicon_score(text)
    if v > 0.15:
        label, probs = "积极", [0.1, 0.2, 0.7]
    elif v < -0.15:
        label, probs = "消极", [0.7, 0.2, 0.1]
    else:
        label, probs = "中性", [0.2, 0.6, 0.2]
    conf = min(0.95, 0.5 + abs(v) / 2)
    probs = [p * (1 - conf) + (conf if config.CLASS_NAMES_CN[i] == label else 0)
             for i, p in enumerate(probs)]
    s = sum(probs)
    return {"label": label, "probs": [p / s for p in probs], "valence": v,
            "modal_weights": None, "engine": "lexicon-baseline"}


def analyze_interview(turns):
    """分析整段访谈（逐句列表），输出逐句时间线与整体心理状态摘要。"""
    timeline = []
    for i, t in enumerate(turns):
        r = analyze_text(t)
        timeline.append({"index": i, "text": t, "label": r["label"],
                         "valence": r["valence"], "probs": r["probs"]})
    vals = [x["valence"] for x in timeline] or [0.0]
    neg_ratio = sum(1 for x in timeline if x["label"] == "消极") / max(1, len(timeline))
    volatility = (sum(abs(vals[i] - vals[i - 1]) for i in range(1, len(vals)))
                  / max(1, len(vals) - 1)) if len(vals) > 1 else 0.0
    return {
        "timeline": timeline,
        "summary": {
            "mean_valence": sum(vals) / len(vals),
            "neg_ratio": neg_ratio,
            "volatility": volatility,
            "n_turns": len(timeline),
        },
        "engine": timeline[0].get("probs") and (_ENGINE or "lexicon-baseline"),
    }
