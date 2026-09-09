# -*- coding: utf-8 -*-
"""心理关注预警：基于情绪时间线统计的规则判定（可解释、可调阈值）。"""
import config


def student_alerts(students):
    """输入课堂各学生统计，输出预警名单 [{track_id, level, reasons}]。"""
    out = []
    for s in students:
        st, reasons = s["stats"], []
        if st["neg_ratio"] > config.ALERT_NEG_RATIO:
            reasons.append("消极情绪时间占比 %.0f%%，超过阈值 %.0f%%"
                           % (st["neg_ratio"] * 100, config.ALERT_NEG_RATIO * 100))
        if st["volatility"] > config.ALERT_VOLATILITY:
            reasons.append("情绪波动度 %.2f 偏高，状态不稳定" % st["volatility"])
        if st.get("max_drop", 0) > config.ALERT_DROP:
            reasons.append("检测到情绪骤降（幅度 %.2f），可能有突发负性事件" % st["max_drop"])
        if reasons:
            level = "重点关注" if len(reasons) >= 2 else "建议关注"
            out.append({"track_id": s["track_id"], "level": level, "reasons": reasons,
                        "stats": st})
    return out


def interview_alert(summary):
    """访谈整体摘要 → 单人关注建议。"""
    reasons = []
    if summary["neg_ratio"] > config.ALERT_NEG_RATIO:
        reasons.append("访谈中消极表达占比 %.0f%%" % (summary["neg_ratio"] * 100))
    if summary["mean_valence"] < -0.25:
        reasons.append("整体情绪价值均值 %.2f，偏消极" % summary["mean_valence"])
    if summary["volatility"] > config.ALERT_VOLATILITY:
        reasons.append("情绪波动明显（%.2f）" % summary["volatility"])
    if not reasons:
        return {"level": "正常", "reasons": ["未触发预警规则，情绪状态平稳"]}
    return {"level": "重点关注" if len(reasons) >= 2 else "建议关注", "reasons": reasons}
