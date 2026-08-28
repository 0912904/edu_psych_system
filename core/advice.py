# -*- coding: utf-8 -*-
"""建议生成：规则模板兜底 + LLM 增强预留接口。"""


def classroom_advice(class_stats, alerts):
    """班级层面教学与关怀建议（规则模板）。"""
    tips = []
    cv = class_stats.get("class_valence", 0)
    if cv < -0.15:
        tips.append("班级整体情绪偏低，建议调整教学节奏，插入互动或小组活动提振氛围。")
    elif cv > 0.3:
        tips.append("班级情绪状态积极，可适当增加挑战性内容，保持学习动力。")
    else:
        tips.append("班级情绪总体平稳，保持当前节奏，注意关注个别波动学生。")
    if alerts:
        tips.append("检测到 %d 名学生触发心理关注预警，建议课后以非正式方式了解情况，"
                    "避免公开点名；若持续多次触发，建议转介学校心理老师。" % len(alerts))
    tips.append("提示：情绪识别结果仅供教师参考，不构成心理评估结论；请结合日常观察综合判断。")
    return tips


def interview_advice(alert):
    """访谈层面关怀建议（规则模板）。"""
    base = {"正常": ["受访者情绪状态平稳，可按计划推进后续交流。"],
            "建议关注": ["受访者存在一定负性情绪，建议后续保持定期沟通，观察变化趋势。"],
            "重点关注": ["受访者负性情绪信号明显，建议尽快安排一对一深入交流，"
                       "必要时联系心理老师介入。"]}
    tips = base.get(alert["level"], [])
    tips.append("提示：本结果由算法生成，仅作辅助参考，不能替代专业心理评估。")
    return tips


def llm_advice(structured_metrics, provider=None):
    """LLM 个性化建议预留接口：输入结构化指标 JSON，返回自然语言建议。

    骨架阶段未接入任何在线模型（涉及费用与隐私策略，待评审）；接入时在此
    调用云 API（DeepSeek/通义等）或本地 ollama，规则版结果作为兜底与护栏。
    """
    raise NotImplementedError("LLM 建议接口待接入，当前使用规则模板")
