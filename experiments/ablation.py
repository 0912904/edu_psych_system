# -*- coding: utf-8 -*-
"""实验三：系统四个设计选择的消融 —— 在真实课堂视频（test.mp4）上重放。

被消融的四项都是我在调试过程中真实改过的地方：
  1) 情绪价值取 argmax 还是按概率加权的期望（曲线是否只在 0/0.9 之间跳）；
  2) 小脸阈值 MIN_FACE_PX（后排糊脸是否参与识别）；
  3) 碎片轨迹过滤 MIN_TRACK_POINTS（几帧误检是否算一名学生）；
  4) 预警统计前是否做 3 点滑动平均（单帧误判是否会触发"情绪骤降"）。
指标是系统真正对外输出的东西：学生人数、班级情绪均值、预警人数、"骤降"最大幅度。
注意：这些数字只说明同一段视频上不同配置的差异，不是模型准确率。
"""
import json
import math
import os
import sys

import numpy as np

import common

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "edu_psych_system"))
import config          # noqa: E402  复用线上阈值，避免实验与系统不一致

CACHE = os.path.join(common.RESULTS, "video_faces.json")


def load_cache():
    if not os.path.exists(CACHE):
        raise SystemExit("先运行 dump_video_faces.py 生成 %s" % CACHE)
    with open(CACHE, encoding="utf-8") as f:
        return json.load(f)


def smooth(vals, win=3):
    """与 core/classroom.py 的 _smooth 完全一致。"""
    if len(vals) < win:
        return list(vals)
    half = win // 2
    return [round(sum(vals[max(0, i - half):i + half + 1])
                  / len(vals[max(0, i - half):i + half + 1]), 3)
            for i in range(len(vals))]


def track(frames, min_face_px, valence_fn):
    """重放线上的认人逻辑（人脸中心距离 < 画面宽 8%），返回各轨迹的情绪序列。"""
    tracks = {}
    small = 0
    for rec in frames:
        w = rec["w"]
        for f in rec["faces"]:
            x, y, fw, fh = f["box"]
            cx, cy = x + fw / 2.0, y + fh / 2.0
            if tracks:
                tid = min(tracks, key=lambda k: math.hypot(
                    tracks[k]["cx"] - cx, tracks[k]["cy"] - cy))
                if math.hypot(tracks[tid]["cx"] - cx,
                              tracks[tid]["cy"] - cy) >= w * 0.08:
                    tid = len(tracks) + 1
            else:
                tid = 1
            if f["size"] < min_face_px:
                small += 1
                continue
            probs = np.array(f["probs"])
            tr = tracks.setdefault(tid, {"cx": cx, "cy": cy, "timeline": []})
            tr["cx"], tr["cy"] = cx, cy
            if tr["timeline"] and tr["timeline"][-1]["t"] == rec["t"]:
                continue
            tr["timeline"].append({"t": rec["t"],
                                   "valence": round(float(valence_fn(probs)), 3)})
    return tracks, small


def stats(timeline, do_smooth):
    vals = (smooth([x["valence"] for x in timeline]) if do_smooth
            else [x["valence"] for x in timeline]) or [0.0]
    neg = sum(1 for x in timeline if x["valence"] < -0.2) / max(1, len(timeline))
    vol = (sum(abs(vals[i] - vals[i - 1]) for i in range(1, len(vals)))
           / max(1, len(vals) - 1)) if len(vals) > 1 else 0.0
    return {"mean_valence": round(sum(vals) / len(vals), 3),
            "neg_ratio": round(neg, 3), "volatility": round(vol, 3),
            "max_drop": round(max((vals[i - 1] - vals[i]
                                   for i in range(1, len(vals))), default=0.0), 3)}


def run(frames, valence="expect", min_face_px=28, min_track_points=5,
        do_smooth=True):
    fn = (common.valence_expect if valence == "expect" else common.valence_argmax)
    tracks, small = track(frames, min_face_px, fn)
    students = [stats(tr["timeline"], do_smooth) for tr in tracks.values()
                if len(tr["timeline"]) >= min_track_points]
    alerts = []
    for st in students:
        r = 0
        r += st["neg_ratio"] > config.ALERT_NEG_RATIO
        r += st["volatility"] > config.ALERT_VOLATILITY
        r += st["max_drop"] > config.ALERT_DROP
        if r:
            alerts.append("重点关注" if r >= 2 else "建议关注")
    mv = [s["mean_valence"] for s in students] or [0.0]
    return {"students": len(students),
            "class_valence": round(sum(mv) / len(mv), 3),
            "small_faces": small,
            "alerts": len(alerts),
            "alerts_high": sum(1 for a in alerts if a == "重点关注"),
            "max_drop_overall": round(max((s["max_drop"] for s in students),
                                          default=0.0), 3),
            "mean_volatility": round(float(np.mean(
                [s["volatility"] for s in students])) if students else 0.0, 3)}


BASE = dict(valence="expect", min_face_px=28, min_track_points=5, do_smooth=True)

VARIANTS = [
    ("系统当前配置（期望情绪值 + 28px + 5点 + 平滑）", {}),
    ("情绪值改回 argmax（第一版做法）", {"valence": "argmax"}),
    ("不过滤小脸（MIN_FACE_PX=0）", {"min_face_px": 0}),
    ("小脸阈值 16px", {"min_face_px": 16}),
    ("小脸阈值 40px", {"min_face_px": 40}),
    ("不过滤碎片轨迹（MIN_TRACK_POINTS=1）", {"min_track_points": 1}),
    ("碎片阈值放宽到 3 点", {"min_track_points": 3}),
    ("预警前不做 3 点平滑", {"do_smooth": False}),
    ("第一版整体配置（argmax + 不过滤 + 不平滑）",
     {"valence": "argmax", "min_face_px": 0, "min_track_points": 1,
      "do_smooth": False}),
]


def main():
    cache = load_cache()
    frames, meta = cache["frames"], cache["meta"]
    rows, out = [], []
    for name, over in VARIANTS:
        cfg = dict(BASE, **over)
        r = run(frames, **cfg)
        out.append({"variant": name, "config": cfg, "result": r})
        rows.append([name, r["students"], r["class_valence"], r["small_faces"],
                     r["alerts"], r["alerts_high"], r["max_drop_overall"],
                     r["mean_volatility"]])
        print("%-40s 学生%2d｜均值%6.3f｜小脸%4d｜预警%2d(重点%d)｜最大骤降%5.2f"
              % (name, r["students"], r["class_valence"], r["small_faces"],
                 r["alerts"], r["alerts_high"], r["max_drop_overall"]),
              flush=True)
    common.save_csv("ablation.csv",
                    ["变体", "学生数", "班级情绪均值", "过小人脸样本",
                     "预警人数", "重点关注", "最大骤降幅度", "平均波动度"], rows)
    common.save_json("ablation.json", {"video": meta, "variants": out})


if __name__ == "__main__":
    main()
