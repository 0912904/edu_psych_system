# -*- coding: utf-8 -*-
"""实验八：头部姿态多线索融合 —— 把项目一的"多模态融合"思想搬回课堂。

表情只是一个线索。YuNet 每张人脸自带 5 个关键点（右眼、左眼、鼻尖、右嘴角、左嘴角），
不用额外模型就能算出三个几何姿态量：
    yaw   = (鼻尖x - 两眼中点x) / 两眼距离            侧头/转头（0 为正脸，越大越偏）
    pitch = (鼻尖y - 两眼中点y) / (嘴中点y - 两眼中点y) 低头/抬头的相对量（正脸约 0.5~0.7）
    roll  = atan2(左眼y - 右眼y, 左眼x - 右眼x)         歪头
本实验回答三个可以检验的问题：
  Q1 转头/低头时，FERplus 是否更"犹豫"、更倾向判中性？（熵、中性概率 vs |yaw| 的相关）
  Q2 表情线索与姿态线索是否提供了不同的信息？（观测级 Spearman 相关，若接近 0 说明互补）
  Q3 融合姿态后，个体"专注度代理"与仅用表情的排序差多大？（Kendall τ）
融合规则（可解释、无训练）：
    attention_i = 1 - clip(|yaw| / 0.6, 0, 1) · 0.6 - clip(|pitch - median| / 0.4, 0, 1) · 0.4
    engagement_i = 0.5 · (valence_i + 1) / 2 + 0.5 · attention_i
注意：没有课堂标签，本实验只能说明"线索之间的关系"与"融合改变了什么"，不是准确率。
"""
import json
import math
import os

import numpy as np

import common

CACHE = os.path.join(common.RESULTS, "video_faces.json")
MIN_FACE_PX = 28


def load_cache():
    if not os.path.exists(CACHE):
        raise SystemExit("先运行 dump_video_faces.py 生成 %s" % CACHE)
    with open(CACHE, encoding="utf-8") as f:
        return json.load(f)


def pose_from_landmarks(lm):
    """lm: [x_re, y_re, x_le, y_le, x_nose, y_nose, x_rm, y_rm, x_lm, y_lm]。"""
    re = np.array(lm[0:2]); le = np.array(lm[2:4]); nose = np.array(lm[4:6])
    rm = np.array(lm[6:8]); lmth = np.array(lm[8:10])
    eye_mid = (re + le) / 2.0
    mouth_mid = (rm + lmth) / 2.0
    eye_dist = float(np.linalg.norm(le - re)) or 1e-6
    face_h = float(mouth_mid[1] - eye_mid[1]) or 1e-6
    yaw = float((nose[0] - eye_mid[0]) / eye_dist)
    pitch = float((nose[1] - eye_mid[1]) / face_h)
    roll = float(math.degrees(math.atan2(le[1] - re[1], le[0] - re[0])))
    return yaw, pitch, roll


def rankdata(a):
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    # 平均并列名次
    uniq, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq)); np.add.at(sums, inv, ranks)
    return sums[inv] / cnt[inv]


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_tau(x, y):
    n = len(x)
    if n < 2:
        return float("nan")
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
    return float(s / (n * (n - 1) / 2))


def build_tracks(frames):
    tracks = {}
    for rec in frames:
        w = rec["w"]
        for f in rec["faces"]:
            if "landmarks" not in f:
                raise SystemExit("缓存缺少 landmarks 字段，请用新版 dump_video_faces.py 重新生成")
            x, y, fw, fh = f["box"]
            cx, cy = x + fw / 2.0, y + fh / 2.0
            tid = None
            if tracks:
                k = min(tracks, key=lambda k: math.hypot(tracks[k]["cx"] - cx,
                                                          tracks[k]["cy"] - cy))
                if math.hypot(tracks[k]["cx"] - cx, tracks[k]["cy"] - cy) < w * 0.08:
                    tid = k
            if tid is None:
                tid = len(tracks) + 1
            tr = tracks.setdefault(tid, {"cx": cx, "cy": cy, "obs": []})
            tr["cx"], tr["cy"] = cx, cy
            if tr["obs"] and tr["obs"][-1]["t"] == rec["t"]:
                continue
            if f["size"] < MIN_FACE_PX:
                continue
            yaw, pitch, roll = pose_from_landmarks(f["landmarks"])
            p = np.array(f["probs"])
            tr["obs"].append({"t": rec["t"], "yaw": yaw, "pitch": pitch, "roll": roll,
                              "valence": common.valence_expect(p),
                              "p_neutral": float(p[0]),
                              "entropy": float(-(np.clip(p, 1e-9, 1) * np.log(np.clip(p, 1e-9, 1))).sum())})
    return {k: v for k, v in tracks.items() if len(v["obs"]) >= 5}


def main():
    cache = load_cache()
    tracks = build_tracks(cache["frames"])
    obs = [o for tr in tracks.values() for o in tr["obs"]]
    yaw = np.array([abs(o["yaw"]) for o in obs])
    pitch = np.array([o["pitch"] for o in obs])
    pitch_dev = np.abs(pitch - np.median(pitch))
    ent = np.array([o["entropy"] for o in obs])
    pneu = np.array([o["p_neutral"] for o in obs])
    val = np.array([o["valence"] for o in obs])
    print("学生 %d 人，观测 %d 次；|yaw| 中位数 %.3f，pitch 中位数 %.3f"
          % (len(tracks), len(obs), np.median(yaw), np.median(pitch)))

    # Q1：姿态偏离 vs 模型犹豫
    q1 = {"spearman_absyaw_entropy": spearman(yaw, ent),
          "spearman_absyaw_pneutral": spearman(yaw, pneu),
          "spearman_pitchdev_entropy": spearman(pitch_dev, ent),
          "spearman_pitchdev_pneutral": spearman(pitch_dev, pneu)}
    # 分箱看趋势：正脸 / 轻微偏转 / 明显偏转
    bins = [(0, 0.15, "正脸 |yaw|<0.15"), (0.15, 0.35, "轻偏 0.15~0.35"), (0.35, 9, "明显偏 >0.35")]
    q1_bins = []
    for lo, hi, name in bins:
        m = (yaw >= lo) & (yaw < hi)
        if m.sum():
            q1_bins.append({"bin": name, "n": int(m.sum()),
                            "mean_entropy": float(ent[m].mean()),
                            "mean_p_neutral": float(pneu[m].mean()),
                            "mean_valence": float(val[m].mean())})
            print("  %-16s n=%3d 熵 %.3f 中性概率 %.3f 情绪值 %.3f"
                  % (name, m.sum(), ent[m].mean(), pneu[m].mean(), val[m].mean()))
    # Q2：表情与姿态的相关
    q2 = {"spearman_valence_absyaw": spearman(val, yaw),
          "spearman_valence_pitchdev": spearman(val, pitch_dev)}
    print("Q1 |yaw|~熵 ρ=%.3f，|yaw|~中性概率 ρ=%.3f；Q2 情绪值~|yaw| ρ=%.3f"
          % (q1["spearman_absyaw_entropy"], q1["spearman_absyaw_pneutral"],
             q2["spearman_valence_absyaw"]))

    # Q3：融合前后的个体排序
    med_pitch = float(np.median(pitch))
    students = []
    for tid, tr in tracks.items():
        v = np.array([o["valence"] for o in tr["obs"]])
        ay = np.array([abs(o["yaw"]) for o in tr["obs"]])
        pd = np.abs(np.array([o["pitch"] for o in tr["obs"]]) - med_pitch)
        att = 1 - np.clip(ay / 0.6, 0, 1) * 0.6 - np.clip(pd / 0.4, 0, 1) * 0.4
        eng = 0.5 * (v + 1) / 2 + 0.5 * att
        students.append({"track_id": tid, "n": len(v),
                         "mean_valence": float(v.mean()),
                         "turned_ratio": float((ay > 0.35).mean()),
                         "head_dev_ratio": float((pd > 0.25).mean()),
                         "attention_proxy": float(att.mean()),
                         "engagement_fused": float(eng.mean()),
                         "engagement_expr_only": float(((v + 1) / 2).mean())})
    tau = kendall_tau([s["engagement_expr_only"] for s in students],
                      [s["engagement_fused"] for s in students])
    rank_e = np.argsort(np.argsort([-s["engagement_expr_only"] for s in students]))
    rank_f = np.argsort(np.argsort([-s["engagement_fused"] for s in students]))
    moved = int((rank_e != rank_f).sum())
    print("Q3 融合前后个体排序 Kendall τ=%.3f，%d/%d 名学生名次变化" % (tau, moved, len(students)))
    for s in sorted(students, key=lambda s: -s["engagement_fused"]):
        print("  学生#%d n=%2d 情绪%6.3f 转头占比%.2f 低/抬头占比%.2f 注意力%.2f 融合%.3f"
              % (s["track_id"], s["n"], s["mean_valence"], s["turned_ratio"],
                 s["head_dev_ratio"], s["attention_proxy"], s["engagement_fused"]))

    common.save_csv("head_pose.csv",
                    ["track_id", "n", "mean_valence", "turned_ratio", "head_dev_ratio",
                     "attention_proxy", "engagement_expr_only", "engagement_fused"],
                    [[s[k] if isinstance(s[k], int) else round(s[k], 4) for k in
                      ("track_id", "n", "mean_valence", "turned_ratio", "head_dev_ratio",
                       "attention_proxy", "engagement_expr_only", "engagement_fused")]
                     for s in students])
    common.save_json("head_pose.json", {
        "video": cache["meta"], "n_students": len(students), "n_obs": len(obs),
        "pose_stats": {"abs_yaw_median": float(np.median(yaw)),
                       "abs_yaw_p90": float(np.percentile(yaw, 90)),
                       "pitch_median": med_pitch},
        "q1_pose_vs_uncertainty": q1, "q1_bins": q1_bins,
        "q2_expression_vs_pose": q2,
        "q3_fusion": {"kendall_tau": tau, "rank_changed": moved},
        "students": students})
    plot(yaw, ent, pneu, val, students)


def plot(yaw, ent, pneu, val, students):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=150)
    axes[0].scatter(yaw, ent, s=8, alpha=0.5)
    axes[0].set_xlabel("|yaw| (nose offset / eye distance)")
    axes[0].set_ylabel("FERplus prediction entropy")
    axes[0].set_title("Q1: head turn vs. model uncertainty", fontsize=10)
    axes[1].scatter(yaw, val, s=8, alpha=0.5, color="darkorange")
    axes[1].set_xlabel("|yaw|"), axes[1].set_ylabel("expected valence")
    axes[1].set_title("Q2: expression cue vs. pose cue", fontsize=10)
    ids = [s["track_id"] for s in students]
    x = np.arange(len(ids))
    axes[2].bar(x - 0.2, [s["engagement_expr_only"] for s in students], 0.4, label="expression only")
    axes[2].bar(x + 0.2, [s["engagement_fused"] for s in students], 0.4, label="expression + pose")
    axes[2].set_xticks(x), axes[2].set_xticklabels(["#%d" % i for i in ids], fontsize=8)
    axes[2].set_ylabel("engagement proxy"), axes[2].legend(fontsize=8)
    axes[2].set_title("Q3: per-student proxy before/after fusion", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(common.RESULTS, "head_pose.png")
    fig.savefig(out)
    print("写出", out)


if __name__ == "__main__":
    main()
