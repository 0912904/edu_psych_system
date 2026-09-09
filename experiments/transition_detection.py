# -*- coding: utf-8 -*-
"""实验九：情绪转折点检测 —— 用变点检测替代"单步骤降"阈值规则。

系统现在的预警看 max_drop（相邻两点平滑值之差 > 0.8）。它只能抓"一帧之内的断崖"，
对"几秒内逐渐滑落"的真实转折无能为力，而且对单帧误判很敏感。本实验比较三种方法：
  A. 骤降规则（现有）：相邻差 > 阈值 τ 即报一次转折
  B. CUSUM：累计偏离均值超过 h 报转折并复位
  C. 二分段变点检测（3 点中值滤波 + binary segmentation，SSE 代价 + BIC 惩罚 + 最小效应量 0.15）
评价分两步：
  1) 合成序列（有真值）：在真实轨迹的噪声水平下生成分段常值 + 高斯噪声 + 偶发单帧尖峰的序列，
     已知变点位置，算三种方法在 ±2 s 容差内的精确率 / 召回率 / F1；
  2) 真实视频：把三种方法应用到每名学生的情绪序列上，输出"第 t 秒情绪由 a 变为 b"的转折表。
注意：合成实验里的 F1 是"检测算法在给定噪声模型下的表现"，不代表课堂上情绪转折的识别准确率。
"""
import json
import math
import os

import numpy as np

import common

CACHE = os.path.join(common.RESULTS, "video_faces.json")
MIN_FACE_PX, MIN_TRACK_POINTS = 28, 5
TOL = 2          # 容差（采样点，1 点 = 1 秒）
N_SYN = 400      # 合成序列条数
SEQ_LEN = 60


# ---------------- 三种检测器 ----------------
def detect_drop(vals, tau=0.8, smooth=True):
    v = smooth3(vals) if smooth else list(vals)
    return [i for i in range(1, len(v)) if abs(v[i] - v[i - 1]) > tau]


def detect_cusum(vals, h=4.0, k=0.5):
    """双边 CUSUM（经典参数 h=4σ, k=0.5σ）；越过 h 报转折并以当前值为新基准。"""
    sigma = robust_sigma(vals)
    v = median3(vals)
    if sigma <= 1e-6:
        return []
    sp, sn, ref, out = 0.0, 0.0, v[0], []
    for i in range(1, len(v)):
        d = (v[i] - ref) / sigma
        sp = max(0.0, sp + d - k)
        sn = max(0.0, sn - d - k)
        if sp > h or sn > h:
            out.append(i)
            sp, sn, ref = 0.0, 0.0, v[i]
    return out


def median3(vals):
    """3 点中值滤波：抹掉单帧尖峰但保留真实的台阶变化（均值平滑做不到这点）。"""
    v = list(vals)
    if len(v) < 3:
        return np.asarray(v, dtype=float)
    return np.array([v[0]] + [sorted(v[i - 1:i + 2])[1] for i in range(1, len(v) - 1)] + [v[-1]])


def detect_binseg(vals, min_seg=3, pen=None, min_delta=0.15):
    """二分段：递归找 SSE 下降最多的切分点，直到收益小于惩罚（BIC：σ² log n）；
    切分前先做 3 点中值滤波，且两侧均值差小于 min_delta（约半个情绪档位）的切分不算转折。"""
    sigma2 = max(robust_sigma(vals) ** 2, 1e-6)
    v = median3(vals)
    n = len(v)
    if n < 2 * min_seg:
        return []
    pen = pen if pen is not None else 2.0 * sigma2 * math.log(n)
    cps = []

    def sse(seg):
        return float(((seg - seg.mean()) ** 2).sum()) if len(seg) else 0.0

    def rec(lo, hi):
        if hi - lo < 2 * min_seg:
            return
        base = sse(v[lo:hi])
        best, best_gain = None, 0.0
        for c in range(lo + min_seg, hi - min_seg + 1):
            gain = base - sse(v[lo:c]) - sse(v[c:hi])
            if gain > best_gain:
                best, best_gain = c, gain
        if best is not None and best_gain > pen:
            if abs(v[lo:best].mean() - v[best:hi].mean()) >= min_delta:
                cps.append(best)
            rec(lo, best)
            rec(best, hi)

    rec(0, n)
    return sorted(cps)


def smooth3(vals):
    v = list(vals)
    if len(v) < 3:
        return v
    return [sum(v[max(0, i - 1):i + 2]) / len(v[max(0, i - 1):i + 2]) for i in range(len(v))]


def robust_sigma(v):
    d = np.diff(np.asarray(v, dtype=float))
    if len(d) == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2)) or float(np.std(d) / math.sqrt(2))


METHODS = [
    ("骤降规则 τ=0.8（现有）", lambda v: detect_drop(v, 0.8)),
    ("骤降规则 τ=0.3", lambda v: detect_drop(v, 0.3)),
    ("CUSUM", detect_cusum),
    ("二分段变点(BIC)", detect_binseg),
]


# ---------------- 合成评测 ----------------
def synth(rng, sigma, n=SEQ_LEN):
    """分段常值 + 噪声 + 单帧尖峰；返回 (序列, 真值变点)。"""
    k = rng.randint(0, 3)                                    # 0~2 个真实转折
    cps = sorted(rng.choice(np.arange(8, n - 8), size=k, replace=False).tolist())
    levels = [rng.uniform(-0.6, 0.6)]
    for _ in cps:
        levels.append(np.clip(levels[-1] + rng.choice([-1, 1]) * rng.uniform(0.3, 0.7), -0.9, 0.9))
    v = np.zeros(n)
    bounds = [0] + cps + [n]
    for lv, a, b in zip(levels, bounds[:-1], bounds[1:]):
        v[a:b] = lv
    v += rng.normal(0, sigma, n)
    for _ in range(rng.randint(0, 3)):                       # 单帧误判尖峰
        i = rng.randint(0, n)
        v[i] += rng.choice([-1, 1]) * rng.uniform(0.5, 0.9)
    return np.clip(v, -1, 1), cps


def prf(pred, true, tol=TOL):
    tp = 0
    used = set()
    for p in pred:
        j = next((t for t in true if abs(t - p) <= tol and t not in used), None)
        if j is not None:
            tp += 1
            used.add(j)
    prec = tp / len(pred) if pred else (1.0 if not true else 0.0)
    rec = tp / len(true) if true else (1.0 if not pred else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


# ---------------- 真实轨迹 ----------------
def real_tracks(frames):
    tracks = {}
    for rec in frames:
        w = rec["w"]
        for f in rec["faces"]:
            x, y, fw, fh = f["box"]
            cx, cy = x + fw / 2.0, y + fh / 2.0
            tid = None
            if tracks:
                k = min(tracks, key=lambda k: math.hypot(tracks[k]["cx"] - cx, tracks[k]["cy"] - cy))
                if math.hypot(tracks[k]["cx"] - cx, tracks[k]["cy"] - cy) < w * 0.08:
                    tid = k
            if tid is None:
                tid = len(tracks) + 1
            tr = tracks.setdefault(tid, {"cx": cx, "cy": cy, "t": [], "v": []})
            tr["cx"], tr["cy"] = cx, cy
            if f["size"] < MIN_FACE_PX or (tr["t"] and tr["t"][-1] == rec["t"]):
                continue
            tr["t"].append(rec["t"])
            tr["v"].append(common.valence_expect(np.array(f["probs"])))
    return {k: v for k, v in tracks.items() if len(v["v"]) >= MIN_TRACK_POINTS}


def describe(v):
    return "积极" if v > 0.15 else "消极" if v < -0.15 else "中性"


def main():
    cache = None
    sigma = 0.08
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            cache = json.load(f)
        tracks = real_tracks(cache["frames"])
        sig = [robust_sigma(tr["v"]) for tr in tracks.values() if len(tr["v"]) >= 8]
        if sig:
            sigma = float(np.median(sig))
        print("真实轨迹 %d 条，逐点噪声 σ 中位数 %.3f（合成实验按此取噪声）" % (len(tracks), sigma))

    # 1) 合成评测
    rng = np.random.RandomState(0)
    data = [synth(rng, sigma) for _ in range(N_SYN)]
    syn_rows, syn_out = [], []
    for name, fn in METHODS:
        P, R, F = [], [], []
        for v, cps in data:
            p, r, f1 = prf(fn(v), cps)
            P.append(p), R.append(r), F.append(f1)
        syn_out.append({"method": name, "precision": float(np.mean(P)),
                        "recall": float(np.mean(R)), "f1": float(np.mean(F))})
        syn_rows.append([name, round(np.mean(P), 4), round(np.mean(R), 4), round(np.mean(F), 4)])
        print("%-22s P=%.3f R=%.3f F1=%.3f" % (name, np.mean(P), np.mean(R), np.mean(F)), flush=True)

    # 噪声水平扫描（只对三种主方法）
    sweep = []
    for s in (0.04, 0.08, 0.12, 0.16, 0.20):
        rng = np.random.RandomState(1)
        d = [synth(rng, s) for _ in range(200)]
        row = {"sigma": s}
        for name, fn in METHODS:
            row[name] = float(np.mean([prf(fn(v), c)[2] for v, c in d]))
        sweep.append(row)

    # 2) 真实视频转折表
    real = []
    if cache:
        for tid, tr in tracks.items():
            v = np.array(tr["v"])
            cps = detect_binseg(v)
            events = []
            bounds = [0] + cps + [len(v)]
            for a, b, c in zip(bounds[:-1], bounds[1:-1], bounds[2:]):
                before, after = float(v[a:b].mean()), float(v[b:c].mean())
                events.append({"t": tr["t"][b], "from": round(before, 3), "to": round(after, 3),
                               "desc": "%s→%s" % (describe(before), describe(after))})
            real.append({"track_id": tid, "n": len(v),
                         "drop_rule_events": len(detect_drop(v, 0.8)),
                         "cusum_events": len(detect_cusum(v)),
                         "binseg_events": events})
            if events:
                print("  学生#%d：" % tid + "；".join("第 %.0f 秒 %s(%.2f→%.2f)" % (e["t"], e["desc"], e["from"], e["to"]) for e in events))
        print("真实视频：骤降规则报 %d 次，CUSUM %d 次，二分段 %d 次"
              % (sum(r["drop_rule_events"] for r in real), sum(r["cusum_events"] for r in real),
                 sum(len(r["binseg_events"]) for r in real)))

    common.save_csv("transition_detection.csv", ["方法", "精确率", "召回率", "F1"], syn_rows)
    common.save_json("transition_detection.json", {
        "synthetic": {"n_sequences": N_SYN, "seq_len": SEQ_LEN, "sigma": sigma,
                      "tolerance_s": TOL, "methods": syn_out, "sigma_sweep": sweep},
        "real_video": {"video": cache["meta"] if cache else None, "tracks": real}})
    plot(sweep, data[:1][0] if data else None)


def plot(sweep, example):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    en = {"骤降规则 τ=0.8（现有）": "drop rule τ=0.8 (current)", "骤降规则 τ=0.3": "drop rule τ=0.3",
          "CUSUM": "CUSUM", "二分段变点(BIC)": "binary segmentation (BIC)"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    for name, _ in METHODS:
        axes[0].plot([r["sigma"] for r in sweep], [r[name] for r in sweep], "o-", label=en[name])
    axes[0].set_xlabel("per-step noise σ"), axes[0].set_ylabel("F1 (±2 s)")
    axes[0].set_title("Change-point detection on synthetic valence sequences", fontsize=10)
    axes[0].legend(fontsize=8), axes[0].grid(alpha=0.3)
    if example is not None:
        v, cps = example
        axes[1].plot(v, color="gray", lw=1, label="sequence")
        for c in cps:
            axes[1].axvline(c, color="green", ls="--", lw=1.2)
        for c in detect_binseg(v):
            axes[1].axvline(c, color="crimson", ls=":", lw=1.5)
        for c in detect_drop(v, 0.8):
            axes[1].plot(c, v[c], "kx", ms=7)
        axes[1].set_title("example: truth (green) / binseg (red) / drop rule (x)", fontsize=10)
        axes[1].set_xlabel("t (s)"), axes[1].set_ylabel("valence"), axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(common.RESULTS, "transition_detection.png")
    fig.savefig(out)
    print("写出", out)


if __name__ == "__main__":
    main()
