# -*- coding: utf-8 -*-
"""实验七：置信度加权聚合框架 —— 把系统里零散的"过滤规则"写成一个带权重的聚合公式，并做消融。

系统当前用三条硬规则处理不可靠的逐帧判断：小脸直接丢弃（<28px）、碎片轨迹直接丢弃
（<5 点）、平滑后再预警。本实验把它们统一成对每次观测 i 的一个连续权重：

    w_i = w_size(s_i) · w_conf(p_i) · w_track(n_i)
      w_size  = clip((s - 16) / (48 - 16), 0, 1)      人脸边长 s：16px 以下不可信，48px 以上饱和
                                                      （拐点来自实验二的尺寸鲁棒性曲线）
      w_conf  = 1 - H(p) / log(8)                     8 类概率的归一化熵：越犹豫权重越低
      w_track = clip(n / 5, 0, 1)                     轨迹长度 n：越短越可能是误检

个人/班级情绪值 = Σ w_i v_i / Σ w_i；再对每名学生做加权 bootstrap 给出情绪均值的 95% 置信区间，
预警只在"区间整体越过阈值"时触发（可信预警）。

评价方式（没有心理标签，所以不评"准确率"，评的是聚合结果对噪声的稳定性）：
  在真实视频缓存上随机把 r% 的观测替换成随机概率向量（模拟误识别），重复 200 次，
  看不同聚合方式下班级情绪均值的偏移 |Δ| 与预警翻转率。越小越稳。
注意：这些数字说明的是"聚合规则对噪声的敏感度"，不是模型在课堂上的准确率。
"""
import json
import math
import os
import sys

import numpy as np

import common

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "edu_psych_system"))
import config          # noqa: E402

CACHE = os.path.join(common.RESULTS, "video_faces.json")
RNG = np.random.RandomState(0)
N_TRIALS = 200
NOISE_RATES = [0.05, 0.10, 0.20]
N_CLASSES = 8


def load_cache():
    if not os.path.exists(CACHE):
        raise SystemExit("先运行 dump_video_faces.py 生成 %s" % CACHE)
    with open(CACHE, encoding="utf-8") as f:
        return json.load(f)


def entropy(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0)
    return float(-(p * np.log(p)).sum())


# ---------------- 三个权重分量 ----------------
def w_size(size, lo=16, hi=48):
    return float(np.clip((size - lo) / float(hi - lo), 0.0, 1.0))


def w_conf(probs):
    return 1.0 - entropy(probs) / math.log(N_CLASSES)


def w_track(n, full=5):
    return float(np.clip(n / float(full), 0.0, 1.0))


def build_tracks(frames):
    """与线上一致的中心距离认人；这里不做任何过滤，过滤交给权重。"""
    tracks = {}
    for rec in frames:
        w = rec["w"]
        for f in rec["faces"]:
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
            tr["obs"].append({"t": rec["t"], "size": f["size"],
                              "probs": np.array(f["probs"], dtype=float)})
    return tracks


# ---------------- 聚合方案 ----------------
def weights_for(track, scheme):
    """返回该轨迹每次观测的权重数组。scheme 见 SCHEMES。"""
    n = len(track["obs"])
    ws = []
    for o in track["obs"]:
        if scheme["kind"] == "hard":            # 系统当前配置：硬阈值 0/1
            w = float(o["size"] >= 28 and n >= 5)
        elif scheme["kind"] == "uniform":       # 第一版：全部等权
            w = 1.0
        else:                                   # 连续权重，可逐项关闭
            w = 1.0
            if scheme.get("size", True):
                w *= w_size(o["size"])
            if scheme.get("conf", True):
                w *= w_conf(o["probs"])
            if scheme.get("track", True):
                w *= w_track(n)
        ws.append(w)
    return np.array(ws)


SCHEMES = [
    ("第一版：等权（无任何过滤）", {"kind": "uniform"}),
    ("系统当前：硬阈值 28px + 5点", {"kind": "hard"}),
    ("置信度加权（完整）", {"kind": "soft"}),
    ("  - 去掉尺寸权重", {"kind": "soft", "size": False}),
    ("  - 去掉熵权重", {"kind": "soft", "conf": False}),
    ("  - 去掉轨迹权重", {"kind": "soft", "track": False}),
    ("  仅尺寸权重", {"kind": "soft", "conf": False, "track": False}),
    ("  仅熵权重", {"kind": "soft", "size": False, "track": False}),
]


def aggregate(tracks, scheme, n_boot=300, rng=RNG):
    """返回 {class_valence, students, alerts, ci_width_mean, coverage}。"""
    per_student = []
    total_w, total_n = 0.0, 0
    for tr in tracks.values():
        ws = weights_for(tr, scheme)
        total_w += ws.sum()
        total_n += len(ws)
        if ws.sum() <= 1e-6:
            continue
        vals = np.array([common.valence_expect(o["probs"]) for o in tr["obs"]])
        mean = float((ws * vals).sum() / ws.sum())
        # 加权 bootstrap：按权重重采样观测，得到均值的 95% 区间
        p = ws / ws.sum()
        idx = rng.choice(len(vals), size=(n_boot, len(vals)), p=p)
        boots = vals[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        neg_ratio = float((ws * (vals < -0.2)).sum() / ws.sum())
        per_student.append({"mean": mean, "ci": (float(lo), float(hi)),
                            "neg_ratio": neg_ratio, "eff_n": float(ws.sum())})
    if not per_student:
        return {"class_valence": 0.0, "students": 0, "alerts": 0,
                "alerts_trusted": 0, "ci_width_mean": 0.0, "coverage": 0.0}
    ew = np.array([s["eff_n"] for s in per_student])
    mv = np.array([s["mean"] for s in per_student])
    return {
        "class_valence": float((ew * mv).sum() / ew.sum()),
        "students": len(per_student),
        # 普通预警：均值/消极占比越过阈值
        "alerts": sum(1 for s in per_student if s["neg_ratio"] > config.ALERT_NEG_RATIO),
        # 可信预警：置信区间上界也在 -0.2 以下（整段区间都偏消极）
        "alerts_trusted": sum(1 for s in per_student if s["ci"][1] < -0.2),
        "ci_width_mean": float(np.mean([s["ci"][1] - s["ci"][0] for s in per_student])),
        "coverage": float(total_w / max(1, total_n)),
    }


def inject_noise(tracks, rate, rng):
    """把 rate 比例的观测替换成随机概率向量（Dirichlet），模拟误识别。"""
    out = {}
    for k, tr in tracks.items():
        obs = []
        for o in tr["obs"]:
            o2 = dict(o)
            if rng.rand() < rate:
                o2["probs"] = rng.dirichlet(np.ones(N_CLASSES) * 0.5)
            obs.append(o2)
        out[k] = {"cx": tr["cx"], "cy": tr["cy"], "obs": obs}
    return out


def main():
    cache = load_cache()
    tracks = build_tracks(cache["frames"])
    n_obs = sum(len(t["obs"]) for t in tracks.values())
    print("轨迹 %d 条，观测 %d 次" % (len(tracks), n_obs))

    rows, out = [], []
    for name, scheme in SCHEMES:
        base = aggregate(tracks, scheme)
        robust = {}
        for r in NOISE_RATES:
            rng = np.random.RandomState(1)
            dev, flips = [], 0
            for _ in range(N_TRIALS):
                noisy = inject_noise(tracks, r, rng)
                res = aggregate(noisy, scheme, n_boot=60, rng=rng)
                dev.append(abs(res["class_valence"] - base["class_valence"]))
                flips += res["alerts"] != base["alerts"]
            robust[r] = {"mean_abs_dev": float(np.mean(dev)),
                         "p95_abs_dev": float(np.percentile(dev, 95)),
                         "alert_flip_rate": flips / N_TRIALS}
        out.append({"scheme": name, "config": scheme, "base": base, "robustness": robust})
        rows.append([name, base["students"], round(base["class_valence"], 3),
                     round(base["coverage"], 3), base["alerts"], base["alerts_trusted"],
                     round(base["ci_width_mean"], 3)]
                    + [round(robust[r]["mean_abs_dev"], 4) for r in NOISE_RATES]
                    + [round(robust[r]["alert_flip_rate"], 3) for r in NOISE_RATES])
        print("%-28s 学生%2d 均值%6.3f 覆盖%.2f 预警%d/可信%d CI宽%.3f | 噪声20%%偏移%.4f 翻转%.2f"
              % (name, base["students"], base["class_valence"], base["coverage"],
                 base["alerts"], base["alerts_trusted"], base["ci_width_mean"],
                 robust[0.2]["mean_abs_dev"], robust[0.2]["alert_flip_rate"]), flush=True)

    header = (["方案", "学生数", "班级情绪均值", "有效覆盖率", "预警数", "可信预警数", "平均CI宽度"]
              + ["噪声%d%%均值偏移" % int(r * 100) for r in NOISE_RATES]
              + ["噪声%d%%预警翻转率" % int(r * 100) for r in NOISE_RATES])
    common.save_csv("confidence_aggregation.csv", header, rows)
    common.save_json("confidence_aggregation.json",
                     {"video": cache["meta"], "n_tracks": len(tracks), "n_obs": n_obs,
                      "n_trials": N_TRIALS, "noise_rates": NOISE_RATES,
                      "weight_def": {"w_size": "clip((s-16)/32,0,1)",
                                     "w_conf": "1-H(p)/log8",
                                     "w_track": "clip(n/5,0,1)"},
                      "schemes": out})
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["uniform", "hard 28px+5pt", "soft (full)", "soft -size",
             "soft -entropy", "soft -track", "size only", "entropy only"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    x = np.arange(len(out))
    for i, r in enumerate(NOISE_RATES):
        axes[0].bar(x + (i - 1) * 0.27, [o["robustness"][r]["mean_abs_dev"] for o in out],
                    width=0.27, label="noise %d%%" % int(r * 100))
    axes[0].set_xticks(x), axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("|Δ class valence| (mean of 200 trials)")
    axes[0].set_title("Sensitivity of class-level valence to injected noise", fontsize=10)
    axes[0].legend(fontsize=8), axes[0].grid(alpha=0.3, axis="y")
    for i, r in enumerate(NOISE_RATES):
        axes[1].bar(x + (i - 1) * 0.27, [o["robustness"][r]["alert_flip_rate"] for o in out],
                    width=0.27, label="noise %d%%" % int(r * 100))
    axes[1].set_xticks(x), axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("alert flip rate")
    axes[1].set_title("Alert stability under injected noise", fontsize=10)
    axes[1].legend(fontsize=8), axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = os.path.join(common.RESULTS, "confidence_aggregation.png")
    fig.savefig(out_path)
    print("写出", out_path)


if __name__ == "__main__":
    main()
