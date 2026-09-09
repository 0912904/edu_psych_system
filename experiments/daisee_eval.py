# -*- coding: utf-8 -*-
"""实验十一（二）：DAiSEE 评估 —— 系统信号与"投入度真值"的关系，以及基于系统信号的片段级分类。

输入是 daisee_extract.py 生成的 experiments/data/daisee_frames.npz（每段 10 帧 × 24 列）。
所有实验严格使用 DAiSEE 官方 Train / Validation / Test 划分（按人划分，无同一学生跨集泄漏）。

三层实验，对应三种强弱不同的结论：
  A. 零样本代理相关：不训练任何参数，直接把系统已有的输出（期望效价、熵、姿态、人脸尺寸、
     效价变化）和 4 个标签做 Spearman 相关。回答"系统现在算出来的东西，和真值有没有关系"。
  B. 片段级分类（固定特征 + 轻量分类器）：把每段 10 帧聚合成 31 维特征，训练类别加权的
     softmax 回归 / 小型 MLP（3 个随机种子），在 Test 上报 macro-F1 / 均衡准确率，并与
     "全预测多数类"基线对比。回答"这些线索能否学出可用的判断"。
  C. 特征组消融：分别去掉 表情 / 熵 / 姿态 / 尺寸 / 时序 五组，看 macro-F1 掉多少。
     回答"哪类线索对哪个状态最重要"，与前面头姿融合、置信度聚合两个实验互相印证。

不做的事：不改标签、不合并划分、不报未加权准确率当主指标（4 类极不均衡，Engagement 中
0/1 级只占 5.8%，多数类基线的准确率就接近 50%）。
"""
import os
import sys

import numpy as np

import common

LABELS = ["Boredom", "Engagement", "Confusion", "Frustration"]
LABELS_CN = {"Boredom": "无聊", "Engagement": "投入", "Confusion": "困惑", "Frustration": "沮丧"}
SEEDS = [0, 1, 2]
CACHE = os.path.join(common.DATA, "daisee_frames.npz")

VAL_VEC = np.array([common.VALENCE[c] for c in common.FER_LABELS_CN], dtype=np.float32)

FEATURE_GROUPS = {
    "表情": ["val_mean", "val_min", "val_max", "val_std", "p_neutral", "p_happiness", "p_surprise",
           "p_sadness", "p_anger", "p_disgust", "p_fear", "p_contempt"],
    "熵/置信": ["ent_mean", "ent_std", "pmax_mean", "low_conf_ratio"],
    "姿态": ["yaw_mean", "yaw_absmean", "yaw_std", "pitch_mean", "pitch_std", "roll_absmean",
           "pose_move"],
    "尺寸/检出": ["face_w_mean", "face_w_std", "found_ratio", "det_score_mean", "center_move"],
    "时序": ["val_slope", "val_change", "flip_count"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


# ----------------------------------------------------------------------------- 特征聚合
def pose_from_landmarks(lm):
    """lm: (N,10) 五点 [右眼, 左眼, 鼻, 右嘴角, 左嘴角]，与 head_pose.py 一致的代理量。"""
    re, le, nose = lm[:, 0:2], lm[:, 2:4], lm[:, 4:6]
    mouth = (lm[:, 6:8] + lm[:, 8:10]) / 2
    eye_mid = (re + le) / 2
    eye_dist = np.linalg.norm(le - re, axis=1) + 1e-6
    face_h = np.linalg.norm(mouth - eye_mid, axis=1) + 1e-6
    yaw = (nose[:, 0] - eye_mid[:, 0]) / eye_dist
    pitch = (nose[:, 1] - eye_mid[:, 1]) / face_h
    roll = np.degrees(np.arctan2(le[:, 1] - re[:, 1], le[:, 0] - re[:, 0]))
    return yaw, pitch, roll


def aggregate(frames):
    """frames: (T,24) -> dict 特征。T 帧中未检出的帧不参与统计。"""
    found = frames[:, 0] > 0.5
    n = int(found.sum())
    f = {k: 0.0 for k in FEATURES}
    f["found_ratio"] = n / len(frames)
    if n == 0:
        f["ent_mean"] = np.log(8)
        return f
    fr = frames[found]
    probs = fr[:, 16:24]
    val = probs @ VAL_VEC
    ent = -(probs * np.log(probs + 1e-9)).sum(1)
    pmax = probs.max(1)
    yaw, pitch, roll = pose_from_landmarks(fr[:, 6:16])
    cx = fr[:, 1] + fr[:, 3] / 2
    cy = fr[:, 2] + fr[:, 4] / 2

    f.update({
        "val_mean": val.mean(), "val_min": val.min(), "val_max": val.max(), "val_std": val.std(),
        "ent_mean": ent.mean(), "ent_std": ent.std(), "pmax_mean": pmax.mean(),
        "low_conf_ratio": float((pmax < 0.5).mean()),
        "yaw_mean": yaw.mean(), "yaw_absmean": np.abs(yaw).mean(), "yaw_std": yaw.std(),
        "pitch_mean": pitch.mean(), "pitch_std": pitch.std(), "roll_absmean": np.abs(roll).mean(),
        "pose_move": float(np.abs(np.diff(yaw)).mean() + np.abs(np.diff(pitch)).mean()) if n > 1 else 0.0,
        "face_w_mean": fr[:, 3].mean(), "face_w_std": fr[:, 3].std(),
        "det_score_mean": fr[:, 5].mean(),
        "center_move": float(np.hypot(np.diff(cx), np.diff(cy)).mean()) if n > 1 else 0.0,
        "val_slope": float(np.polyfit(np.arange(n), val, 1)[0]) if n > 1 else 0.0,
        "val_change": float(np.abs(np.diff(val)).mean()) if n > 1 else 0.0,
        "flip_count": float((np.diff(probs.argmax(1)) != 0).sum()),
    })
    for i, c in enumerate(common.FER_LABELS):
        f["p_" + c] = probs[:, i].mean()
    return {k: float(v) for k, v in f.items()}


def load():
    z = np.load(CACHE, allow_pickle=False)
    frames, y, split = z["frames"], z["y"], z["split"]
    X = np.array([[aggregate(fr)[k] for k in FEATURES] for fr in frames], dtype=np.float32)
    return X, y, split, z["clip_id"]


# ----------------------------------------------------------------------------- 统计工具
def rankdata(a):
    a = np.asarray(a, dtype=np.float64)
    order = a.argsort()
    ranks = np.empty(len(a))
    ranks[order] = np.arange(1, len(a) + 1)
    # 处理并列：取平均秩
    uniq, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / cnt)[inv]


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def spearman_ci(x, y, seed=0, n_boot=500):
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        rs.append(spearman(x[idx], y[idx]))
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def macro_f1_bacc(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    f1s, recs = [], []
    for c in range(k):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        if cm[c, :].sum() == 0:      # 测试集里没有该类，不计入 macro
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        recs.append(rec)
    return float(np.mean(f1s)), float(np.mean(recs)), cm


# ----------------------------------------------------------------------------- 分类器（PyTorch）
def train_clf(Xtr, ytr, Xva, yva, k, seed, hidden=0, epochs=300, lr=1e-2, wd=1e-3):
    """类别加权 softmax 回归（hidden=0）或单隐层 MLP；用验证集 macro-F1 早停。"""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    cnt = np.bincount(ytr, minlength=k).astype(np.float32)
    w = torch.tensor(np.where(cnt > 0, cnt.sum() / (k * np.maximum(cnt, 1)), 0.0), dtype=torch.float32)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xv = torch.tensor(Xva)
    d = Xtr.shape[1]
    if hidden:
        net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(),
                                  torch.nn.Dropout(0.3), torch.nn.Linear(hidden, k))
    else:
        net = torch.nn.Linear(d, k)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    best, best_state = -1.0, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss_fn(net(Xt[idx]), yt[idx]).backward()
            opt.step()
        if ep % 5 == 4:
            net.eval()
            with torch.no_grad():
                pv = net(Xv).argmax(1).numpy()
            f1, _, _ = macro_f1_bacc(yva, pv, k)
            if f1 > best:
                best, best_state = f1, {a: b.clone() for a, b in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()

    def predict(X):
        with torch.no_grad():
            return net(torch.tensor(X)).argmax(1).numpy()
    return predict, best


def standardize(Xtr, *others):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return [(x - mu) / sd for x in (Xtr,) + others]


def to_binary(y, label):
    """DAiSEE 文献常用二分：Engagement 低(0/1) vs 高(2/3)；其余状态 无(0) vs 有(1~3)。"""
    return (y >= 2).astype(np.int64) if label == "Engagement" else (y >= 1).astype(np.int64)


# ----------------------------------------------------------------------------- 主流程
def main():
    X, Y, split, ids = load()
    tr, va, te = split == "Train", split == "Validation", split == "Test"
    print("片段数 train/val/test =", tr.sum(), va.sum(), te.sum(), "| 特征维度", X.shape[1])
    report = {"n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
              "features": FEATURES, "label_dist": {}, "zero_shot": {}, "classification": {},
              "ablation": {}}

    for j, lab in enumerate(LABELS):
        report["label_dist"][lab] = {sp: np.bincount(Y[m, j], minlength=4).tolist()
                                    for sp, m in (("train", tr), ("val", va), ("test", te))}

    # ---- A. 零样本代理相关（全体片段，无训练）
    zs_rows = []
    probes = ["val_mean", "val_min", "ent_mean", "pmax_mean", "yaw_absmean", "pitch_mean",
              "pose_move", "face_w_mean", "center_move", "val_change", "flip_count", "found_ratio"]
    for j, lab in enumerate(LABELS):
        report["zero_shot"][lab] = {}
        for feat in probes:
            x = X[:, FEATURES.index(feat)]
            r = spearman(x, Y[:, j])
            lo, hi = spearman_ci(x, Y[:, j])
            report["zero_shot"][lab][feat] = {"rho": round(r, 4), "ci95": [round(lo, 4), round(hi, 4)]}
            zs_rows.append([lab, feat, round(r, 4), round(lo, 4), round(hi, 4)])
        print("零样本 Spearman [%s]:" % lab,
              ", ".join("%s=%.3f" % (f, report["zero_shot"][lab][f]["rho"]) for f in probes[:6]))
    common.save_csv("daisee_zero_shot.csv", ["label", "feature", "spearman_rho", "ci_lo", "ci_hi"], zs_rows)

    # ---- B. 分类（4 类 + 二分），3 种子
    Xtr_s, Xva_s, Xte_s = standardize(X[tr], X[va], X[te])
    clf_rows = []
    for j, lab in enumerate(LABELS):
        report["classification"][lab] = {}
        for task in ("4class", "binary"):
            k = 4 if task == "4class" else 2
            conv = (lambda y: y) if task == "4class" else (lambda y: to_binary(y, lab))
            ytr, yva, yte = conv(Y[tr, j]), conv(Y[va, j]), conv(Y[te, j])
            maj = np.bincount(ytr, minlength=k).argmax()
            f1_maj, bacc_maj, _ = macro_f1_bacc(yte, np.full(len(yte), maj), k)
            res = {"majority": {"macro_f1": round(f1_maj, 4), "bacc": round(bacc_maj, 4),
                                "acc": round(float((yte == maj).mean()), 4)}}
            for name, hidden in (("softmax", 0), ("mlp64", 64)):
                f1s, baccs, accs, cms = [], [], [], []
                for s in SEEDS:
                    pred, _ = train_clf(Xtr_s, ytr, Xva_s, yva, k, s, hidden=hidden)
                    p = pred(Xte_s)
                    f1, bacc, cm = macro_f1_bacc(yte, p, k)
                    f1s.append(f1); baccs.append(bacc); accs.append(float((p == yte).mean())); cms.append(cm)
                res[name] = {"macro_f1": round(float(np.mean(f1s)), 4), "macro_f1_std": round(float(np.std(f1s)), 4),
                             "bacc": round(float(np.mean(baccs)), 4), "bacc_std": round(float(np.std(baccs)), 4),
                             "acc": round(float(np.mean(accs)), 4),
                             "cm_seed0": cms[0].tolist()}
                clf_rows.append([lab, task, name, res[name]["macro_f1"], res[name]["macro_f1_std"],
                                 res[name]["bacc"], res[name]["acc"], res["majority"]["macro_f1"],
                                 res["majority"]["bacc"], res["majority"]["acc"]])
                print("分类 [%s/%s/%s] macroF1=%.3f±%.3f bacc=%.3f acc=%.3f | 多数类 F1=%.3f bacc=%.3f acc=%.3f"
                      % (lab, task, name, res[name]["macro_f1"], res[name]["macro_f1_std"], res[name]["bacc"],
                         res[name]["acc"], f1_maj, bacc_maj, res["majority"]["acc"]))
            report["classification"][lab][task] = res
    common.save_csv("daisee_classification.csv",
                    ["label", "task", "model", "macro_f1", "macro_f1_std", "bacc", "acc",
                     "majority_macro_f1", "majority_bacc", "majority_acc"], clf_rows)

    # ---- C. 特征组消融（二分任务、MLP、3 种子，报 macro-F1 相对全特征的变化）
    abl_rows = []
    for j, lab in enumerate(LABELS):
        ytr, yva, yte = (to_binary(Y[m, j], lab) for m in (tr, va, te))
        full = report["classification"][lab]["binary"]["mlp64"]["macro_f1"]
        report["ablation"][lab] = {"full": full}
        for g, feats in FEATURE_GROUPS.items():
            keep = [i for i, f in enumerate(FEATURES) if f not in feats]
            f1s = []
            for s in SEEDS:
                pred, _ = train_clf(Xtr_s[:, keep], ytr, Xva_s[:, keep], yva, 2, s, hidden=64)
                f1s.append(macro_f1_bacc(yte, pred(Xte_s[:, keep]), 2)[0])
            m = float(np.mean(f1s))
            report["ablation"][lab]["-" + g] = {"macro_f1": round(m, 4), "delta": round(m - full, 4)}
            abl_rows.append([lab, g, round(m, 4), round(m - full, 4)])
            print("消融 [%s] 去掉 %s: macroF1 %.3f (%+.3f)" % (lab, g, m, m - full))
    common.save_csv("daisee_ablation.csv", ["label", "removed_group", "macro_f1", "delta_vs_full"], abl_rows)

    common.save_json("daisee_eval.json", report)
    plot(report)


def plot(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    font = common.zh_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # 1) 零样本相关热力图
    probes = list(next(iter(report["zero_shot"].values())).keys())
    mat = np.array([[report["zero_shot"][l][f]["rho"] for f in probes] for l in LABELS])
    ax = axes[0]
    im = ax.imshow(mat, cmap="coolwarm", vmin=-0.3, vmax=0.3, aspect="auto")
    ax.set_xticks(range(len(probes))); ax.set_xticklabels(probes, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels([LABELS_CN[l] if font else l for l in LABELS])
    for i in range(4):
        for k in range(len(probes)):
            ax.text(k, i, "%.2f" % mat[i, k], ha="center", va="center", fontsize=7)
    ax.set_title("零样本 Spearman 相关（系统信号 vs 真值）" if font else "Zero-shot Spearman")
    fig.colorbar(im, ax=ax, fraction=0.03)

    # 2) 分类 vs 多数类
    ax = axes[1]
    x = np.arange(4); w = 0.26
    for i, (name, col) in enumerate((("majority", "#bbbbbb"), ("softmax", "#6baed6"), ("mlp64", "#08519c"))):
        vals = [report["classification"][l]["binary"][name]["macro_f1"] for l in LABELS]
        ax.bar(x + (i - 1) * w, vals, w, label=name, color=col)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + 0.01, "%.2f" % v, ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([LABELS_CN[l] if font else l for l in LABELS])
    ax.set_ylim(0, 1); ax.set_ylabel("macro-F1 (Test)")
    ax.set_title("二分任务：系统特征分类 vs 多数类基线" if font else "Binary classification vs majority")
    ax.legend(fontsize=8)

    # 3) 消融
    ax = axes[2]
    groups = list(FEATURE_GROUPS.keys())
    for i, l in enumerate(LABELS):
        d = [report["ablation"][l]["-" + g]["delta"] for g in groups]
        ax.plot(range(len(groups)), d, marker="o", label=LABELS_CN[l] if font else l)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups if font else ["expr", "entropy", "pose", "size", "temporal"])
    ax.set_ylabel("Δ macro-F1")
    ax.set_title("去掉一组特征后的变化（越负越重要）" if font else "Ablation: drop one group")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(common.RESULTS, "daisee_eval.png")
    fig.savefig(path, dpi=150)
    print("写出", path)


if __name__ == "__main__":
    main()
