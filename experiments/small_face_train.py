# -*- coding: utf-8 -*-
"""实验十：小脸鲁棒训练 —— 针对实验二发现的"28px 以下准确率骤降"做闭环。

实验二只是量出了 FERplus 在小脸上的退化曲线，本实验验证一个可检验的假设：
    H：训练时把清晰人脸随机降采样再放大（模拟后排小脸），模型在小脸测试子集上的退化会明显变缓。
做法：在 FER2013 训练集（28,709 张）上从零训练同一个小型 CNN，两种数据增广：
    A. 常规增广（随机平移 + 水平翻转）
    B. 常规增广 + 多尺度退化（p=0.5 随机缩到 12~40px 再放大回 48px）
两组各跑 2 个随机种子，在测试集上按人脸边长 12/16/20/24/28/32/48px 评估准确率、macro-F1 与
"判为中性"比例，并与实验二的 FERplus 曲线并列（不同模型/标签体系，只作退化形态参考）。
预处理与线上系统一致：灰度 → 缩放 → 直方图均衡。
需要 GPU 环境（PyTorch）。
"""
import json
import os
import time

import numpy as np

import common

SIZES = [12, 16, 20, 24, 28, 32, 48]
IMG = 48
EPOCHS = int(os.environ.get("EPS_EPOCHS", 20))
SEEDS = [0, 1]
CLASSES = common.FER2013_CLASSES


def load_split(cv2, split):
    xs, ys = [], []
    cache = os.path.join(common.DATA, "fer2013_%s.npz" % split)
    if os.path.exists(cache):
        d = np.load(cache)
        return d["x"], d["y"]
    for img, cls in common.iter_fer2013(cv2, split):
        if img.shape != (IMG, IMG):
            img = cv2.resize(img, (IMG, IMG))
        xs.append(img)
        ys.append(CLASSES.index(cls))
    x, y = np.stack(xs).astype(np.uint8), np.array(ys, dtype=np.int64)
    np.savez_compressed(cache, x=x, y=y)
    return x, y


def degrade(cv2, img, s):
    """清晰图 → s×s 小脸 → 放大回 IMG（与实验二一致）。"""
    small = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (IMG, IMG))


def build_model(torch, n_classes=7):
    nn = torch.nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.MaxPool2d(2))
    return nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 256),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4),
                         nn.Linear(256, n_classes))


class Augment:
    def __init__(self, cv2, rng, multiscale):
        self.cv2, self.rng, self.multiscale = cv2, rng, multiscale

    def __call__(self, img):
        cv2, rng = self.cv2, self.rng
        if self.multiscale and rng.rand() < 0.5:
            img = degrade(cv2, img, int(rng.randint(12, 41)))
        if rng.rand() < 0.5:
            img = img[:, ::-1]
        dx, dy = rng.randint(-4, 5), rng.randint(-4, 5)
        img = np.roll(np.roll(img, dy, 0), dx, 1)
        return cv2.equalizeHist(np.ascontiguousarray(img))


def to_tensor(torch, batch):
    x = torch.from_numpy(np.stack(batch).astype(np.float32) / 255.0).unsqueeze(1)
    return (x - 0.5) / 0.5


def train_one(torch, cv2, x_tr, y_tr, multiscale, seed, device):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    aug = Augment(cv2, rng, multiscale)
    model = build_model(torch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, epochs=EPOCHS,
                                                steps_per_epoch=(len(x_tr) + 127) // 128)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    n = len(x_tr)
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(n)
        tot, correct, t0 = 0.0, 0, time.time()
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb = to_tensor(torch, [aug(x_tr[j]) for j in idx]).to(device)
            yb = torch.from_numpy(y_tr[idx]).to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            correct += int((out.argmax(1) == yb).sum())
        print("    [%s seed=%d] ep %2d loss %.3f train-acc %.3f (%.0fs)"
              % ("多尺度" if multiscale else "常规", seed, ep + 1, tot / n, correct / n,
                 time.time() - t0), flush=True)
    return model


def evaluate(torch, cv2, model, x_te, y_te, device):
    model.eval()
    rows = {}
    with torch.no_grad():
        for s in SIZES:
            preds = []
            for i in range(0, len(x_te), 512):
                imgs = [cv2.equalizeHist(degrade(cv2, im, s) if s < IMG else im) for im in x_te[i:i + 512]]
                out = model(to_tensor(torch, imgs).to(device))
                preds.extend(out.argmax(1).cpu().tolist())
            preds = np.array(preds)
            acc, per, _, _ = common.metrics([CLASSES[i] for i in y_te],
                                            [CLASSES[i] for i in preds], CLASSES)
            f1 = float(np.mean([per[c]["f1"] for c in CLASSES]))
            rows[s] = {"accuracy": acc, "macro_f1": round(f1, 4),
                       "neutral_pred_ratio": round(float((preds == CLASSES.index("neutral")).mean()), 4)}
    return rows


def main():
    import cv2
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| epochs:", EPOCHS)
    x_tr, y_tr = load_split(cv2, "train")
    x_te, y_te = load_split(cv2, "test")
    print("train %d / test %d" % (len(x_tr), len(x_te)))

    results = {"常规增广": [], "常规+多尺度退化": []}
    for name, ms in (("常规增广", False), ("常规+多尺度退化", True)):
        for seed in SEEDS:
            model = train_one(torch, cv2, x_tr, y_tr, ms, seed, device)
            r = evaluate(torch, cv2, model, x_te, y_te, device)
            results[name].append(r)
            print("  >>> %s seed=%d: " % (name, seed)
                  + " ".join("%dpx=%.3f" % (s, r[s]["accuracy"]) for s in SIZES), flush=True)

    # 汇总（均值±标准差）
    summary, rows = {}, []
    for name, runs in results.items():
        summary[name] = {}
        for s in SIZES:
            acc = [r[s]["accuracy"] for r in runs]
            f1 = [r[s]["macro_f1"] for r in runs]
            neu = [r[s]["neutral_pred_ratio"] for r in runs]
            summary[name][s] = {"acc_mean": float(np.mean(acc)), "acc_std": float(np.std(acc)),
                                "f1_mean": float(np.mean(f1)), "f1_std": float(np.std(f1)),
                                "neutral_pred_ratio": float(np.mean(neu))}
            rows.append([name, s, round(np.mean(acc), 4), round(np.std(acc), 4),
                         round(np.mean(f1), 4), round(np.std(f1), 4), round(np.mean(neu), 4)])
    ferplus = None
    p = os.path.join(common.RESULTS, "size_robustness.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            ferplus = {r["face_px"]: r for r in json.load(f)["rows"]}

    a, b = summary["常规增广"], summary["常规+多尺度退化"]
    ret = {s: b[s]["acc_mean"] / max(1e-6, b[48]["acc_mean"]) for s in SIZES}
    ret_a = {s: a[s]["acc_mean"] / max(1e-6, a[48]["acc_mean"]) for s in SIZES}
    print("\n人脸px | 常规 acc | 多尺度 acc | Δ | 常规保留率 | 多尺度保留率")
    for s in SIZES:
        print("%5d | %.4f | %.4f | %+.4f | %.3f | %.3f"
              % (s, a[s]["acc_mean"], b[s]["acc_mean"], b[s]["acc_mean"] - a[s]["acc_mean"],
                 ret_a[s], ret[s]))

    common.save_csv("small_face_train.csv",
                    ["训练增广", "face_px", "acc_mean", "acc_std", "macro_f1_mean", "macro_f1_std",
                     "neutral_pred_ratio"], rows)
    common.save_json("small_face_train.json", {
        "epochs": EPOCHS, "seeds": SEEDS, "n_train": int(len(x_tr)), "n_test": int(len(x_te)),
        "sizes": SIZES, "summary": summary, "runs": {k: [{str(s): r[s] for s in SIZES} for r in v]
                                                     for k, v in results.items()},
        "retention_vs_48px": {"常规增广": ret_a, "常规+多尺度退化": ret},
        "ferplus_reference": ferplus})
    plot(summary, ferplus)


def plot(summary, ferplus):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    for name, lab, st in (("常规增广", "small CNN, standard aug", "o-"),
                          ("常规+多尺度退化", "small CNN, + multi-scale degradation aug", "s-")):
        m = [summary[name][s]["acc_mean"] for s in SIZES]
        sd = [summary[name][s]["acc_std"] for s in SIZES]
        axes[0].errorbar(SIZES, m, yerr=sd, fmt=st, capsize=3, label=lab)
        axes[1].plot(SIZES, [summary[name][s]["neutral_pred_ratio"] for s in SIZES], st, label=lab)
    if ferplus:
        px = [s for s in SIZES if s in ferplus]
        axes[0].plot(px, [ferplus[s]["accuracy"] for s in px], "^--", color="gray",
                     label="FERplus ONNX (exp. 2, reference)")
        axes[1].plot(px, [ferplus[s]["neutral_pred_ratio"] for s in px], "^--", color="gray",
                     label="FERplus ONNX (reference)")
    axes[0].axvline(28, color="crimson", ls=":", lw=1)
    axes[0].set_xlabel("face size (px)"), axes[0].set_ylabel("accuracy (FER2013 test)")
    axes[0].set_title("Accuracy vs. face resolution", fontsize=10)
    axes[1].set_xlabel("face size (px)"), axes[1].set_ylabel("predicted-neutral ratio")
    axes[1].set_title("Collapse-to-neutral vs. face resolution", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3), ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(common.RESULTS, "small_face_train.png")
    fig.savefig(out)
    print("写出", out)


if __name__ == "__main__":
    main()
