# -*- coding: utf-8 -*-
"""实验二：人脸尺寸鲁棒性曲线 —— 为什么系统里要有 MIN_FACE_PX 阈值。

做法：把 FER2013 测试图先降采样到 s×s（模拟教室后排的小脸），再按系统的流程放大
到 64×64 送入 FERplus，观察准确率、macro-F1 与"全判成中性"的比例随 s 的变化。
这不是新方法，只是把系统里那个拍脑袋的阈值换成一条能看的曲线。
"""
import numpy as np

import common

SIZES = [12, 16, 20, 24, 28, 32, 40, 48, 64]


def sample(cv2, every=3):
    """均匀抽样测试集（tar 内按类别聚集，必须跨类抽样）。"""
    data = []
    for i, (img, cls) in enumerate(common.iter_fer2013_test(cv2)):
        if i % every == 0:
            data.append((img, cls))
    return data


def main(every=3):
    cv2, _, fer = common.load_models()
    data = sample(cv2, every)
    fer_to_2013 = {v: k for k, v in common.FER2013_TO_FER.items()}
    rows = []
    for s in SIZES:
        y_true, y_pred, neutral = [], [], 0
        for img, cls in data:
            small = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
            probs = common.predict_probs(fer, common.preprocess(cv2, small))
            k = int(np.argmax(probs))
            neutral += k == 0
            y_pred.append(fer_to_2013.get(k, common.FER_LABELS[k]))
            y_true.append(cls)
        acc, per, _, _ = common.metrics(y_true, y_pred, common.FER2013_CLASSES)
        f1 = round(float(np.mean([per[c]["f1"] for c in common.FER2013_CLASSES])), 4)
        rows.append([s, acc, f1, round(neutral / len(data), 4)])
        print("人脸 %2dpx：准确率 %.4f｜macro-F1 %.4f｜判为中性比例 %.2f%%"
              % (s, acc, f1, 100.0 * neutral / len(data)), flush=True)

    common.save_csv("size_robustness.csv",
                    ["face_px", "accuracy", "macro_f1", "neutral_pred_ratio"], rows)
    common.save_json("size_robustness.json", {
        "n_samples": len(data), "sizes": SIZES,
        "rows": [dict(zip(["face_px", "accuracy", "macro_f1",
                           "neutral_pred_ratio"], r)) for r in rows]})
    plot(rows)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    px = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.plot(px, [r[1] for r in rows], "o-", label="accuracy")
    ax.plot(px, [r[2] for r in rows], "s-", label="macro-F1")
    ax.plot(px, [r[3] for r in rows], "^--", label="predicted-neutral ratio")
    ax.axvline(28, color="crimson", ls=":", lw=1.2)
    ax.text(29, 0.16, "MIN_FACE_PX=28", color="crimson", fontsize=8)
    ax.set_xlabel("face size (px, before upscaling to 64x64)")
    ax.set_ylabel("value")
    ax.set_title("FERplus robustness vs. face resolution (FER2013 subset)",
                 fontsize=10)
    ax.grid(alpha=0.3), ax.legend(fontsize=8)
    fig.tight_layout()
    common.ensure_dirs()
    out = common.os.path.join(common.RESULTS, "size_robustness.png")
    fig.savefig(out)
    print("写出", out)


if __name__ == "__main__":
    import sys
    main(every=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
