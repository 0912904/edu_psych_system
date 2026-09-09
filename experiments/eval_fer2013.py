# -*- coding: utf-8 -*-
"""实验一：FERplus ONNX 在 FER2013 测试集上的 baseline 评估。

目的：给系统里"表情识别这一环到底有多准"一个可查证的数字，而不是只说"能跑"。
说明：FERplus 有 contempt（轻蔑）类，FER2013 没有，因此该类预测一律计为错误，
并单独统计其出现频次；这是模型标签体系与评测集不一致带来的固有损失。
"""
import numpy as np

import common


def main(limit=None, equalize=True):
    cv2, _, fer = common.load_models()
    y_true, y_pred, confs = [], [], []
    for img, cls in common.iter_fer2013_test(cv2, limit=limit):
        probs = common.predict_probs(fer, common.preprocess(cv2, img, equalize=equalize))
        y_pred.append(common.FER_LABELS[int(np.argmax(probs))])
        y_true.append(common.FER2013_CLASSES[
            common.FER2013_CLASSES.index(cls)])
        confs.append(float(max(probs)))

    # 把 FERplus 标签名换成 FER2013 的类别名，便于直接比对
    fer_to_2013 = {v: k for k, v in common.FER2013_TO_FER.items()}
    y_pred = [fer_to_2013.get(common.FER_LABELS.index(p), p) for p in y_pred]

    acc, per, cm, pred_classes = common.metrics(
        y_true, y_pred, common.FER2013_CLASSES)
    contempt = sum(1 for p in y_pred if p == "contempt")
    macro_f1 = round(float(np.mean([per[c]["f1"] for c in common.FER2013_CLASSES])), 4)

    print("样本数 %d｜总体准确率 %.4f｜macro-F1 %.4f｜预测为 contempt 的比例 %.2f%%"
          % (len(y_true), acc, macro_f1, 100.0 * contempt / max(1, len(y_true))))
    for c in common.FER2013_CLASSES:
        d = per[c]
        print("  %-10s P=%.3f R=%.3f F1=%.3f n=%d"
              % (c, d["precision"], d["recall"], d["f1"], d["support"]))

    common.save_json("fer2013_baseline.json", {
        "n_samples": len(y_true), "accuracy": acc, "macro_f1": macro_f1,
        "contempt_pred_ratio": round(contempt / max(1, len(y_true)), 4),
        "mean_confidence": round(float(np.mean(confs)), 4),
        "equalize_hist": equalize, "per_class": per,
        "confusion_matrix": {"rows_true": common.FER2013_CLASSES,
                             "cols_pred": pred_classes, "matrix": cm.tolist()}})
    common.save_csv("fer2013_baseline.csv",
                    ["class", "precision", "recall", "f1", "support"],
                    [[c, per[c]["precision"], per[c]["recall"],
                      per[c]["f1"], per[c]["support"]]
                     for c in common.FER2013_CLASSES]
                    + [["overall(accuracy)", acc, "", macro_f1, len(y_true)]])
    plot_cm(cm, common.FER2013_CLASSES, pred_classes, acc)
    return acc


def plot_cm(cm, rows, cols, acc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    norm = cm / np.maximum(1, cm.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=150)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.set_xlabel("predicted"), ax.set_ylabel("ground truth")
    ax.set_title("FERplus on FER2013 test (row-normalized), acc=%.3f" % acc,
                 fontsize=10)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if cm[i, j]:
                ax.text(j, i, "%.2f" % norm[i, j], ha="center", va="center",
                        fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    common.ensure_dirs()
    out = common.os.path.join(common.RESULTS, "fer2013_confusion_matrix.png")
    fig.savefig(out)
    print("写出", out)


if __name__ == "__main__":
    import sys
    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
