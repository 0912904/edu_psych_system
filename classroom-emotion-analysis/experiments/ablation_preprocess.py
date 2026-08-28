# -*- coding: utf-8 -*-
"""实验五：预处理消融 —— 直方图均衡与插值方式到底有没有用。

系统里对人脸裁剪做的是"灰度 → resize 64×64 → equalizeHist"。这里在 FER2013
测试集子集上把 equalizeHist 开/关、三种插值方式组合跑一遍，避免"因为教程这么
写"而留下的处理步骤。

一个实测到的约束：emotion-ferplus-8.onnx 的全连接层维度写死为 64×64（换成
48×48 会直接报 MatMul 维度不匹配 K_A=2304 vs K_B=4096），所以输入尺寸不是可
调超参，只能固定 64×64——这也是"小脸必须先放大"这个问题绕不开的原因。
"""
import numpy as np

import common
from size_robustness import sample

INTERP = ["INTER_LINEAR", "INTER_AREA", "INTER_CUBIC"]


def main(every=3):
    cv2, _, fer = common.load_models()
    data = sample(cv2, every)
    fer_to_2013 = {v: k for k, v in common.FER2013_TO_FER.items()}
    rows, out = [], []
    for interp in INTERP:
        for eq in (True, False):
            flag = getattr(cv2, interp)
            y_true, y_pred = [], []
            for img, cls in data:
                g = cv2.resize(img, (64, 64), interpolation=flag)
                if eq:
                    g = cv2.equalizeHist(g)
                blob = g.reshape(1, 1, 64, 64).astype("float32")
                probs = common.predict_probs(fer, blob)
                y_pred.append(fer_to_2013.get(int(np.argmax(probs)), "contempt"))
                y_true.append(cls)
            acc, per, _, _ = common.metrics(y_true, y_pred,
                                            common.FER2013_CLASSES)
            f1 = round(float(np.mean([per[c]["f1"]
                                      for c in common.FER2013_CLASSES])), 4)
            rows.append([interp, "on" if eq else "off", acc, f1])
            out.append({"interpolation": interp, "equalize_hist": eq,
                        "accuracy": acc, "macro_f1": f1})
            print("插值 %-12s｜直方图均衡 %-3s：准确率 %.4f｜macro-F1 %.4f"
                  % (interp, "on" if eq else "off", acc, f1), flush=True)
    common.save_csv("ablation_preprocess.csv",
                    ["interpolation", "equalize_hist", "accuracy", "macro_f1"],
                    rows)
    common.save_json("ablation_preprocess.json",
                     {"n_samples": len(data), "input_size_fixed": 64,
                      "note": "FERplus 全连接层维度写死 64x64，输入尺寸不可调",
                      "rows": out})


if __name__ == "__main__":
    import sys
    main(every=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
