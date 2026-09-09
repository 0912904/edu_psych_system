# -*- coding: utf-8 -*-
"""实验六：直方图均衡在"小脸"条件下是否仍然无用。

实验五在原始 48×48 图上发现 equalizeHist 反而略微掉点。但系统面对的是教室后排
被放大的小脸，对比度情况不同，所以这里把图先降采样到 16/20/24/28/40 px 再放大，
分别比较开/关直方图均衡，看这一步在真正的使用条件下是帮忙还是添乱。
"""
import numpy as np

import common
from size_robustness import sample

SIZES = [16, 20, 24, 28, 40]


def main(every=4):
    cv2, _, fer = common.load_models()
    data = sample(cv2, every)
    fer_to_2013 = {v: k for k, v in common.FER2013_TO_FER.items()}
    rows, out = [], []
    for s in SIZES:
        line = {}
        for eq in (True, False):
            y_true, y_pred = [], []
            for img, cls in data:
                small = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
                probs = common.predict_probs(
                    fer, common.preprocess(cv2, small, equalize=eq))
                y_pred.append(fer_to_2013.get(int(np.argmax(probs)), "contempt"))
                y_true.append(cls)
            acc, per, _, _ = common.metrics(y_true, y_pred,
                                            common.FER2013_CLASSES)
            f1 = round(float(np.mean([per[c]["f1"]
                                      for c in common.FER2013_CLASSES])), 4)
            line["on" if eq else "off"] = (acc, f1)
        rows.append([s, line["on"][0], line["on"][1],
                     line["off"][0], line["off"][1],
                     round(line["on"][0] - line["off"][0], 4),
                     round(line["on"][1] - line["off"][1], 4)])
        out.append({"face_px": s, "acc_eq_on": line["on"][0],
                    "f1_eq_on": line["on"][1], "acc_eq_off": line["off"][0],
                    "f1_eq_off": line["off"][1]})
        print("%2dpx｜均衡 on acc %.4f f1 %.4f｜off acc %.4f f1 %.4f｜"
              "差值 acc %+.4f f1 %+.4f"
              % (s, line["on"][0], line["on"][1], line["off"][0],
                 line["off"][1], rows[-1][5], rows[-1][6]), flush=True)
    common.save_csv("ablation_equalize_small.csv",
                    ["face_px", "acc_eq_on", "f1_eq_on", "acc_eq_off",
                     "f1_eq_off", "acc_delta", "f1_delta"], rows)
    common.save_json("ablation_equalize_small.json",
                     {"n_samples": len(data), "rows": out})


if __name__ == "__main__":
    import sys
    main(every=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
