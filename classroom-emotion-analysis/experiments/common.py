# -*- coding: utf-8 -*-
"""实验公用部分：模型加载、预处理、FER2013 测试集读取、指标计算。

与线上系统 core/classroom.py 保持完全一致的预处理（灰度 → resize 64×64 →
直方图均衡 → 直接送入 FERplus），否则实验结论无法说明系统本身。
"""
import io
import json
import math
import os
import tarfile
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
MODELS = os.environ.get(
    "EPS_MODELS_DIR",
    os.path.normpath(os.path.join(HERE, "..", "models")))

# FERplus 输出顺序（与系统内一致）
FER_LABELS = ["neutral", "happiness", "surprise", "sadness",
              "anger", "disgust", "fear", "contempt"]
FER_LABELS_CN = ["中性", "开心", "惊讶", "伤心", "生气", "厌恶", "恐惧", "轻蔑"]
VALENCE = {"中性": 0.0, "开心": 0.9, "惊讶": 0.2, "伤心": -0.8,
           "生气": -0.7, "厌恶": -0.6, "恐惧": -0.6, "轻蔑": -0.4}

# FER2013（clip-benchmark/wds_fer2013）的 7 类，按 classnames.txt 顺序
FER2013_CLASSES = ["angry", "disgusted", "fearful", "happy",
                   "neutral", "sad", "surprised"]
# FER2013 类别 → FERplus 类别索引（contempt 在 FER2013 中没有对应类别）
FER2013_TO_FER = {"angry": 4, "disgusted": 5, "fearful": 6, "happy": 1,
                  "neutral": 0, "sad": 3, "surprised": 2}

WDS_URL = ("https://huggingface.co/datasets/clip-benchmark/wds_fer2013/"
           "resolve/main/test/%d.tar")


def ensure_dirs():
    for d in (RESULTS, DATA):
        os.makedirs(d, exist_ok=True)


def load_models():
    """返回 (cv2, 人脸检测器工厂, FERplus 网络)。"""
    import cv2
    det = os.path.join(MODELS, "face_detection_yunet_2023mar.onnx")
    fer = os.path.join(MODELS, "emotion-ferplus-8.onnx")
    for p in (det, fer):
        if not os.path.exists(p):
            raise SystemExit("缺少模型：%s（设置 EPS_MODELS_DIR 指向模型目录）" % p)
    return cv2, det, cv2.dnn.readNetFromONNX(fer)


def softmax(logits):
    m = float(max(logits))
    e = [math.exp(float(v) - m) for v in logits]
    s = sum(e) or 1.0
    return [v / s for v in e]


def preprocess(cv2, gray_or_bgr, size=64, equalize=True):
    """与系统一致的人脸预处理：灰度 → resize → 直方图均衡。"""
    img = gray_or_bgr
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (size, size))
    if equalize:
        img = cv2.equalizeHist(img)
    return img.reshape(1, 1, size, size).astype("float32")


def predict_probs(fer_net, blob):
    fer_net.setInput(blob)
    return softmax(fer_net.forward()[0])


def valence_expect(probs):
    """按各类概率加权的期望情绪价值（系统当前用法）。"""
    return sum(probs[i] * VALENCE[FER_LABELS_CN[i]] for i in range(len(probs)))


def valence_argmax(probs):
    """只取最大类别的情绪价值（系统第一版用法，用于消融对比）。"""
    return VALENCE[FER_LABELS_CN[int(np.argmax(probs))]]


def download_fer2013_test(shards=4):
    """下载 FER2013 测试集（webdataset 分片）到 data/，返回本地 tar 路径列表。"""
    ensure_dirs()
    paths = []
    for i in range(shards):
        dst = os.path.join(DATA, "fer2013_test_%d.tar" % i)
        if not os.path.exists(dst) or os.path.getsize(dst) < 1024:
            print("下载 FER2013 测试集分片 %d ..." % i, flush=True)
            with urllib.request.urlopen(WDS_URL % i, timeout=180) as r, \
                    open(dst, "wb") as f:
                f.write(r.read())
        paths.append(dst)
    return paths


def iter_fer2013_test(cv2, limit=None):
    """逐个产出 (灰度图 ndarray, 真实类别名)。"""
    n = 0
    for tar_path in download_fer2013_test():
        with tarfile.open(tar_path) as tf:
            members = {}
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                key, ext = os.path.splitext(m.name)
                members.setdefault(key, {})[ext.lower()] = m
            for key in sorted(members):
                grp = members[key]
                cls_m = grp.get(".cls")
                img_m = next((grp[e] for e in (".webp", ".jpg", ".jpeg", ".png")
                              if e in grp), None)
                if cls_m is None or img_m is None:
                    continue
                idx = int(tf.extractfile(cls_m).read().decode().strip())
                buf = np.frombuffer(tf.extractfile(img_m).read(), dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                yield img, FER2013_CLASSES[idx]
                n += 1
                if limit and n >= limit:
                    return


def metrics(y_true, y_pred, classes):
    """返回 (总体准确率, 每类 precision/recall/f1/support, 混淆矩阵)。

    y_true / y_pred 为类别名；y_pred 允许出现 classes 之外的类别（如 contempt），
    这类预测一律记为错误，并在混淆矩阵中单独成列。
    """
    pred_classes = classes + [c for c in dict.fromkeys(y_pred) if c not in classes]
    ti = {c: i for i, c in enumerate(classes)}
    pi = {c: i for i, c in enumerate(pred_classes)}
    cm = np.zeros((len(classes), len(pred_classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[ti[t], pi[p]] += 1
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(1, len(y_true))
    per = {}
    for c in classes:
        tp = cm[ti[c], pi[c]]
        fp = cm[:, pi[c]].sum() - tp
        fn = cm[ti[c], :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                  "f1": round(f1, 4), "support": int(cm[ti[c], :].sum())}
    return round(acc, 4), per, cm, pred_classes


def save_json(name, obj):
    ensure_dirs()
    path = os.path.join(RESULTS, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print("写出", path)
    return path


def save_csv(name, header, rows):
    ensure_dirs()
    path = os.path.join(RESULTS, name)
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(",".join(str(h) for h in header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print("写出", path)
    return path


def zh_font():
    """给 matplotlib 找一个能显示中文的字体，找不到则返回 None（图内改用英文）。"""
    from matplotlib import font_manager
    for name in ("Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
                 "Source Han Sans CN", "SimHei", "Microsoft YaHei"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return None
