# -*- coding: utf-8 -*-
"""实验十一（一）：DAiSEE 逐秒特征抽取 —— 用与线上系统完全一致的感知链路处理公开数据集。

DAiSEE（Gupta et al., 2016）：112 名学生在线学习时的网络摄像头视频，共 9068 段 10 秒片段，
每段由多名标注者给出 4 个情感状态的 0~3 级标签：Boredom / Engagement / Confusion / Frustration。
它是目前唯一带真值的公开"学习者投入度"视频数据集，正好补上课堂视频"没有真值、不能报准确率"的缺口。

本脚本只负责感知：对每段视频按系统默认的 1 fps 抽 10 帧，跑 YuNet 检测（取最大人脸）+ FERplus，
把每帧的 检测框 / 置信度 / 5 个关键点 / 8 类概率 原样存下来（experiments/data/daisee_frames.npz，
不入库），聚合与评估放在 daisee_eval.py。这样保证：
    1. 评估用的表情、姿态、尺寸、熵四类线索全部来自系统已有的输出，没有为数据集单独加模型；
    2. 特征抽取只需跑一次，后续所有实验（零样本代理、消融、分类）都在缓存上做，可完全复现。
数据集路径由环境变量 EPS_DAISEE_DIR 指定（默认 experiments/data/DAiSEE），目录结构见其 README.txt。
"""
import csv
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import common

DAISEE = os.environ.get("EPS_DAISEE_DIR", os.path.join(common.DATA, "DAiSEE"))
SPLITS = ["Train", "Validation", "Test"]
LABELS = ["Boredom", "Engagement", "Confusion", "Frustration"]
SAMPLE_FPS = 1            # 与 core/classroom.py 的 analyze_video 默认值一致
WORKERS = int(os.environ.get("EPS_WORKERS", max(1, (os.cpu_count() or 4) // 2)))
OUT = os.path.join(common.DATA, "daisee_frames.npz")

_state = {}


def _init():
    cv2, det_path, fer_net = common.load_models()
    cv2.setNumThreads(1)
    _state["cv2"] = cv2
    _state["det"] = cv2.FaceDetectorYN.create(det_path, "", (320, 320))
    _state["fer"] = fer_net


def read_labels():
    """返回 {clip_id: [b, e, c, f]}，clip_id 不含扩展名。"""
    out = {}
    with open(os.path.join(DAISEE, "Labels", "AllLabels.csv"), encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    for r in rows[1:]:
        if len(r) < 5 or not r[0].strip():
            continue
        out[os.path.splitext(r[0].strip())[0]] = [int(x) for x in r[1:5]]
    return out


def list_clips():
    """返回 [(split, clip_id, path)]。"""
    clips = []
    for sp in SPLITS:
        for p in sorted(glob.glob(os.path.join(DAISEE, "DataSet", sp, "*", "*", "*.*"))):
            if os.path.splitext(p)[1].lower() in (".avi", ".mp4"):
                clips.append((sp, os.path.splitext(os.path.basename(p))[0], p))
    return clips


def process_clip(args):
    """一段视频 → (10, 22) 数组：[found, x, y, w, h, score, lm×10, probs×8]，未检出的帧全 0。"""
    split, cid, path = args
    cv2, det, fer = _state["cv2"], _state["det"], _state["fer"]
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = int(round(fps / SAMPLE_FPS)) or 1
    rows = np.zeros((10, 24), dtype=np.float32)
    fidx, sec = 0, 0
    while sec < 10:
        ok = cap.grab()
        if not ok:
            break
        if fidx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                h, w = frame.shape[:2]
                det.setInputSize((w, h))
                _, faces = det.detect(frame)
                if faces is not None and len(faces):
                    f = max(faces, key=lambda a: a[2] * a[3])     # 单人数据集：取最大人脸
                    x, y, fw, fh = [int(v) for v in f[:4]]
                    x0, y0 = max(0, x), max(0, y)
                    face = frame[y0:y + fh, x0:x + fw]
                    if face.size:
                        probs = common.predict_probs(fer, common.preprocess(cv2, face))
                        rows[sec, 0] = 1
                        rows[sec, 1:5] = f[:4]
                        rows[sec, 5] = f[14]
                        rows[sec, 6:16] = f[4:14]
                        rows[sec, 16:24] = probs
            sec += 1
        fidx += 1
    cap.release()
    return split, cid, rows


def main():
    labels = read_labels()
    clips = [c for c in list_clips() if c[1] in labels]
    print("DAiSEE 目录:", DAISEE)
    print("有标签的片段:", len(clips), "| 进程数:", WORKERS, flush=True)
    limit = int(os.environ.get("EPS_LIMIT", 0))
    if limit:
        clips = clips[:limit]
    t0 = time.time()
    ids, splits, ys, feats = [], [], [], []
    with Pool(WORKERS, initializer=_init) as pool:
        for i, (sp, cid, rows) in enumerate(pool.imap_unordered(process_clip, clips, chunksize=8), 1):
            ids.append(cid); splits.append(sp); ys.append(labels[cid]); feats.append(rows)
            if i % 200 == 0 or i == len(clips):
                el = time.time() - t0
                print("  %5d/%d  %.0fs  预计剩余 %.0fs" % (i, len(clips), el, el / i * (len(clips) - i)), flush=True)
    feats = np.stack(feats)
    common.ensure_dirs()
    np.savez_compressed(OUT, clip_id=np.array(ids), split=np.array(splits),
                        y=np.array(ys, dtype=np.int64), frames=feats,
                        columns=np.array(["found", "x", "y", "w", "h", "score"]
                                         + ["lm%d" % i for i in range(10)]
                                         + ["p_" + c for c in common.FER_LABELS]))
    found = feats[:, :, 0]
    print("写出", OUT)
    print("帧级检出率 %.4f | 全 10 帧都检出的片段占比 %.4f | 耗时 %.0fs"
          % (found.mean(), (found.sum(1) == 10).mean(), time.time() - t0))


if __name__ == "__main__":
    main()
