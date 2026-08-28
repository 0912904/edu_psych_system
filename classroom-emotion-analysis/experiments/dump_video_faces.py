# -*- coding: utf-8 -*-
"""把真实课堂视频（test.mp4）的检测结果与 FERplus 概率一次性缓存下来。

后面的消融实验都在这份缓存上重放，避免每换一个阈值就重跑一遍检测 + 识别；
缓存里只有人脸框、尺寸和 8 类概率，不含任何画面像素，可以安全地放进仓库。
"""
import os
import sys
import time

import numpy as np

import common

VIDEO = os.environ.get("EPS_VIDEO", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "edu_psych_system",
    "uploads", "test.mp4"))
OUT = os.path.join(common.RESULTS, "video_faces.json")


def main(video=VIDEO, sample_fps=1, max_seconds=120):
    cv2, det_path, fer = common.load_models()
    detector = cv2.FaceDetectorYN.create(det_path, "", (320, 320))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = int(round(fps / sample_fps)) or 1
    frames, fidx, t_det, t_fer = [], 0, 0.0, 0.0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok or fidx / fps > max_seconds:
            break
        if fidx % step == 0:
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            a = time.time()
            _, faces = detector.detect(frame)
            t_det += time.time() - a
            rec = {"t": round(fidx / fps, 1), "w": w, "h": h, "faces": []}
            for face in (faces if faces is not None else []):
                x, y, fw, fh = [int(v) for v in face[:4]]
                roi = frame[max(0, y):y + fh, max(0, x):x + fw]
                if roi.size == 0:
                    continue
                a = time.time()
                probs = common.predict_probs(
                    fer, common.preprocess(cv2, roi))
                t_fer += time.time() - a
                rec["faces"].append({
                    "box": [x, y, fw, fh],
                    "size": int(min(fw, fh)),
                    "score": round(float(face[14]), 4),
                    "probs": [round(float(p), 5) for p in probs]})
            frames.append(rec)
        fidx += 1
    cap.release()
    n_faces = sum(len(f["faces"]) for f in frames)
    meta = {"video": os.path.basename(video), "fps": round(fps, 2),
            "sampled_frames": len(frames), "faces": n_faces,
            "duration_s": round(fidx / fps, 1),
            "wall_s": round(time.time() - t0, 2),
            "detect_s": round(t_det, 2), "ferplus_s": round(t_fer, 2),
            "labels": common.FER_LABELS}
    print(meta)
    common.save_json("video_faces.json", {"meta": meta, "frames": frames})
    return meta


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else VIDEO)
