# -*- coding: utf-8 -*-
"""实验四：各阶段耗时剖析 —— 一段课堂视频的时间花在哪里。

逐秒抽帧的完整链路拆成解码、检测、裁剪+预处理、FERplus 前向四段计时，
用于回答"为什么 36 秒视频要跑二十多秒"以及"该优化哪一段"。
"""
import os
import sys
import time

import common

VIDEO = os.environ.get("EPS_VIDEO", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "edu_psych_system",
    "uploads", "test.mp4"))


def main(video=VIDEO, sample_fps=1, max_seconds=120):
    cv2, det_path, fer = common.load_models()
    detector = cv2.FaceDetectorYN.create(det_path, "", (320, 320))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = int(round(fps / sample_fps)) or 1
    t = {"decode": 0.0, "detect": 0.0, "preprocess": 0.0, "ferplus": 0.0}
    fidx, n_frames, n_faces = 0, 0, 0
    t0 = time.time()
    while True:
        a = time.time()
        ok, frame = cap.read()
        t["decode"] += time.time() - a
        if not ok or fidx / fps > max_seconds:
            break
        if fidx % step == 0:
            n_frames += 1
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            a = time.time()
            _, faces = detector.detect(frame)
            t["detect"] += time.time() - a
            for face in (faces if faces is not None else []):
                x, y, fw, fh = [int(v) for v in face[:4]]
                a = time.time()
                roi = frame[max(0, y):y + fh, max(0, x):x + fw]
                if roi.size == 0:
                    t["preprocess"] += time.time() - a
                    continue
                blob = common.preprocess(cv2, roi)
                t["preprocess"] += time.time() - a
                a = time.time()
                common.predict_probs(fer, blob)
                t["ferplus"] += time.time() - a
                n_faces += 1
        fidx += 1
    cap.release()
    wall = time.time() - t0
    total = sum(t.values()) or 1.0
    res = {"video": os.path.basename(video),
           "resolution": "%dx%d" % (w, h),
           "duration_s": round(fidx / fps, 1),
           "sampled_frames": n_frames, "faces_processed": n_faces,
           "wall_s": round(wall, 2),
           "realtime_factor": round(wall / max(0.1, fidx / fps), 3),
           "sampled_fps": round(n_frames / wall, 2),
           "stages": {k: {"seconds": round(v, 3),
                          "share": round(v / total, 4),
                          "ms_per_call": round(
                              1000.0 * v / max(1, n_frames if k in
                                               ("decode", "detect") else n_faces), 3)}
                      for k, v in t.items()}}
    for k, v in res["stages"].items():
        print("%-11s %6.2fs  占比 %5.1f%%  单次 %7.2fms"
              % (k, v["seconds"], 100 * v["share"], v["ms_per_call"]))
    print("整段 %.1fs 视频用 %.2fs（实时倍率 %.2fx），抽样帧率 %.2f fps"
          % (res["duration_s"], wall, res["realtime_factor"], res["sampled_fps"]))
    common.save_json("profile.json", res)
    common.save_csv("profile.csv", ["stage", "seconds", "share", "ms_per_call"],
                    [[k, v["seconds"], v["share"], v["ms_per_call"]]
                     for k, v in res["stages"].items()])
    plot(res)
    return res


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = res["stages"]
    keys = ["decode", "detect", "preprocess", "ferplus"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
    bars = ax.bar(keys, [st[k]["seconds"] for k in keys],
                  color=["#8fa8c8", "#4f7dbd", "#9cc79b", "#e08a5d"])
    for b, k in zip(bars, keys):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                "%.1f%%" % (100 * st[k]["share"]), ha="center", va="bottom",
                fontsize=8)
    ax.set_ylabel("seconds")
    ax.set_title("Stage cost on a %ss classroom video (%s, 1 fps sampling)"
                 % (res["duration_s"], res["resolution"]), fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    common.ensure_dirs()
    out = os.path.join(common.RESULTS, "profile.png")
    fig.savefig(out)
    print("写出", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else VIDEO)
