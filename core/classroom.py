# -*- coding: utf-8 -*-
"""课堂模式：多人脸表情时间线与班级聚合。

优先复用 Multimodalv2.0-edu 的 YuNet 人脸检测 + FERplus 表情识别 ONNX 模型
（需 opencv-python）；缺环境或未上传视频时，生成带标记的模拟时间线，保证骨架
可运行、前端可联调。输出结构：
    {students: [{track_id, timeline:[{t, emotion, valence}], stats}], class_stats}
"""
import hashlib
import math
import os
import random
import shutil
import struct
import tempfile

import config

_CV_TMP = os.path.join(tempfile.gettempdir(), "eps_cv")

_FER_LABELS = ["中性", "开心", "惊讶", "伤心", "生气", "厌恶", "恐惧", "轻蔑"]
_VALENCE = {"开心": 0.9, "惊讶": 0.2, "中性": 0.0,
            "伤心": -0.8, "生气": -0.7, "厌恶": -0.6, "恐惧": -0.6, "轻蔑": -0.4}

MIN_FACE_PX = 28        # 小于此像素的人脸（教室后排）表情不可靠，只计人数不进时间线
MIN_TRACK_POINTS = 5    # 少于此点数的轨迹多为误检，不作为一名学生输出


def _ascii_path(path, cache=True):
    """OpenCV 在 Windows 上读不了含中文的路径，必要时复制到纯英文临时路径。

    返回 (可用路径, 是否为临时副本)；模型文件较大，复制后按大小缓存复用。
    """
    try:
        path.encode("ascii")
        return path, False
    except UnicodeEncodeError:
        pass
    os.makedirs(_CV_TMP, exist_ok=True)
    dst = os.path.join(_CV_TMP, hashlib.md5(path.encode("utf-8")).hexdigest()[:12]
                       + os.path.splitext(path)[1].lower())
    if not (cache and os.path.exists(dst)
            and os.path.getsize(dst) == os.path.getsize(path)):
        shutil.copyfile(path, dst)
    return dst, True


def _try_cv2_models():
    try:
        import cv2  # noqa
        det = os.path.join(config.EDU_MODELS_DIR, "face_detection_yunet_2023mar.onnx")
        fer = os.path.join(config.EDU_MODELS_DIR, "emotion-ferplus-8.onnx")
        if os.path.exists(det) and os.path.exists(fer):
            return cv2, _ascii_path(det)[0], _ascii_path(fer)[0]
    except Exception:
        pass
    return None, None, None


def analyze_video(video_path, sample_fps=1, max_seconds=120):
    """逐秒多人脸表情分析；环境不足时抛 RuntimeError 由上层降级为模拟。"""
    cv2, det_path, fer_path = _try_cv2_models()
    if cv2 is None:
        raise RuntimeError("缺少 opencv/ONNX 模型，请在 edu 环境下运行")
    detector = cv2.FaceDetectorYN.create(det_path, "", (320, 320))
    fer = cv2.dnn.readNetFromONNX(fer_path)
    video_path, is_copy = _ascii_path(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = int(round(fps / sample_fps)) or 1
    tracks, fidx, small = {}, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok or fidx / fps > max_seconds:
            break
        if fidx % step == 0:
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
            for face in (faces if faces is not None else []):
                x, y, fw, fh = [int(v) for v in face[:4]]
                cx, cy = x + fw / 2, y + fh / 2

                def _dist(k):                   # 按人脸中心距离认人（含纵向，教室有前后排）
                    return math.hypot(tracks[k]["cx"] - cx, tracks[k]["cy"] - cy)

                tid = min(tracks, key=_dist) \
                    if tracks and min(_dist(k) for k in tracks) < w * 0.08 \
                    else len(tracks) + 1
                if min(fw, fh) < MIN_FACE_PX:    # 后排小脸表情不可靠，只计数不识别
                    small += 1
                    continue
                roi = frame[max(0, y):y + fh, max(0, x):x + fw]
                if roi.size == 0:
                    continue
                g = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (64, 64))
                g = cv2.equalizeHist(g)         # 拉开对比度，小脸更容易看出表情
                fer.setInput(g.reshape(1, 1, 64, 64).astype("float32"))
                emo, val, conf = _emotion(fer.forward()[0])
                tr = tracks.setdefault(tid, {"cx": cx, "cy": cy, "timeline": []})
                tr["cx"], tr["cy"] = cx, cy
                t = round(fidx / fps, 1)
                if tr["timeline"] and tr["timeline"][-1]["t"] == t:
                    continue                    # 同一时刻同一人只保留一条，避免时间线错位
                tr["timeline"].append({"t": t, "emotion": emo,
                                       "valence": val, "conf": conf,
                                       # 归一化人脸框，前端按播放器实际尺寸缩放画框
                                       "box": [round(x / w, 4), round(y / h, 4),
                                               round(fw / w, 4), round(fh / h, 4)]})
        fidx += 1
    cap.release()
    if is_copy:                                 # 临时副本用完即删，不留存学生画面
        try:
            os.remove(video_path)
        except OSError:
            pass
    students = []                               # 点数太少的多为误检，不当成一名学生
    for tr in tracks.values():
        if len(tr["timeline"]) < MIN_TRACK_POINTS:
            continue
        students.append({"track_id": len(students) + 1, "timeline": tr["timeline"],
                         "stats": _stats(tr["timeline"])})
    stats = _class_stats(students)
    stats["small_face_samples"] = small         # 检测到但太小、无法可靠识别表情的人脸次数
    return {"students": students, "class_stats": stats, "engine": "yunet+ferplus"}


def _emotion(logits):
    """FERplus 输出转 (标签, 期望情绪价值, 置信度)。

    只取最大类别会让曲线在“中性 0”和“开心 0.9”之间跳，按概率加权的期望值更能
    体现表情的细微变化，也不容易被单帧误判带偏。
    """
    m = max(logits)
    exp = [math.exp(float(v) - m) for v in logits]
    s = sum(exp) or 1.0
    p = [v / s for v in exp]
    top = max(range(len(p)), key=lambda i: p[i])
    val = sum(p[i] * _VALENCE[_FER_LABELS[i]] for i in range(len(p)))
    return _FER_LABELS[top], round(val, 3), round(p[top], 3)


def probe_duration(video_path):
    """尽力探测视频时长（秒）：优先 opencv，退化为读 mp4/mov 的 mvhd 头，都不行返回 None。"""
    try:
        import cv2
        cap = cv2.VideoCapture(_ascii_path(video_path)[0])
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps > 0 and frames > 0:
            return round(frames / fps, 1)
    except Exception:
        pass
    return _mp4_duration(video_path)


def _mp4_duration(video_path, scan_bytes=4 << 20):
    """零依赖解析 mp4/mov 的 mvhd box 取时长（时间刻度与时长），失败返回 None。"""
    try:
        with open(video_path, "rb") as f:
            head = f.read(scan_bytes)
            i = head.find(b"mvhd")
            if i < 0:                       # moov 可能在文件尾部
                f.seek(max(0, os.path.getsize(video_path) - scan_bytes))
                head = f.read(scan_bytes)
                i = head.find(b"mvhd")
                if i < 0:
                    return None
            ver = head[i + 4]
            if ver == 1:
                scale, dur = struct.unpack(">IQ", head[i + 8 + 16:i + 8 + 28])
            else:
                scale, dur = struct.unpack(">II", head[i + 8 + 8:i + 8 + 16])
            if scale:
                return round(dur / scale, 1)
    except Exception:
        pass
    return None


def simulate(n_students=6, seconds=60):
    """模拟课堂时间线（骨架演示/无 GPU 环境联调用，前端明确标注模拟）。

    seconds 可按上传视频的真实时长传入，使模拟时间线与播放进度对齐。
    """
    seconds = max(10, min(int(seconds or 60), 600))
    rng = random.Random(42)
    drop_lo, drop_hi = seconds * 0.33, seconds * 0.58   # 注入“情绪骤降”的时间窗
    students = []
    for sid in range(1, n_students + 1):
        base = rng.uniform(-0.5, 0.6)
        timeline, v = [], base
        for t in range(seconds):
            v = max(-1, min(1, v + rng.uniform(-0.15, 0.15)
                            + (-0.4 if sid == n_students and drop_lo < t < drop_hi else 0)))
            emo = ("开心" if v > 0.35 else "伤心" if v < -0.5
                   else "生气" if v < -0.3 and rng.random() < 0.3 else "中性")
            timeline.append({"t": t, "emotion": emo,
                             "valence": round(_VALENCE[emo] * 0.4 + v * 0.6, 3)})
        students.append({"track_id": sid, "timeline": timeline,
                         "stats": _stats(timeline)})
    return {"students": students, "class_stats": _class_stats(students),
            "engine": "simulated"}


def _smooth(vals, win=3):
    """3 点滑动平均：预警看的是趋势，不该被单帧误判带出一次“骤降”。"""
    if len(vals) < win:
        return list(vals)
    half = win // 2
    return [round(sum(vals[max(0, i - half):i + half + 1])
                  / len(vals[max(0, i - half):i + half + 1]), 3)
            for i in range(len(vals))]


def _stats(timeline):
    vals = _smooth([x["valence"] for x in timeline]) or [0.0]
    neg = sum(1 for x in timeline if x["valence"] < -0.2) / max(1, len(timeline))
    vol = (sum(abs(vals[i] - vals[i - 1]) for i in range(1, len(vals)))
           / max(1, len(vals) - 1)) if len(vals) > 1 else 0.0
    return {"mean_valence": round(sum(vals) / len(vals), 3),
            "neg_ratio": round(neg, 3), "volatility": round(vol, 3),
            "max_drop": round(max((vals[i - 1] - vals[i]
                                   for i in range(1, len(vals))), default=0.0), 3)}


def _class_stats(students):
    if not students:
        return {}
    mv = [s["stats"]["mean_valence"] for s in students]
    return {"n_students": len(students),
            "class_valence": round(sum(mv) / len(mv), 3),
            "neg_student_ratio": round(sum(1 for s in students
                                           if s["stats"]["neg_ratio"] > config.ALERT_NEG_RATIO)
                                       / len(students), 3)}
