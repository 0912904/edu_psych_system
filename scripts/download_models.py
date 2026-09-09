# -*- coding: utf-8 -*-
"""下载系统所需的两个 ONNX 模型到 models/（不随仓库分发权重）。

用法：python3 scripts/download_models.py [目标目录]
两个地址都走 media.githubusercontent.com，因为这两个文件在源仓库里是 git-lfs
指针，用 raw.githubusercontent.com 只会下到 131 字节的文本指针，加载时报
"Failed to parse ONNX model"。
"""
import os
import sys
import urllib.request

MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        200000),
    "emotion-ferplus-8.onnx": (
        "https://media.githubusercontent.com/media/onnx/models/main/validated/"
        "vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
        30000000),
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(dst_dir=None):
    dst_dir = dst_dir or os.path.join(ROOT, "models")
    os.makedirs(dst_dir, exist_ok=True)
    ok = True
    for name, (url, min_size) in MODELS.items():
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst) and os.path.getsize(dst) >= min_size:
            print("已存在，跳过：%s" % dst)
            continue
        print("下载 %s ..." % name, flush=True)
        try:
            with urllib.request.urlopen(url, timeout=180) as r, \
                    open(dst, "wb") as f:
                f.write(r.read())
        except Exception as e:                       # noqa: BLE001
            ok = False
            print("下载失败：%s\n  原因：%s\n  可手动下载后放到 %s"
                  % (url, e, dst_dir))
            continue
        size = os.path.getsize(dst)
        if size < min_size:
            ok = False
            print("文件异常（仅 %d 字节，疑似 git-lfs 指针）：%s" % (size, dst))
        else:
            print("完成：%s（%.1f MB）" % (dst, size / 1048576.0))
    if ok:
        print("\n模型就绪。启动系统前设置："
              "\n  Linux/macOS: export EDU_MODELS_DIR=%s"
              "\n  Windows:     set EDU_MODELS_DIR=%s" % (dst_dir, dst_dir))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
