# -*- coding: utf-8 -*-
"""教育心理情绪分析系统 —— 零依赖 HTTP 服务（http.server + SQLite）。

REST 接口：
  GET  /api/ping                    健康检查
  POST /api/interview               访谈分析 {turns:[句子,...]}
  GET  /api/classroom/demo          课堂模拟数据分析（无环境依赖）
  POST /api/classroom/video         课堂视频分析；请求体为视频原始字节，文件名由
                                    ?name= 或 X-File-Name 传入；兼容旧的 JSON
                                    请求体 {file_b64, name}
  GET  /api/history                 分析历史
  GET  /api/alerts                  预警记录
静态资源：/ → web/index.html；/uploads/<文件名> → 已上传视频（支持 Range 边下边播）
"""
import base64
import json
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from core import interview, classroom, alerts, advice, db

VIDEO_TYPES = {".mp4": "video/mp4", ".webm": "video/webm",
               ".mov": "video/quicktime", ".mkv": "video/x-matroska"}
CHUNK = 1 << 20


def _interview(body):
    turns = [t.strip() for t in body.get("turns", []) if t.strip()]
    if not turns:
        raise ValueError("turns 不能为空")
    r = interview.analyze_interview(turns)
    al = alerts.interview_alert(r["summary"])
    tips = advice.interview_advice(al)
    db.save_session("interview", r.get("engine"), r["summary"],
                    [al] if al["level"] != "正常" else [])
    return {"result": r, "alert": al, "advice": tips}


def _classroom_demo():
    r = classroom.simulate()
    al = alerts.student_alerts(r["students"])
    tips = advice.classroom_advice(r["class_stats"], al)
    db.save_session("classroom", r["engine"], r["class_stats"], al)
    return {"result": r, "alerts": al, "advice": tips}


def _safe_video_name(name):
    """只取文件名本体（防目录穿越），并限定视频扩展名。"""
    name = urllib.parse.unquote(name or "").replace("\\", "/")
    name = os.path.basename(name.split("/")[-1]).strip()
    ext = os.path.splitext(name)[1].lower()
    if ext not in VIDEO_TYPES:
        raise ValueError("仅支持 mp4 / webm / mov / mkv 格式的视频")
    return name


def _analyze_saved_video(path, name):
    """分析已落盘的视频；缺 opencv/ONNX 时降级为模拟数据，但视频仍可播放。"""
    duration = classroom.probe_duration(path)
    notice = None
    try:
        r = classroom.analyze_video(path)
    except Exception as e:                      # 真实链路失败（缺模型/解码失败等）也不阻断演示
        r = classroom.simulate(seconds=int(duration or 60))
        notice = "%s；当前结果为模拟数据，仅供演示，不代表对该视频的真实分析。" % e
    al = alerts.student_alerts(r["students"])
    tips = advice.classroom_advice(r["class_stats"], al)
    db.save_session("classroom", r["engine"], r["class_stats"], al)
    out = {"result": r, "alerts": al, "advice": tips,
           "video": {"url": "/uploads/" + urllib.parse.quote(name), "name": name,
                     "size": os.path.getsize(path), "duration_sec": duration}}
    if notice:
        out["notice"] = notice
    return out


def _classroom_video_b64(body):
    """兼容旧接口：JSON 请求体中的 base64 视频。"""
    name = _safe_video_name(body.get("name", "up.mp4"))
    path = os.path.join(config.UPLOAD_DIR, name)
    with open(path, "wb") as f:
        f.write(base64.b64decode(body["file_b64"]))
    return _analyze_saved_video(path, name)


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200, ctype="application/json", extra=None):
        data = (json.dumps(obj, ensure_ascii=False).encode()
                if ctype == "application/json" else obj)
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8"
                                                  if ctype.startswith("text")
                                                  or ctype == "application/json" else ""))
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    # ---- 已上传视频：支持 Range 边下边播 --------------------------------
    def _send_video(self, name):
        try:
            name = _safe_video_name(name)
        except ValueError:
            return self._send({"error": "not found"}, 404)
        path = os.path.join(config.UPLOAD_DIR, name)
        if not os.path.isfile(path):
            return self._send({"error": "not found"}, 404)
        size = os.path.getsize(path)
        ctype = VIDEO_TYPES[os.path.splitext(name)[1].lower()]
        rng = self.headers.get("Range", "")
        start, end = 0, size - 1
        partial = False
        if rng.startswith("bytes="):
            s, _, e = rng[6:].partition("-")
            try:
                if s:
                    start = int(s)
                    end = int(e) if e else size - 1
                else:                                   # bytes=-N（尾部 N 字节）
                    start = max(0, size - int(e))
                partial = True
            except ValueError:
                partial = False
            if partial and (start >= size or start > end):
                self.send_response(416)
                self.send_header("Content-Range", "bytes */%d" % size)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        end = min(end, size - 1)
        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                buf = f.read(min(CHUNK, left))
                if not buf:
                    break
                self.wfile.write(buf)
                left -= len(buf)

    def do_GET(self):
        try:
            path = urllib.parse.urlparse(self.path).path
            if path == "/api/ping":
                return self._send({"ok": True, "service": "edu-psych-emotion"})
            if path == "/api/classroom/demo":
                return self._send(_classroom_demo())
            if path == "/api/history":
                return self._send(db.history())
            if path == "/api/alerts":
                return self._send(db.alert_history())
            if path.startswith("/uploads/"):
                return self._send_video(path[len("/uploads/"):])
            fp = os.path.join(config.WEB_DIR,
                              "index.html" if path in ("/", "") else path.lstrip("/"))
            fp = os.path.normpath(fp)
            if fp.startswith(config.WEB_DIR) and os.path.isfile(fp):
                ext = os.path.splitext(fp)[1]
                ctype = {".html": "text/html", ".js": "text/javascript",
                         ".css": "text/css"}.get(ext, "application/octet-stream")
                with open(fp, "rb") as f:
                    return self._send(f.read(), ctype=ctype)
            return self._send({"error": "not found"}, 404)
        except Exception as e:
            return self._send({"error": str(e)}, 500)

    def _recv_video(self):
        """按 Content-Length 分块接收视频原始字节并落盘（不整体读入内存）。"""
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        name = (q.get("name", [None])[0] or self.headers.get("X-File-Name")
                or "up.mp4")
        name = _safe_video_name(name)
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0:
            raise ValueError("请求体为空，未收到视频数据")
        path = os.path.join(config.UPLOAD_DIR, name)
        with open(path, "wb") as f:
            left = n
            while left > 0:
                buf = self.rfile.read(min(CHUNK, left))
                if not buf:
                    break
                f.write(buf)
                left -= len(buf)
        return path, name

    def do_POST(self):
        try:
            path = urllib.parse.urlparse(self.path).path
            ctype = (self.headers.get("Content-Type") or "").lower()
            if path == "/api/classroom/video":
                if ctype.startswith("application/json"):
                    n = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(n).decode() or "{}")
                    return self._send(_classroom_video_b64(body))
                fp, name = self._recv_video()
                return self._send(_analyze_saved_video(fp, name))
            if path == "/api/interview":
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n).decode() or "{}")
                return self._send(_interview(body))
            return self._send({"error": "not found"}, 404)
        except ValueError as e:
            return self._send({"error": str(e)}, 400)
        except Exception as e:
            return self._send({"error": str(e)}, 500)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    print("教育心理情绪分析系统: http://localhost:%d" % config.PORT)
    ThreadingHTTPServer((config.HOST, config.PORT), Handler).serve_forever()
