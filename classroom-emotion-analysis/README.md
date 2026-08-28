# 教育心理情绪分析系统（Classroom Emotion Analysis）

面向课堂与访谈场景的情绪分析原型：上传一段课堂视频，逐秒检测人脸、识别表情，输出每位（匿名编号）学生的情绪时间线、班级整体趋势和基于规则的关注建议；视频可在页面内播放，播放游标与情绪曲线双向联动，画面上实时叠加人脸框与中文情绪描述。

后端只用 Python 标准库的 HTTP 服务 + SQLite，模型走 ONNX（OpenCV DNN，CPU 即可），没有 Web 框架和深度学习训练框架的依赖。

> 这是一个学习/实践项目。系统输出是**统计参考**，不构成任何心理诊断，请先读 [docs/ETHICS.md](docs/ETHICS.md)。

## 截图

<!-- TODO: 概览 / 课堂视频（含人脸框）/ 情绪曲线 / 分析历史 四张截图 -->

## 功能

- **课堂模式**：视频上传 → 逐秒抽帧 → YuNet 人脸检测 → 简单跟踪认人 → FERplus 表情识别 → 每人情绪时间线 + 班级聚合 + 关注建议。
- **访谈模式**：单人访谈文本/音视频的情感倾向分析与摘要（缺少模型环境时退化为词典基线，界面明确标注）。
- **页面内播放**：支持 HTTP Range，可拖动进度；播放位置与曲线游标同步；Canvas 叠加层按当前秒画人脸框和"平静专注 / 略显低落 / 明显开心"这类中文描述。
- **匿名化**：只输出"学生#N"，不展示也不留存任何身份信息；OpenCV 需要的临时视频副本用完即删。
- **诚实降级**：缺 OpenCV / 缺模型时用模拟数据保证界面可跑，但结果会在界面上醒目标注为模拟；后排过小的人脸只计数、不参与识别，界面写明"另有 N 次人脸过小未参与识别"。

## 架构与数据流

```
浏览器 (web/index.html, ECharts + Canvas 叠加层)
   │  上传视频 / 请求分析 / Range 播放
   ▼
app.py  (http.server，零框架路由)
   ├─ core/classroom.py   逐秒抽帧 → YuNet 检测 → 中心距离跟踪 → FERplus → 情绪时间线
   ├─ core/interview.py   访谈文本/音视频情感分析
   ├─ core/alerts.py      基于统计的规则预警（可解释、阈值可调）
   ├─ core/advice.py      面向教师的建议文案
   └─ core/db.py          SQLite 存分析历史
```

关键设计（每一条都有实验支撑，见下节）：

- 情绪价值取 **8 类概率加权的期望**，而不是 argmax，避免曲线只在 0 与 0.9 之间跳。
- `MIN_FACE_PX = 28`：小于 28 像素的人脸不参与表情识别（只计数）。
- `MIN_TRACK_POINTS = 5`：只出现几帧的轨迹视为误检，不算一名学生。
- 预警指标先做 3 点滑动平均，展示曲线保持原始值。

## 快速开始

```bash
git clone <this-repo> && cd classroom-emotion-analysis
pip install -r requirements.txt
python3 scripts/download_models.py          # 下载 YuNet + FERplus 到 models/
python3 edu_psych_system/app.py             # 默认 http://127.0.0.1:5100
```

模型放在别处时用环境变量指定：`export EDU_MODELS_DIR=/path/to/models`。
视频建议 30–60 秒、720p 上下、固定机位、正脸可见（代码里单次分析上限 120 秒）。

## 实验与评测

`experiments/` 下是六个可复现脚本，结果（CSV/JSON/PNG）在 `experiments/results/`，完整结论、表格与边界说明见 **[experiments/RESULTS.md](experiments/RESULTS.md)**。

| 实验 | 脚本 | 回答的问题 | 结果文件 |
|---|---|---|---|
| FER2013 baseline | `eval_fer2013.py` | 表情识别这一环到底多准，错在哪 | `fer2013_baseline.{csv,json}`、`fer2013_confusion_matrix.png` |
| 人脸尺寸鲁棒性 | `size_robustness.py` | 小脸阈值该定在哪 | `size_robustness.{csv,json,png}` |
| 系统设计消融 | `dump_video_faces.py` → `ablation.py` | 四个设计选择各自改变了什么输出 | `ablation.{csv,json}` |
| 预处理消融 | `ablation_preprocess.py` | 直方图均衡与插值方式有没有用 | `ablation_preprocess.{csv,json}` |
| 均衡×小脸交互 | `ablation_equalize_small.py` | 上一条的结论在小脸上是否反转 | `ablation_equalize_small.{csv,json}` |
| 阶段耗时剖析 | `profile_pipeline.py` | 时间花在哪，该优化什么 | `profile.{csv,json,png}` |

数据集脚本会自动从 Hugging Face 拉取 FER2013 官方测试集分片（缓存在 `experiments/data/`，不入库）。消融实验需要一段本地课堂视频，仓库不提供（见下）。

## 已知局限

- 表情识别在**小脸、侧脸、遮挡**下明显退化，且整体偏向"中性"；教室后排基本无法可靠识别。
- 公开表情数据集（正脸、裁剪好、表演式表情）与真实课堂之间存在明显域差异，公开集上的指标不能代表课堂表现。
- 跟踪只用人脸中心距离，学生大幅走动或前后排遮挡时会串号；尚未做 IoU 匹配 / 卡尔曼滤波。
- 预警是**阈值规则**，不是学习出来的模型，阈值来自实验而非人工标注的心理标签。
- 单线程 CPU，逐秒抽帧约 5 倍速于实时，适合课后离线分析，暂不支持实时反馈。

## 伦理与隐私

见 [docs/ETHICS.md](docs/ETHICS.md)。要点：输出仅供教师参考、不做诊断；学生匿名编号；不留存人脸画面；真实课堂使用需知情同意与校方授权；**本仓库不包含任何真人视频、分析数据库或模型权重**。

## 目录结构

```
edu_psych_system/     系统源码（app.py / config.py / core / web）
experiments/          实验脚本、结果与 RESULTS.md
scripts/              模型下载脚本
docs/ETHICS.md        伦理与隐私边界
```

## License

MIT，见 [LICENSE](LICENSE)。YuNet 与 FERplus 模型各自遵循其上游许可。
