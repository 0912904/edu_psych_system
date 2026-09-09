---
name: test-edumind-web
description: Run the relocated EduMind web app locally and verify real classroom inference rather than simulated fallback.
---

# EduMind runtime testing

## Devin Secrets Needed
None for the local app. Public internet is needed for model downloads and the ECharts CDN. External interview attention-fusion assets are separate and are not bundled.

## Setup
- From repo root, install the web inference dependencies with `python3 -m pip install opencv-python numpy`; experiment-only packages are not required for this UI flow.
- Run `python3 scripts/download_models.py` from repo root. Confirm actual ONNX files, not Git LFS pointers; rerunning should skip existing files.
- Start `python3 -u app.py` with working directory `<repo>/edu_psych_system`. Default URL is `http://localhost:5100/`.
- Default models are `<repo>/models`; `EDU_MODELS_DIR` can override the directory. SQLite and uploaded videos live under the nested app directory and must remain ignored.

## Primary UI path and evidence
- Click `课堂视频`, choose a video, then `上传并分析`.
- Use a browser-playable H.264 MP4 with at least five sampled seconds of detectable faces; analysis samples approximately once a second and discards shorter tracks.
- Validate fixture detections at YuNet's default 0.9 confidence threshold. Small/downscaled public samples may fall below it even when visibly recognizable; larger face samples may be needed.
- Verify engine `yunet+ferplus`, actual student count/timeline points, face overlays, visible chart lines, and playback/chart synchronization. HTTP 200 alone is insufficient: failures may return simulated data, or no detected tracks.
- Distinguish the zero-track/no-alert state from successful analysis; absence of alerts does not prove any students were analyzed.
- Use `访谈分析` for text analysis, and `分析历史` → `刷新` plus page reload to verify persistence. Without external interview assets expect `lexicon-baseline`, not attention-fusion.
- Rule-based advice is available; LLM advice is only a reserved interface.
- Capture browser console and network evidence. A missing favicon may produce a 404; distinguish it from JS exceptions and API failures.

## Fixture handling
Keep generated media and screenshots outside tracked source directories. For synthetic videos built from public face samples, label them clearly as fixtures and do not claim emotional-classification accuracy from this runtime check.
