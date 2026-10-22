# Test fixtures

## `obb_sample.jpg`

A real aerial photograph used by the real-model integration tests in
`tests/test_ultralytics_obb_source.py`. It detects 166 ships and 3 harbours with
`yolov8n-obb` at `conf=0.5, imgsz=1024`, which makes it useful for exercising
anchor-class selection: multiple classes and a wide confidence spread, exactly
what the GUI's "pick a different anchor" interaction needs.

Committed (194 KB) because a synthetic substitute does not work: `yolov8n-obb` is
trained on DOTA aerial imagery and detects **nothing** in generated shapes —
verified against axis-aligned blocks, a cross, a narrow ellipse, and four rotated
bars at several scales and contrasts.

## Weights are NOT committed

The integration tests need `yolov8n-obb.pt` (6.5 MB). They **skip** when it is
absent, so the suite stays runnable without it. To enable them:

```bash
python -c "import urllib.request; urllib.request.urlretrieve(
  'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-obb.pt',
  '/tmp/yolov8n-obb.pt')"
```

Put it at `/tmp/yolov8n-obb.pt` (the path the tests check) or update `_WEIGHTS`
in the test module.

## Why these tests exist at all

They are slower than the stub-based ones and need a model, but they caught a
defect no stub could: real detector output is float32, and over 166 genuine boxes
the orthogonality residual had a median of 1.2e-6 and a maximum of 4.5e-6. The
`1e-6` tolerance that looked reasonable on paper rejected **57% of real
detections**. Coordinates from a stub are exact, so the bug was invisible.
