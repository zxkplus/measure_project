"""
Demo runner — loads config (respecting config.local.yaml overrides),
runs the full API pipeline via test client, reports generated files.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

# ── 第1步：加载配置（自动合并 config.local.yaml 的覆盖） ──
from measure_api.config import Config
Config.reset()
cfg = Config.load()

# ── 第2步：从配置中获取输出路径 ──
LOG_DIR = cfg.get("log.directory", "logs")
RECORDS_DIR = cfg.get("call_records.directory", "call_records")

# 如果路径是相对的，转为绝对路径（基于项目根）
if not os.path.isabs(LOG_DIR):
    LOG_DIR = os.path.abspath(LOG_DIR)
if not os.path.isabs(RECORDS_DIR):
    RECORDS_DIR = os.path.abspath(RECORDS_DIR)

RESULTS_DIR = os.path.join(os.path.dirname(RECORDS_DIR), "results")

INS_DIR = "/media/industai/data11/data/内外直径"
REF_FILE = "20260701-143514.jpg"

# ── 第3步：启动日志系统 ──
from measure_api.logger import setup_logging, get_logger
setup_logging(cfg.get("log"))
logger = get_logger("demo")

logger.info("=" * 60)
logger.info("Demo: config.local.yaml 加载生效")
logger.info("    日志目录: %s", LOG_DIR)
logger.info("    调用记录: %s", RECORDS_DIR)
logger.info("    参考图:   %s/%s", INS_DIR, REF_FILE)
logger.info("=" * 60)

# ── 第4步：创建 Flask app + test client ──
from measure_api.server import create_app
app = create_app(cfg)
app.config["TESTING"] = True

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RECORDS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def post(client, url, data, query=""):
    resp = client.post(f"{url}{query}",
                       data=json.dumps(data),
                       content_type="application/json")
    return resp.get_json(), resp.headers.get("X-Trace-Id", "")


with app.test_client() as client:
    # 1. Create session
    body, tid = post(client, "/api/session",
                     {"project_dir": tempfile.mkdtemp(prefix="demo_")})
    sid = body["session_id"]
    logger.info("[%s] 会话创建: %s", tid, sid)

    # 2. Load ref
    ref_path = os.path.join(INS_DIR, REF_FILE)
    post(client, f"/api/session/{sid}/reference", {"image_path": ref_path})

    # 3. Template
    post(client, f"/api/session/{sid}/template", {
        "center": [1237.8, 993.6],
        "size": [1616.7, 1591.5],
        "angle_deg": 0.0,
        "preprocessor": "raw",
        "match_score_threshold": 0.5,
        "angle_range_deg": 30,
        "max_matches": 0,
    })

    # 4. Add circle_1
    c1_params = {
        "center": [814.59, 760.89],
        "radius": 484.15,
        "measure_length1": 60.0, "measure_length2": 10.0,
        "num_measures": 12, "sigma": 1.0, "threshold": 5.0,
        "transition": "negative",
        "start_phi": 0.0, "end_phi": 6.283185307179586,
    }
    body, _ = post(client, f"/api/session/{sid}/measurements",
                   {"object_type": "FitCircle", "label": "circle_1",
                    "params": c1_params})
    logger.info("  circle_1: valid=%s  quality=%s", body.get("valid"),
                body.get("quality"))

    # 5. Add circle_2
    c2_params = {
        "center": [831.08, 807.04],
        "radius": 611.24,
        "measure_length1": 120.0, "measure_length2": 10.0,
        "num_measures": 12, "sigma": 1.0, "threshold": 5.0,
        "transition": "all",
        "start_phi": 0.0, "end_phi": 6.283185307179586,
    }
    body, _ = post(client, f"/api/session/{sid}/measurements",
                   {"object_type": "FitCircle", "label": "circle_2",
                    "params": c2_params})
    logger.info("  circle_2: valid=%s  quality=%s", body.get("valid"),
                body.get("quality"))

    # 6. Test params (调参)
    body, _ = post(client, f"/api/session/{sid}/measurements/test",
                   {"object_type": "FitCircle", "label": "test_sigma_2",
                    "params": {**c1_params, "sigma": 2.0}})
    logger.info("  test(sigma=2.0): valid=%s", body.get("valid"))

    # 7. DAG
    dag = client.get(f"/api/session/{sid}/dag").get_json()
    logger.info("  DAG: %d nodes, is_valid=%s", len(dag["nodes"]), dag["is_valid"])

    # 8. Save
    body, _ = post(client, f"/api/session/{sid}/save", {})
    logger.info("  项目已保存: %s", body["saved_to"])

    # 9. Measure on one real image
    ins_img = os.path.join(INS_DIR, "20260701-143518.jpg")
    body, _ = post(client, f"/api/session/{sid}/measure",
                   {"inspection_image": ins_img})
    logger.info("  测量: status=%s  targets=%d  elapsed=%.1fms",
                body.get("status"), body.get("num_targets", 0),
                body.get("elapsed_ms", 0))

    # 10. Session status
    status = client.get(f"/api/session/{sid}").get_json()
    logger.info("  状态: phase=%s  measurements=%d  dag_valid=%s",
                status["phase"], status["num_measurements"],
                status.get("dag_valid"))

    # 11. Delete session
    client.delete(f"/api/session/{sid}")
    logger.info("  会话已删除: %s", sid)

logger.info("=" * 60)
logger.info("Demo 完成")

# ── 第5步：报告生成的文件 ──
print("\n" + "=" * 60)
print("  Demo 结果报告")
print("=" * 60)

print(f"\n日志目录: {LOG_DIR}")
for f in sorted(os.listdir(LOG_DIR)):
    fpath = os.path.join(LOG_DIR, f)
    size = os.path.getsize(fpath)
    print(f"  {f}  ({size:,} bytes)")

print(f"\n调用记录目录: {RECORDS_DIR}")
date_dirs = sorted(os.listdir(RECORDS_DIR))
for ddir in date_dirs:
    dpath = os.path.join(RECORDS_DIR, ddir)
    if os.path.isdir(dpath):
        print(f"\n  {ddir}/")
        for f in sorted(os.listdir(dpath)):
            fpath = os.path.join(dpath, f)
            size = os.path.getsize(fpath)
            print(f"    {f}  ({size:,} bytes)")

# 打印 trace 文件中记录的所有调用
latest = max(
    (os.path.join(RECORDS_DIR, d) for d in date_dirs if os.path.isdir(os.path.join(RECORDS_DIR, d))),
    default=None,
)
if latest:
    traces = [f for f in sorted(os.listdir(latest)) if "trace" in f]
    for t in traces:
        tpath = os.path.join(latest, t)
        with open(tpath) as fh:
            trace = json.load(fh)
        print(f"\n会话调用索引: {t}  ({trace['total_calls']} 次调用)")
        for c in trace["calls"]:
            print(f"  seq={c['seq']:>2d}  {c['function']:40s}"
                  f"  {c['elapsed_ms']:>8.1f}ms")

print(f"\n结果图目录: {RESULTS_DIR}")
print("\n" + "=" * 60)
print("  提示: 编辑 measure_api/config.local.yaml 可调整路径和设置")
print(f"  当前配置: log.directory = {LOG_DIR}")
print(f"            call_records.directory = {RECORDS_DIR}")
print("=" * 60)
