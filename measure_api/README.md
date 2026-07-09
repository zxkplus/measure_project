 # Measure API
 
 工业视觉测量系统后端 SDK 与 REST 服务。
 
 将 GUI 中的建模和测量能力封装成 Python SDK + Flask REST 接口，
 供后端同事通过 HTTP 完成建模调参与批量测量。
 
 ## 架构
 
 ```
 measure_api/
 ├── __init__.py          # 包导出（MeasureProject, SessionReplay）
 ├── __main__.py          # python -m measure_api 启动入口
 ├── run.sh               # Shell 启动脚本
 ├── config.py            # Config 配置加载（单例，YAML）
 ├── config.yaml          # 默认配置文件
 ├── logger.py            # 日志系统（线程安全，按日期分割）
 ├── project.py           # MeasureProject 核心 SDK
 ├── quality.py           # 质量评分（rms / coverage / amplitude）
 ├── visualizer.py        # 可视化快照（base64 PNG）
 ├── schemas.py           # 类型与常量定义
 ├── server.py            # Flask REST 服务
 ├── call_recorder.py     # API 调用记录（可复现）
 ├── replay.py            # SessionReplay 回放工具
 ├── requirements.txt     # 依赖
 ├── config.local.yaml.example
 ├── logs/                # 日志输出
 ├── call_records/        # 调用记录
 └── tests/               # pytest 测试（71 个用例）
 ```
 
 ## 快速开始
 
 ### 1. 安装依赖
 
 ```bash
 pip install -r measure_api/requirements.txt
 ```
 
 ### 2. 启动服务
 
 ```bash
 # 方式一：python -m
 python -m measure_api --port 5000 --host 0.0.0.0
 
 # 方式二：run.sh（支持环境变量覆盖端口、主机、配置文件路径）
 MEASURE_API_PORT=8080 bash measure_api/run.sh
 ```
 
 ### 3. 验证服务
 
 ```bash
 curl http://localhost:5000/api/health
 # → {"status": "ok", "sessions": 0}
 ```
 
 ## 两阶段工作流
 
 ### 建模阶段（Teach）
 
 ```bash
 # 1. 创建会话
 curl -X POST http://localhost:5000/api/session \
   -H "Content-Type: application/json" \
   -d '{"project_dir": "/path/to/project"}'
 # → {"session_id": "abc123", "status": "created"}
 
 # 2. 加载参考图
 curl -X POST http://localhost:5000/api/session/abc123/reference \
   -H "Content-Type: application/json" \
   -d '{"image_path": "/path/to/reference.png"}'
 # → {"width": 2048, "height": 1536, "path": "..."}
 
 # 3. 设置模板（ROI 旋转框）
 curl -X POST http://localhost:5000/api/session/abc123/template \
   -H "Content-Type: application/json" \
   -d '{"center": [1237, 993], "size": [1616, 1591], "angle_deg": 0}'
 # → {"template_shape": [1617, 1591]}
 
 # 4. 添加测量工具（即时测试反馈）
 curl -X POST http://localhost:5000/api/session/abc123/measurements \
   -H "Content-Type: application/json" \
   -d '{"object_type": "FitCircle", "label": "c1",
        "params": {"center": [814, 760], "radius": 484, "threshold": 5.0}}'
 # → {"label": "c1", "valid": true, "result": {...}, 
 #     "quality": {"num_edges": 96, "coverage": 0.92, "rms": 0.32}}
 
 # 5. 组合测量
 curl -X POST http://localhost:5000/api/session/abc123/composed \
   -H "Content-Type: application/json" \
   -d '{"composed_type": "PointCircleDistance", "label": "wall",
        "dependencies": {"point_label": "c1", "circle_label": "c2"}}'
 
 # 6. 保存
 curl -X POST http://localhost:5000/api/session/abc123/save
 ```
 
 ### 测量阶段（Run）
 
 ```bash
 # 1. 加载已有项目
 curl -X POST http://localhost:5000/api/session/abc123/load
 
 # 2. 执行测量
 curl -X POST http://localhost:5000/api/session/abc123/measure \
   -H "Content-Type: application/json" \
   -d '{"inspection_image": "/path/to/inspection.png"}'
 # → {"status": "ok", "elapsed_ms": 1245, "targets": [...]}
 ```
 
 ## API 端点总览
 
 ### 会话管理
 
 | 方法 | 端点 | 说明 |
 |------|------|------|
 | POST | `/api/session` | 创建会话 |
 | GET | `/api/session/<sid>` | 获取会话状态 |
 | DELETE | `/api/session/<sid>` | 删除会话 |
 | GET | `/api/sessions` | 列出所有活跃会话 |
 
 ### 建模
 
 | 方法 | 端点 | 说明 |
 |------|------|------|
 | POST | `/api/session/<sid>/reference` | 加载参考图 |
 | POST | `/api/session/<sid>/template` | 设置模板（ROI） |
 
 ### 测量对象 CRUD
 
 | 方法 | 端点 | 说明 |
 |------|------|------|
 | POST | `/api/session/<sid>/measurements` | 创建 + 即时测试 |
 | PUT | `/api/session/<sid>/measurements/<label>` | 更新 + 即时测试 |
 | DELETE | `/api/session/<sid>/measurements/<label>` | 删除（级联） |
 | GET | `/api/session/<sid>/measurements` | 列表 |
 | GET | `/api/session/<sid>/measurements/<label>` | 详情 |
 | POST | `/api/session/<sid>/measurements/test` | 只测试不保存 |
 
 ### 组合测量
 
 | 方法 | 端点 | 说明 |
 |------|------|------|
 | POST | `/api/session/<sid>/composed` | 添加组合 |
 | DELETE | `/api/session/<sid>/composed/<label>` | 删除组合 |
 | GET | `/api/session/<sid>/composed` | 列表 |
 
 ### DAG 与持久化
 
 | 方法 | 端点 | 说明 |
 |------|------|------|
 | GET | `/api/session/<sid>/dag` | 依赖关系图 |
 | POST | `/api/session/<sid>/save` | 保存项目 |
 | POST | `/api/session/<sid>/load` | 加载项目 |
 | POST | `/api/session/<sid>/measure` | 执行完整测量流程 |
 
 ### 可选参数
 
 - `?include_visual=true` — 在响应中包含 base64 可视化快照
 
 ## 支持的测量对象类型
 
 ### 原始测量（直接检测图像边缘）
 
 | 类型 | 说明 | 关键参数 |
 |------|------|----------|
 | `EdgePoint` | 单边边缘点 | `row, col, angle, length1, length2, threshold` |
 | `EdgePair` | 双边边缘对（宽度） | `row, col, angle, length1, length2, threshold` |
 | `FitLine` | 拟合直线 | `start, end, num_measures, threshold` |
 | `FitCircle` | 拟合圆 | `center, radius, num_measures, threshold` |
 | `TemplateMatchPoint` | 模板匹配定位点 | `row, col, template_size, preprocessor_type` |
 
 ### 组合测量（依赖其他测量对象）
 
 | 类型 | 说明 | 依赖项 |
 |------|------|--------|
 | `TwoPointsLine` | 两点连线 | `point_a_label, point_b_label` |
 | `TwoPointsDistance` | 两点距离 | `point_a_label, point_b_label` |
 | `PointLineDistance` | 点到线的距离 | `point_label, line_label` |
 | `TwoLinesAngle` | 两线夹角 | `line_a_label, line_b_label` |
 | `PointCircleDistance` | 点到圆的距离 | `point_label, circle_label` |
 
 ## 配置
 
 ### config.yaml（默认配置）
 
 日志、调用记录、Flask 服务等设置均通过 `measure_api/config.yaml` 管理。
 
 ```yaml
 log:
   directory: "logs"
   level: "INFO"
   backup_days: 30
 
 call_records:
   enabled: true
   directory: "call_records"
   retention_days: 90
   write_mode: "async"
 
 server:
   host: "0.0.0.0"
   port: 5000
   max_sessions: 10
 ```
 
 ### 本地覆盖
 
 创建 `measure_api/config.local.yaml` 覆盖默认配置中的对应字段。
 参考 `config.local.yaml.example`。
 
 ### 运行时热更新
 
 ```python
 from measure_api.config import Config
 from measure_api.logger import update_log_level
 
 cfg = Config.load()
 cfg.reload()                     # 重新加载配置文件
 cfg.set("log.level", "DEBUG")    # 或直接设置
 update_log_level("DEBUG")        # 热更新日志等级
 ```
 
 ## 日志系统
 
 线程安全，按日期分割，自动清理旧日志。
 
 ```
 logs/
 ├── measure_api.log             # 主日志（按日期轮转）
 └── measure_api.error.log       # 仅 ERROR 及以上
 ```
 
 每条日志包含 `[trace_id]`，可与调用记录文件双向关联。
 
 ## 调用记录（可复现）
 
 每次 API 调用自动记录入参、出参、耗时，存入 `call_records/` 目录。
 
 ```
 call_records/
 └── 2026-07-09/
     ├── session_sess-abc_trace.json             # 全链路时序索引
     ├── 20260709_153000_001_create_session.json  # 单次调用记录
     └── ...
 ```
 
 ### 回放
 
 ```python
 from measure_api.replay import SessionReplay
 
 replay = SessionReplay.from_trace(
     "call_records/2026-07-09/session_sess-abc_trace.json"
 )
 results = replay.replay_all()
 results = replay.replay_upto(seq=4)
 results = replay.replay_upto(call_id="trk-0004")
 ```
 
 HTTP 响应头部带 `X-Trace-Id`，出问题直接把 trace ID 发过来即可定位。
 
 ## 项目文件结构
 
 一个项目是一个目录：
 
 ```
 my_project/
 ├── config.json          # 测量工具定义、组合关系（纯 JSON，可读）
 ├── template.npz         # 模板像素数据（由原有 MultiTargetWorkflow 处理）
 └── reference.png        # 参考图备份
 ```
 
 ## Python SDK 直接使用
 
 不需要 Flask 服务，团队成员可直接 import 使用：
 
 ```python
 from measure_api import MeasureProject
 
 project = MeasureProject("/path/to/project")
 project.load_reference("reference.png")
 project.set_template(center=(100, 100), size=(160, 160), angle_deg=0)
 
 # 添加测量 → 即时反馈
 result = project.add_measurement("FitCircle", "c1", {
     "center": (100, 100), "radius": 60, "measure_length1": 30,
 })
 print(result["quality"])  # {"num_edges": 96, "rms": 0.32}
 
 # 批量测量
 results = project.measure("inspection.png")
 for t in results["targets"]:
     print(t["measurements"]["c1"]["radius"])
 ```
 
 ## 开发
 
 ```bash
 # 运行测试
 pytest measure_api/tests/ -v
 
 # 运行指定测试文件
 pytest measure_api/tests/test_project.py -v
 
 # 运行单个测试
 pytest measure_api/tests/test_project.py::test_save_load -v
 ```
 
 ### 测试覆盖
 
 | 文件 | 用例数 | 覆盖内容 |
 |------|--------|----------|
 | test_config.py | 9 | 配置加载、合并、点分语法、热更新 |
 | test_logger.py | 8 | 日志写入、等级过滤、trace_id、线程安全 |
 | test_project.py | 24 | 状态机、CRUD、组合、DAG、级联、保存加载 |
 | test_server.py | 18 | 所有端点、错误处理、trace header |
 | test_call_recorder.py | 6 | 同步/异步、trace 索引、清理 |
 | test_replay.py | 6 | 从 trace 重建、按序/按 ID 回放 |
 
 ## 文件清单
 
 ```
 measure_api/
 ├── __init__.py
 ├── __main__.py
 ├── run.sh
 ├── config.py
 ├── config.yaml
 ├── config.local.yaml.example
 ├── logger.py
 ├── project.py
 ├── quality.py
 ├── visualizer.py
 ├── schemas.py
 ├── server.py
 ├── call_recorder.py
 ├── replay.py
 ├── requirements.txt
 ├── README.md
 ├── logs/
 │   └── .gitkeep
 ├── call_records/
 │   └── .gitkeep
 └── tests/
     ├── __init__.py
     ├── conftest.py
     ├── test_config.py
     ├── test_logger.py
     ├── test_project.py
     ├── test_server.py
     ├── test_call_recorder.py
     └── test_replay.py
 ```
