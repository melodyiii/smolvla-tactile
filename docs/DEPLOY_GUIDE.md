# SO-101 部署 / 评测脚本技术文档

> 本文档详细说明 **数据采集 → 训练 → 推理部署** 的完整数据流，  
> 以及部署脚本与采数据脚本 (`test.sh`) 之间的硬件对齐关系。

> 补充阅读：如果你需要一份面向 **项目介绍 / 科研实习包装 / 面试答辩** 的整合版说明，
> 请直接看 `docs/TACTILE_PROJECT_PLAYBOOK.md`。

---

## 目录

1. [硬件配置一览](#1-硬件配置一览)
2. [数据流全链路](#2-数据流全链路)
3. [关键对齐点](#3-关键对齐点)
4. [部署脚本架构](#4-部署脚本架构)
5. [评测脚本详解 (eval_real_robot.py)](#5-评测脚本详解)
6. [录像回放功能](#6-录像回放功能)
7. [启动参数大全](#7-启动参数大全)
8. [需要你提供的信息](#8-需要你提供的信息)
9. [问题排查](#9-问题排查)

---

## 1. 硬件配置一览

以下硬件参数取自你的采数据脚本 `test.sh`，部署脚本已完全对齐：

| 硬件            | 参数                                            | 脚本对齐 |
|:----------------|:------------------------------------------------|:---------|
| 机械臂          | SO-101 Follower, `/dev/ttyACM0`                 | ✅        |
| RealSense 相机  | D435i, serial `243522072793`, **仅 RGB 流**      | ✅        |
|                 | 640×480, 30fps, `use_depth=False`               | ✅        |
| 侧面相机        | OpenCV USB, index `0`                           | ✅        |
|                 | 640×480, 30fps                                  | ✅        |
| 触觉传感器 (左)  | `/dev/ttyUSB0`, baudrate `115200`               | ✅        |
| 触觉传感器 (右)  | `/dev/ttyUSB1`, baudrate `115200`               | ✅        |
| 遥操作器        | SO-101 Leader, `/dev/ttyACM1` (仅采数据)         | N/A      |

### 机器人标识

- **`robot.name = "so_follower"`** — LeRobot 框架中 SO101Follower 的注册名
- **`robot.id = None`** — 你采数据时未设置 `--robot.id`，默认为 `None`
- **标定文件路径**: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/None.json`
- 部署脚本不再设置 `id="so101_follower"`，保持 `id=None` 以复用你的标定文件

---

## 2. 数据流全链路

```
┌──────────────┐    lerobot-record     ┌─────────────────┐    training script    ┌──────────────┐
│  真实硬件      │ ──────────────────→  │  LeRobotDataset  │ ──────────────────→  │  Policy 模型  │
│  (robot+cam)  │   test.sh, 30fps     │  (Parquet+MP4)   │  delta_timestamps    │  (.pt ckpt)   │
└──────────────┘                       └─────────────────┘                       └──────────────┘
                                                                                        │
                                                                                        ▼
┌──────────────┐    eval_real_robot.py  ┌─────────────────┐  ← 来自同一个 robot   ┌──────────────┐
│  评测录像      │ ←──────────────────  │  推理循环 30Hz   │ ←───观测+动作──────── │  真实硬件      │
│  (可选录制)    │   --record_dataset   │  select_action   │                       │  (robot+cam)  │
└──────────────┘                       └─────────────────┘                       └──────────────┘
```

### 2.1 采数据阶段 (`test.sh`)

```bash
lerobot-record \
  --robot.type=so101_tactile \
  --robot.cameras='{
    depth: {type: realsense_rgbd, serial_number: "243522072793", ...fps: 30},
    side:  {type: opencv, index_or_path: 0, ...fps: 30}
  }' \
  --dataset.fps=30 \
  --dataset.repo_id=natsuu/shape-based-assembly-hexagon-04
```

**产出的观测 key**:

| 硬件源               | 原始 key          | Dataset feature key                    | Shape         |
|:---------------------|:------------------|:---------------------------------------|:--------------|
| 6 DOF 关节角          | `shoulder_pan.pos` … `gripper.pos` | `observation.state`             | `(6,)`        |
| RealSense RGB         | `realsense_rgb`   | `observation.images.realsense_rgb`     | `(480,640,3)` |
| RealSense Depth (8bit)| `realsense_depth`  | `observation.images.realsense_depth`  | `(480,640,3)` |
| Side Camera           | `side`            | `observation.images.side`              | `(480,640,3)` |
| 遥操作 (leader 关节角) | —                 | `action`                              | `(6,)`        |

> `realsense_depth` 是 8-bit 伪彩深度图（3 通道，用于 H264 视频编码）。  
> 原始 16-bit 毫米深度另存在 `depth_mm/` 目录中。

### 2.2 训练阶段

训练脚本通过 `LeRobotDataset` 加载数据，将 dataset features 转换为 policy features：

```python
# 训练脚本中
dataset_metadata = LeRobotDatasetMetadata("natsuu/shape-based-assembly-hexagon-04")
features = dataset_to_policy_features(dataset_metadata.features)

# 自动转换:
# "observation.images.realsense_rgb"  →  PolicyFeature(type=VISUAL,  shape=(3,480,640))
# "observation.images.realsense_depth" → PolicyFeature(type=VISUAL,  shape=(3,480,640))
# "observation.images.side"           →  PolicyFeature(type=VISUAL,  shape=(3,480,640))
# "observation.state"                 →  PolicyFeature(type=STATE,   shape=(6,))
# "action"                            →  PolicyFeature(type=ACTION,  shape=(6,))
```

**SmolVLA 额外处理**:
- 图像 resize + padding 到 `(512, 512)`
- State 零填充到 `max_state_dim=32`
- Action chunk: `chunk_size=50`，每次预测 50 步动作序列
- `delta_timestamps["action"] = [0/30, 1/30, 2/30, ..., 49/30]` （50 步 @30fps = 1.67s）

**DiffusionPolicy**:
- `n_obs_steps=2`, `horizon=16`, `n_action_steps=8`
- 每 8 步做一次扩散推理

### 2.3 推理阶段

推理循环（30Hz 节拍）：

```python
while step < max_steps:
    t_start = time.monotonic()

    # ① 读取观测 (robot.get_observation() → numpy dict)
    raw_obs = robot.get_observation()
    # raw_obs = {
    #     "shoulder_pan.pos": 150.3,  "gripper.pos": 45.0, ...
    #     "realsense_rgb": ndarray(480,640,3),
    #     "realsense_depth": ndarray(480,640,3),
    #     "side": ndarray(480,640,3),
    # }

    # ② 转换为推理格式 (numpy → torch, HWC→CHW, [0,1] 归一化)
    obs = prepare_observation_for_inference(raw_obs, device, task=task, robot_type="so_follower")
    # obs = {
    #     "observation.images.realsense_rgb": Tensor(1,3,480,640),  [0,1]
    #     "observation.images.realsense_depth": Tensor(1,3,480,640),
    #     "observation.images.side": Tensor(1,3,480,640),
    #     "observation.state": Tensor(1,6),
    #     "task": "Grasp the hexagon...",
    #     "robot_type": "so_follower",
    # }

    # ③ 预处理 (归一化 state, resize image, tokenize language)
    obs = preprocess(obs)

    # ④ 推理 (SmolVLA: 从 action queue 弹出, queue 空时做一次完整推理)
    action = policy.select_action(obs)   # → Tensor(1, 6)

    # ⑤ 后处理 (反归一化 → 回到度数空间)
    action = postprocess(action)

    # ⑥ 转换为机器人动作 + 安全检查 + 发送
    robot_action = make_robot_action(action, dataset_features)
    robot.send_action(robot_action)

    # ⑦ 节拍控制 (维持 30Hz)
    elapsed = time.monotonic() - t_start
    if (sleep_t := 1/30 - elapsed) > 0:
        time.sleep(sleep_t)
```

---

## 3. 关键对齐点

### 3.1 FPS 对齐 — 30Hz

| 环节       | FPS   | 说明                                                   |
|:-----------|:------|:-------------------------------------------------------|
| 采数据      | 30    | `test.sh --dataset.fps=30`                             |
| 训练        | 30    | `delta_timestamps` 以 1/30s 为间隔                      |
| **推理**    | **30**| `--fps 30`（默认），action chunk 的时序语义与训练一致      |

> ⚠️ **如果推理频率与训练 fps 不一致**（例如 10Hz），action chunk 中每步动作间隔
> 本应是 33ms (1/30s)，但实际间隔变成 100ms (1/10s)，导致动作执行速度
> 变成预期的 1/3，机械臂运动会明显变慢变拖沓。

SmolVLA 使用 action chunk (50步)，首次推理可能需要 100-200ms，但后续 49 步
直接从 queue 弹出，耗时极少。30Hz 节拍下 50 步 ≈ 1.67s 才做一次完整推理，
实际平均频率接近 30Hz。

### 3.2 标定文件对齐

- 采数据时 `robot.id = None`（未设置）→ 标定文件 `~/.cache/.../so_follower/None.json`
- 部署脚本不设置 `id`（保持 `None`）→ 自动找到同一标定文件
- 使用 `--skip_calibration` 跳过标定交互，直接复用

### 3.3 相机 key 名对齐

采数据的相机名 `depth` + `side` 决定了 dataset 中的 key 名：
- `depth` → RealSense RGBD → 自动产生 `realsense_rgb` 和 `realsense_depth` 两个观测流
- `side` → OpenCV → 产生 `side` 观测流

推理时必须使用完全相同的相机名和配置，否则 `build_dataset_frame` / `make_robot_action`
找不到匹配的 feature key，会丢失观测数据或发送错误的动作。

### 3.4 robot_type 对齐

| 来源                | 值              |
|:--------------------|:----------------|
| `info.json` (数据集) | `"so_follower"` |
| `robot.name` (代码)  | `"so_follower"` |
| 推理脚本 `robot_type` | `"so_follower"` |

---

## 4. 部署脚本架构

```
deploy/
├── eval_real_robot.py        # 统一评测脚本（支持 3 种策略 + 录像回放）
├── deploy_tactile_vla.py     # SmolVLA + 触觉传感器 单独部署
├── deploy_vla_only.py        # SmolVLA（无触觉）单独部署
├── deploy_diffusion.py       # DiffusionPolicy 单独部署
├── tactile_modules.py        # 触觉编码器 + 投影器
├── tactile_sensor_reader.py  # 触觉传感器异步读取器
└── safety.py                 # 动作安全检查 + 范围钳位

scripts/
├── run_eval.sh               # 一键评测 (推荐入口)
├── run_tactile_vla.sh        # 单独运行触觉 VLA
├── run_vla_only.sh           # 单独运行 VLA
└── run_diffusion.sh          # 单独运行 Diffusion
```

### 4.1 三种策略模式

| 模式           | 策略                | 观测输入                                 | 特殊组件              |
|:---------------|:-------------------|:-----------------------------------------|:---------------------|
| `vla_only`     | SmolVLA            | RGB + Side + State + Task                | 无                   |
| `tactile_vla`  | SmolVLA + 触觉      | RGB + Side + State + Task + 触觉          | TactileSensorReader  |
| `diffusion`    | DiffusionPolicy    | RGB + Side + State                       | 无 (不需 Task)       |

> **注意**: 三种策略均不使用深度图 (`realsense_depth`)。推理时 RealSense 只开 RGB 流。

### 4.2 Checkpoint 格式

**SmolVLA (vla_only / tactile_vla)**:
```python
ckpt = {
    "model": {                          # policy.model (VLAFlowMatching) 的 state_dict
        "vlm_with_expert.*": ...,       # VLM + action expert
        "state_proj.*": ...,            # state → embedding projector
        "action_head.*": ...,           # action head
        "tactile_encoder.*": ...,       # (仅 tactile_vla)
        "tactile_proj.*": ...,          # (仅 tactile_vla)
    },
    "tactile_encoder": ...,             # (仅 tactile_vla) 独立存储的触觉编码器权重
    "tactile_proj": ...,                # (仅 tactile_vla) 独立存储的触觉投影器权重
    "epoch": int,
    "loss": float,
}
```

**DiffusionPolicy**:
- 方式 A: HuggingFace 格式目录 (`--ckpt_dir outputs/diffusion_so101/`)
- 方式 B: 裸 state_dict (`--ckpt model.pt`，需配合 `--state_dim --action_dim`)

---

## 5. 评测脚本详解

### 5.1 运行流程

```
make_robot()          # 创建机器人、连接相机
    ↓
load_model()          # 加载策略 + pre/post processor
    ↓
[创建 LeRobotDataset]  # (如果 --record_dataset)
    ↓
┌──────────────────────────────────────────┐
│  for episode in range(n_episodes):       │
│    run_episode()                         │
│      - 读取 obs → 推理 → 发送 action     │
│      - [每步录制 frame → add_frame]       │
│      - 询问 success/fail/retry           │
│      - [save_episode]                    │
│    统计成功率                             │
└──────────────────────────────────────────┘
    ↓
[dataset.finalize()]   # 编码视频、保存 meta
    ↓
保存 eval_results.json
```

### 5.2 关键参数

| 参数                    | 默认值                     | 说明                              |
|:------------------------|:--------------------------|:----------------------------------|
| `--model`               | (必选)                    | `tactile_vla` / `vla_only` / `diffusion` |
| `--ckpt`                | (必选)                    | Checkpoint 路径                    |
| `--task`                | `""`                      | 语言指令 (VLA 必填)                |
| `--fps`                 | `30`                      | 推理频率，需与训练 fps 一致          |
| `--skip_calibration`    | `false`                   | 跳过标定（复用采数据标定）           |
| `--record_dataset`      | `false`                   | 录制评测数据集                      |
| `--record_repo_id`      | `natsuu/eval_replay`      | 录制数据集名称                      |
| `--n_episodes`          | `10`                      | 评测 episode 数量                   |
| `--max_steps_per_episode`| `300`                    | 每个 episode 最大步数               |

---

## 6. 录像回放功能

### 6.1 启用录制

```bash
# 方式一: 通过 run_eval.sh
RECORD=1 RECORD_REPO=natsuu/eval_replay ./scripts/run_eval.sh

# 方式二: 直接调用
python deploy/eval_real_robot.py \
    --model vla_only \
    --ckpt path/to/ckpt.pt \
    --task "Grasp the hexagon..." \
    --record_dataset \
    --record_repo_id natsuu/eval_replay \
    --skip_calibration
```

### 6.2 录制内容

每步录制的数据帧：

| Feature key                          | 来源              | 格式                   |
|:-------------------------------------|:------------------|:----------------------|
| `observation.state`                  | 机器人关节角        | `float32, (6,)`       |
| `observation.images.realsense_rgb`   | RealSense RGB     | `video, (480,640,3)`  |
| `observation.images.realsense_depth` | RealSense Depth   | `video, (480,640,3)`  |
| `observation.images.side`            | 侧面相机           | `video, (480,640,3)`  |
| `action`                             | 策略输出的关节动作   | `float32, (6,)`       |
| `task`                               | 语言任务指令        | `string`              |

### 6.3 查看录制视频

```bash
# 使用 LeRobot 自带的数据集可视化工具
python -m lerobot.scripts.lerobot_dataset_viz \
    --repo_id natsuu/eval_replay \
    --episode_index 0

# 数据集保存位置 (默认):
# ~/.cache/huggingface/lerobot/natsuu/eval_replay/
#   ├── data/          # Parquet (state, action)
#   ├── videos/        # MP4 (RGB, depth, side)
#   └── meta/          # info.json, stats.json
```

---

## 7. 启动参数大全

### run_eval.sh 环境变量

```bash
# 基本
MODEL=tactile_vla           # 策略类型
CKPT=train/ckpt.pt          # Checkpoint
TASK="Grasp the hexagon..."  # 任务指令
DEVICE=cuda                  # 计算设备

# 硬件
ROBOT_PORT=/dev/ttyACM0
RS_SERIAL=243522072793
TACTILE_LEFT_PORT=/dev/ttyUSB0
TACTILE_RIGHT_PORT=/dev/ttyUSB1
TACTILE_BAUDRATE=115200
TACTILE_MOCK=                # 非空=使用 mock 触觉

# 评测
FPS=30                       # 推理频率
N_EPISODES=10
MAX_STEPS=300
MAX_REL=10.0                 # 安全: 每步最大变化度数
OUTPUT=eval_results.json

# 标定 & 录制
SKIP_CAL=1                   # 1=跳过标定
RECORD=0                     # 1=录制评测数据集
RECORD_REPO=natsuu/eval_replay
```

### 典型启动命令

```bash
# 评测 SmolVLA (无触觉), 跳过标定, 录制回放
SKIP_CAL=1 RECORD=1 MODEL=vla_only \
  CKPT=train/ckpt.pt \
  TASK="Grasp the hexagon-shaped workpiece, move it above the board, align it with the corresponding slot, and insert it." \
  ./scripts/run_eval.sh

# 评测 SmolVLA + 触觉, mock 模式
SKIP_CAL=1 TACTILE_MOCK=1 MODEL=tactile_vla \
  CKPT=train/ckpt_stage4_smolvla_final.pt \
  TASK="Grasp the hexagon..." \
  ./scripts/run_eval.sh

# 评测 DiffusionPolicy (从目录加载)
SKIP_CAL=1 RECORD=1 MODEL=diffusion \
  CKPT=outputs/diffusion_so101/ \
  N_EPISODES=5 MAX_STEPS=500 \
  ./scripts/run_eval.sh
```

---

## 8. 已确认的信息 & 调试修复记录

以下信息已通过分析训练代码和试运行确认：

### 8.1 Checkpoint 文件 ✅

- **路径**: `train/ckpt_stage4_smolvla_final.pt`
- **结构**: `{"epoch": int, "model": state_dict, "tactile_encoder": state_dict, "tactile_proj": state_dict, "optimizer": ..., "loss": float}`
- `ckpt["model"]` 是 `VLAFlowMatching` 的 state_dict → 加载到 `policy.model.load_state_dict()`
- 加载验证: **0 missing, 0 unexpected keys** ✅

### 8.2 训练细节 ✅

- **观测图像**: 仅 `observation.images.side` + `observation.images.realsense_rgb`（**不使用 depth**）
- **Task 指令**: `"Grab a piece of cloth lying haphazardly in the box."` (来自 `tasks.parquet`)
- **SmolVLA**: 图像 resize 到 512×512 (SigLIP 内部), chunk_size=50, max_state_dim=32
- **DiffusionPolicy**: 图像 resize 到 96×96, n_obs_steps=2, horizon=16, n_action_steps=8
- **数据 fps=30**, 训练 stride=3 (选取 anchor 帧), action chunk 使用原始 30fps 帧间距

### 8.3 调试修复记录

| 序号 | 问题 | 修复 | 文件 |
|:-----|:-----|:-----|:-----|
| 1 | DiffusionConfig 图像 shape `(3,480,640)` 与训练不符 | 改为 `(3,96,96)`, 补 n_obs_steps/horizon/n_action_steps | `eval_real_robot.py`, `deploy_diffusion.py` |
| 2 | `logit_scale` shape 不匹配: 模型 `[1]` vs ckpt `[]` (标量) | `torch.zeros(1)` → `torch.tensor(0.0)` | `tactile_modules.py` |
| 3 | tactile_encoder state_dict 前缀不匹配 | `encoder.encoder.load_state_dict()` → `encoder.load_state_dict()` | `tactile_modules.py` |
| 4 | RealSense 深度流超时 (推理不需要 depth) | 改用 `RealSenseCameraConfig(use_depth=False)`, key 从 `"depth"` 改为 `"realsense_rgb"` | 所有 4 个 deploy 脚本 |
| 5 | RealSenseCamera(use_depth=False) 返回 3-tuple, get_observation 只解包 2-tuple | 添加 `len(latest)` 判断兼容 2/3-tuple | `so_follower.py` |
| 6 | warmup_s 未传给 RealSenseRGBDCamera 构造函数 | 在 `make_cameras_from_configs` 传递 warmup_s | `cameras/utils.py` |
| 7 | `get_observation()` 返回电机 float, `prepare_observation_for_inference` 期望 numpy | 改用 `build_inference_frame()` 自动组装 state array | `eval_real_robot.py` |
| 8 | SmolVLA base input_features 为 camera1/2/3, 与训练实际 realsense_rgb/side 不匹配 | 加载后覆写 `policy.config.input_features` | 所有 SmolVLA deploy 脚本 |

> **注意**: 以上修复仅影响推理/部署脚本，**数据采集脚本 (`test.sh`) 完全不受影响**。
> `test.sh` 继续使用 `realsense_rgbd` 类型（开启深度流），采集数据格式不变。

---

## 9. 问题排查

### Q: 启动时卡在 "Move to middle of range" 标定流程

**原因**: 缺少标定文件或 `id` 不匹配  
**解决**: 加 `--skip_calibration`（或 `SKIP_CAL=1`），前提是采数据时已完成标定

### Q: Action 执行很慢 / 机械臂动作拖沓

**原因**: `--fps` 设置低于训练 fps（例如 10Hz 训练数据是 30fps）  
**解决**: 确保 `--fps 30` 与训练数据 fps 一致

### Q: 找不到 `deploy` 模块 (`ModuleNotFoundError: No module named 'deploy'`)

**原因**: Python path 不包含项目根目录  
**解决**: 使用 `run_eval.sh` 启动（已自动设置 `PYTHONPATH`），  
或手动 `export PYTHONPATH=/home/natsuu/melody/lerobot:$PYTHONPATH`

### Q: `observation.images.realsense_rgb` key 找不到

**原因**: 相机配置的 key 名与采数据不一致  
**解决**: 推理时相机 key 为 `realsense_rgb`（RealSense RGB-only）和 `side`（OpenCV）。  
训练不使用 depth，推理也不开深度流。

### Q: RealSense 相机 "No device connected" / 超时

**原因**: 上次运行崩溃（segfault）后相机 USB 设备挂起  
**解决**: 物理拔插 RealSense USB 线缆，等待 5 秒后重连。  
验证: `conda run -n lero python -c "import pyrealsense2 as rs; print(len(rs.context().query_devices()))"`

### Q: 录制的视频在哪？

```bash
# 默认路径
~/.cache/huggingface/lerobot/natsuu/eval_replay/

# 查看
python -m lerobot.scripts.lerobot_dataset_viz --repo_id natsuu/eval_replay --episode_index 0
```
