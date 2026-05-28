# 基于 LeRobot 的 SO-101 触觉项目说明书

> 面向项目归档、科研实习申请、组会汇报和面试答辩的整合版文档。
> 重点回答 4 个问题：
> 1. 触觉是怎么接进 LeRobot 采集链路的。
> 2. 遥操作是怎么和采集同步起来的。
> 3. SmolVLA 和 SmolVLA + Tactile 是怎么做真机部署的。
> 4. 这个项目如何包装成一段能经得住追问的具身智能科研实习经历。

---

## 1. 项目一句话

本项目基于 LeRobot 和 SO-101 机械臂，扩展了双 16x16 触觉阵列的采集与真机部署能力：

- 在 **数据采集阶段**，实现了视觉、状态、动作和双触觉的同步记录。
- 在 **真机部署阶段**，实现了 `SmolVLA` 和 `SmolVLA + Tactile` 两种模式。
- 在 **工程实现层面**，不是重写整个 LeRobot，而是沿着它已有的 `Config -> Robot Factory -> Robot Wrapper -> Record Loop -> Deploy Script` 这些扩展点做最小侵入式改造。

这也是这个项目最适合对外强调的地方：
你不是“从零手搓一套机器人框架”，而是“在 LeRobot 这个真实开源机器人框架上完成多模态硬件接入、采集链路扩展和真机部署”。

---

## 2. 触觉如何接入 LeRobot 采集链路

## 2.1 设计原则

这个项目的触觉接入有 4 个明确原则：

1. 不 fork 整个 LeRobot 数据集格式。
2. 不破坏 LeRobot 原有的 state / action / image 采集流程。
3. 触觉接入尽量沿用 LeRobot 的机器人抽象，而不是在采集脚本里硬编码。
4. 触觉原始数据优先保真保存，后处理放到训练或部署阶段再做。

因此，最终选择的方案不是把 tactile 直接写进标准 `observation.features`，而是：

- 标准模态继续写进 `LeRobotDataset` 的 parquet + mp4。
- 触觉原始矩阵按 episode / frame index 另存成 sidecar `.npy` 文件。

这是一个很重要的答辩点，因为它体现了你对“框架兼容性”和“数据工程成本”的权衡。

---

## 2.2 不是重写 LeRobot，而是扩展 LeRobot

从代码结构上看，你的改造主要落在 5 个点：

1. 给 `SOFollowerConfig` 增加 `TactileSensorConfig`，并注册新的机器人类型 `so101_tactile`。
2. 在机器人工厂里，让 `so101_tactile` 被实例化为 `TactileSO101Robot`。
3. 新增 `TactileSensor` 串口驱动，负责 16x16 触觉阵列读取。
4. 新增 `TactileSO101Robot` 包装类，把 tactile 传感器挂到 `SO101Follower` 观测流上。
5. 在 `lerobot-record` 的 `record_loop` 末端增加 tactile sidecar 写盘逻辑。

对应到当前仓库里的关键文件：

| 功能 | 文件 |
|:-----|:-----|
| tactile 配置定义 | `src/lerobot/robots/so_follower/config_so_follower.py` |
| robot 工厂分发 | `src/lerobot/robots/utils.py` |
| tactile 串口驱动 | `src/lerobot/common/robot_devices/sensors/tactile.py` |
| 带触觉的 SO-101 机器人包装 | `src/lerobot/common/robot_devices/robots/tactile_so101.py` |
| 官方采集脚本二次扩展 | `src/lerobot/scripts/lerobot_record.py` |
| 你的采集入口脚本 | `test.sh` |

对外介绍时，可以把这部分概括成：

> 我沿着 LeRobot 的配置系统、机器人工厂和采集主循环扩展了一个新的 tactile robot 类型，而不是绕过框架直接写独立采集程序。

---

## 2.3 第一步：把触觉接入 RobotConfig 和工厂

为了让 LeRobot 能像识别 `so101_follower` 一样识别你的触觉机器人，首先要让配置系统认识它。

核心做法：

- 在 `config_so_follower.py` 中增加 `TactileSensorConfig`。
- 在 `SOFollowerRobotConfig` 上注册 `so101_tactile`。
- 在 `robots/utils.py` 里增加一个分支：当 `config.type == "so101_tactile"` 时，返回 `TactileSO101Robot(config)`。

这一步的意义是：

- `lerobot-record` 不需要知道 tactile 细节。
- 上层脚本只要传 `--robot.type=so101_tactile`，LeRobot 就能走完整的 connect / get_observation / send_action 流程。

这是一种很典型的“基于框架扩展新硬件”的实现方式，也是科研实习面试里很加分的点。

---

## 2.4 第二步：实现 TactileSensor 串口驱动

触觉传感器底层驱动在 `src/lerobot/common/robot_devices/sensors/tactile.py`。

它做了 4 件事：

1. 通过串口读取 16 行、每行 16 个整数，组装成 `16x16` 压力矩阵。
2. 在初始化时收集前 30 帧，计算一个静态中值基线。
3. 内部可做阈值裁剪、归一化和 EMA 平滑，用于显示或调试。
4. 对上层暴露 `get_raw_frame()`，返回最新的原始 `16x16 float32` 矩阵。

注意一个非常关键的设计决策：

- **采集与训练/部署真正使用的是 raw tactile frame**。
- 阈值、噪声缩放和可视化热图，不是主数据通路。

这意味着你保留下来的是“原始触觉压力图”，而不是被某种前处理固化后的版本。这样后续训练时可以自由尝试：

- 原始值归一化
- 对数压缩
- 时序滤波
- 触觉 patch/token 化
- CNN / MLP / Transformer / GRU 等不同编码器

如果老师问“为什么保存 raw tactile 而不是处理后的 tactile”，一个标准回答是：

> 因为采集阶段的目标是保真留底，训练阶段再决定前处理方式。这样不会把早期的经验性阈值设计写死进数据集。

---

## 2.5 第三步：重写 Robot Wrapper，而不是重写 Record Loop 主体

你没有直接改 `lerobot-record` 的主逻辑去读串口，而是新建了 `TactileSO101Robot`，继承自 `SO101Follower`。

这个类做了两件核心工作：

### 1. 在 connect / disconnect 生命周期里管理 tactile 设备

- 先调用父类连接机器人本体和相机。
- 再连接左、右触觉传感器。
- 如果某个环节失败，做 best-effort rollback，避免串口或相机资源泄漏。

这体现的是系统工程能力，不是“能跑就行”的脚本式写法。

### 2. 在 get_observation() 里把 tactile 数据挂到观测字典里

每次上层调用 `robot.get_observation()` 时：

- 先拿到 `SO101Follower` 原本的观测：关节状态、相机图像等。
- 再额外读取：
  - `__tactile_raw_left__`
  - `__tactile_raw_right__`

这里特意用了“双下划线隐藏键”的方式，而不是直接新增 `observation.tactile_left` 标准 feature。

原因是：

- 这样不会干扰 LeRobot 原有的 feature 推断逻辑。
- 不会强行修改官方 parquet/video schema。
- 只在你自定义的 record 保存逻辑里消费这些键。

面试时可以把这件事总结成：

> 我把 tactile 接入点放在 Robot abstraction，而不是散落在采集脚本各处，这样上层仍然只需要面向 `robot.get_observation()` 编程。

---

## 2.6 第四步：为什么 tactile 没进标准 LeRobotDataset feature

这是这个项目里最值得讲清楚的技术取舍。

LeRobot 的标准 dataset feature 主要分两类：

- `observation.state`
- `observation.images.xxx`
- `action`

这些 feature 最终会被转换成 parquet 和 mp4。

你的 tactile 没有被放进 `robot.observation_features`，所以它不会自动出现在：

- `meta/info.json` 的标准 feature 描述中
- parquet 数据表中
- 视频编码流程中

你选择的方案是 **sidecar tactile 文件夹**。

实际数据集结构看起来像这样：

```text
data/Grabbing-soybeans-hardly-02/
├── data/
├── meta/
├── tactile_raw_left/
├── tactile_raw_right/
└── videos/
```

`meta/info.json` 里仍然只有标准 feature，例如：

- `action`
- `observation.state`
- `observation.images.depth` 或 `observation.images.realsense_rgb`
- `observation.images.side`

而 tactile 单独保存在：

```text
tactile_raw_left/episode-000000/frame-000123.npy
tactile_raw_right/episode-000000/frame-000123.npy
```

### 这种设计的优点

1. 最大程度兼容官方 LeRobotDataset 工具链。
2. 不需要改 parquet schema 或视频编码逻辑。
3. tactile 原始矩阵保存简单直接，几乎没有额外框架负担。
4. 后续可以在训练数据加载阶段按 frame index 自由组织时序窗口。

### 这种设计的代价

1. 训练 tactile 模型时，需要自己写 sidecar loader。
2. 触觉不是标准 dataset feature，现成可视化工具不会自动显示。
3. 数据集不再是“纯官方 schema”，而是“官方主干 + 自定义 tactile sidecar”。

如果被问“为什么不直接改 LeRobotDataset schema”，回答可以是：

> 因为我的目标是尽量复用官方采集、保存、视频编码和评测工具链。直接改 schema 的工程成本更高，还会破坏和现有工具的兼容性；sidecar 方案更适合快速研究迭代。

---

## 2.7 第五步：遥操作是怎么和采集同时完成的

你的采集本质上还是官方 `lerobot-record` 的控制环。

在 `test.sh` 中，采集命令的核心是：

```bash
lerobot-record \
  --robot.type=so101_tactile \
  --robot.port=/dev/ttyACM0 \
  --robot.tactile_left.port=/dev/ttyUSB0 \
  --robot.tactile_right.port=/dev/ttyUSB1 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --dataset.fps=30
```

这条链路里有两个设备：

- `Follower`：被控制、被拍摄、被采样的机械臂。
- `Leader`：遥操作输入设备。

在每个 30Hz 采集步里，`record_loop` 做的是：

1. `robot.get_observation()` 读取 follower 的当前观测。
2. `teleop.get_action()` 读取 leader 当前关节角度。
3. 把 leader 动作经过 processor 后发给 follower。
4. 把本步 `observation + action + task` 写入 LeRobotDataset。
5. 从 observation 里额外取出 tactile raw frame，写成 sidecar `.npy`。

所以所谓“遥操作同时采集”，其实不是两个并发程序互相同步，而是 **同一个 record loop 在每个控制周期同时完成动作采集和观测采集**。

这是非常值得强调的，因为它说明：

- 动作和观测天然共享同一个 step index。
- tactile sidecar 也和这个 step index 对齐。
- 不需要额外做跨进程时间戳对齐。

如果老师问“你的同步是怎么做的”，最稳妥的回答是：

> 不是在采集后离线时间对齐，而是直接把 tactile 读出嵌进 LeRobot 的单步 record loop，让 state、image、action、tactile 共用同一个 frame index。

---

## 2.8 采集阶段的真实数据流

你可以把整个数据流讲成下面这条主线：

```text
SO101 Leader 读取关节角度
        -> 形成 teleop action
        -> 发送给 SO101 Follower

SO101 Follower 同时输出：
        -> 关节状态
        -> RealSense / side camera 图像
        -> 左右触觉 raw 16x16

record_loop 每步执行：
        -> add_frame(state, image, action, task)
        -> save tactile_raw_left/right as sidecar npy
        -> 下一步
```

这样对外表达，逻辑会很清楚：
这是一个“用 leader 生成动作、用 follower 输出多模态观测、再由同一控制周期完成同步存档”的系统。

---

## 2.9 Linux 环境下如何接入 LeRobot 做采集

这一段是科研实习里很容易被追问的，因为它直接暴露你是不是只会跑 notebook，还是能真正把代码接到 Linux 机器人环境里。

### 推荐说法

官方 LeRobot 推荐用 `conda` 管理环境，项目本地脚本则使用一个已经配好的环境名 `lero`。从工程角度讲，环境名不是重点，重点是：

- Python 环境要能装 LeRobot 和硬件依赖。
- 要能访问串口、相机和 RealSense。
- 要能在项目根目录正确解析本地 `deploy/` 和 `src/` 模块。

### 建议的 Linux 接入流程

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot

conda create -y -n lerobot python=3.12
conda activate lerobot
conda install ffmpeg -c conda-forge

pip install -e .
pip install -e ".[feetech]"
```

如果完全按你当前项目本地脚本来跑，还需要注意：

- `test.sh` 假设有 `conda activate lero`
- `run_eval.sh` / `run_tactile_vla.sh` 也默认尝试激活 `lero`

因此本地项目里要么：

- 直接维护一个叫 `lero` 的环境

要么：

- 把这些脚本里的环境名改成你自己的环境名

### Linux 侧的 4 个常见问题

1. **串口权限**

   如果 `/dev/ttyACM0`、`/dev/ttyUSB0`、`/dev/ttyUSB1` 打不开，通常要确认自己是否在 `dialout` 组里。

2. **USB 设备命名变化**

   插拔后 `ttyUSB0/1` 可能互换，部署前要重新确认左右触觉端口。

3. **RealSense 依赖与设备连接**

   部署前要确认 RealSense 被系统识别，否则 `pyrealsense2` 初始化会失败。

4. **本地模块导入路径**

   你的部署脚本通过 `export PYTHONPATH="$PWD:$PYTHONPATH"` 保证能导入项目根目录下的 `deploy` 包。

如果被问“你在 Linux 上做了什么”，不要只回答“跑了个脚本”，而是要强调：

- 配置 Python 环境
- 接串口设备
- 确认相机 / RealSense
- 维护 calibration 文件
- 处理 `PYTHONPATH` 和本地模块导入
- 做控制频率、设备端口和安全阈值对齐

---

## 3. 如何做真机部署：SmolVLA 与 SmolVLA + Tactile

## 3.1 部署模式概览

你当前项目里实际有 3 种部署模式：

| 模式 | 脚本 | 输入 | 说明 |
|:-----|:-----|:-----|:-----|
| `vla_only` | `deploy/deploy_vla_only.py` | `state + realsense_rgb + side + task` | 纯 SmolVLA |
| `tactile_vla` | `deploy/deploy_tactile_vla.py` | `state + realsense_rgb + side + task + tactile` | SmolVLA + 触觉 token 注入 |
| `diffusion` | `deploy/deploy_diffusion.py` | `state + image` | 另一条策略基线 |

你对外讲项目时，主角应当是前两种：

- `SmolVLA`
- `SmolVLA + Tactile`

---

## 3.2 纯 SmolVLA 真机部署流程

纯视觉版本的核心流程是：

1. 机器人读取当前观测。
2. `build_inference_frame` / `prepare_observation_for_inference` 把 numpy 观测变成 tensor。
3. 走 LeRobot 官方 `preprocess`。
4. `policy.select_action(observation)` 输出动作。
5. 走 `postprocess` 反归一化。
6. 通过安全层把动作发给真实机械臂。

这条链路的关键特点：

- 控制频率默认是 `30Hz`。
- `SmolVLA` 不是每步完整前向，而是 action chunk queue 模式。
- 单次模型推理生成 50 步动作，再逐步从队列里弹出。

这也是一个很关键的答辩点：

> SmolVLA 真机部署并不是 30Hz 每一步都完整跑一次大模型，而是通过 action chunk 机制降低平均推理开销。

---

## 3.3 SmolVLA + Tactile 真机部署流程

触觉版部署的核心思想不是重写 SmolVLA 主体，而是：

1. 单独加载 tactile encoder 和 tactile projector。
2. 单独启动一个 `TactileSensorReader` 后台线程。
3. 在 SmolVLA 的 `embed_prefix` 上做 monkey-patch。
4. 把 tactile embedding 追加到视觉/语言/state prefix 后面，作为额外 token 输入专家网络。

这是一个非常值得讲的工程点，因为它说明：

- 你没有粗暴改官方模型源代码。
- 你通过“外接模态 token”的方式扩展了已有 VLA。
- 改动小、实验快、复用性高。

### tactile 运行时链路

部署时的 tactile 数据流是：

```text
左右触觉传感器
    -> TactileSensorReader 后台 50Hz 读取
    -> 维护 16 帧滑动窗口
    -> get_window() 返回 [1, 16, 2, 16, 16]
    -> DualTactileGridEncoder 编码为 [1, 512]
    -> TactileMLPProjector 投影为 [1, 7680]
    -> reshape 成 8 个 960 维 tactile tokens
    -> 拼接到 SmolVLA prefix
```

### 为什么部署阶段是 16 帧窗口，而不是单帧 tactile

因为你这里的 tactile encoder 不是静态编码器，而是：

- `CNN` 提取每帧空间特征
- `GRU` 聚合 16 帧时序信息

也就是说，这套 tactile 模块显式利用了触觉的时间动态，而不只是瞬时接触图。

如果被问“为什么不用单帧 tactile”，回答可以是：

> 触觉在抓取、压紧、滑移判断里本来就带有明显时序性，单帧只能看到接触分布，16 帧窗口才能捕获接触建立和变化趋势。

---

## 3.4 真机部署时你需要记住的关键数字

这些数字最好背熟，因为它们是最容易被老师追问的运行参数。

### 采集和部署共同参数

- 数据采集 fps：`30`
- 真机部署控制频率：`30Hz`
- Side camera：`640x480 @ 30fps`
- RealSense：`640x480 @ 30fps`

### SmolVLA 关键参数

- `chunk_size = 50`
- `n_action_steps = 50`
- 30Hz 下一个 chunk 覆盖大约 `50 / 30 = 1.67s`

### tactile 关键参数

- tactile 读取频率：`50Hz`
- tactile 时间窗口：`16 帧`
- tactile 输入 shape：`[B, T, 2, 16, 16]`
- tactile token 数：`8`
- token hidden dim：`960`

### 设备

- 项目脚本默认 `device=cuda`
- 当前仓库没有把 GPU 型号写死成某张具体卡
- 启动脚本会直接打印 `torch.cuda.get_device_name(0)`

所以最准确的说法是：

> 部署默认跑在第一张 CUDA 卡上，脚本会在启动时打印实际 GPU 型号，但仓库本身没有把型号写死。

这个回答比“跑在 4090 上”之类未经核实的说法安全得多。

---

## 3.5 Linux 下如何部署真机模型

### 方式 1：统一评测入口

```bash
MODEL=tactile_vla \
CKPT=train/ckpt_stage4_smolvla_final.pt \
TASK="grasp the cloth from the box" \
FPS=30 \
SKIP_CAL=1 \
./scripts/run_eval.sh
```

这个入口的好处是：

- 参数统一
- 可以切换 `vla_only` / `tactile_vla` / `diffusion`
- 可以可选录制评测数据集

### 方式 2：单独跑 tactile 版

```bash
CKPT=train/ckpt_stage4_smolvla_final.pt \
TASK="grasp the cloth from the box" \
./scripts/run_tactile_vla.sh
```

### 部署前必须检查的 5 件事

1. 机器人串口是否正确，例如 `/dev/ttyACM0`
2. 左右 tactile 端口是否正确，例如 `/dev/ttyUSB0` 和 `/dev/ttyUSB1`
3. RealSense 是否能正常被识别
4. calibration 文件是否存在，是否和当前 robot id 对齐
5. `FPS` 是否和训练数据一致，尤其是 `30Hz`

如果老师问“Linux 上怎么完成部署”，你应该按这个顺序答：

> 环境准备 -> 硬件识别 -> calibration -> 启动脚本 -> 频率与 schema 对齐 -> 真机安全检查。

---

## 3.6 这个项目里最容易被追问的部署细节

### 1. 为什么要保证推理频率和训练 fps 一致

因为 SmolVLA 预测的是有时间语义的 action chunk。

如果训练时动作是按 30fps 学的，而部署时你用 10Hz 去执行，那每个 action 之间的实际时间间隔就变了，动作会明显变慢、变钝，甚至偏离训练语义。

### 2. 为什么触觉 reader 是 50Hz，而控制频率是 30Hz

因为 tactile 是一个额外模态，读取频率高一些有利于构造更密的时序窗口；控制环仍保持和数据采集对齐的 30Hz。它们不是必须完全同频。

### 3. 为什么用 monkey-patch `embed_prefix`

因为这是在不大幅侵入基础模型实现的前提下，把 tactile token 注入到 SmolVLA prefix 的最小改动方案。

### 4. 为什么 latency 统计只记录部分 step

因为 `SmolVLA` 的 `select_action()` 有 action queue。只有 queue 为空、真正触发一次新的 chunk 推理时，那次 latency 才有代表性；单纯从 queue 弹动作不代表一次完整模型前向。

---

## 4. 如何把这个项目写进简历和项目介绍

## 4.1 最推荐的项目标题

以下三种写法都可以，按你投递的岗位选择。

### 偏工程版本

`基于 LeRobot 的 SO-101 双触觉采集与 SmolVLA 真机部署系统`

### 偏具身智能研究版本

`面向 SO-101 的视觉-触觉多模态操作系统：LeRobot 采集扩展与 SmolVLA 真机部署`

### 偏科研实习版本

`基于开源机器人框架 LeRobot 的多模态数据采集与 VLA 真机部署实践`

---

## 4.2 简历里可以直接写的 4 条 bullet

你可以直接改成自己口吻后放进简历。

### 版本 A：偏技术实现

- 基于 LeRobot 扩展 `so101_tactile` 机器人类型，完成双 16x16 触觉阵列在 SO-101 上的 Linux 串口接入、机器人抽象封装和采集链路打通。
- 重写 LeRobot 的 `Robot Wrapper + record loop` 扩展点，实现遥操作、视觉、关节状态与双触觉数据在 30Hz 控制环中的同步采集，并将触觉以 sidecar `.npy` 形式按 episode/frame index 对齐保存。
- 实现 `SmolVLA` 与 `SmolVLA + Tactile` 两套真机部署流程，完成 tactile encoder / projector 加载、50Hz 触觉滑窗构造及 tactile token 注入到 VLA prefix 的部署方案。
- 处理 Linux 环境部署中的串口设备、RealSense 相机、校准文件、推理频率和安全动作约束等系统工程问题，完成真实机械臂端到端运行。

### 版本 B：偏科研实习风格

- 在 LeRobot 开源具身智能框架上扩展视觉-触觉多模态采集与真机部署能力，完成 SO-101 机械臂的双触觉传感器接入、数据对齐和策略部署。
- 设计并实现基于 sidecar tactile 存储的多模态采集方案，在不破坏官方 LeRobotDataset 主 schema 的前提下保留高保真触觉原始数据。
- 完成 SmolVLA 在真实机器人上的 30Hz 部署，并进一步实现触觉时序窗口编码与多模态 token 级融合。
- 具备 Linux 环境下机器人系统集成、硬件调试、数据工程和真实平台安全控制经验。

---

## 4.3 面试时 30 秒版本怎么讲

> 我做的是一个基于 LeRobot 的 SO-101 多模态操作项目。核心工作不是重新写一套机器人框架，而是在 LeRobot 现有采集和部署链路上扩展了双触觉模态。具体来说，我实现了触觉串口驱动、带触觉的 robot wrapper、遥操作同步采集，以及 SmolVLA 和 SmolVLA+Tactile 的真机部署。项目重点是把视觉、状态、动作和触觉在真实 Linux 硬件环境里稳定跑通。

---

## 4.4 面试时 2 分钟版本怎么讲

> 这个项目基于 Hugging Face 的 LeRobot 框架，平台是 SO-101 机械臂。我做了两部分工作。
>
> 第一部分是数据采集。我没有绕开 LeRobot 单独写脚本，而是沿着它的配置系统、机器人工厂和 record loop 扩展了一个 `so101_tactile` 机器人类型。底层用串口驱动读取双 16x16 触觉阵列，在 `robot.get_observation()` 里把左右触觉原始矩阵挂进去。采集时继续使用官方 `lerobot-record`，让 leader 遥操作产生动作、follower 输出状态和图像，同时把 tactile raw frame 以 sidecar `.npy` 按 frame index 写盘。这样标准 LeRobotDataset 主干保持兼容，触觉又能完整保真保存。
>
> 第二部分是真机部署。我实现了 `SmolVLA` 和 `SmolVLA + Tactile` 两种模式。纯视觉模式沿用 LeRobot 的 preprocess / select_action / postprocess 流程；触觉模式则额外启动 50Hz 触觉 reader，构造 16 帧时间窗口，通过 CNN+GRU 编码成 tactile feature，再投影成 8 个 token，monkey-patch 到 SmolVLA prefix 里。整个系统在 Linux 上完成了串口、相机、校准和 30Hz 控制环对齐，重点体现的是多模态具身系统工程能力，而不只是单纯训一个模型。

---

## 4.5 科研实习里你最该强调的能力点

如果你投的是具身智能科研实习，老师通常更关心这些点：

1. **你能不能在真实 Linux 环境里把硬件跑起来。**
2. **你是不是会沿着现有框架扩展，而不是只会写 demo。**
3. **你能不能把数据采集、模型输入和部署推理串成完整链路。**
4. **你是否理解多模态系统里的同步、接口设计、部署约束和安全问题。**

因此，这个项目的包装重点不应该只是：

- “我做了触觉模型”

而应该是：

- “我在真实机器人平台上完成了多模态硬件接入、数据工程和 VLA 真机部署。”

---

## 4.6 哪些话可以说，哪些话不要乱说

这是“防止拷打”最重要的一段。

### 可以明确说的

- 我扩展了 LeRobot 的机器人抽象，让双触觉可以进入采集与部署链路。
- 我实现了 tactile 数据的 sidecar 保存方案。
- 我实现了 SmolVLA 和 SmolVLA+Tactile 的真机部署。
- 我知道 action chunk、fps 对齐、tactile 窗口和 token 注入这些部署细节。
- 我处理过 Linux 下串口、相机、校准、环境变量和模块导入问题。

### 不建议过度声称的

- 不要轻易说“我重写了 LeRobot 框架”。更准确的是“我在 LeRobot 上做了扩展”。
- 不要轻易说“我完整实现了触觉训练全流程”，除非你能清楚拿出 tactile sidecar loader、训练脚本和实验结果。
- 不要轻易说“我设计了一个新的 foundation model”。更准确的是“我在现有 SmolVLA 上做了 tactile 模态接入和真机部署”。
- 不要在没核实的情况下说“跑在 4090/A100 上”。准确说法是“默认 CUDA 第一张卡，脚本会打印实际 GPU 型号”。

如果老师继续追问训练部分，而你当前仓库里没有完整训练 loader，可以诚实回答：

> 当前仓库里已经完整实现了 tactile 的采集与真机部署，训练侧需要结合 tactile sidecar 再写一个窗口化 loader；这个思路我是清楚的，但训练脚本不完全在当前仓库里。

这种回答比硬撑更安全。

---

## 5. 常见拷打问题与回答思路

下面这些问题非常高频，建议你提前练熟。

### Q1：为什么不直接把 tactile 写进 LeRobotDataset 的标准 feature？

**答题要点：**

- 目标是最大化复用 LeRobot 官方采集和评测工具链。
- 直接改 schema 会影响 parquet / video / feature 推断逻辑。
- tactile 适合原始矩阵 sidecar 保存，训练时再决定前处理。

**一句话回答：**

> 我优先选择兼容官方主干 schema，把 tactile 作为 sidecar 保存，这样工程侵入更小，也更适合快速研究迭代。

### Q2：你的 tactile 和图像/动作是怎么同步的？

**答题要点：**

- 不是离线时间戳对齐。
- 是在同一个 `record_loop` 里同步完成。
- 共用同一个 frame index。

**一句话回答：**

> 我把 tactile 读出嵌进 LeRobot 的单步采集循环，让 state、image、action 和 tactile 共用同一个 step 和 frame index。

### Q3：为什么要在 Robot Wrapper 层接入 tactile？

**答题要点：**

- 上层应该只依赖 `robot.get_observation()`。
- 让触觉成为设备能力，而不是脚本能力。
- 这样部署和采集都能复用同一套 robot abstraction。

### Q4：为什么部署时触觉是 16 帧窗口，不是单帧？

**答题要点：**

- tactile 编码器里有 GRU。
- 任务里抓取、接触建立、滑移判断都需要时序信息。

### Q5：SmolVLA 的 30Hz 和 chunk_size=50 是什么关系？

**答题要点：**

- 30Hz 是控制环频率。
- `chunk_size=50` 是每次模型生成的动作长度。
- 实际完整推理频率远低于控制频率，因为中间很多步只是从 action queue 弹动作。

### Q6：为什么 tactile reader 是 50Hz，而控制环是 30Hz？

**答题要点：**

- 控制频率要和训练数据 fps 对齐。
- tactile 读取更高频可以给时序窗口更密的采样。
- 两者不必完全相同。

### Q7：你说“重写 LeRobot 的机器人框架”，到底重写了什么？

最稳妥的回答不是“我重写了框架”，而是：

> 我扩展了 LeRobot 的机器人配置、工厂和包装层，实现了一个新的 tactile robot 类型，并修改了 record loop 的保存逻辑让 tactile sidecar 落盘。

### Q8：Linux 上最麻烦的问题是什么？

建议你答：

- 串口端口稳定性和权限
- RealSense 连接与超时
- calibration 文件和 robot id 对齐
- `PYTHONPATH`、本地包导入和环境一致性
- 控制频率和训练 schema 对齐

这样比只说“装依赖比较麻烦”更专业。

---

## 6. 还有哪些知识点需要继续学习

如果你想让这个项目更像一段扎实的具身智能科研经历，建议继续补下面这些知识。

### 1. LeRobot 数据与处理器体系

重点看：

- `LeRobotDataset`
- `dataset_features`
- `build_dataset_frame`
- `preprocess / postprocess`
- `prepare_observation_for_inference`

目标：

你要能讲清“硬件观测 -> dataset feature -> policy feature -> model input”这条映射链。

### 2. 多模态融合方式

重点理解：

- early fusion / late fusion
- token-level fusion
- prefix 注入
- cross-attention
- tactile + vision 的互补性

目标：

你要能解释为什么你这里选择“把 tactile 投影成 token，接到 SmolVLA prefix 上”。

### 3. 时序建模

重点理解：

- 为什么 tactile 用 GRU
- 单帧 vs 滑窗
- 历史观测对抓取任务的重要性

### 4. Linux 机器人系统基础

至少要熟：

- 串口和 USB 设备管理
- `udev` / `dialout`
- 相机驱动与 RealSense 设备检查
- conda / editable install / PYTHONPATH
- 基本日志排障能力

### 5. 机器人安全与控制

至少要会讲：

- `max_relative_target` 的安全意义
- 为什么部署时要做 action range validation
- 为什么控制频率、动作尺度和真实机械臂安全强相关

### 6. 实验方法论

如果你后续要补科研味，应该继续准备：

- 成功率评测
- 不同模态 ablation
- tactile 窗口长度 ablation
- tactile token 数量 ablation
- latency / throughput / safety tradeoff

---

## 7. 如果想把项目再包装得更强，下一步最值得做什么

下面这些工作做完后，你的项目会更像一段成熟科研/工程项目，而不是单点功能开发。

### 第一优先级

1. 把 tactile sidecar 的训练侧 loader 写进仓库。
2. 补一个清晰的 `train tactile smolvla` 数据加载入口。
3. 形成一张完整的“采集 -> 训练 -> 部署”技术图。

### 第二优先级

1. 给 tactile 数据加一个简单可视化工具。
2. 做 `vla_only` vs `tactile_vla` 的真实任务对比。
3. 记录部署时延、成功率和失败案例。

### 第三优先级

1. 把当前文档中的关键部分同步进 `docs/source/`。
2. 在 README 或项目主页加一个“项目亮点”小节。
3. 做一页式项目介绍图，用于投递和汇报。

---

## 8. 对外最稳妥的项目总结

最后给你一个可以直接在汇报或面试里复用的总结版本。

> 这个项目的核心，不只是给 SO-101 接了两个触觉传感器，而是在 LeRobot 这个真实开源具身智能框架上，把多模态硬件接入、遥操作同步采集和 VLA 真机部署完整打通了。工程上，我扩展了机器人抽象和数据采集链路；模型侧，我实现了 SmolVLA 与触觉 token 融合部署；系统侧，我处理了 Linux 下的串口、相机、校准、频率对齐和安全控制问题。它体现的是一个多模态具身系统从框架扩展到真机落地的完整能力链。