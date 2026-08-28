# Ego-loong Postprocess

这个仓库用于把 Ego-loong / sampler ROS2 bag 数据后处理成手套轨迹、相机/手部 3D 可视化和审核网页。

当前主流程入口是：

```bash
scripts/run_sampler_bag_to_glove_trajectory.sh
```

主流程大致为：

```text
ROS2 bag RGBD/hand_frame 提取
-> RTAB-Map 相机位姿导出并对齐到 RGB 帧
-> LocateAnything 检测手套 bbox
-> bbox 稳定跟踪
-> HaMeR 从 bbox 估计 21 个视觉手关键点
-> aligned depth 做 wrist/root 深度修正
-> 视觉手 + /hand_frame 手套数据融合
-> glove FK + 视觉骨长标定 + wrist root 轻量时间滤波
-> 生成轨迹 JSON/视频
-> 生成自包含 review web / 采集风格 web
-> 可选精简输出包
```

## 仓库结构

```text
cfg/                         配置文件
scripts/                     主流程和辅助脚本
preprocess/                  各处理阶段的 Python 实现
utils/                       通用 IO/媒体/可视化工具
hand_msg_ws/                 ROS2 hand_msg 工作空间
hamer/                       HaMeR 代码和模型相关文件
models--nvidia--LocateAnything-3B/  LocateAnything 模型目录
postprocess_data/            后处理输出目录
datatsets/                   原始 bag 数据目录
requirements.txt             Python 依赖说明
setup.sh                     环境安装/检查脚本
```

## 环境要求

需要把当前 Ubuntu 24.04、ROS 2 Jazzy、CUDA/Python 环境和模型固化成完整
Docker 镜像时，参见 [docker/README.md](docker/README.md)。镜像保留宿主机
NVIDIA 驱动边界，并支持单 session、共享 calibration 批次和一帧 smoke test。

当前管线使用三个 Python/ROS 环境：

```text
/usr/bin/python3                         ROS2 bag 提取
/home/lenovo/miniconda3/envs/locate_anything/bin/python  LocateAnything
/home/lenovo/miniconda3/envs/hamer/bin/python            HaMeR / FK / 可视化
```

ROS 侧默认使用：

```bash
/opt/ros/jazzy/setup.bash
hand_msg_ws/install/setup.bash
```

可以用 `setup.sh` 安装或检查环境：

```bash
bash setup.sh
```

默认安装各自的精确 lock；仅在有意测试新版依赖时，才使用 `USE_LOCKS=0 bash setup.sh` 回到宽松版本约束。

只检查已有环境，不安装依赖：

```bash
SKIP_INSTALL=1 bash setup.sh
```

`setup.sh` 会检查手配置和 Retarget FK 模块。路径可通过 `BASE_HAND_CONFIG`、`RETARGET_ROOT`、`RESOLVE_DRIVER` 覆盖；`RESOLVE_DRIVER` 仅供旧 `/glove` bag 现场求解 state27，已经包含 `/hand_frame` 的新 bag 不使用它。

一帧端到端 smoke test 必须显式提供数据，避免误用失效的示例路径：

```bash
RUN_SMOKE=1 SMOKE_BAG_SESSION=/path/to/session-or-data bash setup.sh
```

### 依赖锁和环境快照

`requirements.txt` 只描述宽松的安装需求；以下文件记录当前已验证机器上的精确运行环境，三个运行环境不会混装：

```text
requirements-locate.lock      LocateAnything Conda 环境的精确 Python 包版本
requirements-hamer.lock       HaMeR Conda 环境的精确 Python 包版本
requirements-ros-system.txt   ROS2/系统 Python 的精确 apt 包版本
environment-info.json         OS、GPU、驱动、CUDA、PyTorch、ROS、模型 SHA-256
```

Python lock 可分别用于对应的 Python 3.10 Conda 环境。PyTorch wheel 为 CUDA 13.0 构建，重建时需要使用兼容的 PyTorch wheel 源：

```bash
conda create -n locate_anything python=3.10.20 pip
conda run -n locate_anything python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements-locate.lock

conda create -n hamer python=3.10.20 pip
conda run -n hamer python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements-hamer.lock
```

`requirements-ros-system.txt` 是 apt/ROS 审计清单，不建议交给 pip；精确 apt 版本能否重装取决于对应 Ubuntu/ROS 软件源快照仍然可用。模型文件不放入 Git，`environment-info.json` 用路径、大小和 SHA-256 验证其版本。

环境或模型更新后，在仓库根目录重新生成全部快照：

```bash
python3 scripts/capture_environment_locks.py
```

只做快速环境检查、暂时不读取数 GB 模型时可加 `--skip_model_hashes`；该模式生成的模型哈希为空，不能作为正式发布快照。

## 模型路径

主流程默认使用以下模型/资源：

```text
LocateAnything:
/home/lenovo/Ego-loong-postprocess/models--nvidia--LocateAnything-3B/resolved

HaMeR checkpoint:
/home/lenovo/Ego-loong-postprocess/hamer/_DATA/hamer_ckpts/checkpoints/new_hamer_weights.ckpt

MANO:
/home/lenovo/Ego-loong-postprocess/hamer/_DATA/data/mano
/home/lenovo/Ego-loong-postprocess/hamer/_DATA/data/mano_mean_params.npz
```

左手数据推荐使用：

```bash
HAMER_HANDEDNESS=all_left
VISUAL_SIDE=hand_l
GLOVE_SIDE=left
```

## 输入数据格式

主流程的输入是一个 bag session 目录，例如：

```text
/home/lenovo/Ego-loong-postprocess/datatsets/bag_0703/2026-07-03T0156.38
```

`BAG_SESSION` 可以传单个 session 根目录，也可以直接传其 `data` 目录。脚本会先解析出统一的数据根目录 `BAG_DATA_DIR`，再寻找 bag：

```text
${BAG_SESSION}/bag
${BAG_SESSION}/data/bag
```

批次格式的数据包由多个 session 共享同一份标定：

```text
<batch>/
├── calibrations/
│   ├── calibration_manifest.json
│   ├── hand_calibration.txt
│   ├── camera_extrinsics.json
│   ├── coordinate_calibration.txt
│   ├── installation_calibration.txt
│   ├── calibration_video/bag
│   └── pinch_calibration/bag
└── <session>/
    └── data/
        ├── bag
        └── map/rtabmap.db
```

传入 `<batch>/<session>` 或 `<batch>/<session>/data` 时，主流程会自动发现相邻的 `<batch>/calibrations`，并让各 session 共享 `hand_calibration.txt`、`camera_extrinsics.json` 和 `calibration_video`。旧格式的 `data/calibration/handcal.txt`、`oak_extrinsics.json` 和 `calib_video_*` 仍然兼容。

批次格式的 RTAB-Map 数据库属于各 session，默认读取：

```text
${BAG_DATA_DIR}/map/rtabmap.db
```

旧格式继续回退到 `${BAG_DATA_DIR}/calibration/rtabmap.db`。

当前默认 `USE_RTABMAP_POSE=1`，也就是后续相机位姿来自 `rtabmap.db` 导出的 Poses，并插值到 RGB 帧。

## 根据一个数据包批量运行后处理

假设收到的数据包路径是：

```text
/data/test_data/2026-08-21T1225.50_20/
```

运行前确认它包含公共标定和至少一个一级 session：

```text
/data/test_data/2026-08-21T1225.50_20/
├── calibrations/
├── 2026-08-21T1226.05/data/bag/
├── 2026-08-21T1226.36/data/bag/
└── 2026-08-21T1227.01/data/bag/
```

进入仓库后，推荐使用下面的命令批量运行完整后处理、质量检查和机器人仿真。某个 session 失败后会继续处理剩余数据：

```bash
cd /home/lenovo/Ego-loong-postprocess

BATCH_ROOT=/data/test_data/2026-08-21T1225.50_20 \
CONTINUE_ON_ERROR=1 \
RUN_QUALITY_CHECK=1 \
RUN_ROBOT_SIMULATION=1 \
COMPACT_OUTPUTS=0 \
scripts/run_sampler_batch_to_glove_trajectory.sh
```

`OVERWRITE` 默认是 `0`。首次运行时所有阶段都没有缓存，因此仍会完整处理；同一条命令再次执行时会自动复用有效缓存，只重算输入、参数、配置或代码发生变化的阶段。这也是中断后继续运行的推荐方式：

```bash
BATCH_ROOT=/data/test_data/2026-08-21T1225.50_20 \
CONTINUE_ON_ERROR=1 \
scripts/run_sampler_batch_to_glove_trajectory.sh
```

只有确认需要忽略所有阶段缓存、从头强制重算整个数据包时才设置：

```bash
BATCH_ROOT=/data/test_data/2026-08-21T1225.50_20 \
CONTINUE_ON_ERROR=1 \
OVERWRITE=1 \
scripts/run_sampler_batch_to_glove_trajectory.sh
```

若当前机器没有 MuJoCo 环境，或者本轮只测试普通后处理，可关闭机器人仿真：

```bash
BATCH_ROOT=/data/test_data/2026-08-21T1225.50_20 \
CONTINUE_ON_ERROR=1 \
RUN_ROBOT_SIMULATION=0 \
scripts/run_sampler_batch_to_glove_trajectory.sh
```

正式大规模运行前，可以限制普通后处理和仿真帧数做快速冒烟测试：

```bash
BATCH_ROOT=/data/test_data/2026-08-21T1225.50_20 \
CONTINUE_ON_ERROR=1 \
MAX_FRAMES=60 \
SIMULATION_MAX_FRAMES=60 \
scripts/run_sampler_batch_to_glove_trajectory.sh
```

批次脚本按名称顺序处理所有包含 `data/bag` 的一级子目录，不递归处理更深层目录。共享 `calibration_video` 默认只处理一次，产物和缓存保存在 `postprocess_data/_batch_calibration/<batch-name>/`，后续 session 直接复用。默认遇到失败即停止；大规模评测应设置 `CONTINUE_ON_ERROR=1`。

以上示例数据包的 session 输出目录形如：

```text
postprocess_data/2026-08-21T1225.50_20_20260821T122605/
postprocess_data/2026-08-21T1225.50_20_20260821T122636/
```

每个已启动的 session 即使中途失败，也会尽量生成：

```text
postprocess_data/<session-name>/outputs/summary.json
```

无论批次是否存在失败，批处理结束或停止前都会聚合已经处理的 session，并写入：

```text
postprocess_data/batch_summaries/<batch-name>_summary.json
```

上面示例对应：

```text
postprocess_data/batch_summaries/2026-08-21T1225.50_20_summary.json
```

可通过 `BATCH_SUMMARY=/path/report.json` 修改位置。报告包含总体通过率、质量门禁与仿真判定分布、失败类别、各阶段完成数、总帧数/总时长，以及感知、标定、腕部残差、RTAB-Map 和仿真安全指标的 min/mean/P50/P95/max。

只处理数据包中的一个 session 时，不使用批次脚本，直接运行主管道；`BATCH_ROOT` 指向公共标定所在的数据包根目录：

```bash
BATCH_ROOT=/data/test_data/2026-08-21T1225.50_20 \
BAG_SESSION=/data/test_data/2026-08-21T1225.50_20/2026-08-21T1226.05 \
RUN_QUALITY_CHECK=1 \
RUN_ROBOT_SIMULATION=1 \
scripts/run_sampler_bag_to_glove_trajectory.sh
```

### 旧格式单 session 命令

典型左手数据处理命令：

```bash
BAG_SESSION=/home/lenovo/Ego-loong-postprocess/datatsets/bag_0703/2026-07-03T0156.38 HAMER_HANDEDNESS=all_left VISUAL_SIDE=hand_l GLOVE_SIDE=left USE_RTABMAP_POSE=1 OVERWRITE=1 scripts/run_sampler_bag_to_glove_trajectory.sh
```

如果不指定 `SESSION_NAME`，脚本会根据 `BAG_SESSION` 自动生成输出目录名。例如：

```text
BAG_SESSION=/.../bag_0703/2026-07-03T0156.38
-> postprocess_data/bag_0703_20260703T015638
```

也可以手动指定输出目录名：

```bash
BAG_SESSION=/home/lenovo/Ego-loong-postprocess/datatsets/bag_0703/2026-07-03T0156.38 SESSION_NAME=bag_0703_20260703T015638 HAMER_HANDEDNESS=all_left VISUAL_SIDE=hand_l GLOVE_SIDE=left USE_RTABMAP_POSE=1 OVERWRITE=1 scripts/run_sampler_bag_to_glove_trajectory.sh
```

## 主流程参数

常用参数通过环境变量传入：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `BAG_SESSION` | sampler 示例路径 | 输入采集根目录或其 `data` 目录，两种写法等价 |
| `BATCH_ROOT` | 自动识别 | 共享 `calibrations` 的批次根；单 session 运行时通常无需指定 |
| `CALIBRATION_DIR` | 自动识别 | 新格式为 `${BATCH_ROOT}/calibrations`，旧格式回退到 session 内 `calibration` |
| `HAND_CALIBRATION_FILE` | 自动识别 | 新格式 `hand_calibration.txt`，旧格式 `handcal.txt` |
| `CAMERA_EXTRINSICS_FILE` | 自动识别 | 新格式 `camera_extrinsics.json`，旧格式 `oak_extrinsics.json` |
| `SESSION_NAME` | 自动生成 | 输出 session 名称 |
| `SESSION` | `postprocess_data/${SESSION_NAME}` | 完整输出目录 |
| `BAG_DIR` | 自动识别 | ROS2 bag 目录 |
| `USE_RTABMAP_POSE` | `1` | 是否使用 `rtabmap.db` 相机位姿 |
| `RTABMAP_DB` | 自动识别 | 优先 `${BAG_DATA_DIR}/map/rtabmap.db`，旧格式回退到 calibration 目录 |
| `RTABMAP_MAX_INTERP_GAP_SEC` | `0.25` | 允许插值跨越的最大 RTAB-Map 节点时间间隔 |
| `RTABMAP_RENDER_VIDEOS` | `0` | 是否生成两段 RTAB-Map 轨迹预览 MP4；默认关闭以加快第 2 阶段，设为 `1` 可恢复 |
| `EXTRACT_IMAGE_WRITE_WORKERS` | `8` | ROS bag 解包时并行进行 PNG 编码和写盘；设为 `1` 可回退串行模式 |
| `LOCATE_DTYPE` | `bf16` | LocateAnything 推理精度；Blackwell 默认使用 BF16，较 FP32 更快且显存占用更低 |
| `LOCATE_ATTN_IMPLEMENTATION` | `sdpa` | LocateAnything 注意力后端；默认显式使用 PyTorch SDPA，避免尝试未安装的 Flash/Magi 后再回退 |
| `LOCATE_BATCH_SIZE` | `16` | LocateAnything 批推理大小；仍输出逐帧检测JSON供 stable bbox 使用，设为 `1` 可回退旧逐帧路径 |
| `HAMER_BATCH_SIZE` | `32` | HaMeR 每次统一推理的手部 crop 数；稳定框带显式左右手标签时跨帧批处理，旧 track 数据缺标签时自动回退为 `1` |
| `HAMER_HANDEDNESS` | `track` | HaMeR 手性策略；根据稳定框的左右手轨迹处理双手 |
| `VISUAL_SIDE` | `hand_l` | 视觉手侧 |
| `GLOVE_SIDE` | `left` | 手套数据侧 |
| `IMAGE_LEFT_PHYSICAL_SIDE` | `left` | 画面左侧手框对应的物理手；当前采集相机使用 `left`，镜像或其他安装方式可设为 `right` |
| `ESTIMATE_SCALE` | `1` | 标定时估计并应用 glove FK 到视觉手的尺度；设为 `0` 仅用于固定尺度 1.0 的对照实验 |
| `FPS` | 自动 | 主流程忽略固定覆盖值，始终从 `rgb_stamp_ns` 估计真实 RGB 标称帧率 |
| `TIME_FILTER_REFERENCE_FPS` | `30` | 旧版每帧 alpha、最大步长和确认帧数的参考语义；实际计算按真实 `dt` 换算 |
| `OVERWRITE` | `0` | `1` 时忽略阶段缓存并强制重算 |
| `CONFIG_ONLY` | `0` | `1` 时只解析并打印输入路径，不创建输出或运行处理阶段 |
| `PARALLEL_HANDS` | `1` | 左右手 fusion、平滑、FK、标定应用和腕部追踪并行执行；设为 `0` 可串行调试 |
| `RENDER_DEBUG_VIDEOS` | `0` | 是否生成 bbox、HaMeR、单手 2D、overlay 和单手 3D 调试视频 |
| `RENDER_STABLE_BBOX_VIDEO` | `1` | 是否保留 Locate stable bbox 视频 |
| `RENDER_HAMER_SMOOTH_VIDEO` | `1` | 是否将左右手平滑 HaMeR 2D 结果合并渲染为一个视频 |
| `RENDER_FINAL_VIDEO` | `0` | 是否额外生成最终双手 3D 视频；默认通过 review web 查看轨迹 |
| `MAX_FRAMES` | 空 | 限制处理帧数，调试用 |
| `RUN_QUALITY_CHECK` | `1` | 主流程结束前运行硬性质量门禁；失败时脚本返回非零状态 |
| `QUALITY_MAX_WRIST_RESIDUAL_P95_M` | `0.070` | 腕部跟踪残差 P95 上限，单位米 |
| `COMPACT_OUTPUTS` | `0` | 是否在质量检查通过后精简输出包；设为 `1` 时要求 `RUN_QUALITY_CHECK=1` |

深度和轨迹平滑相关参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `DEPTH_METHOD` | `robust` | wrist/root 深度估计方式 |
| `DEPTH_ROBUST_INDICES` | `0,5,9,13,17` | 用 wrist 和多个 MCP 点做鲁棒深度估计 |
| `DEPTH_RADIUS` | `8` | 深度局部搜索半径 |
| `DEPTH_ROBUST_INLIER_M` | `0.055` | 深度 inlier 阈值 |
| `VISUAL_2D_SMOOTH_ALPHA` | `0.35` | 2D overlay 平滑系数 |
| `HAMER_BRANCH_JUMP_THRESHOLD_DEG` | `75` | 相邻 HaMeR 手掌法向超过此角度时视为姿态分支跳变候选 |
| `HAMER_BRANCH_BRIDGE_GAP_FRAMES` | `3` | 合并异常分支内部短暂返回正常分支的最大帧数 |
| `HAMER_BRANCH_MAX_REJECT_FRAMES` | `60` | 短期跳入后又返回原分支时，允许拒绝并用前后 HaMeR 姿态插值的最大帧数 |
| `HAMER_GLOBAL_MAX_TRANSLATION_STEP_M` | `0.02` | world-frame wrist translation 的单帧速度软上限，单位米 |
| `HAMER_GLOBAL_W_TRANSLATION_SPEED` | `4000` | 超过平移软上限后的惩罚权重 |
| `HAMER_GLOBAL_W_TRANSLATION_JERK` | `120` | wrist translation 三阶差分（jerk）平滑权重 |
| `HAMER_GLOBAL_TRANSLATION_OUTLIER_THRESHOLD_M` | `0.025` | root 与局部匀速预测偏差超过此值时标记异常并降低观测权重 |
| `HAMER_GLOBAL_MIN_ROOT_OBSERVATION_WEIGHT` | `0.1` | 异常 root 观测的最小保留权重，避免直接删除真实快速动作 |
| `MOTION_FILTER_MIN_TRACK_LENGTH` | `15` | 每只手连续有效轨迹的最少帧数 |
| `MOTION_FILTER_MIN_HAND_VALID_RATIO` | `0.90` | 每只手有效帧比例的 episode 级下限 |
| `MOTION_FILTER_MAX_TERMINAL_INVALID_FRAMES` | `5` | 末尾连续无效帧超过该值时启动尾段清洗 |
| `MOTION_FILTER_TERMINAL_TRIM_LOOKBACK_FRAMES` | `30` | 从末尾失效点向前寻找快速运动起点的窗口长度 |
| `MOTION_FILTER_TERMINAL_TRIM_PRE_ROLL_FRAMES` | `15` | 找到末尾快速运动起点后，再向前保守裁掉的帧数 |
| `MOTION_FILTER_TERMINAL_FAST_TRANSLATION_M` | `0.012` | 尾段快速运动起点的 wrist 平移阈值，单位米/帧 |
| `MOTION_FILTER_TERMINAL_FAST_ROTATION_DEG` | `5.0` | 尾段快速运动起点的 wrist 旋转阈值，单位度/帧 |
| `MOTION_FILTER_SPIKE_SIGMA_MULTIPLIER` | `3.0` | 基于二阶差分的鲁棒尖峰阈值倍数 |
| `MOTION_FILTER_MAX_SPIKE_FRAME_FRACTION` | `0.05` | 非诊断信号允许的最大尖峰帧比例 |
| `MOTION_FILTER_STATIC_ENERGY_THRESHOLD_M` | `0.002` | 单帧双手静止运动能量阈值；静止仅标记，不自动拒绝 |
| `MOTION_FILTER_STATIC_EPISODE_FRACTION` | `0.90` | 判为静止候选所需的双手静止帧比例 |
| `WRIST_TRACK_ALPHA` | `0.25` | wrist 3D/root 时间滤波系数，由子流程读取 |
| `WRIST_TRACK_MAX_STEP_M` | `0.007` | wrist 3D/root 单帧最大步长，由子流程读取 |
| `REVIEW_HAND_DISPLAY_ROTATE_DEG` | `45` | 网页 3D 手部显示旋转参数；当前默认开启 `wrist -> middle_mcp` 竖直对齐时主要用于兼容旧命令 |
| `REVIEW_RGB_WORKERS` | `8` | review web 并行导出 RGB JPEG 的线程数；轨迹和触觉由浏览器 Canvas 实时绘制，不再生成逐帧 JPEG |
| `RUN_ROBOT_SIMULATION` | `1` | 自动运行双 UR5e replay、Mink 全轨迹安全求解和 MuJoCo 视频渲染；设为 `0` 可跳过 |
| `SIMULATION_PYTHON` | `.venv-mujoco/bin/python` | 安装了 MuJoCo/Mink 的 Python；缺失时先运行 `simulation/dual_ur5e/setup.sh` |
| `SIMULATION_CONFIG` | 批量验证推荐配置 | 机器人布局、桌面、碰撞距离、速度和加速度等仿真配置 |
| `SIMULATION_CAMERA_CONFIG` | `viewer_camera.json` | 自动渲染视频使用的 MuJoCo 相机配置 |
| `SIMULATION_GL` | `egl` | 无界面渲染后端；无 EGL 的机器可按环境改为 `osmesa` |
| `SIMULATION_MAX_FRAMES` | `0` | 仿真帧数上限；`0` 表示完整轨迹，仅建议在冒烟测试时设置非零值 |
| `REVIEW_SIMULATION_VIDEO` | 空 | 可选外部 MuJoCo MP4；非空时覆盖网页所用的自动仿真视频 |
| `REVIEW_SIMULATION_SUMMARY` | 空 | 可选外部 summary JSON；非空时覆盖网页所用的自动安全摘要 |
| `REVIEW_SIMULATION_NPZ` | 空 | 可选外部 NPZ；非空时覆盖网页所用的自动同步轨迹 |

## 输出目录

完整输出默认在：

```text
postprocess_data/${SESSION_NAME}/
```

主要目录：

```text
preprocess/                                      只读的 ROS2 bag 提取结果：逐帧 RGB/depth、hand_frame、tf
rtabmap_pose/camera_frames/                      应用 RTAB-Map 后的逐帧相机 JSON，不回写 preprocess/all_data
locateanything_white_glove_with_imu/             LocateAnything 原始 bbox
locateanything_white_glove_with_imu_stable/      稳定 bbox
hamer_from_stable_locateanything_.../per_frame/  HaMeR 逐帧派生 JSON
depth_correct_hamer_force_right/per_frame/       深度修正后的逐帧派生 JSON
fusion_input_force_right_depthroot/              视觉+手套融合输入
glove_fk21_visual_bones_smooth_solve045/         glove FK、标定、wrist tracking 和轨迹
outputs/                                         用户主要查看和交付结果
```

`outputs/` 内默认的重要结果：

```text
outputs/videos/02_stable_bbox.mp4
outputs/videos/04_dual_visual_21kpts_2d_smooth.mp4

outputs/data/trajectory_wristroot_track_cameraoptical.jsonl
outputs/data/locate_bboxes.json
outputs/data/stable_bboxes.json

outputs/summaries/*.json
outputs/web/index.html
outputs/summary.json
outputs/simulation/*_source_dual_ur5e.npz
outputs/simulation/*_mink_dual_ur5e.npz
outputs/simulation/*_mink_dual_ur5e_summary.json
outputs/simulation/*_mink_dual_ur5e.mp4
```

`outputs/summary.json` 是单个 session 的统一机器可读报告。主管道通过退出钩子生成它，因此正常完成、质量门禁失败或中途异常都会尽量留下当前状态。内容包括：

- 总体判定和固定失败分类；
- 轨迹帧数、时长、有效率、静止候选和尾段裁剪；
- 左右手匹配率、视觉覆盖率、深度应用率、标定误差和腕部残差；
- RTAB-Map 覆盖率、插值间隔和缺失位姿；
- Mink/安全审计、最小间距、误差、恢复帧和失败帧；
- 网页与关键产物状态，以及各缓存阶段完成情况；
- 完整 `quality_report` 指标，便于未来增加统计项而不修改历史格式。

也可以对已有 session 手动重建并聚合：

```bash
scripts/generate_session_summary.py \
  --session postprocess_data/SESSION \
  --quality_requested

scripts/aggregate_postprocess_summaries.py \
  --root postprocess_data \
  --output postprocess_data/batch_summaries/all_sessions_summary.json
```

默认仍保留左右手各自的 fusion、平滑、FK、标定和腕部追踪数据及 summary，只生成 stable bbox 和双手 HaMeR 2D 平滑视频，不生成双手 3D MP4；3D 轨迹通过 `outputs/web/index.html` 查看。需要恢复旧版单手/中间调试视频时设置 `RENDER_DEBUG_VIDEOS=1`，需要额外生成双手 3D MP4 时设置 `RENDER_FINAL_VIDEO=1`。

### 稳定手掌动作坐标系

在构建动作坐标系前，`OptimizeHamerGlobalTrajectory.py` 固定 glove FK 的 wrist-relative 局部手形和现有尺度，仅在共享 world frame 中平滑每帧 wrist translation 与 palm `SO(3)` orientation。当前骨长标定尚未可靠，因此不使用 FK→RGB 重投影，也不允许优化器改变手指局部姿态或尺度。平移部分对偏离局部匀速预测的 root 观测进行鲁棒降权，并联合使用二阶平滑、jerk 正则和 2 cm/frame 速度软约束；旋转部分保留 glove orientation 观测、速度/二阶平滑和超过 41°/frame 的跳变惩罚。每帧的 root 权重、预测残差和异常标记保存在 `optimized_trajectory` 中。

最终轨迹中每只手的 `hands.<side>.palm_frame` 使用统一的右手系 `ego_loong_glove_fk_palm_v5`，直接继承 `optimized_trajectory` 的全局时序优化 wrist root 与 palm rotation。`+x` 由 wrist 指向四个 MCP 的均值，`+z` 为 glove/FK 手形的手背法向，`+y = +z × +x`。最终局部手指姿态继续来自 FK。HaMeR 掌方向不进入最终 policy action，仅用于视觉诊断、深度定位和异常过滤。

该字段同时提供 camera/world 下的 wrist 和 palm pose、四元数、旋转矩阵、连续 6D rotation，以及未平滑/平滑后的帧间旋转和角速度。最终 21 点始终来自 FK 局部姿态；HaMeR+depth 只通过上游 wrist tracker 提供根位置。对应统计写入：

```text
outputs/summaries/stable_palm_frame_summary.json
```

### 轨迹质量过滤和末尾坏帧清洗

`FilterTrajectoryQuality.py` 在 episode 和 frame 两级检查 camera、wrist、finger 的平移、旋转、四元数和二阶差分尖峰；当前不启用时长过滤，也不启用 chunk 过滤。持续的真实快速动作不会仅因速度较大被当作尖峰，尖峰判断使用运动量的二阶差分。原始 wrist 信号用于诊断和物理硬阈值检查，最终是否可用主要依据全局优化后的 wrist/finger 轨迹。

若某只手在 episode 末尾连续失效超过 5 帧，过滤器会向前最多 30 帧寻找 wrist 快速运动的起点，再从该起点额外向前回退 15 帧作为实际裁剪点，以清除快速动作开始前后的不稳定过渡。轨迹 JSONL、review web 和后续渲染都使用裁剪后的帧范围；ROS bag、RGB/depth 和其他原始采集文件不会被删除。裁剪起止帧、原因和原始/输出帧数写入：

```text
outputs/summaries/motion_filter_summary.json
```

静止运动能量按每只手 `sqrt(wrist_translation^2 + (0.1 * wrist_rotation_rad)^2 + fingertip_local_rms^2)` 计算；只有左右手同时低于 `0.002 m/frame` 才将该帧视为静止。整段双手静止比例达到 90% 时仅标记为 `static_candidate`，默认不会自动判废。

## 可视化网页

主流程最后会自动生成 review web：

```text
outputs/web/index.html
```

网页包含：

```text
左上：RGB 帧播放；提供仿真视频时与 Robot/MuJoCo 画面左右并排
左下：Head + hand 3D 轨迹播放
右侧：数据基本信息、wrist x/y/z 曲线和当前 wrist 坐标
底部：播放/暂停、重置、时间轴
```

网页使用 RGB 帧序列播放，不依赖浏览器 MP4/H.264 解码。3D 轨迹和触觉热力图由浏览器根据嵌入数据实时绘制，不再生成逐帧图片。生成后的网页包含：

```text
outputs/web/rgb_frames/*.jpg
outputs/web/tactile_hand.png
```

### 自动机器人仿真与 RGB 对比

主管道默认在收集最终轨迹后自动执行以下步骤：

1. 将人体双腕轨迹映射为双 UR5e 初始机器人轨迹。
2. 使用 Mink 对完整重定时轨迹做碰撞约束求解和恢复尝试。
3. 检查机器人自碰撞、双臂互碰、机架、桌面和安全平面的最小间距。
4. 从最终 Mink 轨迹渲染 MP4，并自动接入 review web。

仿真结果为 `FAIL` 时仍保留 NPZ、summary、视频和网页，便于大规模测试时
统计失败率、定位失败帧和回看动作；`PASS` 也不代表允许直接下发真实机器人，
因为负载、抓取物、线缆和未建模环境仍不在当前检查范围内。每个 session 的
仿真阶段有独立内容缓存，输入轨迹、配置、模型资产或仿真代码未变化时会跳过。

仅运行普通后处理、不需要仿真时：

```bash
RUN_ROBOT_SIMULATION=0 scripts/run_sampler_bag_to_glove_trajectory.sh
```

下面的显式参数方式仍可用于将已有或外部仿真结果接入网页。

生成网页时提供已有的 MuJoCo 视频、summary 和 NPZ，顶部 RGB 区域会自动
左右二分显示 RGB 与 Robot；轨迹、触觉、右侧信息和底部控制布局保持不变：

```bash
/home/lenovo/miniconda3/envs/hamer/bin/python scripts/generate_review_web.py \
  --session /path/to/postprocess_session \
  --simulation_video simulation/dual_ur5e/outputs/SESSION_dual_ur5e.mp4 \
  --simulation_summary simulation/dual_ur5e/outputs/SESSION_dual_ur5e_summary.json \
  --simulation_npz simulation/dual_ur5e/outputs/SESSION_dual_ur5e.npz
```

网页会把视频复制为：

```text
outputs/web/robot_simulation.mp4
```

播放、暂停、拖动和重置使用同一条底部时间轴。普通 replay NPZ 使用
`pre_retime_times_sec -> retimed_path_times_sec` 做分段线性精确同步；旧版或
二次 Mink 产物缺少成对时间节点时按总时长同步，并在生成结果的
`simulation.sync_mode` 中标记为 `duration_scaled`。Robot 画面底部同时显示
仿真 `PASS/FAIL`、最小环境间距和重定时后时长。

若 `simulation/dual_ur5e/outputs/` 中存在以 session 名开头的标准
`*_dual_ur5e.mp4`，脚本会自动发现；非标准文件名使用上面的显式参数。
没有视频时页面保持原来的单 RGB 布局。需要关闭自动发现时添加
`--no_simulation_video`。

### 采集风格网页

如果要生成不覆盖旧网页的采集风格页面，使用 `--output_subdir web_collect`：

```bash
/home/lenovo/miniconda3/envs/hamer/bin/python scripts/generate_review_web.py --session /home/lenovo/Ego-loong-postprocess/postprocess_data/07_04_2_20260704T080233 --fps 30 --output_subdir web_collect --hand_display_rotate_deg 45
```

输出：

```text
outputs/web_collect/index.html
outputs/web_collect/rgb_frames/*.jpg
outputs/web_collect/tactile_hand.png
```

`web_collect` 右侧的采集时长、图表、拖动和播放均来自轨迹中的 `timestamp.rgb_stamp_ns`。`--fps` 只在旧轨迹缺少时间戳时作为回退值，不再用帧数除以固定 FPS 计算时长。

如果 session 已经 compact，`preprocess/all_data` 不存在，脚本会尝试复用 `outputs/web/rgb_frames` 生成新的 `web_collect/rgb_frames`。

### 网页 3D 显示参数

网页 3D 面板只影响显示，不修改轨迹 JSON。当前关键参数：

```text
hand_display_scale = 1.50
scene_display_scale = 1.0
align_middle_vertical = true
```

`align_middle_vertical=true` 时，每帧会把 `wrist -> middle_mcp` 显示为竖直方向；同时 wrist 轨迹主运动方向会自动对齐到屏幕横向。

视角方向 `view_d` 由 [scripts/generate_review_web.py](/home/lenovo/Ego-loong-postprocess/scripts/generate_review_web.py) 里的 `oblique_dir` 间接控制：

```python
top = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
oblique_dir = -top * 0.80 - cam_axes[:, 2].astype(np.float64) * 0.18
```

常用调整：

```python
# 更接近正下方俯视，侧向更少
oblique_dir = -top * 0.98 - cam_axes[:, 2].astype(np.float64) * 0.08

# 更斜，侧向更明显
oblique_dir = -top * 0.45 - cam_axes[:, 2].astype(np.float64) * 0.75

# 从上方看，而不是下方看
oblique_dir = top * 0.92 - cam_axes[:, 2].astype(np.float64) * 0.18
```

注意：因为网页默认会做“轨迹横向对齐”和“中指竖直对齐”，轻微修改 `oblique_dir` 时肉眼变化可能不明显。需要观察纯视角变化时，可以临时加 `--no_align_middle_vertical`。

单独重新生成默认网页：

```bash
/home/lenovo/miniconda3/envs/hamer/bin/python scripts/generate_review_web.py --session /home/lenovo/Ego-loong-postprocess/postprocess_data/bag_0703_20260703T015638 --hand_display_rotate_deg 45
```

## 精简输出包

默认主流程保留全部中间产物，便于质量复核、断点续跑和定位问题。

确认需要生成精简交付包时，运行时显式设置：

```bash
COMPACT_OUTPUTS=1
```

主流程会先运行 `scripts/validate_pipeline_quality.py`。只有必需文件、帧数和手部匹配、深度修正、标定误差、腕部残差等检查全部通过，才会调用精简脚本；失败时保留完整目录并以非零状态退出。检查结果写入 `outputs/quality_report.json`。

默认质量门限为：手套数据匹配率不低于 95%，视觉手存在率不低于 90%，深度修正成功率不低于 85%，标定误差中位数不高于 3 cm、P95 不高于 6 cm，腕部跟踪残差 P95 不高于 4 cm。使用 RTAB-Map 时还要求轨迹生成和应用覆盖率均为 100%、缺失帧为 0，实际插值 gap 不超过 0.25 秒。可通过对应的 `QUALITY_*` 环境变量覆盖。

精简运行示例：

```bash
COMPACT_OUTPUTS=1 BAG_SESSION=/path/to/session HAMER_HANDEDNESS=all_left VISUAL_SIDE=hand_l GLOVE_SIDE=left USE_RTABMAP_POSE=1 OVERWRITE=1 scripts/run_sampler_bag_to_glove_trajectory.sh
```

精简逻辑由以下脚本实现：

```bash
scripts/compact_postprocess_session.py
```

它会保留自包含网页、关键轨迹、少量最终视频和 summary，删除可再生成的中间目录和重复大文件。

先 dry-run 查看会删除什么，不实际删除：

```bash
/home/lenovo/miniconda3/envs/hamer/bin/python scripts/compact_postprocess_session.py --session /home/lenovo/Ego-loong-postprocess/postprocess_data/bag_0703_20260703T015638 --dry_run
```

实际精简某个 session：

```bash
/home/lenovo/miniconda3/envs/hamer/bin/python scripts/compact_postprocess_session.py --session /home/lenovo/Ego-loong-postprocess/postprocess_data/bag_0703_20260703T015638
```

如果需要保留原始完整包并生成一个精简对比包，建议先复制 session，再对复制目录执行精简。

## 阶段缓存

主入口为每个阶段在 session 的 `.pipeline_cache/` 下写入 manifest。manifest 包含输入内容哈希、阶段参数、相关代码文件哈希、输出完整性和完成时间。下游依赖 manifest 的稳定语义指纹，不受 `completed_at` 或 JSON 排版变化影响。提取、RTAB 相机元数据、HaMeR 和深度修正分别拥有独立输出目录，后续阶段不会修改提取目录。只有内容、参数、代码或要求的输出发生变化时才重新计算。

网页阶段显式依赖收集阶段 manifest 和最终轨迹文件，因此轨迹变化后不会静默复用旧网页。需要无条件重算全部阶段时使用 `OVERWRITE=1`。

## RTAB-Map 和深度说明

当前管线中：

- 相机位姿来自 `rtabmap.db` 导出的 Poses，并插值到 RGB 帧。
- RTAB-Map 采用强制严格模式：范围外帧、缺失位姿或超过 0.25 秒的定位断流都会使流程失败，不会保留单帧 odom 位姿继续处理。需要回退时设置 `USE_RTABMAP_POSE=0`，整段统一使用原始位姿。
- 位姿应用前会先检查全部帧；重复应用时保留最初的 `c2w_before_rtabmap_pose`，不会用上一次 RTAB-Map 位姿覆盖原始 odom 位姿。
- OAK aligned depth 用于 HaMeR wrist/root 的深度修正。
- 提取器在 registered depth topic 存在时只使用 registered depth 及其 CameraInfo；仅在 registered topic 完全不存在时才回退 raw depth，禁止逐帧混用两种深度源。registered depth 已位于 RGB 坐标系，不使用 depth-to-RGB 外参。
- wrist/root 深度不是直接用单点，而是默认使用 wrist 和多个 MCP 点做鲁棒估计。
- RTAB-Map feature 深度不直接用于每帧手腕深度替代；它更适合作为诊断或稀疏约束来源。

相关脚本：

```text
preprocess/BuildRtabmapCameraTrajectory.py
preprocess/ApplyCameraTrajectoryToPreprocess.py
preprocess/DepthCorrectHandKpts.py
preprocess/DiagnoseRtabmapDepth.py
```

## 常见问题

### Missing bag directory

报错示例：

```text
FileNotFoundError: Missing bag directory: .../bag
```

说明脚本没有找到 ROS2 bag 目录。当前会自动尝试：

```text
${BAG_SESSION}/bag
${BAG_SESSION}/data/bag
```

如果数据结构不同，手动指定：

```bash
BAG_DIR=/actual/path/to/bag BAG_SESSION=/path/to/session scripts/run_sampler_bag_to_glove_trajectory.sh
```

### RTABMAP_DB not found

批次格式默认路径（`BAG_DATA_DIR` 由 `BAG_SESSION` 自动解析）：

```text
${BAG_DATA_DIR}/map/rtabmap.db
```

旧格式会继续尝试 calibration 目录中的 `rtabmap.db`。

如果数据库在别处：

```bash
RTABMAP_DB=/path/to/rtabmap.db USE_RTABMAP_POSE=1 scripts/run_sampler_bag_to_glove_trajectory.sh
```

如果暂时不用 RTAB-Map 位姿：

```bash
USE_RTABMAP_POSE=0 scripts/run_sampler_bag_to_glove_trajectory.sh
```

### 输出太大

完整 session 保留了逐帧 RGB/depth、中间 JSON、多个视频和调试目录，适合调试但体积大。

交付或只看网页时开启：

```bash
COMPACT_OUTPUTS=1
```

或者对已有输出执行：

```bash
/home/lenovo/miniconda3/envs/hamer/bin/python scripts/compact_postprocess_session.py --session /path/to/postprocess_session
```

### 网页打不开或视频不播放

RGB 仍由 `outputs/web/rgb_frames` 播放，轨迹和触觉由 Canvas 绘制；只有可选
Robot 对比画面使用 MP4。确认基础文件存在：

```text
outputs/web/index.html
outputs/web/rgb_frames/00000.jpg
outputs/web/tactile_hand.png
```

启用 Robot 对比时还应存在：

```text
outputs/web/robot_simulation.mp4
```

如果缺失，重新生成网页：

```bash
/home/lenovo/miniconda3/envs/hamer/bin/python scripts/generate_review_web.py --session /path/to/postprocess_session
```

如果打开后只有黑色面板，优先检查浏览器控制台是否有 JS 报错，以及 `index.html` 中是否存在完整的 `Math.min(DURATION,t||0)`。修复脚本后重新生成网页即可。

## 开发注意

- 主入口优先改 `scripts/run_sampler_bag_to_glove_trajectory.sh`。
- 单阶段逻辑在 `preprocess/` 下。
- 用户交付结果集中在 `outputs/`。
- 主流程默认运行质量门禁并保留完整输出；门禁失败时返回非零状态。只有显式设置 `COMPACT_OUTPUTS=1` 且质量检查通过时才会精简。
- `REVIEW_HAND_DISPLAY_ROTATE_DEG`、`hand_display_scale`、`scene_display_scale`、`align_middle_vertical` 都只影响网页 3D 显示，不修改轨迹 JSON。
- `rgb_stamp_ns` 是统一时间真值：网页按逐帧真实时间播放；EMA、步长上限和确认逻辑按 `dt` 换算；恒定帧率 MP4 使用真实标称帧率并对采集停顿重复帧，以保持时间长度。
