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

`BAG_SESSION` 可以传采集根目录，也可以直接传其 `data` 目录。脚本会先解析出统一的数据根目录 `BAG_DATA_DIR`，再寻找 bag：

```text
${BAG_SESSION}/bag
${BAG_SESSION}/data/bag
```

新版数据包通常使用：

```text
${BAG_SESSION}/data/bag
${BAG_SESSION}/data/calibration/rtabmap.db
${BAG_SESSION}/data/calibration/calib_video_*/bag
```

其中 `calib_video_*` 是专门录制的手眼/手套视觉标定 bag。主流程默认 `AUTO_CALIB_VIDEO=1`，会自动选择最新的 `calib_video_*` 目录作为标定输入；如果想手动指定，设置 `CALIB_BAG_SESSION=/path/to/calib_video_xxx`。

RTAB-Map 数据库默认从解析后的数据根目录读取：

```text
${BAG_DATA_DIR}/calibration/rtabmap.db
```

当前默认 `USE_RTABMAP_POSE=1`，也就是后续相机位姿来自 `rtabmap.db` 导出的 Poses，并插值到 RGB 帧。

## 一行运行命令

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
| `SESSION_NAME` | 自动生成 | 输出 session 名称 |
| `SESSION` | `postprocess_data/${SESSION_NAME}` | 完整输出目录 |
| `BAG_DIR` | 自动识别 | ROS2 bag 目录 |
| `USE_RTABMAP_POSE` | `1` | 是否使用 `rtabmap.db` 相机位姿 |
| `RTABMAP_DB` | `${BAG_DATA_DIR}/calibration/rtabmap.db` | RTAB-Map 数据库；`BAG_DATA_DIR` 由脚本自动解析 |
| `RTABMAP_MAX_INTERP_GAP_SEC` | `0.25` | 允许插值跨越的最大 RTAB-Map 节点时间间隔 |
| `HAMER_HANDEDNESS` | `all_left` | HaMeR 手性策略 |
| `VISUAL_SIDE` | `hand_l` | 视觉手侧 |
| `GLOVE_SIDE` | `left` | 手套数据侧 |
| `FPS` | 自动 | 主流程忽略固定覆盖值，始终从 `rgb_stamp_ns` 估计真实 RGB 标称帧率 |
| `TIME_FILTER_REFERENCE_FPS` | `30` | 旧版每帧 alpha、最大步长和确认帧数的参考语义；实际计算按真实 `dt` 换算 |
| `OVERWRITE` | `0` | `1` 时忽略阶段缓存并强制重算 |
| `MAX_FRAMES` | 空 | 限制处理帧数，调试用 |
| `COMPACT_OUTPUTS` | `0` | 是否在质量检查通过后精简输出包；默认保留完整中间产物 |

深度和轨迹平滑相关参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `DEPTH_METHOD` | `robust` | wrist/root 深度估计方式 |
| `DEPTH_ROBUST_INDICES` | `0,5,9,13,17` | 用 wrist 和多个 MCP 点做鲁棒深度估计 |
| `DEPTH_RADIUS` | `8` | 深度局部搜索半径 |
| `DEPTH_ROBUST_INLIER_M` | `0.055` | 深度 inlier 阈值 |
| `VISUAL_2D_SMOOTH_ALPHA` | `0.35` | 2D overlay 平滑系数 |
| `WRIST_TRACK_ALPHA` | `0.25` | wrist 3D/root 时间滤波系数，由子流程读取 |
| `WRIST_TRACK_MAX_STEP_M` | `0.007` | wrist 3D/root 单帧最大步长，由子流程读取 |
| `REVIEW_HAND_DISPLAY_ROTATE_DEG` | `45` | 网页 3D 手部显示旋转参数；当前默认开启 `wrist -> middle_mcp` 竖直对齐时主要用于兼容旧命令 |

## 输出目录

完整输出默认在：

```text
postprocess_data/${SESSION_NAME}/
```

主要目录：

```text
preprocess/                                      ROS2 bag 提取的逐帧 RGB/depth/hand_frame/tf
rtabmap_pose/                                    RTAB-Map 相机位姿导出和应用摘要
locateanything_white_glove_with_imu/             LocateAnything 原始 bbox
locateanything_white_glove_with_imu_stable/      稳定 bbox
hamer_from_stable_locateanything_.../            HaMeR 结果
fusion_input_force_right_depthroot/              视觉+手套融合输入
glove_fk21_visual_bones_smooth_solve045/         glove FK、标定、wrist tracking 和轨迹
outputs/                                         用户主要查看和交付结果
```

`outputs/` 内的重要结果：

```text
outputs/videos/00_rgb_raw.mp4
outputs/videos/01_locate_bboxes.mp4
outputs/videos/02_stable_bbox.mp4
outputs/videos/03_visual_21kpts_raw.mp4
outputs/videos/04_visual_21kpts_2d_smooth.mp4
outputs/videos/06_glove_fk_overlay_wristroot_track.mp4
outputs/videos/07_trajectory_3d_world.mp4
outputs/videos/07b_trajectory_3d_camera_frame.mp4

outputs/data/trajectory_wristroot_track_cameraoptical.jsonl
outputs/data/visual_2d_smooth.jsonl
outputs/data/locate_bboxes.json
outputs/data/stable_bboxes.json

outputs/summaries/*.json
outputs/web/index.html
```

## 可视化网页

主流程最后会自动生成 review web：

```text
outputs/web/index.html
```

网页包含：

```text
左上：RGB 帧播放
左下：Head + hand 3D 轨迹播放
右侧：数据基本信息、wrist x/y/z 曲线和当前 wrist 坐标
底部：播放/暂停、重置、时间轴
```

网页使用帧序列播放，不依赖浏览器 MP4/H.264 解码。生成后的网页是自包含的，RGB 和 3D 帧位于：

```text
outputs/web/rgb_frames/*.jpg
outputs/web/traj_frames/*.jpg
```

### 采集风格网页

如果要生成不覆盖旧网页的采集风格页面，使用 `--output_subdir web_collect`：

```bash
/home/lenovo/miniconda3/envs/hamer/bin/python scripts/generate_review_web.py --session /home/lenovo/Ego-loong-postprocess/postprocess_data/07_04_2_20260704T080233 --fps 30 --output_subdir web_collect --hand_display_rotate_deg 45
```

输出：

```text
outputs/web_collect/index.html
outputs/web_collect/rgb_frames/*.jpg
outputs/web_collect/traj_frames/*.jpg
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

主入口为每个阶段在 session 的 `.pipeline_cache/` 下写入 manifest。manifest 包含输入内容哈希、阶段参数、相关代码文件哈希、输出完整性和完成时间。只有这些指纹与当前运行完全一致且要求的输出仍完整时，阶段才会命中缓存；输入 bag、标定、参数或代码变化会使该阶段以及引用其 manifest 的下游阶段重新计算。

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

默认路径（`BAG_DATA_DIR` 由 `BAG_SESSION` 自动解析）：

```text
${BAG_DATA_DIR}/calibration/rtabmap.db
```

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

网页不是直接播放 MP4，而是播放 `outputs/web/rgb_frames` 和 `outputs/web/traj_frames` 里的帧。确认这些文件存在：

```text
outputs/web/index.html
outputs/web/rgb_frames/00000.jpg
outputs/web/traj_frames/00000.jpg
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
- 主流程默认保留完整输出；只有显式设置 `COMPACT_OUTPUTS=1` 且质量检查通过时才会精简。
- `REVIEW_HAND_DISPLAY_ROTATE_DEG`、`hand_display_scale`、`scene_display_scale`、`align_middle_vertical` 都只影响网页 3D 显示，不修改轨迹 JSON。
- `rgb_stamp_ns` 是统一时间真值：网页按逐帧真实时间播放；EMA、步长上限和确认逻辑按 `dt` 换算；恒定帧率 MP4 使用真实标称帧率并对采集停顿重复帧，以保持时间长度。
