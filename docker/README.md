# Ego-loong 完整开发镜像

该镜像固化当前已验证的 Ubuntu 24.04、CUDA 13.0、ROS 2 Jazzy、
LocateAnything、HaMeR、Retarget 和模型权重。原始数据与处理结果不进入镜像，
运行时通过 volume 挂载。

## 宿主机边界

NVIDIA 内核驱动必须安装在宿主机，不能由普通应用镜像携带。当前 CUDA 13.0
镜像要求 Linux NVIDIA 驱动不低于 `580.65.06`。宿主机还需要 Docker 和
NVIDIA Container Toolkit。

只检查宿主机：

```bash
bash docker/host-setup-ubuntu24.04.sh
```

在全新 Ubuntu 24.04 主机上安装 Docker、Compose v2、Buildx 和 NVIDIA
Container Toolkit。`-E` 会保留当前代理变量，在需要代理访问 GitHub 的机器上必须保留：

```bash
sudo -E INSTALL=1 bash docker/host-setup-ubuntu24.04.sh
```

脚本不会自动替换 NVIDIA 内核驱动；驱动过旧时会停止并提示升级。

## 构建

构建脚本先用 `conda-pack` 快照当前两套 Python 环境，再暂存 Retarget 运行文件，
最后构建包含模型权重的完整镜像：

```bash
bash docker/build-image.sh
```

默认镜像名为 `ego-loong-postprocess:ubuntu24.04-cuda13.0`。

常用覆盖参数：

```bash
IMAGE_TAG=registry.local/ego-loong-postprocess:2026-08-20 \
LOCATE_ENV_PREFIX=/path/to/locate_anything \
HAMER_ENV_PREFIX=/path/to/hamer \
RETARGET_SOURCE=/path/to/Retarget \
bash docker/build-image.sh
```

首次构建生成的环境快照位于 `.docker-assets/envs/`，后续构建会先校验再复用。
Buildx 安装后，增量构建也会复用构建上下文和镜像层。环境更新后：

```bash
FORCE_PACK=1 bash docker/build-image.sh
```

`.docker-assets/` 已加入 `.gitignore`，不会提交环境包或 Retarget 副本。

## 静态 smoke test

```bash
docker run --rm --gpus all --shm-size=16g \
  -v /home/lenovo/.cache/ego-loong-container:/cache \
  ego-loong-postprocess:ubuntu24.04-cuda13.0 smoke
```

它会检查 GPU/CUDA、ROS 2、自定义消息、LocateAnything、HaMeR、MANO、
Retarget 和必需模型文件。

## 处理单个 session

镜像内项目根固定为 `/opt/ego-loong-postprocess`。输出目录必须挂载到其
`postprocess_data`，否则容器退出后输出会留在临时层中。

```bash
docker run --rm --gpus all --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  -v /home/lenovo/Downloads:/data/input:ro \
  -v /home/lenovo/Ego-loong-postprocess/postprocess_data:/opt/ego-loong-postprocess/postprocess_data \
  -v /home/lenovo/.cache/ego-loong-container:/cache \
  -e BATCH_ROOT=/data/input/2026-08-19T1107.36_8 \
  -e BAG_SESSION=/data/input/2026-08-19T1107.36_8/2026-08-19T1109.12 \
  -e SESSION_NAME=2026-08-19T1107.36_8_20260819T110912 \
  -e CALIB_SESSION=/opt/ego-loong-postprocess/postprocess_data/_batch_calibration/2026-08-19T1107.36_8 \
  -e CALIB_OVERWRITE=0 -e OVERWRITE=1 -e COMPACT_OUTPUTS=1 \
  ego-loong-postprocess:ubuntu24.04-cuda13.0 pipeline
```

## 处理整个批次

```bash
docker run --rm --gpus all --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  -v /home/lenovo/Downloads:/data/input:ro \
  -v /home/lenovo/Ego-loong-postprocess/postprocess_data:/opt/ego-loong-postprocess/postprocess_data \
  -v /home/lenovo/.cache/ego-loong-container:/cache \
  -e BATCH_ROOT=/data/input/2026-08-19T1107.36_8 \
  -e CALIB_OVERWRITE=0 -e OVERWRITE=1 -e COMPACT_OUTPUTS=1 -e CONTINUE_ON_ERROR=1 \
  ego-loong-postprocess:ubuntu24.04-cuda13.0 batch
```

当前批次脚本仍按 session 串行执行，共享 calibration 缓存。

## Compose 开发环境

```bash
mkdir -p /home/lenovo/.cache/ego-loong-container
HOST_UID="$(id -u)" HOST_GID="$(id -g)" \
docker compose -f docker/compose.yaml run --rm postprocess
```

## 一帧端到端测试

静态测试通过后，可以挂载一个真实数据包运行：

```bash
docker run --rm --gpus all --shm-size=16g \
  -v /home/lenovo/Downloads:/data/input:ro \
  -v /home/lenovo/Ego-loong-postprocess/postprocess_data:/opt/ego-loong-postprocess/postprocess_data \
  -v /home/lenovo/.cache/ego-loong-container:/cache \
  -e FULL_SMOKE=1 \
  -e SMOKE_BAG_SESSION=/data/input/path/to/session \
  ego-loong-postprocess:ubuntu24.04-cuda13.0 smoke
```

## 离线导出

```bash
docker save ego-loong-postprocess:ubuntu24.04-cuda13.0 \
  | zstd -T0 -10 -o ego-loong-postprocess-ubuntu24-cuda13.tar.zst
```

目标机器使用 `zstd -dc <archive> | docker load` 导入。模型权重和 MANO 文件
只应存放在有权限控制的内部镜像仓库或离线介质中。
