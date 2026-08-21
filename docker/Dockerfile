# syntax=docker/dockerfile:1

ARG BASE_IMAGE=nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE} AS system-base

ARG DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=jazzy
ARG UBUNTU_MIRROR=https://mirrors.ustc.edu.cn/ubuntu
ARG ROS_MIRROR=https://mirrors.ustc.edu.cn/ros2/ubuntu

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN sed -i -E "s#https?://(archive|security).ubuntu.com/ubuntu/?#${UBUNTU_MIRROR}/#g" \
      /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates curl wget gnupg2 lsb-release locales tzdata \
      build-essential cmake ninja-build pkg-config ccache \
      git git-lfs openssh-client rsync unzip zip \
      ffmpeg sqlite3 jq \
      vim nano less tmux htop tree ripgrep \
      procps pciutils usbutils iproute2 iputils-ping net-tools \
      libgl1 libegl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      libgomp1 libjpeg-turbo8 libpng16-16t64 \
      python3 python3-pip python3-venv python3-dev \
    && locale-gen en_US.UTF-8 zh_CN.UTF-8 \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_DISTRO=${ROS_DISTRO} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    PYOPENGL_PLATFORM=egl

COPY .docker-assets/ros/ros2-archive-keyring.gpg /usr/share/keyrings/ros2-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros2-archive-keyring.gpg] ${ROS_MIRROR} $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
      > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-ros-base \
      ros-${ROS_DISTRO}-rosbag2-py \
      ros-${ROS_DISTRO}-rosbag2-storage-sqlite3 \
      ros-${ROS_DISTRO}-rosidl-runtime-py \
      ros-${ROS_DISTRO}-sensor-msgs \
      ros-${ROS_DISTRO}-nav-msgs \
      ros-${ROS_DISTRO}-tf2-msgs \
      python3-colcon-common-extensions \
      python3-numpy python3-opencv python3-scipy python3-yaml \
    && rm -rf /var/lib/apt/lists/*

FROM system-base AS ros-workspace

WORKDIR /opt/ego-loong-postprocess/hand_msg_ws
COPY hand_msg_ws/src ./src
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && colcon build --merge-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 \
    && rm -rf build log

FROM ros-workspace AS python-environments

RUN mkdir -p /opt/conda/envs/locate_anything /opt/conda/envs/hamer
COPY .docker-assets/envs/locate_anything.tar.gz /tmp/locate_anything.tar.gz
COPY .docker-assets/envs/hamer.tar.gz /tmp/hamer.tar.gz
RUN python3 -m tarfile -e /tmp/locate_anything.tar.gz /opt/conda/envs/locate_anything \
    && (tar -xzf /tmp/hamer.tar.gz -C /opt/conda/envs/hamer \
        || test -f /opt/conda/envs/hamer/lib/python3.10/site.py) \
    && python3 -c 'import sys; from pathlib import Path; old="    with open(path, "; new="    if not __import__(\"os\").path.exists(path):\n        return False\n    with open(path, "; [(lambda p: p.write_text(p.read_text().replace(old, new, 1)))(Path(x)) for x in sys.argv[1:]]' \
       /opt/conda/envs/locate_anything/bin/conda-unpack \
       /opt/conda/envs/hamer/bin/conda-unpack \
    && python3 /opt/conda/envs/locate_anything/bin/conda-unpack \
    && python3 /opt/conda/envs/hamer/bin/conda-unpack \
    && rm -f /tmp/locate_anything.tar.gz /tmp/hamer.tar.gz \
    && /opt/conda/envs/locate_anything/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda)' \
    && /opt/conda/envs/hamer/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda)'

FROM python-environments AS application

ARG GIT_COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="Ego-loong postprocess" \
      org.opencontainers.image.description="Ubuntu 24.04 / ROS 2 Jazzy / CUDA 13 Ego-loong postprocess environment" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}"

ENV ROOT=/opt/ego-loong-postprocess \
    ROS_SETUP=/opt/ros/jazzy/setup.bash \
    HAND_MSG_SETUP=/opt/ego-loong-postprocess/hand_msg_ws/install/setup.bash \
    ROS_PYTHON=/usr/bin/python3 \
    LOCATE_PYTHON=/opt/conda/envs/locate_anything/bin/python \
    HAMER_PYTHON=/opt/conda/envs/hamer/bin/python \
    LOCATE_MODEL=/opt/ego-loong-postprocess/models--nvidia--LocateAnything-3B/resolved \
    BASE_HAND_CONFIG=/opt/ego-loong-postprocess/retarget/host/hand_config.json \
    RETARGET_ROOT=/opt/ego-loong-postprocess/retarget/retarget \
    RESOLVE_DRIVER=/opt/ego-loong-postprocess/retarget/resolve_driver \
    PYTHONPATH=/opt/ego-loong-postprocess/hamer:/opt/ego-loong-postprocess \
    XDG_CACHE_HOME=/cache/xdg \
    HF_HOME=/cache/huggingface \
    MPLCONFIGDIR=/cache/matplotlib \
    NUMBA_CACHE_DIR=/cache/numba

WORKDIR ${ROOT}

COPY preprocess ./preprocess
COPY scripts ./scripts
COPY third_party/nvidia_locateanything_batch ./third_party/nvidia_locateanything_batch
COPY hamer/hamer ./hamer/hamer
COPY utils ./utils
COPY tactile/assets/hand_live.png ./tactile/assets/hand_live.png
COPY requirements.txt requirements-locate.lock requirements-hamer.lock requirements-ros-system.txt environment-info.json setup.sh README.md ./
COPY .docker-assets/retarget/retarget ./retarget/retarget
COPY .docker-assets/retarget/hand_config.json ./retarget/host/hand_config.json
COPY .docker-assets/retarget/resolve_driver ./retarget/resolve_driver
COPY docker/entrypoint.sh docker/smoke-test.sh ./docker/

RUN chmod +x scripts/*.sh docker/*.sh retarget/resolve_driver \
    && mkdir -p postprocess_data /cache \
    && /opt/conda/envs/hamer/bin/python -m py_compile preprocess/DepthCorrectHandKpts.py scripts/generate_review_web.py

FROM application AS full

# Keep large, rarely-changing model files in the final layer so code edits reuse
# all preceding layers.
COPY models--nvidia--LocateAnything-3B ./models--nvidia--LocateAnything-3B
RUN model_root="${ROOT}/models--nvidia--LocateAnything-3B" \
    && while IFS= read -r -d '' link; do \
         target="$(readlink "${link}")"; \
         blob="${model_root}/blobs/$(basename "${target}")"; \
         [[ -s "${blob}" ]] || { printf 'Missing model blob for %s\n' "${link}" >&2; exit 1; }; \
         relative="$(realpath --relative-to="$(dirname "${link}")" "${blob}")"; \
         ln -snf "${relative}" "${link}"; \
       done < <(find "${model_root}" -type l -print0) \
    && test -s "${LOCATE_MODEL}/config.json" \
    && test -s "${LOCATE_MODEL}/model-00001-of-00002.safetensors" \
    && test -s "${LOCATE_MODEL}/model-00002-of-00002.safetensors"
COPY hamer/_DATA ./hamer/_DATA

ENTRYPOINT ["/opt/ego-loong-postprocess/docker/entrypoint.sh"]
CMD ["bash"]
