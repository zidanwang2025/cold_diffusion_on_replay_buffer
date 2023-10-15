# Copied from https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/py.Dockerfile

# A Dockerfile that sets up a full gymnasium-robotics install with test dependencies
ARG PYTHON_VERSION=3.9
FROM python:$PYTHON_VERSION

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get -y update && apt-get install -y unzip libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb patchelf ffmpeg cmake swig

# RUN apt-get install -y xserver-xorg-video-fbdev
RUN apt-get install -y xserver-xorg-video-dummy
# (alternative) This installs xserver-xorg-video-dummy as a dependency
# RUN apt-get install -y xpra


# Download mujoco
RUN mkdir -p /usr/local/etc/mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /usr/local/etc/mujoco \
    && rm mujoco.tar.gz
ENV MUJOCO_PY_MUJOCO_PATH="/usr/local/etc/mujoco/mujoco210"
ENV LD_LIBRARY_PATH="/usr/local/etc/mujoco/mujoco210/bin"

# deepmind control suite reads from this env var
ENV MJLIB_PATH="/usr/local/etc/mujoco/mujoco210/bin/libmujoco210.so"

# Compile-time library location
ENV LIBRARY_PATH=$LD_LIBRARY_PATH
RUN ln -s /usr/local/etc/mujoco $HOME/.mujoco \
    && mkdir -p /opt/mujoco_py/mujoco_py/binaries \
    && ln -s /usr/local/etc/mujoco /opt/mujoco_py/mujoco_py/binaries/linux

# Download mujoco
# RUN mkdir /root/.mujoco \
#     && cd /root/.mujoco \
#     && wget -qO- 'https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz' | tar -xzvf -

# ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin"

# Build mujoco-py from source. Pypi installs wheel packages and Cython won't recompile old file versions in the Github Actions CI.
# Thus generating the following error https://github.com/cython/cython/pull/4428
# Takuma -- pinned at the current latest commit (timestamp: Nov 16, 2022)
RUN git clone https://github.com/openai/mujoco-py.git\
    && cd mujoco-py \
    && git checkout a13903b82f1ab316815e63b3575526005b3b2ae1 \
    && pip install -e .


# Copied from the Dockerfile in robosuite:
# ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy
# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY ./vendor/Xdummy-entrypoint /usr/local/entrypoints/bin/Xdummy-entrypoint

# === Project-specific stuff ===
# Install wandb and its (optional) dependencies
RUN pip install \
    wandb \
    kornia \
    imageio \
    imageio-ffmpeg \
    moviepy

RUN pip install \
    torch \
    einops \
    tqdm \
    matplotlib \
    scipy \
    scikit-learn

RUN pip install params_proto termcolor pandas

# This compiles /usr/local/lib/python3.8/dist-packages/mujoco_py/cymj.pyx
RUN python -c 'import mujoco_py'

RUN pip install opencv-python

RUN apt-get update && apt-get install -y --no-install-recommends \
    net-tools \
    xserver-xorg

# NOTE: Without this, singularity fails with
# OSError: [Errno 30] Read-only file system: b'/usr/local/lib/python3.9/dist-packages/mujoco_py/generated/mujocopy-buildlock'
RUN chmod a+rwx -R /mujoco-py/mujoco_py/generated

# When we have --network=host, the entrypoint provided by Gymnasium fails
# since it sees the Xorg server on the other containers.
# COPY ./bin /usr/local/entrypoints/bin
# ENTRYPOINT ["/usr/local/entrypoints/bin/docker_entrypoint"]

# Pin versions
RUN pip install gymnasium==0.28.1

# zidan_diffusion branch (gymansium-robotics 1.2.2)

RUN pip install git+https://github.com/bstadie/Gymnasium-Robotics.git@027e2a6052fab275ffc6736890bcbba543cb5936
RUN pip install h5py
# RUN pip install gymnasium-robotics==1.2.0

ENTRYPOINT ["/usr/local/entrypoints/bin/Xdummy-entrypoint"]
