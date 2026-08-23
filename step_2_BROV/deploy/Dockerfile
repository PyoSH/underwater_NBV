FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        gir1.2-gst-plugins-base-1.0 \
        gir1.2-gstreamer-1.0 \
        gstreamer1.0-libav \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-tools \
        iproute2 \
        iputils-ping \
        libgomp1 \
        netcat-openbsd \
        procps \
        python3-gi \
        python3-opencv \
        python3-pip \
        python3-yaml \
        ros-humble-camera-calibration \
        ros-humble-cv-bridge \
        ros-humble-desktop \
        ros-humble-geometry-msgs \
        ros-humble-image-proc \
        ros-humble-image-transport \
        ros-humble-rclpy \
        ros-humble-sensor-msgs \
        ros-humble-std-msgs \
        ros-humble-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/brov-requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade "pip<27" "setuptools<80" wheel \
    && python3 -m pip install --no-cache-dir -r /tmp/brov-requirements.txt \
    && python3 -m pip install --no-cache-dir --ignore-installed \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.1,<3"

COPY docker/entrypoint.sh /usr/local/bin/brov-entrypoint
COPY docker/ros_env.sh /usr/local/bin/brov-ros-env
RUN chmod +x /usr/local/bin/brov-entrypoint /usr/local/bin/brov-ros-env \
    && printf '\n# BROV ROS 2 overlay\nsource /usr/local/bin/brov-ros-env\n' >> /root/.bashrc

WORKDIR /workspace/deploy

ENTRYPOINT ["/usr/local/bin/brov-entrypoint"]
CMD ["sleep", "infinity"]
