FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV GST_VERSION=1.28.1

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-pip python3-venv python3-gi python3-gi-cairo \
    gir1.2-glib-2.0 gir1.2-gst-plugins-base-1.0 \
    meson ninja-build cmake pkg-config flex bison nasm \
    libglib2.0-dev libgudev-1.0-dev libmount-dev \
    libgl-dev libegl-dev libgles-dev libdrm-dev libgbm-dev \
    libx11-dev libxext-dev libxv-dev libxi-dev libxfixes-dev \
    libwayland-dev libpulse-dev libasound2-dev \
    libopus-dev libvpx-dev libx264-dev libx265-dev \
    libfdk-aac-dev libmp3lame-dev libvorbis-dev libtheora-dev \
    libsrt-gnutls-dev libsoup-3.0-dev libjson-glib-dev \
    libcairo2-dev libpango1.0-dev librsvg2-dev \
    libavcodec-dev libavformat-dev libavutil-dev libavfilter-dev libswscale-dev \
    libplacebo-dev libplacebo254 \
    libopencolorio-dev \
    git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Build GStreamer 1.28.1 monorepo (includes GES, nvcodec, GL plugins)
WORKDIR /build
RUN git clone --branch ${GST_VERSION} --depth 1 \
    https://gitlab.freedesktop.org/gstreamer/gstreamer.git

WORKDIR /build/gstreamer
RUN meson setup build \
    --prefix=/usr/local \
    --buildtype=release \
    -Dgpl=enabled \
    -Dgst-plugins-bad:nvcodec=enabled \
    -Dgst-plugins-base:gl=enabled \
    -Dgst-plugins-base:gl_api=opengl,gles2 \
    -Dgst-plugins-base:gl_platform=egl,glx \
    -Dges=enabled \
    -Dgst-editing-services:pygi-overrides-dir=/usr/lib/python3/dist-packages/gi/overrides \
    -Dintrospection=enabled \
    -Dpython=enabled \
    -Dgst-plugins-rs=disabled \
    -Ddevtools=disabled \
    -Ddoc=disabled \
    -Dexamples=disabled \
    -Dtests=disabled \
    && ninja -C build -j$(nproc) \
    && ninja -C build install

# Build whisper.cpp with CUDA support
WORKDIR /build
RUN git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git
WORKDIR /build/whisper.cpp
RUN cmake -B build \
    -DGGML_CUDA=1 \
    -DCMAKE_CUDA_ARCHITECTURES="60;70;75;80;86;89;90" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    && cmake --build build -j$(nproc) \
    && cmake --install build

# Update library paths
RUN ldconfig

# --- Runtime stage ---
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-gi python3-gi-cairo \
    gir1.2-glib-2.0 \
    libgl1 libegl1 libgles2 libdrm2 libgbm1 \
    libx11-6 libxext6 libxv1 \
    libpulse0 libasound2t64 \
    libopus0 libvpx9 libx264-164 libx265-209 \
    libfdk-aac2 libmp3lame0 libvorbis0a libtheora0 \
    libsrt1.5-gnutls libsoup-3.0-0 libjson-glib-1.0-0 \
    libcairo2 libpango-1.0-0 librsvg2-2 \
    libplacebo254 \
    libopencolorio2.3 \
    ffmpeg \
    git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy built GStreamer from builder
COPY --from=builder /usr/local /usr/local
RUN ldconfig

# Set up GStreamer environment
ENV GST_PLUGIN_PATH=/usr/local/lib/x86_64-linux-gnu/gstreamer-1.0
ENV GI_TYPELIB_PATH=/usr/local/lib/x86_64-linux-gnu/girepository-1.0
ENV LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV GST_GL_API=opengl3
ENV WHISPER_MODEL_DIR=/root/.cache/ave/models

# Install pywhispercpp with CUDA support (uses system whisper.cpp libs)
RUN GGML_CUDA=1 pip install --break-system-packages pywhispercpp

# Install scene detection and vision dependencies
RUN pip install --break-system-packages \
    "scenedetect[opencv]>=0.6" \
    "onnxruntime-gpu>=1.20" \
    "transformers>=4.49" \
    "numpy>=1.26" \
    "Pillow>=10.0" \
    "torch>=2.0"

# Install ave package
WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
RUN pip install --break-system-packages -e ".[dev]"

# Copy test code
COPY tests/ tests/
COPY CLAUDE.md .

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["pytest", "-v"]
