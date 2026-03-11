FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV GST_VERSION=1.28.1
ENV OCIO_VERSION=2.5.1
ENV LIBPLACEBO_VERSION=7.349.0

# Build dependencies
# NOTE: Ubuntu 24.04 ships OCIO 2.1.3 (need >=2.5.1 for built-in ACES 2.0 configs)
# and libplacebo API 338 (need >=349 for stable pl_renderer API). Both built from source.
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
    liblcms2-dev libvulkan-dev glslang-dev libglfw3-dev \
    libyaml-cpp-dev libexpat1-dev libimath-dev libminizip-dev \
    python3-jinja2 \
    git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Build OpenColorIO 2.5.1 from source (Ubuntu 24.04 only has 2.1.3)
WORKDIR /build
RUN git clone --branch v${OCIO_VERSION} --depth 1 \
    https://github.com/AcademySoftwareFoundation/OpenColorIO.git
WORKDIR /build/OpenColorIO
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DOCIO_BUILD_APPS=OFF \
    -DOCIO_BUILD_TESTS=OFF \
    -DOCIO_BUILD_GPU_TESTS=OFF \
    -DOCIO_BUILD_PYTHON=OFF \
    -DOCIO_INSTALL_EXT_PACKAGES=MISSING \
    -DBUILD_SHARED_LIBS=ON \
    && cmake --build build -j$(nproc) \
    && cmake --install build

# Build libplacebo 7.349.0 from source (Ubuntu 24.04 has API 338, need >=349)
WORKDIR /build
RUN git clone --branch v${LIBPLACEBO_VERSION} --depth 1 --recursive \
    https://github.com/haasn/libplacebo.git
WORKDIR /build/libplacebo
RUN meson setup build \
    --prefix=/usr/local \
    --buildtype=release \
    -Ddemos=false \
    -Dtests=false \
    && ninja -C build -j$(nproc) \
    && ninja -C build install

# Refresh library cache after installing OCIO and libplacebo
RUN ldconfig

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

# Build gst-libplacebo plugin
COPY plugins/gst-libplacebo /build/gst-libplacebo
RUN cd /build/gst-libplacebo && meson setup build \
    --prefix=/usr/local \
    --buildtype=release \
    && meson compile -C build \
    && meson install -C build

# Build gst-ocio plugin
COPY plugins/gst-ocio /build/gst-ocio
RUN cd /build/gst-ocio && meson setup build \
    --prefix=/usr/local \
    --buildtype=release \
    && meson compile -C build \
    && meson install -C build

# Update library paths
RUN ldconfig

# --- Runtime stage ---
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime dependencies
# NOTE: OCIO and libplacebo shared libs come from /usr/local (built from source).
# Runtime deps for those libs (lcms2, vulkan) installed via apt.
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
    liblcms2-2 libvulkan1 \
    ffmpeg \
    git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy built artifacts from builder (GStreamer, OCIO, libplacebo, plugins)
COPY --from=builder /usr/local /usr/local
RUN ldconfig

# Smoke test: verify custom GStreamer plugins are registered
RUN gst-inspect-1.0 placebofilter && gst-inspect-1.0 ociofilter

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
