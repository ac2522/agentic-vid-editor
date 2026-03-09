#!/bin/bash
set -e

# Verify GStreamer installation
echo "=== GStreamer Version ==="
gst-inspect-1.0 --version

# Verify GES is available
python3 -c "
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GES', '1.0')
from gi.repository import Gst, GES
Gst.init(None)
GES.init()
print(f'GES initialized: Gst {Gst.version_string()}')
" || echo "WARNING: GES not available"

# Verify whisper.cpp / pywhispercpp
python3 -c "
from pywhispercpp.model import Model
print('pywhispercpp available')
" || echo "WARNING: pywhispercpp not available"

# Verify nvcodec if GPU present
if command -v nvidia-smi &> /dev/null; then
    echo "=== NVIDIA GPU ==="
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    gst-inspect-1.0 nvcodec 2>/dev/null | head -3 || echo "nvcodec plugin not found"
fi

exec "$@"
