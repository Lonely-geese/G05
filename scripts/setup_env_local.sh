#!/usr/bin/env bash
# Reproduce the locked environment on this machine.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}mirrors.aliyun.com,pypi.nvidia.com,pypi.nvidia.cn"
export no_proxy="${no_proxy:+${no_proxy},}mirrors.aliyun.com,pypi.nvidia.com,pypi.nvidia.cn"
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
export UV_HTTP_TIMEOUT=300
export UV_CONCURRENT_DOWNLOADS=12
# egl-probe 1.0.2 uses an old minimum CMake policy version.
export CMAKE_POLICY_VERSION_MINIMUM=3.5
# Keep package files on local disk; the repository may be on slow shared storage.
if [[ ! -e .venv && ! -L .venv ]]; then
    export UV_PROJECT_ENVIRONMENT="${GALAXEAVLA_VENV_DIR:-$HOME/.local/share/venvs/galaxeavla-py310}"
    mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")"
fi
uv sync --frozen --extra dev --inexact --index-strategy unsafe-best-match
if [[ ! -e .venv && ! -L .venv ]]; then
    ln -s "$UV_PROJECT_ENVIRONMENT" .venv
fi
# The existing lockfile omits this runtime requirement of Open3D 0.18.0.
uv pip install --python .venv/bin/python \
    --index-url https://mirrors.aliyun.com/pypi/simple/ 'ipywidgets==8.1.8'

# TorchCodec's CUDA wheel needs NPP, which is not included in the torch wheel.
npp_root="$PWD/.cache/cuda-12.8"
if [[ ! -f "$npp_root/usr/local/cuda-12.8/targets/x86_64-linux/lib/libnppicc.so.12" ]]; then
    mkdir -p "$npp_root"
    npp_deb="$npp_root/libnpp-12-8_12.3.3.100-1_amd64.deb"
    curl --noproxy '*' --fail --location --retry 3 \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/libnpp-12-8_12.3.3.100-1_amd64.deb \
        --output "$npp_deb"
    echo "54febea3b7a793e65318647c0548c0fea2416ef0a7dc70c672c6877f3bcba992  $npp_deb" | sha256sum --check
    dpkg-deb --extract "$npp_deb" "$npp_root"
fi
echo 'Environment installed. Run: source .venv/bin/activate && source .env'
