# 本机 GalaxeaVLA 环境

安装日期：2026-09-18。

## 使用

```bash
cd /mnt/cfs/7rnh3z/kele/GalaxeaVLA
source .venv/bin/activate
source .env
```

项目 `.venv` 是指向 `/root/.local/share/venvs/galaxeavla-py310` 的软链接。
包文件放在本机磁盘，避免共享盘大量小文件读写造成的安装和导入延迟。
迁移到其他机器时需要重新安装环境。

- Python 3.10.16
- PyTorch 2.7.1+cu128 / CUDA runtime 12.8
- 项目锁定依赖、dev 测试依赖，以及 Open3D 缺失的 ipywidgets 8.1.8
- 8 张 NVIDIA A800-SXM4-80GB

`.env` 配置了项目路径、缓存、输出目录，以及 TorchCodec 所需的 CUDA 12 NPP 库路径。
训练输出默认在 `outputs/`，Hugging Face 缓存在 `.cache/huggingface/`。

## 安装与兼容处理

```bash
bash scripts/setup_env_local.sh
```

安装脚本使用现有 `uv.lock`，对国内阿里云和 NVIDIA 镜像采用直连，其他地址沿用当前 shell 的代理设置。
使用 `CMAKE_POLICY_VERSION_MINIMUM=3.5` 兼容 egl-probe 的旧 CMake 配置。
补充安装 ipywidgets，并下载、校验和解包 NVIDIA 官方 NPP 12.8 runtime 至项目 `.cache/cuda-12.8/`。
系统驱动和系统 CUDA Toolkit 未修改。

锁定的 FlashAttention 4 版本不支持 A800/SM80 的反向计算。
`src/g05/models/g05/qwen35/vision.py` 已增加硬件判断：SM90 及以上使用 FA4，
Ampere 优先使用 FA2（如已安装），否则使用 PyTorch SDPA。
当前 A800 环境使用 SDPA；FLA 线性注意力仍使用加速实现。

## 验证与范围

验证日志保存在 `artifacts/environment_setup/`。
测试覆盖核心模块导入、依赖一致性、8 张 GPU 的 BF16 运算、8 卡 NCCL all-reduce、
视觉注意力与 FLA 前反向计算、视频解码、配置解析，以及 15 项现有单元测试。

```bash
uv pip check --python .venv/bin/python
python tools/resolve_config.py r1pro --key model.model_arch
pytest tests/test_recap_windows.py tests/test_action_op_mask_loss.py -q
```

本次未下载模型权重、训练数据或额外仿真场景，未启动完整模型训练。
训练前还需要按 README 准备 checkpoint、数据集路径和对应任务配置。
