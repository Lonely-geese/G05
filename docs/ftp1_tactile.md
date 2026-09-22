# G05 的 FTP-1 风格触觉分支

分支：`feat/ftp1-tactile`。

数据：`/mnt/cfs/7rnh3z/kele/combined_rj45_tactile_0917_0921_shift3`，
LeRobot v2.1，30 Hz，3,430 episodes / 1,006,708 frames。数据已完成
action shift3；配置中所有 `time_offset=0`，不再移动标签。

## 实现与适配边界

参考 [FTP-1 官方实现](https://github.com/michaelyuancb/ftp1-policy)，尤其是
`src/openpi/models_pytorch/t3_tactile_encoder.py`、`ftp1_blocks.py`、
`ftp1_attention_masks.py` 和 `ftp1_pytorch.py`。

| 路径 | 本分支实现 |
| --- | --- |
| 视觉语言 | headDepthCamera、leftCamera、rightCamera → 原 G05 VLM |
| 触觉图像 | leftTactileCamera1/2 → 同类型传感器共享 ViT stem（3 层）→ 共享 ViT trunk（9 层）→ 每路 1 个 CLS token |
| 接触位置 | 两个 token 加入各自 area embedding，再统一投影；area ID 0/1 是本数据集局部标识 |
| 触觉专家 | 独立 24 层 Qwen3.5 full-attention，hidden 512，MLP 2048；KV head 规格与动作专家一致 |
| 动作专家 | 各层读取 `[视觉语言 KV | 触觉 KV | 动作]`；触觉不读视觉语言，视觉语言不读触觉 |

G05 的 VLM 是混合注意力：有视觉 KV 的 6 层读取视觉和触觉，其余 18 层读取
触觉和动作。触觉只在每次 FM 推理开始时编码一次，缓存复用于去噪迭代。
触觉图像独立缩放到 224×224、归一化到 [-1,1]，不用 RGB 颜色抖动或随机裁剪，
也不占用或随机挤掉原来的 3 个 VLM 图像槽。

这是 **FTP-1 融合结构的 Qwen/G05 适配，不是完整 FTP-1 模型复现**。
触觉 ViT 和触觉专家随机初始化；只加载 G05 base 的原有权重，未加载 FTP-1
触觉预训练权重，不兼容直接导入其 Gemma 专家。当前限定两路完整 RGB 触觉输入，
未实现 FTP-1 的跨机器人通用形态映射、缺失传感器 mask 或额外标量触觉模态。

默认只训练连续 FM 动作（不训练 AR 动作 token）；`joint_training=true`
允许 FM 梯度传回 VLM。设为 false 时仅分离 VLM 梯度，触觉分支仍可训练。
FM-only 的部分 VLM 参数没有损失路径，因此任务启用 DDP `find_unused_parameters`。
触觉专家最后一层只有 K/V 被消费，其无损失路径的 Q/O/MLP 参数显式冻结。

动作布局沿用已有 RJ45 配置：左臂和左夹爪训练，右臂/夹爪 loss mask 关闭，
后处理维持当前右侧关节状态。若要双臂训练，需要一并更改 action filter 和
右侧归一化配置，并重新计算统计，不能只改 loss mask。

## 准备与运行

在 G05 根目录执行。脚本优先使用 G05 `.venv`，不存在则使用相邻 GalaxeaVLA
环境；始终通过 `PYTHONPATH` 导入 **G05 本分支**，不导入兄弟仓库的模型代码。
无需复制 `.env`。也可设置 `G05_PYTHON=/path/to/python`。脚本同时复用相邻
GalaxeaVLA 缓存中的 CUDA 12.8 动态库，解决本机全局 CUDA 13 与 TorchCodec 的
NPP 12 依赖不匹配；其他机器可用 `G05_CUDA_LIB` 指定该目录。

```bash
# 首次：只用本 shift3 数据计算全量统计，并检查头/中/尾真实样本
bash run_combined_rj45_tactile.sh prepare

# 已有统计：校验 provenance、checksum 和真实五路相机数据
bash run_combined_rj45_tactile.sh check

# 默认单卡 batch=1，2 个训练 step + 1 个评估 batch，不保存训练权重
bash run_combined_rj45_tactile.sh smoke

# 正式训练默认 8 卡、每卡 batch=4、10 epochs，约 31,460 steps/epoch
bash run_combined_rj45_tactile.sh train

# 例：指定 GPU/卡数/每卡 batch；其余参数走 Hydra override
CUDA_VISIBLE_DEVICES=0,1 NPROC=2 PER_GPU=2 \
  bash run_combined_rj45_tactile.sh train model.max_epochs=null model.max_steps=10000
```

可用环境变量：`TACTILE_DATASET_ROOT`、`G05_TACTILE_STATS`、`G05_OUTPUT_DIR`、
`EXP_NAME`、`G05_PYTHON`、`NPROC`、`PER_GPU`。数据源变更后必须生成新的统计。
默认统计位置：`artifacts/combined_rj45_tactile_shift3/dataset_stats.json`；
默认输出：`outputs/combined_rj45_tactile_shift3/<EXP_NAME>/`。
准备脚本拒绝覆盖已有统计，启动前自动检查统计来源/校验和和解码器依赖，
启动脚本拒绝复用已有输出目录。

基础权重和处理器路径继承现有 RJ45 task，位于相邻 GalaxeaVLA 的
`base_checkpoint/`。首训 `use_meta_device=false`，保证 base checkpoint 中缺失的
触觉参数真实初始化；恢复完整触觉 checkpoint 时可启用 meta 加载。
8 卡默认有效 batch=32（梯度累积 1）；这是初始训练配置，不是已调优结果。
当前继承 `val_set_proportion=0`，评估会重用训练数据，**不能当作独立验证集指标**。

## 输入与检查

处理器输出额外字段：

```python
batch["tactile_pixel_values"] = {
    "tactile_left_1": tensor,  # [B, 1, 3, 224, 224], float [-1,1]
    "tactile_left_2": tensor,
}
```

标准 `policy.forward` / `predict_action` 和分阶段
`prefill(samples, pixel_values, tactile_pixel_values=...)` → `generate_action`
均传递此字段。缺失相机会报错，不会静默退回视觉策略。
AR/VQA 路径不使用触觉。部署前还应针对实际机器人 observation 映射、动作后处理和
控制安全限制单独验证；本分支不启动机器人服务或执行动作。

```bash
PYTHONPATH=src ../GalaxeaVLA/.venv/bin/python -m pytest -q \
  tests/test_tactile_branch.py tests/test_action_op_mask_loss.py
```

覆盖独立触觉编码、确定性相机顺序、稀疏 KV/位置拼接、缓存不被修改、
FM 梯度（含 checkpoint、多 flow sample、VLM 梯度开关）、推理输入敏感性、
checkpoint roundtrip、旧模型关闭分支兼容性以及配置/相机分离。

## 本机验证记录（2026-09-22）

- 相关回归测试：28 项通过（触觉、动作 mask、recap 窗口、battery 分段数据）。
- BF16 autocast 下触觉梯度有限且非零。
- 全量统计已生成，真实索引 0、503354、1006707 均通过五路视频解码、
  3+2 相机分离和 `(32,27)` 动作检查；没有通过随机重试替换测试样本。
- 新增触觉模块共 250,070,784 参数，其中 243,778,560 可训练；checkpoint
  新增 419 个条目，正好对应 G05 base 加载时的 419 个随机初始化条目。
- 双卡 `NPROC=2 PER_GPU=1` 的完整 G05 base 短跑通过：2 个训练 step 的
  跨卡平均 loss 为 0.9756、0.1743，随后完成 1 个 FM 推理评估 batch。
  这只是流程验证，不代表收敛或触觉带来收益；未开始 10 轮正式训练。
  日志：`outputs/combined_rj45_tactile_shift3/ftp1_tactile_smoke_verify_20260922/train.log`。
  未保存短跑模型/优化器 checkpoint。
- 短跑出现 CLS token 梯度 stride 与 DDP bucket 不一致的非致命性能警告，
  不影响反向传播。原 MFU FLOPs 估算尚未包含触觉模块，仅作参考。
- DataLoader 退出清理修复：训练与评估迭代器现在在主线程中显式关闭，
  先停止 pin-memory 线程，再关闭 worker；正常结束和训练/评估异常都会执行。
  评估 worker 延迟到首次评估时启动，persistent worker 仍跨 epoch 复用。
  11 项生命周期测试通过；另以小模型完成双卡 DDP、每卡 10 步与评估数据读取，
  确认每卡 8 个 worker、2 个 pin-memory 线程均已停止，退出码 0，
  未出现原先的 BrokenPipe/ConnectionRefused/不能 join 当前线程异常。
  验证记录：`artifacts/combined_rj45_tactile_shift3/dataloader_cleanup_20260922/`。
  已运行的旧训练进程不会热更新；修复从下次启动生效。
- 完整模型退出复测：停止旧训练后，以修复版运行 8 卡 × 每卡 batch 16，
  完成 10 次更新和一次评估；loss、梯度均有限，launcher 返回 0。
  未再出现 DataLoader 清理异常，已记录的子进程全部退出，8 卡显存均归零。
  本次按要求未重启正式训练。日志与检查结果：
  `artifacts/combined_rj45_tactile_shift3/ftp1_tactile_cleanup_verify_20260922_100800/`。
- 提交前相关回归测试共 39 项通过（触觉、DataLoader 生命周期、动作 mask、
  recap 窗口、battery 分段数据）；检查时将未捕获的线程/析构异常视为失败。
- 额外运行的 `test_serve_policy_dynamic_batching.py` 未通过：既有 collate 对
  不等长文本张量直接 stack。测试及其 inferencer/data_utils 文件与原 HEAD
  一致，本次未修改这个与触觉分支无关的服务端问题；不声称全仓库测试通过。
