# HUMAN 纠错 SFT 基线

本实验从用户指定的 **0912 morning** 模型开始，使用原示范 70% / HUMAN 纠错 30% 的采样比例，学习率 `5e-6`，全局 batch 128，训练 2,000 optimizer steps。沿用 30 Hz、32 步 action chunk、右臂相对关节目标与原夹爪表示，左臂/左夹爪不计入动作损失。它是纠错 SFT 基线，不包含 RECAP 的 value/advantage 更新。

## 已准备的数据

目录：`/mnt/cfs/l1x67e/kele/data/recap_human_sft_shift1_v1`

| 集合 | 原始 rollout | 可训练窗口 |
|---|---:|---:|
| train | 180 | 46,836 |
| val | 20 | 4,408 |

种子为 7；同一 rollout 的全部窗口仅进入一个集合。没有 HUMAN 帧的 `rollout_20260912_201138_fa0ee794` 已排除。旧示范沿用原配置的 episode 划分：550,438 个训练帧、403 个验证帧；不是为这次实验新构造的独立旧任务测试集。

数据通过窗口索引提取，底层保留完整 episode 的图像时间轴和状态上下文：

```text
recap_human_sft_shift1_v1/
  READY.json
  summary.json
  train/、val/
    meta/correction_windows.json
    meta/info.json、episodes.jsonl、episodes_stats.jsonl、tasks.jsonl
    data/chunk-000/episode_*.parquet
    videos/chunk-000/<camera>/episode_*.mp4  # 指向原始视频的软链接
```

这样不需要重编码视频，也不会改变图像帧和时间戳。**原始数据目录必须继续保留。底层总帧数不是可训练样本数**，实际长度由有效窗口索引定义。普通 BaseLerobotDataset 会拒绝这种数据，必须使用 `CorrectionWindowDataset`。

## 过滤和动作时序

窗口需要 H+1=33 个连续 HUMAN 实测状态：当前观察加未来 32 个目标。所有帧满足：

- 相同 HUMAN segment；observation_sequence 连续。
- 实际时间戳递增，相邻间隔不大于 50 ms。
- 状态有限，state_age 在 0–20 ms；camera_skew 在 0–45 ms。
- quality_mask 仅接受 0 或 64；其他标记排除。

这里允许 64 是一个显式的数据处理假设：采集端该 bit 的精确定义尚未确认，因此依赖独立状态/相机检查，不按总体 degraded 标记删除几乎全部人工数据。拖拽轨迹的监督来自实测状态，不使用旧控制命令的年龄筛选目标。具体阈值保存在每个 manifest 中，可通过转换器参数调整。

转换器只在通过检查的 HUMAN 连续区间内部执行一次 `action[t] = state[t+1]`。Dataset 只暴露整个 action chunk 都有效的起点，不使用片段末尾 padding，不允许非零 time_offset 重复 shift。读取错误立即报错，不随机退回未筛选的 POLICY/TRANSITION 帧。不删除静止帧，避免误删插接时的接触保持动作。

已抽查 train episode 0、75、179 的头部/右腕视角，图像中可见拖拽纠正插接和后续回撤。联系图为 `artifacts/recap_dataset_audit/human_sample_contact_sheet.jpg`。这属于抽查，不能解释为逐条确认全部人工动作质量或真实任务成功。

## 重新生成

在仓库根目录运行。为避免覆盖，数据输出和配置输出都必须不存在；调整规则时使用新的版本名。

```bash
.venv/bin/python scripts/utils/prepare_recap_sft.py \
  --source /mnt/cfs/l1x67e/kele/data/recap_assisted_success_rollouts \
  --output /mnt/cfs/l1x67e/kele/data/recap_human_sft_shift1_v1 \
  --base-config configs/data/tianji_0907_0908_0910_shift1_right_arm.yaml \
  --config-output configs/data/tianji_recap_human_sft_v1.yaml
```

`summary.json` 包含规则、排除记录、划分、窗口数和采样倍率；窗口 manifest 保留源 rollout、原始帧区间、行为 checkpoint 和 recap SHA256。READY 只在转换全部完成后写入。

## 混合采样

配置：`configs/data/tianji_recap_human_sft_v1.yaml`。

两个来源共享 `galaxea_r1pro` processor/normalizer，但使用不同 Dataset 类。开启 `use_weight_for_sampling` 和 `use_weight_normalization`。旧数据 weight=1，纠错 weight≈5.0367666，由以下公式计算：

```text
human_weight = 0.3 / 0.7 × old_train_length / human_valid_train_length
```

归一化后的逻辑采样长度约为纠错 179,182、旧示范 418,091，总计 597,273；比例约为 30% / 70%。训练采样器会打乱逻辑索引，纠错样本重复采样、旧示范分散下采样；实际单批次比例会波动。不能直接把 weight 配成 0.3/0.7。

## 启动与验证

启动脚本：`run_g05_recap_human_sft_v1.sh`。

```bash
bash run_g05_recap_human_sft_v1.sh
```

默认 8 卡，每卡 batch 16，warmup 100，cosine schedule，每 500 步保存一次。默认初始化路径：

```text
/mnt/cfs/l1x67e/kele/GenieSim3.0-Dataset/checkpoints/g05_runs/r1pro/
tianji_0907_0908_0910_shift1_right_arm_from_0911_step21505_8gpu_gbs128_extra5ep_formal/
checkpoints/step_21505.pt
```

这个训练 checkpoint 对应导出的 `g05_tianji_0912_morning_epoch5_step21505_inference`。复用同一 run 的 dataset_stats.json，重新初始化优化器和 scheduler。修改初始化时可设置 `RECAP_INIT_WEIGHT` 和 `RECAP_INIT_STATS`，二者应成对提供。

先做单步检查可运行：

```bash
EXP_NAME=recap_sft_smoke bash run_g05_recap_human_sft_v1.sh --dry-run
```

**仓库实际的 `--dry-run` 会加载模型并执行 1 个训练步和 1 个评测步**，不保存 checkpoint；不是只打印配置。底层脚本会进入 test 命名模式。

默认正式输出：

```text
/mnt/cfs/l1x67e/kele/GenieSim3.0-Dataset/checkpoints/g05_runs/r1pro/
tianji_recap_human_sft_v1_from_0912_morning_mix30_lr5e6_steps2000/
```

同名输出存在时脚本拒绝覆盖。要重新做一个实验，用新的 `EXP_NAME`。改变 `RECAP_GPUS` 时应同时调整 batch 或梯度累积以维持所需全局 batch。

独立全量核验：

```bash
.venv/bin/python scripts/utils/verify_recap_sft.py \
  --root /mnt/cfs/l1x67e/kele/data/recap_human_sft_shift1_v1 \
  --stats artifacts/inference_exports/g05_tianji_0912_morning_epoch5_step21505_inference/dataset_stats.json \
  --output artifacts/recap_dataset_audit/sft_integrity.json

PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_recap_windows.py tests/test_action_op_mask_loss.py
```

已核验全部 51,244 个窗口的原始来源、真实时间连续性、shift1 和完整动作 chunk；600 个视频链接与原视频一致，训练/验证 rollout 交集为 0。右臂 target 标量约 0.0023% 超出初始化统计的 min/max，约 0.406% 超出 q01/q99。这是分布检查，不代表这些值被裁掉。

当前训练器每次只评估一批验证样本，不是完整验证集。配置把纠错来源放在首位，使本轮周期评测取自纠错验证集；需要另外用完整任务评测判断旧能力是否退化。最终必须比较 morning 与纠错模型在同场景、同执行设置下的无人工接管成功率、失败类型和耗时，不能仅凭训练 loss 宣称成功率提升。

## 本次运行记录

15 项窗口边界、索引、读取失败和动作损失测试通过；真实 train/val 视频读取与 overfit 索引检查通过；8 卡 1 步训练和 1 批评测试跑正常退出。正式 2,000 步任务已作为独立后台进程启动，完成情况应以正式日志为准。

```bash
tail -f artifacts/recap_dataset_audit/sft_train_v1.log
```

启动时间、PID、输出目录与最近核验状态保存在 `artifacts/recap_dataset_audit/sft_run_v1.json`。其他检查产物：`sft_tests.log`、`sft_integrity.json`、`sft_loader_check.log` 和 `sft_resolved_config.log`（后者实际为单步试跑日志）。
