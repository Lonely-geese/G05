# 八卡触觉训练短跑：50 steps

本次从 G05 base 重新加载，触觉分支随机初始化；未从之前双卡短跑续训。
8 卡 × 每卡 batch 4，梯度累积 1，有效 batch 32，50 个 optimizer steps，
约 1,600 个样本呈现。seed=7，目标学习率 4e-5，BF16 autocast，FM-only，
视觉语言/动作专家/触觉分支联合训练，左臂和左夹爪为训练目标。
实际组合配置为 cosine 调度、固定 warmup=200 步、weight decay=0.03。
因此本次 50 步全部处于 warmup，日志中的更新后学习率从 2e-7 升到 1e-5，
尚未到达目标学习率 4e-5；不能据此判断 warmup 后的训练稳定性。

| Optimizer step | 8 卡平均 FM loss 的区间均值 |
| --- | ---: |
| 1–10 | 0.58542 |
| 11–20 | 0.48517 |
| 21–30 | 0.36895 |
| 31–40 | 0.30556 |
| 41–50 | 0.31684 |

前后 10 步均值下降 45.88%。第一步 0.4930，最后一步 0.2700；最低
0.1555（step 47），最高 0.9611（step 6）。整体向下，但并非单调下降，
step 42 有一次 0.7412 的尖峰。不同 batch 与 FM 噪声/时间采样都会带来波动，
本次没有对单次尖峰的来源做归因。

50 步 loss 和 50 条梯度范数都为有限值；裁剪前梯度范数范围 1.1133–9.5736，
裁剪阈值沿用配置 1.0。训练无 OOM，末尾 FM 推理评估完成。
退出时 DataLoader/pin-memory 线程出现 BrokenPipe/ConnectionRefused/
cannot-join-current-thread 清理异常，但训练已达到 50 步，launcher 最终返回 0。
所有 8 卡显存已释放。这个退出清理问题本次只记录，未改训练器。

结论：本次短跑的训练损失总体下降，未见数值发散；不能据此证明已收敛、
触觉相比纯视觉有收益或真机任务成功率提高。当前没有独立验证集，最后的
单 batch 评估也不能作为泛化结论。未保存模型/优化器 checkpoint。

## 文件

- `loss_curve.png` / `loss_curve.svg`：逐步曲线与 5 步移动平均。
- `loss.csv`：1-based optimizer step 与跨卡平均 loss（日志精度 4 位小数）。
- `summary.json`：区间统计、范围和梯度范数检查。
- `offline_history.json`：本地 W&B 历史；其中 `fm_loss` 是 rank 0 的记录，
  **不是** `loss.csv` 中控制台 all-reduce 后的八卡平均值。
- 原始日志：`../8gpu_50steps_20260922.log`。
- 运行输出：`../../../outputs/combined_rj45_tactile_shift3/ftp1_tactile_8gpu_50steps_20260922/`。

## 复现命令（从 G05 根目录）

使用新的 EXP_NAME 避免复用已有输出目录：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 PER_GPU=4 \
  DRY_RUN_STEPS=50 EXP_NAME=ftp1_tactile_8gpu_50steps_repeat \
  bash run_combined_rj45_tactile.sh smoke
```
