# 每卡 batch 8 / 16 实测（2026-09-22）

8 × NVIDIA A800 80GB；原脚本 smoke 模式，每组 30 次真实优化器更新和一次评估。
沿用任务配置：BF16 autocast、梯度检查点、完整 VLM/动作/触觉分支联合训练、AdamW、梯度累积 1。
两组均从同一基础权重加载，触觉分支按 seed=7 初始化。训练全部处于 200 步 warmup 中。

| 每卡 batch | 全局 batch | 采样显存峰值/卡 | 秒/步 | 总吞吐（样本/秒） | 结果 |
| --- | --- | --- | --- | --- | --- |
| 8 | 64 | 63.05 GiB | 3.45 | 18.55 | 30 步 + 评估完成，退出码 0 |
| 16 | 128 | 72.04 GiB | 6.25 | 20.48 | 30 步 + 评估完成，退出码 0 |

秒/步按第 10 次更新完成到第 30 次更新完成的耗时除以 20 计算，排除初始化及最后评估；包含数据等待和训练日志开销。
显存为 nvidia-smi 每 2 秒采样、取全部 8 卡的最大值，不是 PyTorch 分配器精确瞬时峰值。
两组各 30 条 loss 和梯度范数全部有限，未发生 OOM。评估产物均存在。
退出时有已有的 DataLoader/pin-memory 清理异常（BrokenPipe、ConnectionRefused/Reset、cannot join current thread），launcher 最终均返回 0。
本测试验证当前数据和配置下的短跑容量，不证明长训收敛或所有批次均不会 OOM。batch=16 采样峰值约余 8 GiB，比 batch=8 总吞吐高约 10%。

## 正式训练状态

为释放显存，原正式训练在第 61 步结束，当时尚未到第 2000 步首次 checkpoint。
测试结束后曾按原 batch=4 从基础权重重新启动（不属于 checkpoint 续训）。
随后按用户要求在第 153 步停止，尚未保存 checkpoint；当前正式训练保持停止。
DataLoader 清理修复后又完成完整模型的 8 卡 × batch 16、10 步与评估，
退出码 0，无清理异常，子进程和显存均已释放。
验证结果：[validation.json](../ftp1_tactile_cleanup_verify_20260922_100800/validation.json)。

- 旧实验：`ftp1_tactile_train_restarted_20260922_095930`
- 原始日志仅保留在本机，未纳入 Git。
- 结果：[results.json](results.json)；各卡显存采样：`batch8_gpu.csv` / `batch16_gpu.csv`。

## 复现测试

从 G05 根目录执行，确保相应 GPU 空闲：

```bash
NPROC=8 PER_GPU=8 DRY_RUN_STEPS=30 bash run_combined_rj45_tactile.sh smoke
NPROC=8 PER_GPU=16 DRY_RUN_STEPS=30 bash run_combined_rj45_tactile.sh smoke
```
