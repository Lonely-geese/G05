# 0917 + 0918 RJ45 合并训练

状态：已按用户要求停止本次训练，GPU 已释放，尚未生成 epoch 权重。用户将用 run_combined_rj45.sh 手动启动新实验。原运行记录见 training_status.json 和 train.log。

两个目录通过同一 dataset_group 合并加载，不重复复制视频、不再次 action shift。共 2,615 条轨迹、951,339 帧；每帧等权参与一个 epoch。

- 0917：1,286 条 / 535,558 帧。
- 0918：1,329 条 / 415,781 帧。
- 初始化：`base_checkpoint/g05-base/checkpoints/model_state_dict.pt`，新 optimizer/scheduler，`resume_ckpt=null`。
- 右臂固定：右臂和右夹爪屏蔽 FM loss，AR tokenizer 丢弃这两部分的 token；推理反归一化后强制输出当前观测的右臂/右夹爪位置。保留双臂状态和三路相机作为输入。
- 不训练原数据的 Cartesian pose 字段；模型使用每侧 7 个关节和 1 个夹爪维度，并保持 base 模型的 padding 布局。
- norm：对两份数据所有帧重新计算所有模型 state/action 字段的统计，stats_downsample_rate=1，按 32 步动作窗口及相对关节变换计算。没有复用旧 norm。被屏蔽的右侧 action 使用 dummy 归一化以避免放大静止噪声，但统计仍完整计算和保存。
- 10 epochs，8 卡 DDP，BF16，梯度累积 1。优先全局 batch 256（每卡 32）；如显存不足则全局 batch 128（每卡 16）。
- 学习率 4e-5、warmup 200 steps、cosine、weight decay 0.03。
- batch 256：每 epoch 3,717 steps，总计 37,170 steps；batch 128：每 epoch 7,433 steps，总计 74,330 steps。
- DistributedSampler 每 epoch 补齐 5 个样本，末批每卡 6 个样本；保留末批不丢数据。
- 每 epoch 保存一个 `checkpoints/step_<epoch*steps_per_epoch>.pt`。前 9 个含 optimizer/scheduler，最后一个为训练入口保存的推理权重；`last.pt` 指向最新。
- 全部数据用于训练。每 epoch 的评估使用训练数据，仅作诊断，不是独立验证集。

## 文件

- `dataset_stats.json`：新 norm。
- `norm_provenance.json`：来源、规模和 norm SHA256。
- `prepare.log`：全量统计和跨数据集边界样本验证。
- `right_arm_tests.log`：右臂 loss 屏蔽/推理保持测试。
- `smoke256.log`：8 卡 batch 256 短测，首步 CUDA OOM。
- `batch_selection.json`：降到 batch 128 的证据和最终参数。
- `train.log`：正式训练日志。
- `training_status.json`：运行 PID、输出目录、epoch 权重映射以及完成/失败状态。

配置：`configs/task/combined_rj45_0917_0918.yaml`、`configs/data/combined_rj45_0917_0918.yaml`。
入口：`run_combined_rj45.sh`；后台监督程序：`tools/launch_combined_rj45_training.py`。

查看日志：
```bash
tail -f /mnt/cfs/7rnh3z/kele/GalaxeaVLA/artifacts/combined_rj45_0917_0918/train.log
```

正式输出：`outputs/combined_rj45_0917_0918/left_arm_base_gbs128_ep10_20260919/`。
正式保存步数：7,433、14,866、22,299、29,732、37,165、44,598、52,031、59,464、66,897、74,330。

启动核验：正式 batch 128 已连续完成至少 10 个训练步骤，loss 有限，无 CUDA OOM。见 `startup_verification.json`。全量 norm 的独立均值核对见 `norm_verification.json`。训练尚在进行，不能把启动核验视为 10 epochs 已完成。
