# cleaned_rj45 训练准备

数据：`/mnt/cfs/7rnh3z/kele/0917/cleaned_rj45`，1,286 条轨迹、535,558 帧。
初始化：`base_checkpoint/g05-base/checkpoints/model_state_dict.pt`；使用同目录提供的 Qwen processor 和 action_tokenizer。
原始数据已 action shift 1，本次不再次移位。

## 正式训练参数

- 8 张 GPU，每卡 batch 16，梯度累积 1，全局 batch 128。
- 5 epochs，4,185 steps/epoch，共 20,925 optimizer steps。
- 每个 epoch 的最后一个 batch 为 8 个样本；DistributedSampler 全局补齐 2 个样本。
- 从 base 权重初始化新 optimizer/scheduler，学习率 4e-5，warmup 200，cosine。
- BF16 混合精度；右臂和右夹爪不参与 FM loss，也不生成对应 AR 训练 token。
- 保留双臂 proprio 和三路相机；推理后处理将右臂和右夹爪输出替换为当前观测状态。
- 全量重算 action/state norm，stats_downsample_rate=1，32-step horizon；文件 `dataset_stats.json`，来源和校验值见 `norm_provenance.json`。
- 全部数据用于训练；评估使用训练数据，仅用于诊断，不是独立验证指标。
- 每 4,185 steps 保存一次，文件为 step_4185.pt、step_8370.pt、step_12555.pt、step_16740.pt、step_20925.pt。

配置：`configs/task/cleaned_rj45.yaml`、`configs/data/cleaned_rj45.yaml`。
启动脚本：`run_cleaned_rj45.sh`。

## 用户手动后台启动

```bash
cd /mnt/cfs/7rnh3z/kele/GalaxeaVLA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup bash run_cleaned_rj45.sh train \
  > artifacts/cleaned_rj45_setup/train.log 2>&1 < /dev/null &
echo $! > artifacts/cleaned_rj45_setup/train.pid
```

脚本自动激活 `.venv` 和 `.env`，并校验本次新计算的 norm。
正式输出目录：
`outputs/cleaned_rj45/cleaned_rj45_left_arm_base_gbs128_ep5_20260918/`

查看日志：

```bash
tail -f artifacts/cleaned_rj45_setup/train.log
```

若同名输出目录已经存在，脚本会拒绝覆盖。新实验可通过 `EXP_NAME=新名称` 更换目录；这不是断点恢复命令。

## 少步数验证

`smoke.log` 是首步训练和评估验证。`smoke3.log` 是连续 3 步训练、最后一步评估的验证日志。
验证使用原始 base checkpoint 和真实 8 卡 batch 128，不保存训练权重。
正式训练尚未启动，等待用户手动执行上面的命令。

验证结果（2026-09-18）：连续 3 个 optimizer steps + 1 次评估通过，退出码 0。
三步 loss 分别为 6.9502、7.0577、7.0883，均为有限值；短测不代表模型收敛质量。
所有训练进程已退出，正式训练未启动。
热身后的两步约 2–2.5 秒/步，20,925 步纯训练约 11.6–14.5 小时；
计入启动、评估和 checkpoint 写入后，暂估 12–16 小时。
此估计仅基于两步稳态样本，长时间 I/O 变化可能影响耗时。
