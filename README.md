# 基于 Residual ConvLSTM 的海表温度次日预测

本项目使用 NOAA OISST 日尺度海表温度（Sea Surface Temperature, SST）数据，研究如何根据连续 10 天的海温场预测下一天海温。项目从老师提供的 ConvLSTM 复现代码出发，重建了可复现的数据、训练和评估流程，并针对强持续性基线、局部变化塌缩和过早停止等问题进行了受控改进。

当前核心实验已经完成：**R3 是整体温度误差最低的模型，R7 是兼顾整体精度与局部海温变化的推荐模型。**

## 项目亮点

- 按真实日期排序并检查重复、缺失和不连续日期；
- 归一化参数只由训练期数据计算，避免数据泄漏；
- 陆地 NaN 不参与损失和指标，所有主要结果只统计有效海洋格点；
- 使用 persistence（直接用最后一天预测下一天）作为强基线；
- 将直接预测完整海温场改为 Residual ConvLSTM 预测日变化量；
- 增加局部日变化异常损失，避免模型只学习全海域统一升降温；
- 增加最少训练轮数和多指标检查点，保留后期才形成的空间变化能力；
- 保存完整配置、逐轮历史、物理单位指标和可视化结果。

## 研究任务

一天的 SST 数据是一张二维海温网格。默认任务使用前 10 天预测第 11 天：

```text
输入 X：[B, T, C, H, W] = [B, 10, 1, 128, 128]
目标 Y：[B, C, H, W]    = [B, 1, 128, 128]
```

其中 `B` 是 batch size，`T` 是时间长度，`C=1` 表示海温通道。

默认数据范围和划分：

| 项目 | 设置 |
| --- | --- |
| 数据 | NOAA OISST V2.1，2020 年日尺度数据 |
| 空间区域 | 100°E–132°E，16°N–48°N |
| 网格大小 | 128×128 |
| 日期数量 | 366 天 |
| 训练/验证/测试样本 | 246 / 73 / 37 |
| 测试日期 | 2020-11-25 至 2020-12-31 |
| 归一化 | 训练集全局 Min-Max |
| 损失与指标区域 | 有效海洋格点 |

数据按目标日期顺序划分，验证集和测试集可以使用分界点之前已经观测到的历史海温，但标签日期不会跨集合重复。

## 方法

### Persistence 基线

海温相邻两天通常非常接近，因此最重要的基线不是随机预测，而是：

```text
下一日预测 = 最后一个输入日
```

如果模型不能超过 persistence，就不能证明它学到了有用的次日变化。

### Residual ConvLSTM

直接 ConvLSTM 需要重新生成完整海温场，而真正需要学习的主要是相邻两日的小变化。Residual ConvLSTM 改为：

```text
下一日预测 = 最后一个输入日 + ConvLSTM 预测的日变化量
```

残差输出头采用标准差 `1e-3` 的小随机初始化，使初始预测接近 persistence，同时从第一批数据开始就能把梯度传入 ConvLSTM 主体。

### 局部日变化异常损失

仅优化完整海温 MSE 时，模型可能只学会全海域统一升温或降温。为强调局部冷暖结构，项目增加：

```text
日变化 = 下一日海温 - 最后一个输入日海温
局部异常 = 日变化 - 当天海洋格点平均日变化
总损失 = 海温预测 MSE + λ × 局部异常 MSE
```

R7 使用 `λ=0.5`。

### 最少训练轮数与多检查点

实验发现，平均升降温通常在前几轮学会，而局部空间结构到约第 22 轮才开始形成。最终训练器支持：

- `--min-epochs`：达到指定轮数前不触发早停；
- `best_objective.pt`：验证集组合目标最低；
- `best_rmse.pt`：验证集 RMSE 最低；
- `best_anomaly.pt`：验证集局部异常损失最低；
- `best_correlation.pt`：验证集日变化相关性最高；
- `last.pt`：训练最后一轮；
- `best.pt`：兼容旧命令，等同于最佳组合目标检查点。

模型选择只使用验证集，测试集不参与调参、早停或检查点选择。

## 主要结果

所有表格均采用相同的时间划分、海洋掩码和物理单位评估协议。

| 实验 | 主要变化 | RMSE（℃） | MAE（℃） | Skill | 变化幅度比 | 变化相关性 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Persistence | 最后一天直接预测下一天 | 0.276034 | 0.181800 | 0% | — | — |
| 直接 ConvLSTM | 直接生成下一日完整海温 | 0.284106 | 0.188126 | -2.92% | — | — |
| R1 | Residual ConvLSTM | 0.268121 | 0.173600 | +2.87% | — | — |
| R2 | 低学习率与持续衰减 | 0.282119 | 0.188824 | -2.20% | 0.0110 | -0.1490 |
| R3 | 小随机残差头，精度导向 | **0.266321** | **0.172299** | **+3.52%** | 0.0038 | -0.0748 |
| R4 | 异常损失权重 1.0 | 0.270073 | 0.177451 | +2.16% | 0.2338 | **+0.2014** |
| R5 | 异常损失权重 0.25 | 0.266385 | 0.172371 | +3.50% | 0.0085 | -0.1330 |
| R6 | 异常损失权重 0.5，旧早停 | 0.266534 | 0.172580 | +3.44% | 0.0177 | -0.1417 |
| **R7** | **权重 0.5 + 最少 50 轮 + 多检查点** | **0.268501** | **0.175043** | **+2.73%** | **0.2606** | **+0.1859** |

指标解释：

- `Skill = 1 - 模型 RMSE / persistence RMSE`，大于 0 表示超过 persistence；
- 变化幅度比为“预测日变化标准差 / 真实日变化标准差”，越接近 1 越好；
- 变化相关性衡量预测与真实局部冷暖变化的一致程度。

### 最终结论

1. **直接 ConvLSTM 没有超过 persistence。** 海温次日预测必须认真对待强持续性基线。
2. **Residual ConvLSTM 明显更适合该任务。** R1 首次获得正 Skill，证明学习日变化比重建完整海温更有效。
3. **较低学习率并不一定更稳定或更优。** R2 的低学习率和持续衰减使模型停留在接近 persistence 的弱修正状态。
4. **整体 RMSE 好不代表学会了空间变化。** R3 的 RMSE 最低，但预测变化幅度只有真实值的 0.384%，且相关性为负。
5. **局部异常损失能够缓解空间塌缩。** R4 将变化幅度比提升到 0.234，并把相关性变为正数，但权重 1.0 带来更大暖偏差。
6. **平均变化和局部空间变化的学习速度不同。** R5/R6 过早选择第 2 轮模型，丢失了第 22 轮以后逐渐形成的空间能力。
7. **R7 是当前最佳综合模型。** 相比 R3，R7 的 RMSE 仅增加 0.82%，变化幅度提高约 67.9 倍，相关性由负转正，同时仍保持 +2.73% Skill。

因此，本项目将 **R3 作为精度上限模型**，将 **R7 第 47 轮最佳组合目标/最佳 RMSE 检查点作为推荐主结果**。

## 快速开始

### 1. 安装环境

推荐 Python 3.10 或 3.11。先按照服务器 CUDA 版本安装 PyTorch，再安装其余依赖：

```bash
pip install -r requirements.txt
```

检查 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 2. 预处理数据

仓库已经包含 2020 年原始 NetCDF 文件，可生成规范 NPZ：

```bash
python prepare_data.py \
  --input-dir 2020 \
  --output data/processed/sst_2020_east_china_sea.npz \
  --lon-min 100 --lon-max 132 \
  --lat-min 16 --lat-max 48
```

预处理会按日期排序、检查重复和缺失日期、提取区域、保留陆地 NaN，并保存 `sst`、`dates`、`lon` 和 `lat`。

老师提供的旧 `data.npy` 仍可通过 `--start-date 2020-01-01` 读取，但正式实验建议使用 NPZ。

### 3. 训练推荐的 R7

```bash
python train.py \
  --data data/processed/sst_2020_east_china_sea.npz \
  --model residual-convlstm \
  --residual-readout-init-std 1e-3 \
  --normalization minmax \
  --loss-mask ocean \
  --change-anomaly-weight 0.5 \
  --seq-len 10 \
  --batch-size 4 \
  --hidden-dims 16 16 \
  --epochs 200 \
  --min-epochs 50 \
  --patience 20 \
  --learning-rate 1e-3 \
  --lr-scheduler none \
  --num-workers 4 \
  --seed 42 \
  --device cuda \
  --amp \
  --output-root outputs \
  --run-name residual-convlstm-r7
```

### 4. 评估检查点

推荐首先评估最佳组合目标：

```bash
python evaluate.py \
  --checkpoint outputs/residual-convlstm-r7/best_objective.pt \
  --data data/processed/sst_2020_east_china_sea.npz \
  --device cuda \
  --amp \
  --save-examples 3
```

也可以将检查点替换为 `best_rmse.pt`、`best_anomaly.pt`、`best_correlation.pt` 或 `last.pt`。每种检查点默认写入独立的 `evaluation_<checkpoint>/` 目录，不会覆盖其他评估结果。

## 输出内容

```text
outputs/<run-name>/
├── config.json
├── history.csv
├── best.pt
├── best_objective.pt
├── best_rmse.pt
├── best_anomaly.pt
├── best_correlation.pt
├── last.pt
├── evaluation_best_objective/
│   ├── metrics.json
│   └── examples.png
└── ...
```

`examples.png` 包含前一天、真实下一天、模型预测、真实日变化、预测日变化和绝对误差。

## 项目结构

```text
.
├── 2020/                         # 原始 OISST NetCDF 数据
├── prepare_data.py               # 日期校验、区域提取和 NPZ 生成
├── train.py                      # 训练、验证、早停和多检查点保存
├── evaluate.py                   # 测试指标和可视化
├── sst_forecasting/
│   ├── data.py                   # 数据读取、归一化、滑动窗口
│   ├── models.py                 # ConvLSTM、Residual ConvLSTM、CNN
│   ├── metrics.py                # 掩码损失、Skill 和日变化指标
│   └── utils.py                  # 随机种子、设备和文件工具
├── tests/test_core.py            # 核心单元测试
├── experiment_records/           # 可版本控制的实验配置、指标和图片
├── ConvLSTM_pytorch-master/      # 上游 ConvLSTM 参考实现
├── convlstm_*.py                 # 老师提供的旧版脚本
└── requirements.txt
```

## 实验记录

根 README 只总结最终项目，具体受控改动、失败原因和逐轮经验保存在各实验目录：

| 实验 | 说明 |
| --- | --- |
| [直接 ConvLSTM 基线](experiment_records/convlstm_minmax_t10_seed42/) | 可信评估流程下未超过 persistence |
| [R1](experiment_records/residual_convlstm_minmax_t10_seed42/) | Residual ConvLSTM 首次超过 persistence |
| [R2](experiment_records/residual_convlstm_r2_low_lr_t10_seed42/) | 低学习率与持续衰减失败 |
| [R3](experiment_records/residual_convlstm_r3_gradient_start_t10_seed42/) | 小随机残差头，整体精度最佳 |
| [R4](experiment_records/residual_convlstm_r4_anomaly_weight1_t10_seed42/) | 强异常损失恢复空间变化 |
| [R5](experiment_records/residual_convlstm_r5_anomaly_weight025_t10_seed42/) | 异常损失权重过弱 |
| [R6](experiment_records/residual_convlstm_r6_anomaly_weight05_t10_seed42/) | 发现过早停止和单检查点问题 |
| [R7](experiment_records/residual_convlstm_r7_multicheckpoint_weight05_t10_seed42/) | 最少训练轮数与多检查点，最终综合模型 |

实验目录保留小型 `config.json`、`history.csv`、`metrics.json` 和图片；大型模型权重通过 `.gitignore` 排除。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖日期排序、训练期归一化、时间划分、张量形状、Residual 初始化、掩码损失、日变化诊断、异常损失、学习率调度和最少训练轮数逻辑。

## 局限与可选扩展

当前结论基于 2020 年数据、一个时间划分和随机种子 42，适合完成本次小型复现与改进项目，但不应直接解释为跨年份、跨海域的普遍结论。后续可选扩展包括：

- 使用多个随机种子报告均值和标准差；
- 使用多年数据进行独立年份验证和测试；
- 系统比较 Min-Max、Z-score 和无归一化；
- 进行海洋掩码和时间窗口长度消融；
- 引入风场、海流、海表高度等外部物理变量。

## 数据、引用与许可证说明

- 数据：NOAA/NCEI 1/4° Daily Optimum Interpolation Sea Surface Temperature (OISST), Version 2.1；
- ConvLSTM：Shi et al., 2015, *Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting*；
- 参考实现：Andrea Palazzi, `ndrplz/ConvLSTM_pytorch`。

`ConvLSTM_pytorch-master/` 中的上游参考实现保留其 MIT License。仓库其余研究代码目前未在根目录声明独立许可证；若计划公开复用或接受外部贡献，建议由仓库所有者补充根级许可证。
