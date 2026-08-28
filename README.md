# ConvLSTM 海表温度预测研究

本项目使用 NOAA OISST 日尺度海表温度（Sea Surface Temperature, SST）数据，研究 ConvLSTM 对下一日海温空间分布的预测能力，并系统比较归一化、海洋掩码、历史窗口长度和基线模型对结果的影响。

> 当前研究版由老师提供的复现代码整理而来。原始脚本仍保留在仓库根目录，正式实验统一使用 `prepare_data.py`、`train.py` 和 `evaluate.py`。

## 1. 研究主线

核心问题：

1. ConvLSTM 能否优于“明天等于今天”的持续性预测？
2. ConvLSTM 是否优于直接把多天海温作为通道输入的普通卷积模型？
3. 无归一化、Min-Max 归一化和 Z-score 标准化对训练稳定性与预测误差有何影响？
4. 将陆地 NaN 填 0 并计入损失，与使用海洋掩码相比会产生多大偏差？
5. 使用 3、5、7、10、14 或 30 天历史数据时，下一日预测效果如何变化？

推荐研究题目：

> 归一化、海洋掩码与时间窗口长度对 ConvLSTM 海表温度预测性能的影响

## 2. 公平实验原则

每组对比实验必须尽量只改变一个变量：

- 固定相同的空间区域和日期范围；
- 固定训练集、验证集和测试集的时间边界；
- 归一化参数只能由训练期数据计算；
- 测试指标只在有效海洋格点上计算；
- 归一化与未归一化模型使用相同的无界线性输出层；
- 使用相同的随机种子、优化器和早停规则；
- 至少与持续性预测比较；
- 最终关键实验建议使用 3 个随机种子，报告均值和标准差。

## 3. 数据含义

OISST 是 NOAA 发布的 0.25° 日尺度最优插值海温产品。每天的 `.nc` 文件包含一张规则经纬度网格上的海温分析场，并非某个固定时刻的原始卫星照片。

当前区域：

- 经度：100°E–132°E；
- 纬度：16°N–48°N；
- 空间大小：128×128；
- 2020 年共有 366 天。

一天的数据可以表示为 `[H, W] = [128, 128]`。一个 10 天预测样本为：

```text
X: [T, C, H, W] = [10, 1, 128, 128]
Y: [C, H, W]    = [1, 128, 128]
```

训练时 DataLoader 再增加 batch 维：

```text
X_batch: [B, T, C, H, W]
Y_batch: [B, C, H, W]
```

## 4. 研究版解决的原始问题

老师提供的原始代码适合展示基本流程，但存在以下会影响实验可信度的问题：

1. 按完整路径字符串排序文件，月份可能变成 `1, 10, 11, 12, 2, ...`；
2. 未验证重复日期、缺失日期和时间连续性；
3. 将陆地 NaN 当作 0℃ 并计入 MSE/RMSE；
4. 最后一层直接使用 `tanh` 限制的隐藏状态，未归一化时无法表示真实温度范围；
5. 整个训练集一次送入 GPU，显存被时间步激活占用；
6. 训练、验证、测试数据全部提前放入 GPU；
7. 没有保存验证集最优模型、早停、训练历史和随机种子；
8. 没有持续性预测和普通卷积基线；
9. 只报告一个包含陆地的整体 RMSE；
10. 模型、数据处理、训练和评估重复写在同一个脚本中。

## 5. 项目结构

```text
.
├── 2020/                         # 原始 OISST NetCDF 文件
├── data.npy                      # 老师版本生成的处理数据
├── prepare_data.py               # 日期校验、区域提取、生成规范 NPZ
├── train.py                      # ConvLSTM/Residual ConvLSTM/CNN 训练、验证、早停
├── evaluate.py                   # 海洋指标、持续性基线、结果图
├── sst_forecasting/
│   ├── data.py                   # 数据读取、归一化、滑动窗口、DataLoader
│   ├── models.py                 # ConvLSTM、残差 ConvLSTM 和普通卷积基线
│   ├── metrics.py                # 掩码损失和物理单位指标
│   └── utils.py                  # 随机种子、设备和文件工具
├── tests/test_core.py            # 张量形状、掩码、归一化测试
├── requirements.txt
├── ConvLSTM_pytorch-master/      # 上游参考实现与 MIT License
└── convlstm_*.py                 # 老师提供的 legacy 脚本
```

## 6. 环境安装

推荐 Python 3.10 或 3.11。服务器上的 PyTorch 应根据 CUDA 版本使用官方安装命令，随后安装其余依赖：

```bash
pip install -r requirements.txt
```

检查 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 7. 数据预处理

### 7.1 从原始 NetCDF 重建规范数据

```bash
python prepare_data.py \
  --input-dir 2020 \
  --output data/processed/sst_2020_east_china_sea.npz \
  --lon-min 100 --lon-max 132 \
  --lat-min 16 --lat-max 48
```

预处理会：

1. 从文件名提取 8 位日期并按日期排序；
2. 检查重复日期和缺失日期；
3. 提取指定经纬度区域；
4. 保留 NaN，不把陆地伪装成 0℃；
5. 保存 `sst`、`dates`、`lon` 和 `lat`。

生成的 `sst` 形状统一为 `[day, height, width]`。

### 7.2 兼容老师提供的 data.npy

```bash
python train.py --data data.npy --start-date 2020-01-01
```

研究版训练代码可直接读取现有 `data.npy`，但正式实验应优先使用重新按日期生成的 `.npz` 文件。

## 8. 训练可信 ConvLSTM 基线

RTX 4090 建议先从 batch size 4 或 8 开始：

```bash
python train.py \
  --data data/processed/sst_2020_east_china_sea.npz \
  --model convlstm \
  --normalization minmax \
  --seq-len 10 \
  --batch-size 4 \
  --hidden-dims 16 16 \
  --epochs 200 \
  --patience 20 \
  --seed 42 \
  --amp \
  --run-name convlstm-minmax-t10-seed42
```

输出目录：

```text
outputs/convlstm-minmax-t10-seed42/
├── best.pt       # 验证集最优模型
├── config.json   # 完整实验配置
└── history.csv   # 每轮训练和验证指标
```

## 9. 训练普通卷积基线

```bash
python train.py \
  --data data/processed/sst_2020_east_china_sea.npz \
  --model cnn \
  --normalization minmax \
  --seq-len 10 \
  --batch-size 8 \
  --epochs 200 \
  --patience 20 \
  --seed 42 \
  --run-name cnn-minmax-t10-seed42
```

CNN 会把 10 天当作 10 个输入通道，用于判断循环记忆是否真正带来收益。

## 10. 训练 Residual ConvLSTM

普通 ConvLSTM 需要直接生成完整的下一日海温场，但相邻两日的 SST 通常非常相似。
Residual ConvLSTM 改为学习日变化量：

```text
下一日预测 = 最后一个输入日 + ConvLSTM 预测的日变化量
```

其输出层初始化为 0，因此未经训练的初始预测严格等于持续性预测。训练命令：

```bash
python train.py \
  --data data/processed/sst_2020_east_china_sea.npz \
  --model residual-convlstm \
  --normalization minmax \
  --seq-len 10 \
  --batch-size 4 \
  --hidden-dims 16 16 \
  --epochs 200 \
  --patience 20 \
  --seed 42 \
  --amp \
  --run-name residual-convlstm-minmax-t10-seed42
```

### R2：降低学习率并自动衰减

R1 在第 10 轮取得最佳结果，之后验证损失明显震荡。R2 保持数据、模型和掩码不变，
只调整优化过程：初始学习率降为 `3e-4`，验证损失停滞 6 轮后学习率减半，最低降到
`1e-5`，早停等待增加到 40 轮。

```bash
python train.py \
  --data data/processed/sst_2020_east_china_sea.npz \
  --model residual-convlstm \
  --normalization minmax \
  --loss-mask ocean \
  --seq-len 10 \
  --batch-size 4 \
  --hidden-dims 16 16 \
  --epochs 300 \
  --patience 40 \
  --learning-rate 3e-4 \
  --lr-scheduler plateau \
  --lr-patience 6 \
  --lr-factor 0.5 \
  --min-learning-rate 1e-5 \
  --seed 42 \
  --amp \
  --run-name residual-convlstm-r2-minmax-t10-seed42
```

`history.csv` 会记录每轮实际使用的 `learning_rate`。模型选择始终只依据验证损失，
测试集不参与学习率调整或早停。

### R3：恢复较大学习率并启动主干梯度

R2 表明从 `3e-4` 开始并持续降低学习率会让模型停留在近似常数变化预测。
R3 恢复 R1 已验证有效的固定 `1e-3`，并将残差输出头从严格全零改为标准差
`1e-3` 的极小随机初始化。

全零输出头虽然能让初始模型严格等于持续性预测，但第一批数据反向传播时，梯度无法
穿过零权重到达 ConvLSTM 主体。极小随机初始化仍使初始修正接近 0，同时让循环主体
从第一批数据开始学习。

```bash
python train.py \
  --data data/processed/sst_2020_east_china_sea.npz \
  --model residual-convlstm \
  --residual-readout-init-std 1e-3 \
  --normalization minmax \
  --loss-mask ocean \
  --seq-len 10 \
  --batch-size 4 \
  --hidden-dims 16 16 \
  --epochs 200 \
  --patience 20 \
  --learning-rate 1e-3 \
  --lr-scheduler none \
  --seed 42 \
  --amp \
  --run-name residual-convlstm-r3-gradient-start-t10-seed42
```

`--residual-readout-init-std 0` 保留原来的严格持续性初始化，可用于复现 R1/R2。
R3 与 R1 只有残差头初始化不同。

### R4：增加局部日变化异常损失

R3 的整体 RMSE 和 Skill 已经优于持续性预测，但预测日变化的空间标准差只有真实值的
约 0.384%，说明改进主要来自接近全海域一致的平均降温修正。R4 保持 R3 的模型、
初始化、学习率和数据划分不变，只增加局部日变化异常损失：

```text
日变化 = 下一日海温 - 最后一个输入日海温
局部异常 = 日变化 - 当天海洋格点的平均日变化
总损失 = 海温预测 MSE + 权重 × 局部异常 MSE
```

推荐先用权重 `1.0` 做单变量对照：

```bash
python train.py \
  --data data/processed/sst_2020_east_china_sea.npz \
  --model residual-convlstm \
  --residual-readout-init-std 1e-3 \
  --normalization minmax \
  --loss-mask ocean \
  --change-anomaly-weight 1.0 \
  --seq-len 10 \
  --batch-size 4 \
  --hidden-dims 16 16 \
  --epochs 200 \
  --patience 20 \
  --learning-rate 1e-3 \
  --lr-scheduler none \
  --num-workers 4 \
  --seed 42 \
  --device cuda \
  --amp \
  --output-root /home/dataDisk/sn/xya/convlstm-sst-runs \
  --run-name residual-convlstm-r4-anomaly-loss-t10-seed42
```

`--change-anomaly-weight 0` 完全关闭新损失并复现旧训练目标。训练历史会分别记录总目标、
原始预测 MSE 和局部异常 MSE，便于判断空间结构改善是否以整体 RMSE 变差为代价。
R4 的最佳检查点、学习率调度（若启用）和早停均依据验证集“总损失”，测试集仍不参与
模型选择。

## 11. 测试与持续性基线

```bash
python evaluate.py \
  --checkpoint outputs/convlstm-minmax-t10-seed42/best.pt \
  --data data/processed/sst_2020_east_china_sea.npz \
  --save-examples 3
```

评估输出：

- 海洋格点 RMSE（℃）；
- 海洋格点 MAE（℃）；
- 模型平均偏差 Bias（℃）；
- 持续性预测 RMSE（直接用最后一天预测第 11 天）；
- 持续性预测 MAE（℃）；
- Skill：`1 - model_rmse / persistence_rmse`；
- 真实与预测日变化的 Pearson 相关系数；
- 日变化标准差之比（预测标准差 / 真实标准差）；
- 真实与预测的平均日变化（℃）；
- 测试样本数和首尾日期；
- 若干日期的海温图、真实/预测日变化图和绝对误差图。

```text
Skill > 0：模型优于持续性预测
Skill = 0：与持续性预测相同
Skill < 0：模型不如直接使用最后一天
```

## 12. 核心实验矩阵

| 编号 | 模型 | 归一化 | 海洋掩码 | 窗口 |
|---|---|---|---|---:|
| B0 | 持续性预测 | 无要求 | 有 | 10 |
| B1 | CNN | Min-Max | 有 | 10 |
| C1 | ConvLSTM | 无 | 有 | 10 |
| C2 | ConvLSTM | Min-Max | 有 | 10 |
| C3 | ConvLSTM | Z-score | 有 | 10 |
| C4 | ConvLSTM | Min-Max | 无 | 10 |
| R1 | Residual ConvLSTM | Min-Max | 有 | 10 |
| R2 | Residual ConvLSTM + LR 衰减 | Min-Max | 有 | 10 |
| R3 | Residual ConvLSTM + 小随机残差头 | Min-Max | 有 | 10 |
| R4 | R3 + 局部日变化异常损失 | Min-Max | 有 | 10 |
| R5 | R3 + 异常损失权重 0.25 | Min-Max | 有 | 10 |
| R6 | R3 + 异常损失权重 0.5 | Min-Max | 有 | 10 |

掩码消融实验使用：

```bash
python train.py ... --loss-mask all
```

正式海洋结果使用默认设置：

```bash
python train.py ... --loss-mask ocean
```

时间窗口实验只改变 `--seq-len`：

```text
3、5、7、10、14、30
```

## 13. 数据划分

默认按目标日期进行时间顺序划分：

```text
前 70% 日期：训练目标
中间 20% 日期：验证目标
最后 10% 日期：测试目标
```

验证和测试样本可以使用边界之前已经观测到的历史海温作为输入，但标签日期不会跨集合重复。这符合“一天前预报”场景。

如果以后使用多年数据，建议固定同一区域，并采用独立年份划分，例如：

```text
2013–2020：训练
2021：验证
2022：测试
```

## 14. 原始代码与引用

ConvLSTM 参考实现：

- Andrea Palazzi, `ndrplz/ConvLSTM_pytorch`，MIT License；
- Shi et al., 2015, *Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting*。

数据来源：

- NOAA/NCEI 1/4° Daily Optimum Interpolation Sea Surface Temperature (OISST), Version 2.1；
- https://www.ncei.noaa.gov/products/optimum-interpolation-sst

## 15. 开发状态

- `teacher-original`：老师提供的初始版本；
- `research/improved-baseline`：规范化研究版本；
- 已完成：可信 ConvLSTM 基线，测试 RMSE 0.2841℃，未超过持续性基线 0.2760℃；
- 已完成：R1 Residual ConvLSTM 获得 +2.87% 测试 Skill，但预测日变化幅度偏弱；
- 已完成：R2 低初始学习率实验失败，测试 Skill -2.20%，变化幅度比仅 0.011；
- 已完成：R3 用极小随机残差头获得当前最佳 RMSE 0.2663℃ 和 +3.52% Skill，
  但日变化空间幅度仍严重塌缩；
- 已完成：R4 使用异常损失权重 1.0，将变化幅度比从 0.0038 提高到 0.2338，
  相关系数由负转正，但测试 Skill 从 R3 的 +3.52% 降到 +2.16%；
- 当前目标：并行运行权重 0.25（R5）和 0.5（R6），寻找整体精度与空间变化的折中；
- 后续目标：进行归一化、掩码和时间窗口消融实验。
