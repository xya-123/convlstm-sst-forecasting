# 🌊 ConvLSTM 海表温度预测项目

本项目使用 ConvLSTM（卷积长短期记忆网络 对 2020 年每日海表温度（SST）数据进行时序预测。  
模型通过 过去 10 天的 SST 数据 预测 第 11 天的 SST，并在测试集上计算 RMSE 误差。

---

## 📁 项目结构

```
project/
│
├── 2020/                     # 存放原始 NetCDF 文件（按月份分文件夹）
│   ├── 01/
│   │   ├── oisst-avhrr-v02r01.20200101.nc
│   │   ├── ...
│   └── ...
│
├── data.py                   # 数据提取与预处理脚本（生成 data.npy）
├── convlstm_guiyihua.py      # 模型训练、验证与测试（含归一化处理）
└── README.md                 # 项目说明文档
```

---

## 🧩 功能概述

该项目主要包括两个部分：  
1. `data.py`：数据提取与 `.npy` 文件生成  
2. `convlstm_guiyihua.py`：ConvLSTM 模型训练、验证与测试（包含归一化与反归一化）

---

## 📊 一、数据处理（`data.py`）

### 1. 数据来源
文件夹 `2020/` 中包含 2020 年每日平均 SST 的 NetCDF 文件：  
```
oisst-avhrr-v02r01.20200101.nc
oisst-avhrr-v02r01.20200102.nc
...
```
每个文件包含变量：
- `lon`：经度 (0–360)
- `lat`：纬度 (-90–90)
- `sst`：海表温度（Sea Surface Temperature）

### 2. 提取区域范围
从经纬度中提取：
- 经度：100°E–132°E
- 纬度：16°N–48°N
- 区域大小：128×128

### 3. 输出结果
提取后，将每日 SST 数据保存为：
```
data.npy  # shape = (365, 1, 1, 128, 128)
```
其中：
- 第 1 维为时间（日序）
- 最后两维为经纬度格点

---

## 🧠 二、模型训练（`convlstm_guiyihua.py`）

### 1. 数据集构建
- 输入：连续 10 天 SST
- 输出：第 11 天 SST
- 数据集形状：
  ```
  X.shape = (356, 10, 1, 128, 128)
  Y.shape = (356, 1, 128, 128)
  ```

### 2. 数据划分
按比例分为：
- 训练集：70%
- 验证集：20%
- 测试集：10%

---

## ⚙️ 三、归一化与反归一化

### 1. 归一化策略
- 仅使用 训练集 计算最小值与最大值；
- 验证集与测试集使用相同归一化参数；
- 归一化公式：
  ```python
  x_norm = (x - min_train) / (max_train - min_train)
  ```

### 2. 反归一化计算误差
在测试阶段使用训练集的 min/max 还原预测结果：
```python
x_real = x_norm * (max_train - min_train) + min_train
```
然后计算 RMSE：
```python
rmse = sqrt(mean_squared_error(Y_true.flatten(), Y_pred.flatten()))
```

---

## 🧮 四、模型结构（ConvLSTM）

模型由两层卷积 LSTM 组成：
```python
ConvLSTM(
    input_dim=1,
    hidden_dim=[8, 1],
    kernel_size=(3, 3),
    num_layers=2
)
```

- 第一层提取时空特征；
- 第二层输出下一时间步的 SST。

---

## 🔧 五、训练与验证

- 优化器：Adam (`lr=1e-3`)  
- 损失函数：MSELoss  
- 训练轮数：1000  
- 每轮输出：
  ```
  Epoch [x/x] | Train Loss: ... | Val Loss: ...
  ```

---

## 🧾 六、测试与结果

### 1. 测试方法
- 使用测试集进行预测；
- 预测结果反归一化；
- 计算并输出结果。

### 2. 测试结果
1) 原始代码
- 数据集: 使用**中国东海**海域在2020年1月1日至2020年12月31日的海表温数据中的前70%的数据作为训练集，后10%的数据作为测试集
- RMSE：0.7186

2) 去除归一化操作
- 数据集: 使用**中国东海**海域在2020年1月1日至2020年12月31日的海表温数据中的前70%的数据作为训练集，后10%的数据作为测试集
- RMSE：15.7762

3) 原始代码 + 更改数据集
- 数据集: 使用**中国南海**海域在2013年1月1日至2022年12月31日的海表温数据中的前9年的数据作为训练集，后1年的数据作为测试集
- RMSE: 0.1437

4) 去除归一化操作 + 更改数据集
- 数据集: 使用**中国南海**海域在2013年1月1日至2022年12月31日的海表温数据中的前9年的数据作为训练集，后1年的数据作为测试集
- RMSE: 0.1486

### 3. 结果分析
1) 归一化操作的作用
归一化操作可以避免不同样本间数值尺度差异导致训练不稳定，并确保模型能够聚焦于数据的空间—时间结构而非绝对量级，从而提升收敛速度与泛化性能。
在中国东海海域数据集中，存在大量的陆地数据，原始数据为NaN，这些数据在预处理过程中会被填0，如果不归一化，填0的数据会被模型认为数据中存在大量温度为0℃的信息，影响信息的分布。使用归一化操作后所有数据被压缩到0～1之间，模型能够更专注于数据的空间—时间结构，而不是绝对的温度值。
而在中国南海海域数据集中，陆地数据极少，不归一化也不会明显影响模型的预测能力。

2) 更改数据集的作用
使用时间跨度更长的数据集让模型学习到更多样的海表温度变化趋势，从而提高预测准确性。
---

## 💻 七、运行说明

1. 生成数据文件
   ```bash
   python data.py
   ```
   输出：`data.npy`

2. 训练并测试模型
   ```bash
   python convlstm_guiyihua.py
   ```
   输出：
   - 每轮训练与验证损失；
   - 最终测试集 RMSE。

---

## 🧱 八、环境依赖

推荐环境：
- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- 依赖安装：
  ```bash
  pip install numpy torch xarray netCDF4 scikit-learn
  ```

---

## 📚 九、引用
1, Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W.-K., and Woo, W-C. 2015. Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting. In Proceedings of the 29th International Conference on Neural Information Processing Systems (NIPS ’15), Vol. 1. MIT Press, Cambridge, MA, 802–810. DOI: https://dl.acm.org/doi/10.5555/2969239.2969329

2, Palazzi, A., Yu, H., and Pini, S. ConvLSTM PyTorch Implementation. https://github.com/ndrplz/ConvLSTM_pytorch