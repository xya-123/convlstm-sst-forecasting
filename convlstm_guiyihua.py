#代码来源：https://github.com/ndrplz/ConvLSTM_pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



# ========= 工具函数 =========
def fit_minmax(x):
    """在训练集上拟合 min/max（忽略 NaN）"""
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    # 防止极端情况
    if np.isclose(x_max, x_min):
        x_max = x_min + 1e-8
    return float(x_min), float(x_max)

def transform_minmax(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

def inverse_minmax(x_norm, x_min, x_max):
    return x_norm * (x_max - x_min) + x_min

# ========= 1) 加载原始数据 =========
data = np.load('data.npy')  # 原始: (365, 1, 1, 128, 128)
print("原始数据形状:", data.shape)

# ========= 2) 构造“过去10天预测第1天”的样本 =========
seq_len = 10
X_list, Y_list = [], []
for i in range(len(data) - seq_len):
    X_list.append(data[i:i+seq_len])        # (10, 1, 1, 128, 128)
    Y_list.append(data[i+seq_len])          # (1, 1, 128, 128)
X = np.array(X_list).squeeze(2)             # → (N, 10, 1, 128, 128)
Y = np.array(Y_list).squeeze(2)             # → (N, 1, 128, 128)
print("构造后的数据集形状:", X.shape, Y.shape)  # 与你现有脚本一致:contentReference[oaicite:2]{index=2}

# ========= 3) 先切分，再在训练集上拟合归一化参数 =========
N = len(X)
train_end = int(N * 0.7)
val_end   = int(N * 0.9)

X_train_raw, Y_train_raw = X[:train_end], Y[:train_end]
X_val_raw,   Y_val_raw   = X[train_end:val_end], Y[train_end:val_end]
X_test_raw,  Y_test_raw  = X[val_end:], Y[val_end:]

# —— 拟合仅使用训练集（可以把输入与标签拼一起求全局 min/max）——
#   注意：这里对“原始数据”计算 min/max（未把 NaN 改 0），避免 NaN=0 影响统计量
train_stack = np.concatenate([X_train_raw.reshape(train_end, -1),
                              Y_train_raw.reshape(train_end, -1)], axis=1)
train_min, train_max = fit_minmax(train_stack)
print(f"训练集拟合的 min/max: {train_min:.4f}, {train_max:.4f}")

# ========= 4) NaN 统一置 0（按你的要求），再用训练集 min/max 归一化 =========
def nan_to_zero(a):
    a = a.copy()
    a[np.isnan(a)] = 0.0
    return a

X_train = nan_to_zero(X_train_raw)
Y_train = nan_to_zero(Y_train_raw)
X_val   = nan_to_zero(X_val_raw)
Y_val   = nan_to_zero(Y_val_raw)
X_test  = nan_to_zero(X_test_raw)
Y_test  = nan_to_zero(Y_test_raw)

# 用训练集的 min/max 对三份数据做同尺度归一化
X_train = transform_minmax(X_train, train_min, train_max)
Y_train = transform_minmax(Y_train, train_min, train_max)
X_val   = transform_minmax(X_val,   train_min, train_max)
Y_val   = transform_minmax(Y_val,   train_min, train_max)
X_test  = transform_minmax(X_test,  train_min, train_max)
Y_test  = transform_minmax(Y_test,  train_min, train_max)

# ========= 5) 转 Tensor =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
Y_train = torch.tensor(Y_train, dtype=torch.float32, device=device)
X_val   = torch.tensor(X_val,   dtype=torch.float32, device=device)
Y_val   = torch.tensor(Y_val,   dtype=torch.float32, device=device)
X_test  = torch.tensor(X_test,  dtype=torch.float32, device=device)
Y_test  = torch.tensor(Y_test,  dtype=torch.float32, device=device)

# ========= 6) 定义 ConvLSTM 模型（保持不变） =========
# 你的脚本里已经有 ConvLSTM 定义，训练骨架也类似:contentReference[oaicite:3]{index=3}:contentReference[oaicite:4]{index=4}
model = ConvLSTM(input_dim=1,
                 hidden_dim=[8, 1],
                 kernel_size=(3, 3),
                 num_layers=2,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ========= 7) 训练 + 验证 =========
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out_train, _ = model(X_train)                  # 输出序列
    pred_train = out_train[0][:, -1, :, :, :]      # 取最后一帧
    loss = criterion(pred_train, Y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        out_val, _ = model(X_val)
        pred_val = out_val[0][:, -1, :, :, :]
        val_loss = criterion(pred_val, Y_val)

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")

# ========= 8) 测试：用训练集 min/max 反归一化后计算 RMSE =========
model.eval()
with torch.no_grad():
    out_test, _ = model(X_test)
    pred_test_norm = out_test[0][:, -1, :, :, :].cpu().numpy()
    Y_test_norm    = Y_test.cpu().numpy()

# 反归一化（严格使用训练集的 min/max）
pred_test = inverse_minmax(pred_test_norm, train_min, train_max)
Y_test_gt = inverse_minmax(Y_test_norm,    train_min, train_max)

rmse = sqrt(mean_squared_error(Y_test_gt.flatten(), pred_test.flatten()))
print(f"✅ 测试集 RMSE（使用训练集 min/max 反归一化）: {rmse:.4f}")