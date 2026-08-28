import numpy as np
import matplotlib.pyplot as plt

# 读取npy文件
data = np.load("data.npy")  # 或改为你的文件路径
print("数据维度:", data.shape)
data = data[0,0,0]

# 如果是(1, 200, 200)，取出第一个时间步
if data.ndim == 3 and data.shape[0] == 1:
    data = data[0]

# 经纬度范围
lon = np.linspace(100, 150, data.shape[1])
lat = np.linspace(0, 50, data.shape[0])
lon2d, lat2d = np.meshgrid(lon, lat)

# 绘图
plt.figure(figsize=(7, 6))
contour = plt.contourf(lon2d, lat2d, data, levels=np.arange(0, 30, 1),
                       cmap="RdYlBu_r", extend="both")
plt.colorbar(contour, label="(℃)")
plt.xlabel("longitude (°E)")
plt.ylabel("latitude (°N)")
plt.title("2020-1-1 Sea surface temperature (SST)")

# 保存图像
plt.savefig("sst_19930101.png", dpi=300, bbox_inches="tight")

# 显示图像
plt.show()
