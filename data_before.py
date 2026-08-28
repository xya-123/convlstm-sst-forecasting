import os
import numpy as np
import xarray as xr

# 设置主目录
base_dir = '2020'

# 收集所有 nc 文件路径
nc_files = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.nc'):
            nc_files.append(os.path.join(root, file))

# 按文件名排序（确保时间顺序）
nc_files.sort()

# 读取并提取 sst 变量
sst_list = []
for f in nc_files:
    print(f"正在处理 {f} ...")
    ds = xr.open_dataset(f)
    sst = ds['sst'].values
    sst_list.append(sst)
    ds.close()

# 合并所有 sst 数据为一个 NumPy 数组
sst_all = np.stack(sst_list, axis=0)  # shape: (time, lat, lon)

# 保存为 data.npy
np.save('data.npy', sst_all)

print(f"处理完成，共 {len(sst_list)} 个文件，已保存为 data.npy")
