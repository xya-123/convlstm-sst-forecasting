import os
import numpy as np
import xarray as xr

# ========== 参数设置 ==========
base_dir = '2020'  # 主文件夹路径
lon_min, lon_max = 100, 132
lat_min, lat_max = 16, 48

# ========== 收集所有 .nc 文件 ==========
nc_files = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.nc'):
            nc_files.append(os.path.join(root, file))

# 按日期排序，确保时间顺序一致
nc_files.sort()

print(f"共找到 {len(nc_files)} 个 nc 文件。")

# ========== 预处理：读取第一个文件确定索引范围 ==========
sample = xr.open_dataset(nc_files[0])
lon = sample['lon'].values
lat = sample['lat'].values

# 找出经纬度对应索引
lon_index = np.where((lon >= lon_min) & (lon <= lon_max))[0]
lat_index = np.where((lat >= lat_min) & (lat <= lat_max))[0]

print(f"经度索引范围: {lon_index[0]} - {lon_index[-1]} ({len(lon_index)} 个点)")
print(f"纬度索引范围: {lat_index[0]} - {lat_index[-1]} ({len(lat_index)} 个点)")

# 确认是 200×200 区域
print(f"区域大小: {len(lat_index)} × {len(lon_index)}")

sample.close()

# ========== 循环读取所有文件并提取区域数据 ==========
sst_list = []

for i, f in enumerate(nc_files):
    print(f"[{i+1}/{len(nc_files)}] 处理 {os.path.basename(f)} ...")
    ds = xr.open_dataset(f)
    sst = ds['sst'].values[0][0]

    # 提取子区域
    sst_sub = sst[lat_index.min():lat_index.max()+1, lon_index.min():lon_index.max()+1]
    sst_list.append(sst_sub)
    ds.close()

# ========== 组合成一个 NumPy 数组 ==========
# shape: (time, 1, 1, lat, lon)
sst_all = np.array(sst_list)[:, np.newaxis, np.newaxis, :, :]

print(f"数据形状: {sst_all.shape}")

# ========== 保存为 .npy 文件 ==========
np.save('data.npy', sst_all)
print("✅ 已保存为 data.npy")
