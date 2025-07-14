import os
import glob
import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

# ---------- 配置 ----------
data_root = "data/ffhq/ffhq_dinov2_base_npz/ffhq_256"
n_components = 256           # 例：把 768 → 256
batch_rows   = 1024          # 每个文件本身就是 1024 行
dtype        = np.float32    # Dinov2 一般是 float32
# --------------------------

# 1) 构建增量 PCA 对象
ipca = IncrementalPCA(
    n_components=n_components,
    batch_size=batch_rows   # <= 行数即可
)

# 2) 第一遍：partial_fit
file_list = sorted(glob.glob(os.path.join(data_root, "*.npz")))
for i, path in tqdm(enumerate(file_list)):
    with np.load(path, mmap_mode='r') as npz:
        X = npz['patch_tokens']                     # (1024, 768)
        ipca.partial_fit(X.squeeze(0).astype(dtype))    # 自动居中

print("Done partial_fit.")
print("Explained variance ratio (first 10):", ipca.explained_variance_ratio_[:10])

# 3) 第二遍：transform 并写出
out_dir = os.path.join(data_root, "pca_%d" % n_components)
os.makedirs(out_dir, exist_ok=True)

for path in file_list:
    with np.load(path, mmap_mode='r') as npz:
        X = npz['patch_tokens'].astype(dtype)
        Z = ipca.transform(X.squeeze())                # (1024, n_components)
        save_path = os.path.join(out_dir, os.path.basename(path))
        np.savez_compressed(save_path, Z)

print("All batches transformed →", out_dir)
