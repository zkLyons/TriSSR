# 单独热力图
import numpy as np
import matplotlib.pyplot as plt

# 1. 载入数据 (保持原逻辑)
try:
    freq = np.load("out/freq_out.npy")[0]
    mamba = np.load("out/mamba_out.npy")[0]
    time = np.load("out/time_out.npy")[0]
    fusion = np.load("out/fusion_out.npy")[0]
except FileNotFoundError:
    # 为了演示，如果找不到文件，生成随机数据
    print("未找到数据文件，使用随机数据演示效果...")
    freq = np.random.rand(10, 256)
    mamba = np.random.rand(10, 256)
    time = np.random.rand(10, 256)
    fusion = np.random.rand(10, 256)

# 2. 数据预处理
def to_2d(x):
    if x.ndim == 1:
        return x[np.newaxis, :]
    return x

features = [to_2d(freq), to_2d(mamba), to_2d(time), to_2d(fusion)]

# 定义标签
labels = ["Frequency Feature", "State-space Feature", "Time Feature", "Fused Feature"]

# 3. 计算相关性矩阵
feature_flat = [f.flatten() for f in features]
corr_matrix = np.corrcoef(feature_flat)

# 4. 开始绘图
fig, ax = plt.subplots(figsize=(10, 9))  # 稍微增加高度以容纳底部标签

# 绘制热力图
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

# 设置标题
ax.set_title('Inter-Feature Correlation', fontsize=22, fontweight='bold', pad=24)

# 设置坐标轴刻度
ax.set_xticks(range(4))
ax.set_yticks(range(4))

# === 关键修改：增大字号(16) + 加粗(bold) ===
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=18, fontweight='bold')
ax.set_yticklabels(labels, fontsize=18, fontweight='bold')

# 添加 Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=14) # Colorbar 刻度也稍微大一点

# 5. 在格子中添加数值
for i in range(4):
    for j in range(4):
        text_color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
        ax.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                ha="center", va="center", 
                color=text_color, 
                fontsize=16,    # 数值字号也同步增大
                fontweight='bold') # 数值加粗

# 保存和显示
# 使用 bbox_inches='tight' 自动剪裁白边，防止大字体溢出
plt.tight_layout()
plt.savefig("out/correlation_heatmap_bold1.png", dpi=300, bbox_inches='tight')
plt.show()