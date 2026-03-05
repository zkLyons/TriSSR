import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. 全局字体与清晰度设置 (【修改点：调大字号】) ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.weight': 'bold',          # 全局字体加粗，看起来更清晰
    'font.size': 16,                # 全局基础字号 (9 -> 14)
    'axes.titlesize': 22,           # 子图标题字号 (11 -> 20)
    'axes.labelsize': 18,           # 坐标轴标签字号 (9 -> 15)
    'xtick.labelsize': 14,          # x轴刻度字号 (8 -> 12)
    'ytick.labelsize': 14,          # y轴刻度字号 (8 -> 12)
    'figure.dpi': 200               # 屏幕显示清晰度
})

# --- 2. 载入数据 (保持不变) ---
try:
    freq = np.load("out/freq_out.npy")[0]
    mamba = np.load("out/mamba_out.npy")[0]
    time = np.load("out/time_out.npy")[0]
    fusion = np.load("out/fusion_out.npy")[0]
except FileNotFoundError:
    print("未找到数据文件，生成随机数据用于演示...")
    freq = np.random.rand(20, 256) # 稍微改大一点形状以便观察
    mamba = np.random.rand(20, 256)
    time = np.random.rand(20, 256)
    fusion = np.random.rand(20, 256)

def to_2d(x):
    if x.ndim == 1:
        return x[np.newaxis, :]
    return x

freq = to_2d(freq)
mamba = to_2d(mamba)
time = to_2d(time)
fusion = to_2d(fusion)

features = [freq, mamba, time, fusion]
titles = ["(a) Frequency Feature", "(b) State-space Feature", "(c) Time Feature", "(d) Fused Feature"]

# --- 3. 绘图逻辑优化 ---

# 【修改点：增大画布】
# 字体变大了，画布必须跟着变大，否则字会挤在一起。
fig = plt.figure(figsize=(18, 14)) 

for i, (feature, title) in enumerate(zip(features, titles)):
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    
    # 创建网格
    x = np.arange(feature.shape[1])
    y = np.arange(feature.shape[0])
    X, Y = np.meshgrid(x, y)
    
    # 3D表面图
    surf = ax.plot_surface(X, Y, feature, cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)
    
    # 【修改点：标题位置】
    # 之前的 pad=-100 可能会导致大字体直接插入图中，改为 -20 或 0
    # y=1.02 可以让标题稍微往上提一点
    ax.set_title(title, fontweight='bold', pad=10, y=0.99) 
    
    # 【修改点：增加 labelpad】
    # 字体大了，标签需要离轴远一点，不然会和刻度数字重叠
    ax.set_xlabel('Feature Dim', labelpad=12, fontweight='bold')
    ax.set_ylabel('Seq Len', labelpad=12, fontweight='bold')
    ax.set_zlabel('Value', labelpad=12, fontweight='bold')

    # 调整视角
    ax.view_init(elev=30, azim=-60)
    
    # 刻度设置 (可选：减少刻度数量，避免大字体拥挤)
    # ax.locator_params(axis='x', nbins=5)
    # ax.locator_params(axis='y', nbins=5)

# 【修改点：放松间距】
# 字体变大后，子图之间需要更多空间
plt.subplots_adjust(
    left=0.05,
    right=0.8,
    bottom=0.05,
    top=0.92,
    wspace=-0.1,  # 水平间距：从 -0.25 改为 0.1 (拉开左右距离)
    hspace=0.1   # 垂直间距：适当增加
)

# 保存图片
plt.savefig("out/3d_feature_visualization_large_font.png", dpi=300, bbox_inches='tight')
print("图片已保存至 out/3d_feature_visualization_large_font.png")
plt.show()