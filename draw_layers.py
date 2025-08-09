# layer

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Prepare the data
# beauty
# data = {
#     'Layers': [1, 2, 3],
#     'ndcg@10': [0.0491, 0.0474, 0.0400],
#     'ndcg@20': [0.0557, 0.0542, 0.0469],
#     'mrr@10': [0.0404, 0.0389, 0.0318],
#     'mrr@20': [0.0422, 0.0408, 0.0336],
#     'hit@10': [0.0774, 0.0753, 0.0669],
#     'hit@20': [0.1038, 0.1023, 0.0944],
#     'GPU_RAM_GB': [1.25, 2.24, 3.25]
# }
# video
data = {
    'Layers': [1, 2, 3],
    'ndcg@10': [0.067, 0.0654, 0.0594],
    'ndcg@20': [0.0795, 0.0654, 0.0731],
    'mrr@10': [0.0522, 0.0499, 0.044],
    'mrr@20': [0.0556, 0.0538, 0.0478],
    'hit@10': [0.1158, 0.1165, 0.1101],
    'hit@20': [0.1656, 0.1729, 0.1651],
    'GPU_RAM_GB': [2.86, 5.73, 8.91]
}
# Set global font and font size for better readability in a manuscript
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'DejaVu Sans'
})
plt.rcParams['axes.unicode_minus'] = False

df = pd.DataFrame(data)

# Melt the performance data to be compatible with seaborn
df_melted_performance = df.melt(id_vars=['Layers'],
                                value_vars=['ndcg@10', 'ndcg@20', 'mrr@10', 'mrr@20', 'hit@10', 'hit@20'],
                                var_name='Metric',
                                value_name='Value')

# 主要调整：缩小图表尺寸，使其更紧凑
fig, ax1 = plt.subplots(figsize=(8, 5))  # 从 (10, 6) 改为 (8, 5)

# Set different line styles and markers to match the user's reference image
linestyles = ['-', '--', ':', '-.', '--', '-.']
markers = ['o', 'o', '^', '^', 'v', 'v']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Loop through each metric to plot the lines with specific styles
for i, metric in enumerate(df_melted_performance['Metric'].unique()):
    sns.lineplot(data=df_melted_performance[df_melted_performance['Metric'] == metric],
                 x='Layers', y='Value', ax=ax1,
                 linestyle=linestyles[i], marker=markers[i], color=colors[i],
                 label=metric)

ax1.set_xlabel('Number of Layers', fontsize=14, fontweight='bold')
ax1.set_ylabel('Performance Metric Value', fontsize=14,fontweight='bold')
ax1.set_xticks(df['Layers'])
# 调整坐标轴刻度标签的字体大小
ax1.tick_params(axis='both', labelsize=10)
ax1.grid(False)

# 进一步缩小x轴范围，使图形更紧凑
ax1.set_xlim(0.8, 3.2)  # 从 (0.9, 3.1) 改为 (0.8, 3.2)

# Adjust y-axis range for ax1
min_perf_val = df_melted_performance['Value'].min()
max_perf_val = df_melted_performance['Value'].max()
ax1.set_ylim(min_perf_val * 0.9, max_perf_val * 1.1)

# Create a second y-axis
ax2 = ax1.twinx()
sns.lineplot(data=df, x='Layers', y='GPU_RAM_GB', marker='s', color='red', linestyle='--', ax=ax2, label='GPU RAM (GB)')
ax2.set_ylabel('GPU RAM (GB)', fontsize=12, labelpad=5, fontweight='bold')  # 减少 labelpad
# 同时调整右侧Y轴刻度标签的字体大小
ax2.tick_params(axis='y',  labelsize=10)
ax2.grid(False)

# Adjust y-axis range for ax2
min_ram_val = df['GPU_RAM_GB'].min()
max_ram_val = df['GPU_RAM_GB'].max()
ax2.set_ylim(min_ram_val * 0.9, max_ram_val * 1.1)

# Get all handles and labels for the legend
handles, labels = ax1.get_legend_handles_labels()
handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
all_handles = handles + handles_ax2
all_labels = labels + labels_ax2

# 优化图例位置和大小，增加右侧距离避免与Y轴标签重合
ax1.legend(all_handles, all_labels, title='Metrics', loc='center left', 
          bbox_to_anchor=(1.15, 0.75), fontsize=10, frameon=True, 
          fancybox=True, shadow=True, title_fontsize=11)
ax2.get_legend().remove()

# 调整标题
plt.title('Impact of Layers on Performance and GPU RAM(video)', fontsize=16, fontweight='bold', pad=10)

# 使用更紧凑的布局
plt.tight_layout()
plt.savefig('optimized_plot_for_manuscript.png', dpi=1200, bbox_inches='tight')
plt.show()  # 改为 show() 以便查看结果