# learning rate
import matplotlib.pyplot as plt
import numpy as np

# 使用一个没有网格的样式
plt.style.use('seaborn-v0_8-white')

# 数据
# learning_rates = np.array([0.1, 0.01, 0.001, 0.0001])
# ndcg_10 = np.array([0.0373, 0.0519, 0.0574, 0.0670])
# ndcg_20 = np.array([0.0486, 0.0638, 0.0574, 0.0795])
# mrr_10 = np.array([0.0264, 0.0377, 0.0433, 0.0522])
# mrr_20 = np.array([0.0294, 0.0410, 0.0467, 0.0556])
# hit_10 = np.array([0.0739, 0.0988, 0.1043, 0.1158])
# hit_20 = np.array([0.1185, 0.1463, 0.1548, 0.1656])
# 数据
# dataset:video
learning_rates = [0.1, 0.01, 0.001, 0.0001]
ndcg_10 = np.array([0.0373, 0.0519, 0.0574, 0.0670])
ndcg_20 = np.array([0.0486, 0.0638, 0.0574, 0.0795])
mrr_10 = np.array([0.0264, 0.0377, 0.0433, 0.0522])
mrr_20 = np.array([0.0294, 0.0410, 0.0467, 0.0556])
hit_10 = np.array([0.0739, 0.0988, 0.1043, 0.1158])
hit_20 = np.array([0.1185, 0.1463, 0.1548, 0.1656])


# 设置全局字体大小
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 创建图形和子图
# 调整 figsize，将宽度从 10 减小到 8 或 7
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制折线图
ax.plot(learning_rates, ndcg_10, marker='o', linestyle='-', linewidth=2, markersize=7, label='NDCG@10')
ax.plot(learning_rates, ndcg_20, marker='s', linestyle='--', linewidth=2, markersize=7, label='NDCG@20')
ax.plot(learning_rates, mrr_10, marker='^', linestyle='-.', linewidth=2, markersize=8, label='MRR@10')
ax.plot(learning_rates, mrr_20, marker='D', linestyle=':', linewidth=2, markersize=6, label='MRR@20')
ax.plot(learning_rates, hit_10, marker='v', linestyle='-', linewidth=2, markersize=8, label='HIT@10')
ax.plot(learning_rates, hit_20, marker='<', linestyle='--', linewidth=2, markersize=8, label='HIT@20')

# 设置x轴为对数刻度
ax.set_xscale('log')
ax.set_xlabel('Learning Rate', fontsize=14, fontweight='bold')
ax.set_ylabel('Metric Value', fontsize=14, fontweight='bold')
ax.set_title('Model Performance vs. Learning Rate (Video Dataset)', fontsize=16, fontweight='bold')

# 设置x轴刻度标签
ax.set_xticks(learning_rates)
ax.set_xticklabels([str(lr) for lr in learning_rates])

# 显式地设置 x 轴范围，使其更紧凑
# 设置为略大于或等于数据范围
ax.set_xlim([min(learning_rates) / 2, max(learning_rates) * 1.2])


# 设置图例
ax.legend(loc='upper right', frameon=True, shadow=False, fontsize=11, ncol=2)

# 添加最优值标注（简化版）
best_ndcg20_value = ndcg_20.max()
best_lr_index = ndcg_20.argmax()
best_lr = learning_rates[best_lr_index]

ax.axvline(x=best_lr, color='red', linestyle='--', alpha=0.7, linewidth=1.5)


# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('beauty_learning rate.pdf', bbox_inches='tight')
plt.savefig('beauty_learning rate.png', dpi=600, bbox_inches='tight')

plt.show()
