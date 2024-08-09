import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr

# 加载数据
bs = np.load('/media/2T/yongtong/Rena/rena_learning/bs_9000.npy')       # 样本数据，形状为 (1000, 52)
label = np.load('/media/2T/yongtong/Rena/rena_learning/label_9000.npy') # 标签数据，形状为 (1000, 25)


# --------------- 样本数据清洗 -------------------

bs1 = bs[:, :19]
bs2 =  bs[:, 24]

bs = np.hstack((bs1, bs2[:, np.newaxis]))



# --------------- 标签数据清洗 -------------------
# label_1 = label[:, :13]
# label_2 = label[:, -3]
# label  = np.hstack( (label_1, label_2[:, np.newaxis]) ) # 所有运动，但是排除不动的舵机

label = label[:, :13]  # 仅头部和面部运动 

# label = label[:, :10]  # 仅面部运动 



# 联合分布的降维可视化
pca = PCA(n_components=2)
bs_pca = pca.fit_transform(bs)
label_pca = pca.fit_transform(label)

plt.scatter(bs_pca[:, 0], bs_pca[:, 1], label='Sample Data', alpha=0.5)

plt.scatter(label_pca[:, 0], label_pca[:, 1], label='Label Data', alpha=0.5)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.title('PCA Visualization of Sample and Label Data')
plt.show()


# 计算关联性
correlation_results = []
for i in range(bs.shape[1]):
    for j in range(label.shape[1]):
        pearson_corr, _ = pearsonr(bs[:, i], label[:, j])
        spearman_corr, _ = spearmanr(bs[:, i], label[:, j])
        correlation_results.append({
            'Feature Index': i,
            'Label Index': j,
            'Pearson Correlation': pearson_corr,
            'Spearman Correlation': spearman_corr
        })

# 转换为DataFrame以便查看
correlation_df = pd.DataFrame(correlation_results)
print(correlation_df.sort_values(by='Pearson Correlation', ascending=False).head())

# 根据关联性选择最相关的特征和标签对
top_correlations = correlation_df.sort_values(by='Pearson Correlation', ascending=False).head(10)
print("Top 10 Correlations:")
print(top_correlations)