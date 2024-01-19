from pyexpat import features
import numpy as np
from sklearn.decomposition import PCA

data = np.loadtxt('code\\reference.txt', dtype=np.float64) # reference.txt 有 100 行，每行是一个 84 维向量
print(f'data.shape: {data.shape}')

# PCA 降维
dimension = 50 # 降维后的维度
pca = PCA(n_components=dimension)
features_reduced = pca.fit_transform(data)
print(f'x_reduced.shape: {features_reduced.shape}')
print(f'pca.explained_variance_ratio_: {pca.explained_variance_ratio_}') # 降维后每个特征的方差占比

# 将降维后数据保存到文件中
outputFileName = f'code\\features_reduced-{dimension}.txt'
np.savetxt(outputFileName, features_reduced, fmt='%f', delimiter=' ')