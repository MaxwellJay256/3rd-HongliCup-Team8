from cvxpy import vec
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col

fileName = f'code\\reference1.txt'
df = np.loadtxt(fileName)  # 100*48
df = (df - np.mean(df, axis=0)) / np.std(df, axis=0)

w1 = -0.8
w2 = 0.2

k1 = np.zeros(48)
k2 = np.zeros(48)

vector_names = ['' for _ in range(48)]
xtitile = ['max', 'min', 'median', 'mean', 'var', 'peak', 'peak2peak', 'rms']
ytitile = ['Accel-X', 'Accel-Y', 'Accel-Z', 'Gyro-X', 'Gyro-Y', 'Gyro-Z']

for i in range(6):
    for j in range(8):
        ave = np.zeros(10)
        y = (df.T)[8 * i + j]
        for k in range(10):
            z = y[k * 10:k * 10 + 10]
            ave[k] = z.mean()
            k1[8 * i + j] += np.std(z, axis=0) / 10 + (np.max(z, axis=0) - np.min(z, axis=0)) / 10
        k2[8 * i + j] = np.max(ave, axis=0) - np.min(ave, axis=0)

        # 命名每个向量
        vector_names[8 * i + j] = f'{ytitile[i]}-{xtitile[j]}'

# 创建一个包含向量名称和分数的列表
score_list = list(zip(vector_names, k1, k2, w1 * k1 + w2 * k2))

num = np.arange(1,49)
print(num)

# 对列表根据分数进行排序
sorted_scores = sorted(score_list, key=lambda x: x[3], reverse=True)

# 打印排序后的向量名称、排名和分数
for rank, (name, k1_value, k2_value, score) in enumerate(sorted_scores, start=1):
    print(f'{rank} & {name} & {k1_value:.2f} & {k2_value:.2f} & {score:.2f} \\\\ \\hline')