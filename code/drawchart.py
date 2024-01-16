import numpy as np
import matplotlib.pyplot as plt
df = np.loadtxt('code\\reference\\1\\1.txt')


fig, axs = plt.subplots(nrows=df.shape[0], ncols=1, figsize=(8, 2*df.shape[0]))
for i in range(df.shape[0]):
    x = np.arange(df.shape[1])
    y = df[i, :]
    axs[i].plot(x, y)
    axs[i].set_title(f'Line {i+1}')
plt.tight_layout()
plt.show()


'''
#箱型图
fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot(df.T, labels=[f'{i+1}' for i in range(df.shape[0])])
plt.show()
'''