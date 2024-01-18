import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col

fileName = f'code\\reference1.txt'
df = np.loadtxt(fileName)

df = (df - np.mean(df, axis=0)) / np.std(df, axis=0)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:10]
fig, axs = plt.subplots(nrows=6, ncols=8, figsize=(16, 48))
x = np.arange(10)
xtitile = ['max', 'min', 'median', 'mean', 'var', 'peak', 'peak2peak', 'rms']
ytitile = ['Accel-X','Accel-Y','Accel-Z','Gyro-X','Gyro-Y','Gyro-Z']
for j in range(8):
    axs[0][j].set_title(xtitile[j])
for j in range(6):
    axs[j][0].set_ylabel(ytitile[j])
for k in range(6):
    for j in range(8):
        for i in range(10):
            y = (df.T)[8*k+j]
            z = y[i*10:i*10+10]
            axs[k][j].plot(x,z,'o',color=colors[i])
            axs[k][j].axhline(z.mean(),color=colors[i])
plt.tight_layout()
plt.savefig(f'code\\images\\features-1-normalized.png')
plt.show()