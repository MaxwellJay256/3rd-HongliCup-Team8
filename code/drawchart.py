import numpy as np
import matplotlib.pyplot as plt

# 读取数据
dataType = 'reference'
subject = '1'
trial = '4'

fileName = f'code\\{dataType}\\{subject}\\{trial}.txt'
df = np.loadtxt(fileName)

# 折线图
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 4))
fig.suptitle(f'{dataType} {subject}-{trial}')
subtitles = ['Accel', 'Gyro']
Axes = ['X','Y','Z']
for i in range(2):
    for j in range(3):
        x = np.arange(df.shape[1])
        y = df[3*i+j, :]
        axs[i].plot(x, y, label=Axes[j])
    axs[i].set_title(subtitles[i])
plt.tight_layout()
plt.legend()
plt.savefig(f'code\\images\\{dataType}-{subject}-{trial}.png')
plt.show()

# 箱型图
fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle(f'{dataType} {subject}-{trial}')
ax.boxplot(df.T, labels=[f'{i+1}' for i in range(df.shape[0])])
plt.savefig(f'code\\images\\{dataType}-{subject}-{trial}-boxplot.png')
plt.show()