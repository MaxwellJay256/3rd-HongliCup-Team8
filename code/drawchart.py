import numpy as np
import matplotlib.pyplot as plt

# 读取数据
dataType = 'reference'
subject = '1'
trial = '2'

fileName = f'code\\{dataType}\\{subject}\\{trial}.txt'
df = np.loadtxt(fileName)

# 折线图
fig, axs = plt.subplots(nrows=df.shape[0], ncols=1, figsize=(8, 2*df.shape[0]))
fig.suptitle(f'{dataType} {subject}-{trial}')
subtitles = ['Accel X', 'Accel Y', 'Accel Z', 'Gyro X', 'Gyro Y', 'Gyro Z']
for i in range(df.shape[0]):
    x = np.arange(df.shape[1])
    y = df[i, :]
    axs[i].plot(x, y)
    axs[i].set_title(subtitles[i])
plt.tight_layout()
plt.show()

# 箱型图
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(f'{dataType} {subject}-{trial}')
ax.boxplot(df.T, labels=[f'{i+1}' for i in range(df.shape[0])])
plt.show()
