import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

def map_cosine_similarity_to_0_1(cosine_similarity):
    return (cosine_similarity + 1) / 2



# 读取数据
df = np.loadtxt('code\\testreference.txt')
num = [11,3,7,20,10,17,0,22,19,5,4,18]


fileName = f'code\\reference1.txt'
df1 = np.loadtxt(fileName)  # 100*48
df = (df - np.mean(df1, axis=0)) / np.std(df1, axis=0)
df1 = (df1 - np.mean(df1, axis=0)) / np.std(df1, axis=0)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:10]

name = ['Accel-Y-mean','Accel-X-mean','Accel-X-rms','Accel-Z-var','Accel-Y-median','Accel-Z-min','Accel-X-max','Accel-Z-peak2peak','Accel-Z-mean','Accel-X-peak','Accel-X-var','Accel-Z-median']

data = df[:,num]
data1 = df1[:,num]

for i in range(100):
    fig, axs = plt.subplots(nrows=1, ncols=12, figsize=(18, 5), sharey=True)
    for j in range(12):
        axs[j].set_title(name[j])
        axs[j].axhline(np.array([data[2*i,j]]),color=colors[0])
        axs[j].axhline(np.array([data[2*i+1,j]]),color=colors[1])
        axs[j].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # 隐藏 x 轴刻度
        axs[j].set_ylim(-3, 5)  # 设置 y 轴范围
    plt.tight_layout()
    plt.savefig(f'code\\images\\test-features-{i}.png')



for i in range(100):
    input1 = torch.Tensor(data[2*i]).unsqueeze(0)
    input2 = torch.Tensor(data[2*i+1]).unsqueeze(0)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    output = cos(input1, input2).item()
    if map_cosine_similarity_to_0_1(output) > 0.9:
        flag = '是'
    else:
        flag = '否'
    print(f'{i+1} & {output:.2f} & {map_cosine_similarity_to_0_1(output):.3f} & {flag} \\\\ \\hline ')
    
cnt = 0
for i in range(100):
    for j in range(100):
        input1 = torch.Tensor(data1[i]).unsqueeze(0)
        input2 = torch.Tensor(data1[j]).unsqueeze(0)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(input1, input2).item()
        output = map_cosine_similarity_to_0_1(output)
        if i//10 == j//10:
            #color1 = colors[0]
            if output > 0.9:
                cnt += 1
        else:
            #color1 = colors[1]
            if output <= 0.9:
                cnt += 1
        #plt.plot(np.array([i%10]),np.array([output]),'o',color=color1)
#plt.show()
print(cnt/10000)
