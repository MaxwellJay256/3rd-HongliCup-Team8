import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

all_data = np.loadtxt("code\\features_reduced-50.txt")

m1 = all_data.max(axis=0)
m2 = all_data.min(axis=0)
all_data = 2*(all_data-m2)/(m1-m2)-1

# 构建正样本对
positive_pairs = []
for person_id in range(10):
    for i in range(9):
        for j in range(i + 1, 10):
            positive_pairs.append((all_data[person_id*10+i], all_data[person_id*10+j], 1.0))

# 构建负样本对
negative_pairs = []
for person_id in range(10):
    for i in range(10):
        other_person_id = (person_id + np.random.randint(1, 10)) % 10  # 随机选择另外一个个体
        random_sample = np.random.randint(10)  # 随机选择一个样本
        negative_pairs.append((all_data[person_id*10+i], all_data[other_person_id*10+random_sample], 0.0))

# 将正负样本合并
all_pairs = positive_pairs + negative_pairs


# 打乱样本顺序
np.random.shuffle(all_pairs)

# 划分训练集和测试集
train_pairs = all_pairs[:80]
test_pairs = all_pairs[80:]

# 获取训练集和测试集的数据和标签
train_pairs_data = np.array([(pair[0], pair[1]) for pair in train_pairs])
train_labels = np.array([pair[2] for pair in train_pairs])

test_pairs_data = np.array([(pair[0], pair[1]) for pair in test_pairs])
test_labels = np.array([pair[2] for pair in test_pairs])

# 转换为 PyTorch 的 Tensor
train_pairs_data = torch.Tensor(train_pairs_data)
train_labels = torch.Tensor(train_labels)

test_pairs_data = torch.Tensor(test_pairs_data)
test_labels = torch.Tensor(test_labels)

# pairs_data 和 labels 可以用于训练 Siamese 网络


# 假设你已经有了 pairs_data 和 labels
# pairs_data 是一个形状为 (num_samples, 2, num_features) 的数组
# labels 是一个形状为 (num_samples,) 的数组

# 定义 Siamese 网络
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # 定义 Siamese 网络的结构，可以根据需要调整
        # 这里以一个简单的全连接层为例
        self.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8)
        )

    def forward_one(self, x):
        # 前向传播一个样本
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        # 前向传播两个样本并返回其输出
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# 初始化 Siamese 网络
num_features = train_pairs_data.size(2)
siamese_model = SiameseNetwork()

# 定义损失函数和优化器
criterion = nn.MarginRankingLoss(margin=1.0)  # MarginRankingLoss 用于 Siamese 网络
optimizer = optim.Adam(siamese_model.parameters(), lr=0.001)

# 将数据转换为 DataLoader
batch_size = 32

train_dataset = TensorDataset(train_pairs_data[:, 0, :], train_pairs_data[:, 1, :], train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_pairs_data[:, 0, :], test_pairs_data[:, 1, :], test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 训练 Siamese 网络
num_epochs = 100
for epoch in range(num_epochs):
    for batch_data1, batch_data2, batch_labels in train_dataloader:
        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        output1, output2 = siamese_model(batch_data1, batch_data2)

        # 计算损失
        loss = criterion(output1, output2,  batch_labels.view(-1, 1))

        # 反向传播和优化
        loss.backward()
        optimizer.step()

    print(f'Training - Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 在测试集上评估
siamese_model.eval()  # 切换到评估模式
total_test_loss = 0.0

with torch.no_grad():
    for batch_data1, batch_data2, batch_labels in test_dataloader:
        output1, output2 = siamese_model(batch_data1, batch_data2)
        test_loss = criterion(output1, output2,  batch_labels.view(-1, 1))
        total_test_loss += test_loss.item()

average_test_loss = total_test_loss / len(test_dataloader)
print(f'Testing - Average Loss: {average_test_loss}')
# 保存模型
torch.save(siamese_model.state_dict(), 'siamese_model.pth')

# 加载模型
loaded_model = SiameseNetwork()
loaded_model.load_state_dict(torch.load('siamese_model.pth'))
loaded_model.eval()

cnt = 0
for i in range(100):
    for j in range(100):
        feature_vector1 = all_data[i]
        feature_vector2 = all_data[j]
        feature_vector1 = torch.Tensor(feature_vector1)
        feature_vector2 = torch.Tensor(feature_vector2)
        with torch.no_grad():
            output1 = loaded_model.forward_one(feature_vector1)
            output2 = loaded_model.forward_one(feature_vector2)
        # 例如，使用欧氏距离
        euclidean_distance = torch.norm(output1 - output2)
        if euclidean_distance < 8:
            if i//10 == j//10:
                cnt += 1
        else:
            if i//10 != j//10:
                cnt += 1
        output1 = output1.view(1, -1)
        output2 = output2.view(1, -1)
        # 或使用余弦相似性
        cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
        threshold = 0.7  # 根据需要调整
        if cosine_similarity > threshold:
            if i//10 == j//10:
                cnt += 1
        else:
            if i//10 != j//10:
                cnt += 1
        
print(cnt/10000)
torch.save(siamese_model.state_dict(), f'code\\models\\siamese_model-{cnt/10000}.pth')
'''
feature_vector1 = all_data[0]
feature_vector2 = all_data[15]
feature_vector1 = torch.Tensor(feature_vector1)
feature_vector2 = torch.Tensor(feature_vector2)
with torch.no_grad():
    output1 = loaded_model.forward_one(feature_vector1)
    output2 = loaded_model.forward_one(feature_vector2)
output1 = output1.view(1, -1)
output2 = output2.view(1, -1)
cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
threshold = 0.5  # 根据需要调整
print(cosine_similarity)
if cosine_similarity > threshold:
    print("这两个特征向量来自同一个人")
else:
    print("这两个特征向量来自不同的人")
'''