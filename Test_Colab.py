import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

# 定义训练设备
device = torch.device("cuda:0")

# 添加SummaryWriter
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

# 模型
class FirstNet(nn.Module):
    def __init__(self):
        super(FirstNet, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding = 1),
            nn.Conv2d(32, 64, 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding = 1),
            nn.Conv2d(128, 256, 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256*7*7, 256),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.Dropout(),
            nn.Linear(64,10),

        )

    def forward(self, output):
        output = self.module(output)
        return output


train_data = torchvision.datasets.MNIST("data", train = True, transform = torchvision.transforms.ToTensor(),
                                        download = True)
test_data = torchvision.datasets.MNIST("data", train = False, transform = torchvision.transforms.ToTensor(),
                                       download = True)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为: {}".format(train_data_size))
print("测试数据集的长度为: {}".format(test_data_size))


# 利用 DataLoader 加载数据集
train_data_loader = DataLoader(train_data, batch_size = 128)
test_data_loader = DataLoader(test_data, batch_size = 128)

# 记录训练次数
train_step = 0
# 测试次数
test_step = 0

# 新建神经网络
Fn = FirstNet()
Fn = Fn.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 0.01
optim = torch.optim.SGD(Fn.parameters(), lr = learning_rate)

# 轮数
epoch = 50

for i in range(epoch):
    print("-------第{}轮训练开始------".format(i+1))

    # 训练步骤开始
    Fn.train()
    for data in train_data_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = Fn(imgs)
        # 优化器调优
        optim.zero_grad()
        loss = loss_fn(outputs, targets)
        loss.backward()
        optim.step()
        train_step += 1
        if train_step % 100 == 0:
            print("训练次数: {}, Loss: {}".format(train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), train_step)

    # 测试步骤开始
    Fn.eval()
    total_test_loss = 0
    accuracy = 0
    total_accuracy = 0
    with torch.no_grad():  # 不累计梯度
        for data in test_data_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = Fn(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
        print("整体测试集上的Loss: {}".format(total_test_loss))
        print("整体测试集上的正确率为: {}".format(torch.true_divide(total_accuracy, test_data_size)))
        test_step += 1
        writer.add_scalar("test_loss", total_test_loss, test_step)
        writer.add_scalar("test_accuracy", torch.true_divide(total_accuracy, test_data_size), test_step)
        torch.save(Fn.state_dict(), "Fn_{}.pth".format(test_step))

writer.close()

# 经过 300 个 epoch， 最高测试集正确率为 99.57%。
