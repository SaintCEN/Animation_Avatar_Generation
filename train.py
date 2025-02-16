import os
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 包装数据集
class FaceDataset(Dataset):
    def __init__(self, tags_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.tags_df = pd.read_csv(tags_file, delimiter='\t', header=None)

    def __len__(self):
        return len(os.listdir(self.root_dir))  # 返回数据集大小

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"{idx}.jpg")  # 获取图像文件名
        img = Image.open(img_name).convert('RGB')  # 打开图像并转换为RGB格式
        if self.transform:
            img = self.transform(img)  # 应用数据增强
        return img

# 定义生成器网络
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, img_shape[0] * img_shape[1] * img_shape[2]),
            nn.Tanh()  # 使用Tanh激活函数，使输出在[-1, 1]范围内
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)  # 将输出reshape为图像形状
        return img

# 定义判别器网络
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(img_shape, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)  # 将图像展平为一维向量
        validity = self.model(img_flat)  # 判别器输出
        return validity

# 定义训练函数
def train():
    # 定义训练参数
    img_shape = (3, 64, 64)  # 图像形状 (通道数, 高度, 宽度)
    latent_dim = 100  # 潜在向量的维度
    epochs = 70  # 训练轮数
    batch_size = 64  # 批量大小
    learning_rate = 0.0002

    # 数据增强和预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 调整图像大小为64x64
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
    ])

    # 读入数据
    dataset = FaceDataset(tags_file='train.csv', root_dir='./img', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化生成器和判别器
    generator = Generator(latent_dim, img_shape)
    discriminator = Discriminator(img_shape[0] * img_shape[1] * img_shape[2])
    generator.to(device)
    discriminator.to(device)

    # 定义损失函数和优化器
    loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # 训练开始
    for epoch in range(1, epochs + 1):
        print(f"第{epoch}次训练开始")
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device)
            valid = torch.ones(imgs.size(0), 1).to(device)  # 真实图像的标签
            fake = torch.zeros(imgs.size(0), 1).to(device)  # 生成图像的标签

            # 训练生成器
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim).to(device)  # 随机噪声
            gen_imgs = generator(z)  # 生成图像
            g_loss = loss(discriminator(gen_imgs), valid)  # 生成器损失
            g_loss.backward()
            optimizer_G.step()

            # 训练判别器
            optimizer_D.zero_grad()
            real_loss = loss(discriminator(imgs), valid)  # 真实图像的损失
            fake_loss = loss(discriminator(gen_imgs.detach()), fake)  # 生成图像的损失
            d_loss = (real_loss + fake_loss) / 2  # 判别器总损失
            d_loss.backward()
            optimizer_D.step()

            # 打印损失
            if i % 10 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # 每个 epoch 保存一次模型
        torch.save(generator.state_dict(), './generator.pth')
        torch.save(discriminator.state_dict(), './discriminator.pth')

# 只有在直接运行 train.py 时才执行训练逻辑
if __name__ == "__main__":
    train()