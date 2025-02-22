# 测试网络
import torch
from torchvision.utils import save_image
from train import Generator

# 加载保存的生成器模型
latent_dim = 100
img_shape = (3,64,64)
generator = Generator(latent_dim, img_shape)
generator.load_state_dict(torch.load("generator.pth"))
generator.eval()  # 设置为评估模式，关闭 dropout 和 batch normalization

# 生成随机噪声向量
num_samples = 1  # 生成图像的数量
z = torch.randn(num_samples, latent_dim)

# 通过生成器生成图像
with torch.no_grad():
    generated_images = generator(z)

# 将生成的图像保存到文件中
save_image(generated_images, "generated_images.png", nrow=5, normalize=True)

# 可选：显示生成的图像
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(num_samples):
    axes[i // 5, i % 5].imshow(generated_images[i].permute(1, 2, 0).cpu().numpy())
    axes[i // 5, i % 5].axis("off")
plt.show()