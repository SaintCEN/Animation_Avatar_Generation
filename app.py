from flask import Flask, send_file
from flask_cors import CORS  # 导入 CORS
import torch
from PIL import Image
import io
import logging
import numpy as np
import torch.nn
from torch import nn

app = Flask(__name__)
CORS(app)  # 启用 CORS 支持

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

# 加载保存的生成器模型
latent_dim = 100
img_shape = (3, 64, 64)
generator = Generator(latent_dim, img_shape)
generator.load_state_dict(torch.load("generator.pth"))
generator.eval()  # 设置为评估模式

# 生成图片的函数
def generate_image(generator):
    num_samples = 1
    z = torch.randn(num_samples, latent_dim)  # 生成随机噪声
    with torch.no_grad():
        generated_images = generator(z)
        # 将张量转换为 PIL 图像
        generated_images = generated_images.squeeze(0).permute(1, 2, 0).cpu().numpy()
        generated_images = (generated_images + 1) / 2 * 255  # 将范围从 [-1, 1] 转换为 [0, 255]
        generated_images = np.clip(generated_images, 0, 255).astype(np.uint8)  # 确保像素值在 0-255 范围内
        return Image.fromarray(generated_images)

# API 路由：生成图片并返回图片 URL
@app.route('/generate-image', methods=['GET'])
def generate_image_api():
    try:
        image = generate_image(generator)
        img_io = io.BytesIO()  # 创建一个内存缓冲区
        image.save(img_io, format='PNG')  # 将图像保存为 PNG 格式
        img_io.seek(0)  # 将指针移动到文件开头
        return send_file(img_io, mimetype='image/png')  # 返回图像文件
    except Exception as e:
        logging.error(f"Error generating image: {e}")
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)