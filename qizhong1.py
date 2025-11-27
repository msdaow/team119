import torch

# 定义参数
in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = 3
batch_size = 1

# 创建输入张量
input = torch.randn(batch_size,
    in_channels,
    width,
    height)

# 创建卷积层
conv_layer = torch.nn.Conv2d(in_channels,
    out_channels,
    kernel_size=kernel_size)

# 前向传播
output = conv_layer(input)

# 打印形状信息
print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)