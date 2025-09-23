import torch
import numpy as np


tensor_from_list = torch.tensor([[0, 8], [8, 0]])
print("从 list 创建的张量:\n", tensor_from_list)

np_array = np.array([[2, 6], [1, 9]])
tensor_from_numpy = torch.from_numpy(np_array)
print("从 numpy 数组创建的张量:\n", tensor_from_numpy)

import torch

x_zeros = torch.zeros([4, 6])
print("全0张量：")
print(x_zeros)

x_ones = torch.ones([1, 3, 6])
print("\n全1张量：")
print(x_ones)

import torch

# ========== 1. zeros_like / ones_like：基于已有张量形状生成全0/全1张量 ==========
# 先创建一个“输入张量”（示例形状为 2行3列）
input_tensor = torch.tensor([[1, 2, 3],
                            [4, 5, 6],
                             [6,7,8]])
print("输入张量:")
print(input_tensor)
print("形状:", input_tensor.shape, "\n")

# 返回与 input_tensor 形状相同的全0张量
x_zeros_like = torch.zeros_like(input_tensor)
print(" 结果:")
print(x_zeros_like, "\n")

# 返回与 input_tensor 形状相同的全1张量
x_ones_like = torch.ones_like(input_tensor)
print(" 结果:")
print(x_ones_like, "\n")


# ========== 2. 指定形状创建张量==========

shape_tensor_zeros = torch.zeros(6, 2)  # 创建 3行2列 的全0张量
print(" 创建的全0张量:")
print(shape_tensor_zeros, "\n")

shape_tensor_ones = torch.ones(3, 2)   # 创建 3行2列 的全1张量
print(" 创建的全1张量:")
print(shape_tensor_ones, "\n")


# ========== 3. 依赖另外一个张量的形状创建张量 ==========
# 先创建一个“参考张量” x
x = torch.tensor([[10, 20],
                 [30, 40],
                 [50, 60]])
print("参考张量 x:")
print(x)
print("x 的形状:", x.shape, "\n")

# 方式1：用 zeros_like/ones_like 直接复用形状
y_like_zeros = torch.zeros_like(x)
print(" y_like_zeros:")
print(y_like_zeros, "\n")

# 方式2：先获取 x 的形状（x.size() 或 x.shape），再用形状创建张量
y_ones_with_size = torch.ones(x.size())  # x.size() 等价于 x.shape
print(" y_ones_with_size:")
print(y_ones_with_size)

import torch

# 1维张量
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
print("1维张量形状：", tensor_1d.shape)  # 等价于 tensor_1d.size()

# 2维张量
tensor_2d = torch.tensor([[1,2,3,4,5],
                          [6,7,8,9,10],
                          [11,12,13,14,15]])
print("2维张量形状：", tensor_2d.shape)

# 3维张量
tensor_3d = torch.randn(4, 5, 3)
print("3维张量形状：", tensor_3d.shape)

import torch

# 创建演示用张量（指定为浮点型）
x = torch.tensor([1, 2, 9], dtype=torch.float32)  # 关键：添加 dtype=torch.float32
y = torch.tensor([4, 3, 6])
x_neg = torch.tensor([-8, -2, 12])

# 其他运算保持不变
z_add = x + y
print("加法: x + y =", z_add)

z_sub = x - y
print("减法: x - y =", z_sub)

z_pow = x.pow(2)
print("求幂: x² =", z_pow)

z_mul = x.mul(y)
print("乘法: x.mul(y) =", z_mul)

z_div = x.div(y)
print("除法: x.div(y) =", z_div)

z_abs = x_neg.abs()
print("绝对值: x_neg.abs() =", z_abs)

z_sum = x.sum()
print("求和: x.sum() =", z_sum)

# 现在可以正常计算均值了
z_mean = x.mean()
print("求均值: x.mean() =", z_mean)

import torch

# 1. transpose：交换两个维度
x_trans = torch.zeros([2, 3])
print("transpose 前形状:", x_trans.shape)
x_trans = x_trans.transpose(0, 1)
print("transpose 后形状:", x_trans.shape, "\n")

# 2. view：重塑张量形状
x_view = torch.arange(0, 6)  # 生成0-5的张量
print("view 前形状:", x_view.shape)
x_view = x_view.view(2, 3)
print("view 后形状:", x_view.shape, "\n")

# 3. squeeze：移除指定维度（维度大小为1时）
x_squeeze = torch.zeros([1, 2, 3])
print("squeeze 前形状:", x_squeeze.shape)
x_squeeze = x_squeeze.squeeze(0)
print("squeeze 后形状:", x_squeeze.shape, "\n")

# 4. unsqueeze：新增指定维度
x_unsqueeze = torch.zeros([2, 3])
print("unsqueeze 前形状:", x_unsqueeze.shape)
x_unsqueeze = x_unsqueeze.unsqueeze(1)
print("unsqueeze 后形状:", x_unsqueeze.shape, "\n")

# 5. cat：拼接多个张量（在dim=1维度拼接）
x_cat = torch.zeros([2, 1, 3])
y_cat = torch.zeros([2, 3, 3])
z_cat = torch.zeros([2, 2, 3])
w_cat = torch.cat([x_cat, y_cat, z_cat], dim=1)
print("cat 后形状:", w_cat.shape)

import torch

# —————— 1. 张量移动到不同设备 ——————
x = torch.tensor([1.0, 2.0])
print("原始张量的设备：", x.device)

# 移动到CPU（默认已在CPU，演示 .to('cpu') 用法）
x_cpu = x.to('cpu')
print("移动到CPU后的设备：", x_cpu.device)



# —————— 2. 张量的梯度计算 ——————
# 创建带梯度追踪的张量
x = torch.tensor([[1., 0.], [-1., 1.]], requires_grad=True)
print("输入张量 x:\n", x)

# 计算 z = 所有元素的平方和
z = x.pow(2).sum()
print("z = x².sum() 的值：", z)

# 反向传播求梯度（dz/dx）
z.backward()

# 查看x的梯度
print("x的梯度 x.grad:\n", x.grad)