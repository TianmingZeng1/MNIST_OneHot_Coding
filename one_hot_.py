import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()  # 加载数据集
target = digits.target  # 获取目标标签

def one_hot_encoder(x):
    """One-Hot Encodes a target variable
    
    Args:
        x: Array containing target values
    
    Returns:
        An array of shape (n_samples, n_classes) with one-hot encoded target values.
    """
    # 获取类的数量，这里是10（数字0到9）
    nclasses = 10
    # 初始化一个形状为 (样本数量, 类别数量) 的全零数组
    out = np.zeros((x.shape[0], nclasses))
    # 对于每个样本，将对应的类索引位置设为1
    for i, x_ in enumerate(x):
        out[i, x_] = 1
    return out

# 测试函数
target_oh = one_hot_encoder(target)
print(f"target value: {target[40]}, corresponding one-hot vector: {target_oh[40,:]}")

#for i in range(50):
 #   print(f"Sample {i}, target value: {target[i]}")
