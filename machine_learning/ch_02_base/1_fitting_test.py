import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.preprocessing import PolynomialFeatures  # 构建多项式特征
from sklearn.model_selection import train_test_split  # 划分训练集和测试集
from sklearn.metrics import mean_squared_error  # 均方误差损失函数

'''
1. 生成数据
2. 划分训练集和测试集
3. 定义模型（线性回归模型）
4. 训练模型
5. 预测结果，计算误差
'''

# 1. 生成数据
X = np.linspace(-3, 3, 300)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, 300).reshape(-1, 1)

# 2. 划分训练集和测试集
print(train_test_split(X, y, test_size=0.2, random_state=42))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 定义模型（线性回归模型）
# 4. 训练模型
# 5. 预测结果，计算误差
