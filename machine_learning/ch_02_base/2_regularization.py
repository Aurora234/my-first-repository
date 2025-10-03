import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge  # 线性回归模型
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
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 生成数据
X = np.linspace(-3, 3, 300).reshape(-1, 1)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, 300).reshape(-1, 1)

fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax[0, 0].plot(X, y, "gs")
ax[0, 1].plot(X, y, "yo")
ax[0, 2].plot(X, y, "yo")
# plt.show()
# 2. 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
poly20 = PolynomialFeatures(degree=20)
x_train = poly20.fit_transform(x_train)
x_test = poly20.fit_transform(x_test)

# 3.定义模型
model = LinearRegression()

# 4.训练模型
model = model.fit(x_train, y_train)

# 5.预测结果，计算误差
y_pred1 = model.predict(x_test)
test_loss1 = mean_squared_error(y_test, y_pred1)

ax[0, 0].plot(X, model.predict(poly20.fit_transform(X)), "r")
ax[0, 0].text(-3, 1, f"测试误差:{test_loss1:.4f}")

ax[1, 0].bar(np.arange(21), model.coef_.reshape(-1))

# 3.定义模型
lasso = Lasso(alpha=0.01)

# 4.训练模型
lasso = lasso.fit(x_train, y_train)

# 5.预测结果，计算误差
y_pred1 = lasso.predict(x_test)
test_loss2 = mean_squared_error(y_test, y_pred1)

ax[0, 1].plot(X, lasso.predict(poly20.fit_transform(X)), "r")
ax[0, 1].text(-3, 1, f"测试误差:{test_loss2:.4f}")

ax[1, 1].bar(np.arange(21), lasso.coef_.reshape(-1))

# 3.定义模型
ridge = Ridge(alpha=1)

# 4.训练模型
ridge = ridge.fit(x_train, y_train)

# 5.预测结果，计算误差
y_pred1 = ridge.predict(x_test)
test_loss2 = mean_squared_error(y_test, y_pred1)

ax[0, 2].plot(X, ridge.predict(poly20.fit_transform(X)), "r")
ax[0, 2].text(-3, 1, f"测试误差:{test_loss2:.4f}")

ax[1, 2].bar(np.arange(21), ridge.coef_.reshape(-1))

plt.show()
