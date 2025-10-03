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
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 1. 生成数据
X = np.linspace(-3, 3, 300).reshape(-1, 1)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, 300).reshape(-1, 1)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(X, y, "gs")
ax[1].plot(X, y, "yo")
ax[2].plot(X, y, "yo")
# plt.show()
# 2. 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 定义模型（线性回归模型）
model = LinearRegression()
# 欠拟合
x_train1 = x_train
x_test1 = x_test

# 4. 训练模型
model.fit(x_train1, y_train)  # 模型训练

# 5. 预测结果，计算误差
y_pred1 = model.predict(x_test1)  # 预测
ax[0].plot(np.array(X), model.predict(np.array(X)), "b--")
ax[0].text(-3, 1, f"测试集均方误差：{mean_squared_error(y_test, y_pred1):.4f}")
ax[0].text(-3, 1.3, f"训练集均方误差：{mean_squared_error(y_train, model.predict(x_train1)):.4f}")

# 恰好拟合
poly5 = PolynomialFeatures(degree=5)
x_train2 = poly5.fit_transform(x_train)
x_test2 = poly5.fit_transform(x_test)
# print(x_train2.shape)
# print(x_test2.shape)
model.fit(x_train2, y_train)
y_pred2 = model.predict(x_test2)
test_loss2 = mean_squared_error(y_test, y_pred2)
train_loss2 = mean_squared_error(y_train, model.predict(x_train2))
ax[1].plot(X, model.predict(poly5.fit_transform(X)), "b-")  # 绘制曲线
ax[1].text(-3, 1, f"测试集均方误差：{test_loss2:.4f}")
ax[1].text(-3, 1.3, f"训练集均方误差：{train_loss2:.4f}")

# 过拟合
poly20 = PolynomialFeatures(degree=20)
x_train3 = poly20.fit_transform(x_train)
x_test3 = poly20.fit_transform(x_test)
model.fit(x_train3, y_train)
y_pred3 = model.predict(x_test3)
test_loss3 = mean_squared_error(y_test, y_pred3)
train_loss3 = mean_squared_error(y_train, model.predict(x_train3))
ax[2].plot(X, model.predict(poly20.fit_transform(X)), "b-")  # 绘制曲线
ax[2].text(-3, 1, f"测试集均方误差：{test_loss3:.4f}")
ax[2].text(-3, 1.3, f"训练集均方误差：{train_loss3:.4f}")


plt.show()
