import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 本代码是一个最简单的线形回归问题，优化函数为momentum

rate = 1e-2 # learning rate
def da(y,y_p,x):
    return (y-y_p)*(-x)

def db(y,y_p):
    return (y-y_p)*(-1)

def draw_hill(x,y):
    a = np.linspace(-10,10,100)
    print(a)
    b = np.linspace(-10,10,100)
    x = np.array(x)
    y = np.array(y)

    allSSE = np.zeros(shape=(len(a),len(b)))
    for ai in range(0,len(a)):
        for bi in range(0,len(b)):
            a0 = a[ai]
            b0 = b[bi]
            tmp = y - (a0*x + b0)
            tmp = tmp**2 # 对矩阵内的每一个元素平方
            SSE = sum(tmp)/2
            allSSE[ai][bi] = SSE

    a,b = np.meshgrid(a, b)

    return [a,b,allSSE]
# simulated data
x = [30	,35,37,	59,	70,	76,	88,	100]
y = [1100,	1423,	1377,	1800,	2304,	2588,	3495,	4839]


# 数据归一化
x_max = max(x)
x_min = min(x)
y_max = max(y)
y_min = min(y)
# x_mean = np.mean(x)
for i in range(0,len(x)):
    x[i] = (x[i] - x_min)/(x_max - x_min)
    y[i] = (y[i] - y_min)/(y_max - y_min)

[ha,hb,hallSSE] = draw_hill(x,y)

# 初始化a,b值
a = -10
b = -10
fig4 = plt.figure(4,figsize=(12,8))
plt.subplot(2,2,2)
plt.contourf(ha,hb,hallSSE,15,alpha=0.75,cmap=plt.cm.hot)
C = plt.contour(ha,hb,hallSSE,15,colors='black')
plt.clabel(C,inline=True)
plt.xlabel('a')
plt.ylabel('b')
plt.xticks()
plt.yticks()
# plt.show()
# plot bowl
ax = fig4.add_subplot(2, 2, 1, projection='3d')
ax.plot_surface(ha, hb, hallSSE, rstride=2, cstride=2, cmap='rainbow')
plt.ion() # iteration on
all_a = []
all_b = []
all_loss = []
all_step = []
for step in range(1,500):
    loss = 0
    all_da = 0
    all_db = 0
    for i in range(0,len(x)):
        y_p = a*x[i] + b
        loss = loss + (y[i] - y_p)*(y[i] - y_p)/2
        all_da = all_da + da(y[i],y_p,x[i])
        all_db = all_db + db(y[i],y_p)

    a = a - rate*all_da
    b = b - rate*all_db

    all_a.append(a)
    all_b.append(b)
    all_loss.append(loss)
    all_step.append(step)

    # plot gradient descent point
    ax.scatter(a, b, loss, color='black')

    plt.subplot(2,2,2)
    plt.scatter(a,b,loss,color='green')

    # plot lines
    plt.subplot(2,2,3)
    plt.plot(x,y)
    plt.plot(x,y,'o')
    x_ = np.linspace(0, 1, 2)
    y_draw = a * x_ + b
    plt.plot(x_,y_draw)

    # plot losses
    plt.subplot(2,2,4)
    plt.plot(all_step,all_loss,color='orange')
    plt.xlabel("step")
    plt.ylabel("loss")

    if step%10 == 0:
        print("step: ", step, " loss: ", loss)
        plt.show()
        plt.pause(0.01)
plt.show()
plt.pause(99999999999)