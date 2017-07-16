import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
# 本代码是一个最简单的线形回归问题，优化函数为 momentum SGD

rate = 1e-2 # learning rate
def da(y,y_p,x):
    return (y-y_p)*(-x)

def db(y,y_p):
    return (y-y_p)*(-1)

def draw_hill(x,y):
    a = np.linspace(-20,20,100)
    print(a)
    b = np.linspace(-20,20,100)
    x = np.array(x)
    y = np.array(y)

    allSSE = np.zeros(shape=(len(a),len(b)))
    for ai in range(0,len(a)):
        for bi in range(0,len(b)):
            a0 = a[ai]
            b0 = b[bi]
            tmp = y - (a0*x + b0)
            tmp = tmp**2 # 对矩阵内的每一个元素平方
            SSE = sum(tmp)/(2*len(x))
            allSSE[ai][bi] = SSE

    a,b = np.meshgrid(a, b)

    return [a,b,allSSE]

def shuffle_data(x,y):
    # 随机打乱x，y的数据，并且保持x和y一一对应
    seed = random.random()
    random.seed(seed)
    random.shuffle(x)
    random.seed(seed)
    random.shuffle(y)

def get_batch_data(x,y,batch=3):
    shuffle_data(x,y)
    x_new = x[0:batch]
    y_new = y[0:batch]
    return [x_new,y_new]
# simulated data
x = [30	 ,   35,   37,	 59,   70,	 76,   88,	100]
y = [1100, 1423, 1377, 1800, 2304, 2588, 3495, 4839]


# 数据归一化
x_max = max(x)
x_min = min(x)
y_max = max(y)
y_min = min(y)
for i in range(0,len(x)):
    x[i] = (x[i] - x_min)/(x_max - x_min)
    y[i] = (y[i] - y_min)/(y_max - y_min)

[ha,hb,hallSSE] = draw_hill(x,y)

# 初始化a,b值
a = 10
b = -20
fig4 = plt.figure(4,figsize=(12,8))

# 绘制等高线图
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
last_va = 0 # momentum
last_vb = 0
gamma = 0.9
for step in range(1,500):
    loss = 0
    all_da = 0
    all_db = 0
    # mini-batch gd 的精华在此
    batch_size = 4
    [x_new, y_new] = get_batch_data(x,y,batch=batch_size)
    for i in range(0,len(x_new)):
        y_p = a*x_new[i] + b
        loss = loss +(y_new[i] - y_p)*(y_new[i] - y_p)/2

        all_da = all_da + da(y_new[i],y_p,x_new[i])
        all_db = all_db + db(y_new[i],y_p)

    va = gamma * last_va + rate*all_da
    vb = gamma * last_vb + rate*all_db

    a = a - va
    b = b - vb

    last_va = va
    last_vb = vb

    all_a.append(a)
    all_b.append(b)
    all_loss.append(loss)
    all_step.append(step)

    # plot gradient descent point
    ax.scatter(a, b, loss/batch_size, color='black')

    # plot on contour
    plt.subplot(2,2,2)
    plt.scatter(a,b,loss/batch_size,color='blue',marker='.',linewidths=0.1)

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