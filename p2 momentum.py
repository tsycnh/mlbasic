import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

rate = 1e-2
def da(y,y_p,x):
    return (y-y_p)*(-x)

def db(y,y_p):
    return (y-y_p)*(-1)

def draw_hill(x,y):
    loss = 0
    a = np.linspace(-10,10,100)
    print(a)
    b = np.linspace(-10,10,100)
    x = np.array(x)
    y = np.array(y)

    fig = plt.figure(89)
    ax = Axes3D(fig)
    # a,b = np.meshgrid(a, b)

    allSSE = np.zeros(shape=(len(a),len(b)))
    for ai in range(0,len(a)):
        for bi in range(0,len(b)):
            a0 = a[ai]
            b0 = b[bi]
            tmp = y - (a0*x + b0)
            tmp = tmp**2 # 对矩阵内的每一个元素平方
            SSE = sum(tmp)/2
            allSSE[ai][bi] = SSE

    print(allSSE)
    a,b = np.meshgrid(a, b)

    #ax.scatter(a,b,allSSE)
    ax.plot_surface(a, b, allSSE, rstride=1, cstride=1, cmap='rainbow')
    plt.ion()
    plt.draw()
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
b = 10
fig4 = plt.figure(4)
plt.ion()
all_a = []
all_b = []
all_loss = []
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
    if step%10 == 0:
        print("step: ",step," loss: " , loss)

    all_a.append(a)
    all_b.append(b)
    all_loss.append(loss)

    ax = Axes3D(fig4)
    ax.scatter(all_a, all_b, all_loss, color='black')
    ax.plot_surface(ha, hb, hallSSE, rstride=1, cstride=1, cmap='rainbow')
    plt.ion()
    plt.show()

    plt.figure(2)
    plt.plot(x,y)
    plt.plot(x,y,'o')
    x_ = np.arange(0, 1, step=0.01)
    y_draw = a * x_ + b
    plt.plot(x_,y_draw)

    plt.ion()
    plt.show()
    plt.pause(0.01)


