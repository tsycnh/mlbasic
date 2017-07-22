import numpy as np
import matplotlib.pyplot as plt
x = np.array([30	,35,37,	59,	70,	76,	88,	100]).astype(np.float32)
y = np.array([1100,	1423,	1377,	1800,	2304,	2588,	3495,	4839]).astype(np.float32)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

x_max = max(x)
x_min = min(x)
y_max = max(y)
y_min = min(y)

for i in range(0,len(x)):
    x[i] = (x[i] - x_min)/(x_max - x_min)
    y[i] = (y[i] - y_min)/(y_max - y_min)

print(x,y)
a = 1
b = 0
x_ = np.array([0,1])
y_ = a*x_+b
yp = a*x +b
r = sum(np.square(np.round(yp-y,4)))
print(r/16)
plt.scatter(x,y)
plt.xlabel(u"x")
plt.ylabel(u"y")
plt.plot(x_,y_,color='green')
plt.pause(3333)
