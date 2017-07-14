import sys

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1,3,1,projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 500) # theta旋转角从-4pi到4pi，相当于两圈
z = np.linspace(0, 2, 500) # z轴从下到上,从-2到2之间画100个点
r = z # 半径设置为z大小
x = r * np.sin(theta) # x和y画圆
y = r * np.cos(theta) # x和y画圆
ax.plot(x, y, z, label='curve')
ax.legend()

plt.show()