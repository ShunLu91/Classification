import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist


def rang(x, y, jump):
  while x < y:
    yield x
    x += jump

if __name__ == '__main__':
    # 构造参数方程表示的圆
    theta = np.arange(0, 2*np.pi, 0.01)
    # a = 7.1970 * np.cos(theta)
    # b = 7.1970 * np.sin(theta)
    a = 16.8570 * np.cos(theta)
    b = 16.8570 * np.sin(theta)
    b = -abs(b)
    # 读取原始数据
    data = pd.read_excel('project3.xlsx').values
    x = data[:, 0]
    y = data[:, 1]
    # for i in range(0, 301):
    #     for j in range(0,301):
    #         if round(x[i],2) == round(a[j],2):
    #             if round(y[i],2) == round(b[j],2):
    #                 # c = round(y[i],2) - round(b[j],2)
    #                 print(x[i])
    z = np.polyfit(x, y, 5) # 用10次多项式拟合
    p1 = np.poly1d(z)
    print(p1) # 在屏幕上打印拟合多项式
    yvals = p1(x)
    k = -0.46630766*x
    for i in range(0, 301):
        for j in range(0, 301):
            if round(x[i],2) == round(a[j], 2):
                if round(yvals[i], 2) > round(b[j], 2):
                    print(x[i])

    # 坐标轴移到中间
    fig = plt.figure('Sine Wave', (10, 8))
    ax = axisartist.Subplot(fig, 1, 1, 1)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, 0)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["x"].set_axis_direction('top')
    ax.axis["y"].set_axis_direction('left')

    plot1 = plt.plot(x, y, 'g', label='original values')
    plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
    plot3 = plt.plot(a, b, 'b')
    plt.axis('equal')
    plt.savefig("filename.png")
    plt.show()