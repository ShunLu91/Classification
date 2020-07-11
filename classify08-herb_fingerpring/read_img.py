import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np

path = '../dataset/herb_fingerprint/data/0/000_______JZZKM1802012.png'
lena = mpimg.imread(path)  # 读取和代码处于同一目录下的 lena.png
print(lena.shape)
lena2 = lena[50:750, 100:550, :]

plt.imshow(lena2)  # 显示图片
plt.axis('off')  # 不显示坐标轴
plt.show()