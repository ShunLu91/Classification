import os
import numpy as np
import shutil


def read_img_path(file_path):
    path = os.listdir(file_path)
    for i, _path in enumerate(path):
        if _path.endswith('png'):
            pass
        else:
            path.pop(i)
    path = [os.path.join(file_path, p) for p in path]
    return path


def build_remove(source, target):
    if not os.path.exists(target):
        os.mkdir(target)
        print("build target direction")
    for image in source:
        if image.endswith('png'):
            shutil.move(image, target + '/' + image.split("/")[-1])


if __name__ == '__main__':
    os.chdir("..")
    # 分到train和test中
    src = '../dataset/BreakHis/train'
    dst = '../dataset/BreakHis/val'
    if not os.path.exists(dst):
        os.mkdir(dst)

    for cur_path in os.listdir(os.path.join(src)):
        cur = np.asarray(read_img_path(os.path.join(src, cur_path)))
        np.random.shuffle(cur)
        train = cur[:int(len(cur) * 0.7)]
        val = cur[int(len(cur) * 0.7):]
        build_remove(val, os.path.join(dst, cur_path))
