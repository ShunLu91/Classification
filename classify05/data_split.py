import os
import glob
import random
import shutil


data_dir = '../dataset/rail_surface_defects/IMAGES'
train_dir = '../dataset/rail_surface_defects/train'
val_dir = '../dataset/rail_surface_defects/val'


name_list = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
data_dict = {}
for name in name_list:
    data_dict[name] = []

data = glob.glob('../dataset/rail_surface_defects/IMAGES/*.jpg')

# read path
for _path in data:
    for index in range(len(name_list)):
        if name_list[index] in _path:
            data_dict[name_list[index]].append(_path)
            break

# shuffle and move
for index in range(len(name_list)):
    random.shuffle(data_dict[name_list[index]])
    train_dir_class = os.path.join(train_dir, name_list[index])
    val_dir_class = os.path.join(val_dir, name_list[index])
    if not os.path.exists(train_dir_class):
        os.mkdir(train_dir_class)
    if not os.path.exists(val_dir_class):
        os.mkdir(val_dir_class)

    for num, _path in enumerate(data_dict[name_list[index]]):
        if num < len(data_dict[name_list[index]]) // 2:
            shutil.copyfile(_path, os.path.join(train_dir_class, os.path.basename(_path)))
        else:
            shutil.copyfile(_path, os.path.join(val_dir_class, os.path.basename(_path)))





