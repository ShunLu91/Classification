import os
import glob
import random
import shutil


data_dir = '../dataset/herb_fingerprint'
train_dir = '../dataset/herb_fingerprint/train'
val_dir = '../dataset/herb_fingerprint/val'


name_list = ['0', '1', '2']
data_dict = {}
for name in name_list:
    data_dict[name] = []

data = list()
for _name in name_list:
    _dir = os.path.join(data_dir, _name, '*.png')
    data.extend(glob.glob(_dir))
    data_dict[_name] = glob.glob(_dir)

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
        if num < len(data_dict[name_list[index]]) * 0.7:
            shutil.copyfile(_path, os.path.join(train_dir_class, os.path.basename(_path)))
        else:
            shutil.copyfile(_path, os.path.join(val_dir_class, os.path.basename(_path)))





