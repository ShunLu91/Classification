import os
import shutil


def moveFile(fileDir):
    pathDir = os.listdir(fileDir)  # 列举所有文件夹路径'train', 'val', 'test'
    for _dir in pathDir:
        for _path in os.listdir(os.path.join(fileDir, _dir, 'PNEUMONIA')):
            if not os.path.exists(os.path.join(fileDir, _dir, 'BACTERIA')):
                os.makedirs(os.path.join(fileDir, _dir, 'BACTERIA'))
            if not os.path.exists(os.path.join(fileDir, _dir, 'VIRUS')):
                os.makedirs(os.path.join(fileDir, _dir, 'VIRUS'))
            if 'bacteria' in _path:
                shutil.move(os.path.join(fileDir, _dir, 'PNEUMONIA', _path), os.path.join(fileDir, _dir, 'BACTERIA'))
            if 'virus' in _path:
                shutil.move(os.path.join(fileDir, _dir, 'PNEUMONIA', _path), os.path.join(fileDir, _dir, 'VIRUS'))

    return


if __name__ == '__main__':
    data_dir = '/Users/lushun/Documents/dataset/lung/Data'
    moveFile(data_dir)
