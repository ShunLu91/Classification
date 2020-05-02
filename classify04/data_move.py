import os
import glob
import shutil


pos_list = glob.glob('../dataset/BreakHis/breast/benign/*/*/*/*/*.png')
neg_list = glob.glob('../dataset/BreakHis/breast/malignant/*/*/*/*/*.png')

print(len(pos_list))
print(len(neg_list))

for i in pos_list:
    shutil.copyfile(i, '../dataset/BreakHis/breast/positive/' + os.path.basename(i))
for j in neg_list:
    shutil.copyfile(j, '../dataset/BreakHis/breast/negative/' + os.path.basename(j))
