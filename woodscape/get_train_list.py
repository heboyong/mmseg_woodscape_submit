import glob
import shutil
import random


image_file = (glob.glob('/home/kemove/disk/project/mmseg_woodscape/data/real/rgb_images/*.png'))
random.shuffle(image_file)
print(len(image_file))
train_number = 2058
with open('train.txt', 'w') as txt_file:

    for name in image_file[:train_number]:
        name = name.split('/')[-1].split('.')[0]
        txt_file.writelines(name+'\n')

with open('val.txt', 'w') as txt_file:

    for name in image_file[train_number:]:
        name = name.split('/')[-1].split('.')[0]
        txt_file.writelines(name+'\n')