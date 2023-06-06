import glob
import shutil
import random


image_file = (glob.glob('/home/kemove/disk/project/mmseg_woodscape/data/test/rgb_images(test_set)/*.png'))

with open('test.txt', 'w') as txt_file:

    for name in image_file:
        name = name.split('/')[-1].split('.')[0]
        txt_file.writelines(name+'\n')
