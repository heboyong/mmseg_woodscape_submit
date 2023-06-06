import glob
import os
import shutil
import random

root_dir = '/home/kemove/disk/project/mmseg_woodscape/data/syn/'

all_file_dir = os.path.join(root_dir, 'paralleldomain-woodscape')

for scene in os.listdir(all_file_dir):
    print(scene)
    pre_image = os.path.join(all_file_dir, scene, 'previous_images')
    rgb_image = os.path.join(all_file_dir, scene, 'rgb_images')
    rgb_labels = os.path.join(all_file_dir, scene, 'motion_annotations/rgbLabels')
    gt_labels = os.path.join(all_file_dir, scene, 'motion_annotations/gtLabels')

    for item in os.listdir(rgb_image):
        shutil.copy(os.path.join(rgb_image, item), os.path.join(root_dir, 'rgb_images', str(scene) + '_' + item))

    for item in os.listdir(pre_image):
        shutil.copy(os.path.join(pre_image, item), os.path.join(root_dir, 'previous_images', str(scene) + '_' + item))

    for item in os.listdir(gt_labels):
        shutil.copy(os.path.join(gt_labels, item), os.path.join(root_dir, 'motion_annotations/gtLabels', str(scene) + '_' + item))

    for item in os.listdir(rgb_labels):
        shutil.copy(os.path.join(rgb_labels, item), os.path.join(root_dir, 'motion_annotations/rgbLabels', str(scene) + '_' + item))
