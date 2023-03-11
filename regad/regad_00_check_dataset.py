# 1. Run Python File
# 2. paste:
#    export LD_LIBRARY_PATH=/home/kswada/kw/mvtech_ad/regad/venv/lib/python3.8/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

import os

import numpy as np
import glob

from PIL import Image


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# check image
# ----------------------------------------------------------------------------------------------------------------

base_path = '/media/kswada/MyFiles/dataset/mvtec_ad/'

cat = 'bottle'
cat2_list = ['broken_large', 'broken_small', 'contamination']
cat2_idx = 1


# ----------
fpath_img_test_list = np.sort(glob.glob(os.path.join(base_path, f'{cat}/test/{cat2_list[cat2_idx]}/*.png')))
fpath_img_gt_list = np.sort(glob.glob(os.path.join(base_path, f'{cat}/ground_truth/{cat2_list[cat2_idx]}/*.png')))

print(f'len test images: {len(fpath_img_test_list)}')
print(f'len gt images: {len(fpath_img_gt_list)}')


# ----------
img_idx = 0

print(fpath_img_test_list[img_idx])
print(fpath_img_gt_list[img_idx])

im_test = Image.open(fpath_img_test_list[img_idx])
im_gt = Image.open(fpath_img_gt_list[img_idx])

im_test.show()
im_gt.show()

print(im_test.format, im_test.size, im_test.mode)
print(im_gt.format, im_gt.size, im_gt.mode)


# ---------
for idx in range(5):
    im = Image.open(fpath_img_test_list[idx])
    im.show()


# ---------
print(np.array(im_test))
print(np.array(im_test).shape)
