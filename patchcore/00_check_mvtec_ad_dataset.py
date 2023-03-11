# export LD_LIBRARY_PATH=/home/kswada/kw/mvtech_ad/patchcore/venv/lib/python3.8/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

import os
import glob
import random
import cv2


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# check image resolution
# ----------------------------------------------------------------------------------------------------------------

# set dataset path
dataset_path = '/media/kswada/MyFiles/dataset/mvtec_ad'

# this is all categories
category_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


# ----------
# train by category

for cat in category_list:

    img_paths = glob.glob(os.path.join(dataset_path, cat, 'train/good/*.png'))

    print(f'------------------------------------------------------\n')
    print(f'category: {cat}  number of files: {len(img_paths)}')

    # 10 samples
    img_paths_sample = random.sample(img_paths, 10)

    for imgpath in img_paths_sample:
        filename = os.path.basename(imgpath)
        img = cv2.imread(imgpath)
        print(f'      category: {cat}  img: {filename} - shape: {img.shape}')
        # cv2.imshow(f'img: {filename}', img)
        # cv2.waitKey(0)

    # cv2.destroyAllWindows()


