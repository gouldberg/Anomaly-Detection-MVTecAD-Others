
import os
import shutil
import glob
import cv2


# ----------
# all test/defective_type into test/bad

base_path = '/home/kswada/kw/mvtech_ad/ocr_gan'

data_path = os.path.join(base_path, 'data')

texture_classes = ["carpet", "grid", "leather", "tile", "wood"]
object_classes = ["cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"]
others = ["bottle"]

mvtec_class = texture_classes + object_classes + others

for cls_obj in mvtec_class:
    print(f'processing -- {cls_obj}')
    folder_obj_cls = os.path.join(data_path, cls_obj, 'test')
    folder_obj = os.path.join(data_path, cls_obj, 'test', 'bad')

    if os.path.exists(folder_obj):
        shutil.rmtree(folder_obj)

    os.makedirs(folder_obj)

    folder_list = glob.glob(os.path.join(folder_obj_cls, '*'))
    defective_cls_list = list(set([fold.split('/')[-1] for fold in folder_list]) - set(['good']))

    img_count = 0
    for defective_cls in defective_cls_list:
        print(f'processing -- {cls_obj} - {defective_cls}')
        defective_cls = defective_cls_list[0]
        img_files = glob.glob(os.path.join(folder_obj_cls, defective_cls, '*png'))

        for img_file in img_files:
            img_count += 1
            img = cv2.imread(img_file)
            save_name = os.path.join(folder_obj, str(img_count).zfill(3) + '.png')
            cv2.imwrite(save_name, img)
