import numpy as np
import cv2
import os


if __name__ == '__main__':
    num = 0
    mask_path = 'DUT-RGBD/train_data/train_masks'
    mask_list = os.listdir(mask_path)
    gts_path = os.path.join('DUT-RGBD/train_data/train_gts')
    if not os.path.exists(gts_path):
        os.mkdir(gts_path)

    for mask in mask_list:
        m = cv2.imread(os.path.join(mask_path, mask), cv2.IMREAD_GRAYSCALE)
        ret2, th2 = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = th2[..., np.newaxis]

        cv2.imwrite(os.path.join('DUT-RGBD/train_data/train_gts', mask), result)
        num += 1

        if num % 100 == 0:
            print(num)

    mask_path = 'DUT-RGBD/test_data/test_masks'
    mask_list = os.listdir(mask_path)
    gts_path = os.path.join('DUT-RGBD/test_data/test_gts')
    if not os.path.exists(gts_path):
        os.mkdir(gts_path)

    for mask in mask_list:
        m = cv2.imread(os.path.join(mask_path, mask), cv2.IMREAD_GRAYSCALE)
        ret2, th2 = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = th2[..., np.newaxis]

        cv2.imwrite(os.path.join('DUT-RGBD/test_data/test_gts', mask), result)
        num += 1
        if num % 100 == 0:
            print(num)

    print('finished')