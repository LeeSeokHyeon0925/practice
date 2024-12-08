import cv2, os
import numpy as np

VOC_COLORMAP = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 0, 128],
                [128, 128, 0], [128, 128, 128], [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192],
                [128, 0, 64], [128, 0, 192], [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128],
                [0, 192, 0], [0, 192, 128], [128, 64, 0]]

def mini_batch_training(image, gt, batch_size, image_size):
    batch_image = np.zeros((batch_size, image_size, image_size, 3))
    batch_gt = np.zeros((batch_size, image_size, image_size, 1))

    rand_num = np.random.randint(0, image.shape[0], size=batch_size)

    for it in range(batch_size):
        temp = rand_num[it]
        batch_image[it, :, :] = (image[temp, :, :, :] / 255.0) * 2 - 1
        batch_gt[it] = gt[temp, :, :, 0:1]

    return batch_image, batch_gt

def load_semantic_seg_data(image_path, gt_path, image_size):
    image_names = os.listdir(image_path)
    gt_names = os.listdir(gt_path)

    images = np.zeros(shape=(len(image_names), image_size, image_size, 3), dtype=np.uint8)
    gts = np.zeros(shape=(len(gt_names), image_size, image_size, 3), dtype=np.uint8)

    for it in range(len(image_names)):
        print('image %d / %d' % (it + 1, len(image_names)))
        image = cv2.imread(image_path + image_names[it])
        image = cv2.resize(image, (image_size, image_size))
        images[it, :, :, :] = image

    for it in range(len(gt_names)):
        print('gt %d / %d' % (it + 1, len(gt_names)))
        gt = cv2.imread(gt_path + gt_names[it])
        gt_index = np.zeros(shape=(gt.shape[0], gt.shape[1], 3), dtype=np.uint8)

        for ic in range(len(VOC_COLORMAP)):
            code = VOC_COLORMAP[ic]
            gt_index[np.where(np.all(gt == code, axis=-1))] = ic

        gt_index = cv2.resize(gt_index, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        gts[it, :, :, :] = gt_index

    return images, gts