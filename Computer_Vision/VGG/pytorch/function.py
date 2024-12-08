import cv2, os
import numpy as np

def read_gt(gt_txt, num_image):
    cls = np.zeros(num_image)
    f = open(gt_txt, 'r')
    lines = f.readlines()

    for it in range(len(lines)):
        cls[it] = int((lines[it])[:-1]) - 1

    f.close()

    return cls

def mini_batch_training_zip(z_file, z_file_list, cls, batch_size, image_size=128):
    batch_image = np.zeros((batch_size, image_size, image_size, 3))
    batch_cls = np.zeros(batch_size)

    rand_num = np.random.randint(0, len(z_file_list), size=batch_size)

    for it in range(batch_size):
        temp = rand_num[it]
        image_temp = z_file.read(z_file_list[temp])
        image_temp = cv2.imdecode(np.fromstring(image_temp, np.uint8), 1)
        image_temp = cv2.resize(image_temp, (image_size, image_size))
        image_temp = image_temp.astype(np.float32)

        batch_image[it, :, :] = (image_temp / 255.0) * 2 - 1
        batch_cls[it] = cls[temp]

    return batch_image, batch_cls