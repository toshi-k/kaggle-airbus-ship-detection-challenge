import math

import numpy as np

from lib.visualize import coord2_img
from lib.predict import numpy_sigmoid, predict_boxes_numpy


def is_far(predict_x, predict_y, target, step):
    dist = math.sqrt((predict_x - target.x) ** 2 + (predict_y - target.y) ** 2)
    return dist > step * 1.5


def return_only_negative(predicted_tensors, index, dbox_params, img_size=768):

    fp_boxes = dict()

    for l, input_tensor in enumerate(predicted_tensors):

        step = img_size / input_tensor.shape[2]

        x_points = np.arange(step/2-0.5, img_size, step)
        y_points = np.arange(step/2-0.5, img_size, step)

        for x, x_point in enumerate(x_points):
            for y, y_point in enumerate(y_points):
                for j in range(len(dbox_params)):
                    fp_boxes[(l, x, y, j)] = numpy_sigmoid(input_tensor[6 * j, x, y])

    # negative mining
    # fp_boxes = [key for key, item in fp_boxes.items()]
    negative_boxes = {key: item for key, item in sorted(fp_boxes.items(), key=lambda x: x[1], reverse=True)[:512]}

    positive_boxes = list()

    return positive_boxes, negative_boxes, index


def calc_matching_degree(img1, img2):
    area_intersect = np.sum(img1 * img2)
    area_union = np.sum(img1) + np.sum(img2) - area_intersect

    if area_union < 1e-5:
        return 0

    matching_degree = area_intersect / area_union
    return matching_degree


def search_boxes(predicted_tensors, target, index, dbox_params, img_size=768):

    if np.isnan(target.x) and np.isnan(target.y):
        return return_only_negative(predicted_tensors, index, dbox_params, img_size=768)

    target_img = coord2_img(target.x, target.y, target.height, target.width, target.rotate) / 255.0
    positive_boxes = []

    fp_boxes = dict()

    best_degree = 0
    best_box = None

    for l, input_tensor in enumerate(predicted_tensors):

        step = img_size / input_tensor.shape[2]

        x_points = np.arange(step/2-0.5, img_size, step)
        y_points = np.arange(step/2-0.5, img_size, step)

        for x, x_point in enumerate(x_points):
            for y, y_point in enumerate(y_points):

                # for i in range(len(dbox_params)):

                i = (dbox_params.rotate_vars - target.rotate).abs().idxmin()

                # print('i: {}'.format(i))

                dbox_param = dbox_params.iloc[i]

                confidence, predict_x, predict_y, predict_height, predict_width, predict_rotate = predict_boxes_numpy(
                    input_tensor, i, x, y, x_point, y_point, dbox_param, step
                )

                for j in range(len(dbox_params)):
                    if j != i:
                        fp_boxes[(l, x, y, j)] = numpy_sigmoid(input_tensor[6 * j, x, y])

                if is_far(predict_x, predict_y, target, step):
                    fp_boxes[(l, x, y, i)] = numpy_sigmoid(input_tensor[6 * i, x, y])
                    continue

                img_dbox = coord2_img(predict_x, predict_y, predict_height, predict_width, predict_rotate, img_size)
                img_dbox_norm = img_dbox / 255

                matching_degree = calc_matching_degree(target_img, img_dbox_norm)

                if matching_degree > best_degree:

                    best_degree = matching_degree
                    best_box = (l, x, y, i)

                if matching_degree > 0.7:
                    positive_boxes.append((l, x, y, i))

                if matching_degree < 0.4:
                    fp_boxes[(l, x, y, i)] = numpy_sigmoid(input_tensor[6 * i, x, y])

    if len(positive_boxes) == 0 and best_box is not None:
        positive_boxes.append(best_box)

    num_train = 0

    for positive_box in positive_boxes:

        if len(positive_box) == 0:
            continue

        if positive_box in fp_boxes:
            fp_boxes.pop(positive_box)

        num_train += 1

    # negative mining
    # fp_boxes = [key for key, item in fp_boxes.items()]
    negative_boxes = {key: item for key, item in sorted(fp_boxes.items(), key=lambda x: x[1], reverse=True)[:512]}

    return positive_boxes, negative_boxes, index
