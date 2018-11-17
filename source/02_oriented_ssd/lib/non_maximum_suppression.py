import numpy as np
from collections import deque

from lib.predict import predict_boxes_numpy, predict_conf_numpy
from lib.visualize import coord2_img
from lib.search_best_box import calc_matching_degree


def thresholding(predicted_tensors, dbox_param, threshold=0.7, img_size=768):

    dict_conf = get_confidences(predicted_tensors, dbox_param, img_size)
    list_tuple = [key for key, item in dict_conf.items() if item > threshold]

    return list_tuple


def get_confidences(predicted_tensors, dbox_params, img_size):

    dict_conf = dict()

    for l, input_tensor in enumerate(predicted_tensors):

        step = img_size / input_tensor.shape[2]

        x_points = np.arange(step / 2 - 0.5, img_size, step)
        y_points = np.arange(step / 2 - 0.5, img_size, step)

        for x, _ in enumerate(x_points):
            for y, _ in enumerate(y_points):
                for i in range(len(dbox_params)):

                    # confidence
                    dict_conf[(l, x, y, i)] = predict_conf_numpy(input_tensor, i, x, y)

    return dict_conf


def non_maximum_suppression(predicted_tensors, dbox_params, threshold, threshold2, overlap, img_size=768):

    dict_conf = get_confidences(predicted_tensors, dbox_params, img_size)
    dict_conf = sorted(dict_conf.items(), key=lambda x: x[1], reverse=True)

    list_tuple_candidate = deque([(key, item) for key, item in dict_conf if item > threshold])

    list_tuple = list()
    list_predicted_img = list()

    max_conf = 0

    while True:

        if len(list_tuple_candidate) == 0 or len(list_tuple) > 20:
            if max_conf > threshold2:
                return list_tuple, list_predicted_img
            else:
                return list(), list()

        max_deg = 0
        pred, conf = list_tuple_candidate.popleft()

        max_conf = max(max_conf, conf)

        predicted_tensor = predicted_tensors[pred[0]]

        step = img_size / predicted_tensor.shape[2]

        x_points = np.arange(step / 2 - 0.5, img_size, step)
        y_points = np.arange(step / 2 - 0.5, img_size, step)
        dbox_param = dbox_params.iloc[pred[3]]

        _, predict_x, predict_y, predict_height, predict_width, predict_rotate = predict_boxes_numpy(
            predicted_tensor, pred[3], pred[1], pred[2], x_points[pred[1]], y_points[pred[2]], dbox_param, step
        )

        img_box = coord2_img(predict_x, predict_y, predict_height, predict_width, predict_rotate, img_size)

        for ref_box in list_predicted_img:

            matching_degree = calc_matching_degree(img_box/255.0, ref_box/255.0)
            max_deg = max(max_deg, matching_degree)

        if max_deg < overlap:
            # when overlap region is small enough
            list_tuple.append(pred)
            list_predicted_img.append(img_box)
