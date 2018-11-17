import math
import numpy as np
import torch
from torch.nn import functional as F


def predict_boxes(input_tensor, i, x, y, x_point, y_point, dbox_param, step, b=0):

    confidence = F.sigmoid(input_tensor[b:b+1, 6 * i, x, y])
    predict_x = F.tanh(input_tensor[b:b+1, 6 * i + 1, x, y]) * step + x_point + dbox_param.x_shifts
    predict_y = F.tanh(input_tensor[b:b+1, 6 * i + 2, x, y]) * step + y_point + dbox_param.y_shifts

    predict_height = torch.exp(input_tensor[b:b + 1, 6 * i + 3, x, y] + math.log2(step) / 1.5) * dbox_param.height_shifts + 2
    predict_width = torch.exp(input_tensor[b:b + 1, 6 * i + 4, x, y] + math.log2(step) / 1.5) * dbox_param.width_shifts + 2
    predict_rotate = torch.atan(input_tensor[b:b+1, 6 * i + 5, x, y]) + dbox_param.rotate_vars

    return confidence, predict_x, predict_y, predict_height, predict_width, predict_rotate


def numpy_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def numpy_softplus(x):
    return np.log(1 + np.exp(x))


def predict_boxes_numpy(input_tensor, i, x, y, x_point, y_point, dbox_param, step):

    confidence = numpy_sigmoid(input_tensor[6 * i, x, y])
    predict_x = np.tanh(input_tensor[6 * i + 1, x, y]) * step + x_point + dbox_param.x_shifts
    predict_y = np.tanh(input_tensor[6 * i + 2, x, y]) * step + y_point + dbox_param.y_shifts

    predict_height = np.exp(input_tensor[6 * i + 3, x, y] + math.log2(step) / 1.5) * dbox_param.height_shifts + 2
    predict_width = np.exp(input_tensor[6 * i + 4, x, y] + math.log2(step) / 1.5) * dbox_param.width_shifts + 2
    predict_rotate = np.arctan(input_tensor[6 * i + 5, x, y]) + dbox_param.rotate_vars

    return confidence, predict_x, predict_y, predict_height, predict_width, predict_rotate


def predict_conf_numpy(input_tensor, i, x, y):
    return numpy_sigmoid(input_tensor[6 * i, x, y])
