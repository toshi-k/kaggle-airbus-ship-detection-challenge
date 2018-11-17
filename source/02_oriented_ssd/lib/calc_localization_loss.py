import numpy as np
from lib.default_box import dbox_params

import torch
from torch import nn
from torch.autograd import Variable

from lib.predict import predict_boxes


def calc_localization_loss(pos, net_out, target, b, img_size=768):

    step = img_size / net_out[pos[0]].shape[2]

    x_points = np.arange(step / 2 - 0.5, img_size, step)
    y_points = np.arange(step / 2 - 0.5, img_size, step)

    x_point = x_points[pos[1]]
    y_point = y_points[pos[2]]

    dbox_param = dbox_params.iloc[pos[3]]

    confidence, predict_x, predict_y, predict_height, predict_width, predict_rotate = predict_boxes(
        net_out[pos[0]], pos[3], pos[1], pos[2], x_point, y_point, dbox_param, step, b=b
    )

    loss_x = nn.SmoothL1Loss()(predict_x, Variable(torch.Tensor([target.x]).cuda()))
    loss_y = nn.SmoothL1Loss()(predict_y, Variable(torch.Tensor([target.y]).cuda()))

    loss_height = nn.SmoothL1Loss()(predict_height, Variable(torch.Tensor([target.height]).cuda()))
    loss_width = nn.SmoothL1Loss()(predict_width, Variable(torch.Tensor([target.width]).cuda()))
    loss_rotate = nn.SmoothL1Loss()(predict_rotate, Variable(torch.Tensor([target.rotate]).cuda()))

    loss = loss_x + loss_y + loss_width + loss_height + loss_rotate

    return loss
