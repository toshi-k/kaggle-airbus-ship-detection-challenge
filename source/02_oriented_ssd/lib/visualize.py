import random
import numpy as np
from functools import partial
from skimage.draw import polygon, polygon_perimeter
from lib.predict import predict_boxes_numpy


def coord_draw(func, mean_x, mean_y, length, width, rotate, img_size=768, img_base=None, weight=1.0):

    if img_base is None:
        img = np.zeros((img_size, img_size), dtype=np.uint8)
    else:
        img = img_base

    if np.isnan(mean_x) and np.isnan(mean_y):
        return img

    mean_x = float(mean_x)
    mean_y = float(mean_y)
    height = float(length)
    width = float(width)
    rotate = float(rotate)

    W2 = np.array([[np.cos(rotate), np.sin(rotate)], [-np.sin(rotate), np.cos(rotate)]])

    c = np.array([[-height/2, -width/2], [height/2, -width/2], [height/2, width/2], [-height/2, width/2]])
    c = (W2 @ c.T).T + np.array([mean_x, mean_y])

    try:
        rr, cc = func(c[:, 0], c[:, 1], shape=(img_size, img_size))
        img[rr, cc] = np.maximum(img[rr, cc].flatten(), int(255 * weight))
    except:
        raise RuntimeError('error in drawing')

    return img


def coord2_img(mean_x, mean_y, length, width, rotate, img_size=768, img_base=None):

    return coord_draw(polygon, mean_x, mean_y, length, width, rotate, img_size=img_size, img_base=img_base)


def coord2_boarder(mean_x, mean_y, length, width, rotate, img_size, img_base=None, weight=1.0):

    polygon_perimeter_clip = partial(polygon_perimeter, clip=True)

    return coord_draw(polygon_perimeter_clip, mean_x, mean_y, length, width, rotate,
                      img_size=img_size, img_base=img_base, weight=weight)


def draw_predicted_boxes(predicted_tensors, dbox_params, img_size=768, rate=1.0, list_tuple=None):

    img_boxes = np.zeros((img_size, img_size), dtype=np.uint8)

    num_error = 0

    for l, input_tensor in enumerate(predicted_tensors):

        step = img_size / input_tensor.shape[2]

        x_points = np.arange(step / 2 - 0.5, img_size, step)
        y_points = np.arange(step / 2 - 0.5, img_size, step)

        for x, x_point in enumerate(x_points):
            for y, y_point in enumerate(y_points):
                for i in range(len(dbox_params)):

                    if random.random() > rate:
                        continue

                    dbox_param = dbox_params.iloc[i]

                    if (list_tuple is not None) and ((l, x, y, i) not in list_tuple):
                        continue

                    confidence, predict_x, predict_y, predict_height, predict_width, predict_rotate = predict_boxes_numpy(
                        input_tensor, i, x, y, x_point, y_point, dbox_param, step
                    )

                    if float(confidence) > 0.01:

                        try:
                            img_boxes = coord2_boarder(predict_x, predict_y, predict_height, predict_width,
                                                       predict_rotate, img_size, img_base=img_boxes, weight=confidence)
                        except:
                            num_error += 1

    if num_error > 0:
        print('error in drawing (x{})'.format(num_error))

    return img_boxes


def draw_mask_from_coords(coords, img_size=768):

    img_boxes = np.zeros((img_size, img_size), dtype=np.uint8)

    for i in range(len(coords)):

        coord = coords.iloc[i]
        img_boxes = coord2_img(
            coord.x,
            coord.y,
            coord.height,
            coord.width,
            coord.rotate,
            img_size=img_size,
            img_base=img_boxes
        )

    return img_boxes
