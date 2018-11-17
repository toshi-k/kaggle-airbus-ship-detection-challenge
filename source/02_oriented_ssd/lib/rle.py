import random
import colorsys
import numpy as np

"""
This script is forked from Paulo's kaggle-kernel !
Run-Length Encode and Decode paulorzp
https://www.kaggle.com/paulorzp/run-length-encode-and-decode
"""


def rle_encode(img):
    """
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        https://www.kaggle.com/paulorzp/run-length-encode-and-decode#293790
    """

    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background

    """

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def rle_decode_color(mask_rle, shape, img=None):

    if img is None:
        img = np.zeros((shape[0]*shape[1], 3))

    if mask_rle == '':
        return img.reshape(shape + (3,))

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = img.reshape(-1, 3)

    r, g, b = colorsys.hsv_to_rgb(random.random(), 1.0, 0.8)

    for lo, hi in zip(starts, ends):

        img[lo:hi, 0] += r
        img[lo:hi, 1] += g
        img[lo:hi, 2] += b

    return img.reshape(shape + (3,))


def rle_decode_color_multi(mask_rles, shape):

    img = np.zeros((shape[0] * shape[1], 3))

    for mask_rle in mask_rles:

        img = rle_decode_color(mask_rle, shape, img)

    return np.swapaxes(img, 0, 1)


def make_encoded_pixels(list_predicted_img):

    if len(list_predicted_img) == 0:
        return ['']

    result = list()

    already_used = np.zeros((768, 768), dtype=np.uint8)

    for img in list_predicted_img:

        img[already_used > 0] = 0

        box_pixels = rle_encode(img/255.0)
        result.append(box_pixels)

        already_used = already_used | img

    result = [box_pixels for box_pixels in result if box_pixels != '']

    if len(result) == 0:
        return ['']

    return result
