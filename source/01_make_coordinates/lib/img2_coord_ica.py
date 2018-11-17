import os
import math
import numpy as np
from PIL import Image
from skimage.draw import polygon, polygon_perimeter
from sklearn.decomposition import FastICA


def img2_coord(img, init=None):

    assert np.max(img) <= 1.0

    if init is None:
        init = np.zeros((img.shape[0] + 200, img.shape[1] + 200))

    init[100:-100, 100:-100] = img

    img = init

    img_size = img.shape[0]

    tile_x = np.tile(np.arange(img_size), (img_size, 1))
    tile_y = tile_x.T

    mean_x = np.sum(img * tile_x) / np.sum(img)
    mean_y = np.sum(img * tile_y) / np.sum(img)

    dist_mean_x = np.abs(mean_x - tile_x) * img
    dist_mean_y = np.abs(mean_y - tile_y) * img

    hypo = np.max(((dist_mean_x * dist_mean_x) + (dist_mean_y * dist_mean_y)))

    diff_mean_x = tile_x[img > 0].flatten() - mean_x
    diff_mean_y = tile_y[img > 0].flatten() - mean_y

    m = np.stack([diff_mean_x, diff_mean_y])

    decomposer = FastICA(2)
    decomposer.fit(m.T)
    Uica = decomposer.mixing_

    # print('ICA vectors')
    norms = np.sqrt((Uica ** 2).sum(axis=0))
    Uica = Uica / np.sqrt((Uica ** 2).sum(axis=0))
    if norms[0] > norms[1]:
        rotate = -np.arctan2(Uica[0, 0], Uica[1, 0])
    else:
        rotate = -np.arctan2(Uica[0, 1], Uica[1, 1])

    # represent between [-math.pi, math.pi]
    if rotate < -math.pi / 2:
        rotate += math.pi
    elif rotate > math.pi / 2:
        rotate -= math.pi

    # print('rotate: {:.2f} [deg]'.format(rotate * 360 / 2 / 3.14))

    aspect_ratio = max(norms) / min(norms)
    # print('aspect ratio: {:.2f}'.format(aspect_ratio))

    width = np.sqrt(hypo / (1 + aspect_ratio**2)) * 2 + 0.25
    # height = np.sqrt(hypo * aspect_ratio**2 / (1 + aspect_ratio**2)) * 2
    height = width * aspect_ratio

    # print('width: {} height: {}'.format(width, height))

    return mean_x, mean_y, height, aspect_ratio, rotate, img_size


def coord2_img(mean_x, mean_y, length, aspect_ratio, rotate, img_size):

    W2 = np.array([[np.cos(rotate), np.sin(rotate)], [-np.sin(rotate), np.cos(rotate)]])

    height = length
    width = height / aspect_ratio

    c = np.array([[-height/2, -width/2], [height/2, -width/2], [height/2, width/2], [-height/2, width/2]])
    c = (W2 @ c.T).T + np.array([mean_y, mean_x])

    img = np.zeros((img_size, img_size), dtype=np.uint8)
    rr, cc = polygon(c[:, 0], c[:, 1])

    index = (rr >= 0) * (rr < img_size) * (cc >= 0) * (cc < img_size)

    img[rr[index], cc[index]] = 255
    return img


def coord2_boarder(mean_x, mean_y, length, aspect_ratio, rotate, img_size):

    W2 = np.array([[np.cos(rotate), np.sin(rotate)], [-np.sin(rotate), np.cos(rotate)]])

    height = length
    width = height / aspect_ratio

    c = np.array([[-height/2, -width/2], [height/2, -width/2], [height/2, width/2], [-height/2, width/2]])
    c = (W2 @ c.T).T + np.array([mean_y, mean_x])

    img = np.zeros((img_size, img_size), dtype=np.uint8)
    rr, cc = polygon_perimeter(c[:, 0], c[:, 1])

    index = (rr >= 0) * (rr < img_size) * (cc >= 0) * (cc < img_size)

    img[rr[index], cc[index]] = 255
    return img


def img2_coord_iter(img, threshold):

    img_re = np.zeros((img.shape[0] + 200, img.shape[1] + 200))

    for _ in range(10):

        mean_x, mean_y, length, aspect_ratio, rotate, img_size_re = img2_coord(img, init=img_re / 255.0)
        img_re = coord2_img(mean_x, mean_y, length, aspect_ratio, rotate, img_size_re)

        truth_img_norm = img
        reconst_img_norm = img_re[100:-100, 100:-100] / 255.0

        area_intersect = np.sum(truth_img_norm * reconst_img_norm)
        area_union = np.sum(truth_img_norm) + np.sum(reconst_img_norm) - area_intersect
        matching_degree = area_intersect / area_union

        # print('matching_degree: {}'.format(matching_degree))

        if matching_degree > threshold:
            break

    mean_x -= 100
    mean_y -= 100

    return mean_x, mean_y, length, aspect_ratio, rotate, img.shape[0]


def main():

    file_name = 'img300_rec20-60_deg45.png'
    path = os.path.join('sample_data', file_name)

    img = np.array(Image.open(path))[:, :, 0] / 255.0

    mean_x, mean_y, length, aspect_ratio, rotate, img_size_re = img2_coord_iter(img, threshold=0.05)

    img_re = coord2_img(mean_x, mean_y, length, aspect_ratio, rotate, img.shape[0])
    img_boarder = coord2_boarder(mean_x, mean_y, length, aspect_ratio, rotate, img.shape[0])

    os.makedirs('_result_sample', exist_ok=True)
    Image.fromarray(img_re).save(os.path.join('_result_sample', file_name), format='PNG')
    Image.fromarray(img_boarder).save(os.path.join('_result_sample', file_name[:-4] + '_boarder.png'), format='PNG')


if __name__ == '__main__':
    main()
