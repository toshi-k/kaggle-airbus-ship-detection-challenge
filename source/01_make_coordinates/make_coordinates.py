import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from PIL import Image

from lib.img2_coord_ica import img2_coord_iter, coord2_img
from lib.log import Logger


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    """

    Args:
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return

    Returns:
        numpy array, 1 - mask, 0 - background

    """

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255

    return img.reshape(shape).T


def main_test():

    i = 5304  # 11, 15, 16, 5398

    image_id = segmentations.iloc[i, 0]

    truth_img = rle_decode(segmentations.iloc[i, 1])

    print(np.max(truth_img))

    coord = img2_coord_iter(truth_img / 255.0, threshold=0.05)
    reconst_img = coord2_img(*coord)

    sse = np.sum((reconst_img - truth_img) ** 2)
    print('sum of squared error: {}'.format(sse))

    os.makedirs('_result_sample', exist_ok=True)
    Image.fromarray(reconst_img).save(os.path.join('_result_sample', image_id[:-4] + '_reconstruct.png'), format='PNG')
    Image.fromarray(truth_img).save(os.path.join('_result_sample', image_id[:-4] + '_truth.png'), format='PNG')


def main():

    logger = Logger('coord_ica')

    list_mean_x = list()
    list_mean_y = list()
    list_height = list()
    list_aspect_ratio = list()
    list_rotate = list()

    num_error = 0
    num_zero_ship = 0

    os.makedirs('_error_imgs', exist_ok=True)

    sse_array = np.array([])

    for i, image_id in tqdm(enumerate(segmentations.ImageId), total=len(segmentations)):

        encoded = segmentations.iloc[i, 1]

        if encoded == '':

            list_mean_x.append(np.nan)
            list_mean_y.append(np.nan)
            list_height.append(np.nan)
            list_aspect_ratio.append(np.nan)
            list_rotate.append(np.nan)
            num_zero_ship += 1
            continue

        truth_img = rle_decode(encoded)

        reconst_img = np.zeros(truth_img.shape)  # initialize

        threshold_iter = 0.95
        threshold_last = 0.6

        truth_img_norm = truth_img / 255.0

        try:

            mean_x, mean_y, height, aspect_ratio, rotate, img_size = img2_coord_iter(truth_img_norm, threshold_iter)
            reconst_img = coord2_img(mean_x, mean_y, height, aspect_ratio, rotate, img_size)
            reconst_img_norm = reconst_img / 255.0

            sse = np.sum((reconst_img_norm - truth_img_norm) ** 2)
            sse_array = np.append(sse_array, sse)

            area_intersect = np.sum(truth_img_norm * reconst_img_norm)
            area_union = np.sum(truth_img_norm) + np.sum(reconst_img_norm) - area_intersect
            matching_degree = area_intersect / area_union

            if matching_degree < threshold_last:
                logger.info('[{}] sse: {} matching_degree: {}'.format(image_id, sse, matching_degree))
                raise RuntimeError

            list_mean_x.append(mean_x)
            list_mean_y.append(mean_y)
            list_height.append(height)
            list_aspect_ratio.append(aspect_ratio)
            list_rotate.append(rotate)

        except (RuntimeError, ValueError):

            num_error += 1

            list_mean_x.append(np.nan)
            list_mean_y.append(np.nan)
            list_height.append(np.nan)
            list_aspect_ratio.append(np.nan)
            list_rotate.append(np.nan)

        if matching_degree < threshold_last:

            try:
                Image.fromarray(reconst_img).save(
                    os.path.join('_error_imgs', image_id[:-4] + '_deg{:.3f}_re.png'.format(matching_degree)))
                Image.fromarray(truth_img).save(
                    os.path.join('_error_imgs', image_id[:-4] + '_deg{:.3f}_truth.png'.format(matching_degree)))
            except:
                pass

    logger.info('mean of reconstruct error: {:.3f}'.format(np.mean(sse_array)))

    logger.info('num zero ship: {0:d} / {1:d}'.format(num_zero_ship, len(segmentations)))
    logger.info('num_error: {0:d} / {1:d}'.format(num_error, len(segmentations)))

    result = pd.DataFrame()
    result['ImageID'] = segmentations.ImageId
    result['x'] = list_mean_y
    result['y'] = list_mean_x
    result['height'] = list_height
    result['width'] = [height / ratio for height, ratio in zip(list_height, list_aspect_ratio)]
    result['rotate'] = list_rotate

    result.to_csv('../../input/coordinates.csv', index=False, float_format='%.4f')


if __name__ == '__main__':

    segmentations = pd.read_csv('../../dataset/train_ship_segmentations_v2.csv')
    print(segmentations.head())
    segmentations = segmentations.fillna('')

    # main_test()
    main()
