import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.autograd import Variable

from lib.load_img import SampleLoader
from lib.default_box import dbox_params
from lib.rle import rle_decode_color_multi, make_encoded_pixels
from lib.visualize import draw_predicted_boxes
from lib.non_maximum_suppression import non_maximum_suppression
from lib.augment import augment_input, aggregate_output


def predict_main(model, threshold, threshold2, overlap):

    list_sample_submission = pd.read_csv('../../dataset/sample_submission_v2.csv')
    list_test_img = list_sample_submission.ImageId.tolist()

    # list_test_img = list_test_img[:100]  # for debug

    dir_save = './_test_v2'
    os.makedirs(dir_save, exist_ok=True)

    model.eval()

    list_ImageId = list()
    list_encodes_pixels = list()

    loader = SampleLoader(
        dir_img='../../dataset/test_v2',
        coord_path='../../input/coordinates.csv'
    )

    for i, target_name in tqdm(enumerate(list_test_img), total=len(list_test_img)):

        img_input, _, original = loader(target_name)
        input_tensor = augment_input(img_input)

        net_out = model.forward(Variable(input_tensor).cuda())

        net_out_numpy = [tensor.cpu().data.numpy() for tensor in net_out]
        net_out_numpy_batch_ave = aggregate_output(net_out_numpy)

        list_tuple, list_predicted_img = non_maximum_suppression(net_out_numpy_batch_ave,
                                                                 dbox_params, threshold, threshold2, overlap)

        encoded_pixels = make_encoded_pixels(list_predicted_img)

        list_ImageId.extend([target_name] * len(encoded_pixels))
        list_encodes_pixels.extend(encoded_pixels)

        if 300 < i < 400:

            img_predicted = draw_predicted_boxes(net_out_numpy_batch_ave, dbox_params, rate=1.0)
            Image.fromarray(img_predicted).save(os.path.join(dir_save, '{}_predited_ave.png'.format(target_name[:-4])))

            Image.fromarray(original).save(os.path.join(dir_save, '{}_original.png'.format(target_name[:-4])))

            img_submit = (rle_decode_color_multi(encoded_pixels, (768, 768)) * 255).astype(np.uint8)
            Image.fromarray(img_submit).save(os.path.join(dir_save, '{}_submit_ossd.png'.format(target_name[:-4])))

            original_submit = 0.5 * original + 0.5 * img_submit
            Image.fromarray(original_submit.astype(np.uint8)).save(
                os.path.join(dir_save, '{}_original_submit.png'.format(target_name[:-4])))

    submission = pd.DataFrame()
    submission['ImageId'] = list_ImageId
    submission['EncodedPixels'] = list_encodes_pixels

    return submission


def main():

    # parse arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='model.pt',
                        help='name of model')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--threshold2', '-g', type=float, default=0.8,
                        help='global threshold')
    parser.add_argument('--overlap', '-o', type=float, default=0.1,
                        help='maximum overlap degree')
    params = parser.parse_args()

    # load model

    model = torch.load(os.path.join('_models', params.model))

    # predict

    submission = predict_main(model, threshold=params.threshold, threshold2=params.threshold2, overlap=params.overlap)

    # save submission

    os.makedirs('_submission', exist_ok=True)
    submission.to_csv('_submission/submit_th{}_gth{}_ov{}.csv'.format(
        params.threshold, params.threshold2, params.overlap), index=False)


if __name__ == '__main__':
    main()
