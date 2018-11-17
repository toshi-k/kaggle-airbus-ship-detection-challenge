import os
import warnings
import random
import math
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as transforms


class SampleLoader:

    def __init__(self, dir_img, coord_path=None, use_augmentation=False):

        self.dir_img = dir_img
        self.use_augmentation = use_augmentation

        if coord_path is not None:
            self.have_coord = True
            self.coord = pd.read_csv(coord_path)
        else:
            self.have_coord = False

        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, target_name):

        try:
            img = Image.open(os.path.join(self.dir_img, target_name))
        except OSError:
            warnings.warn('catch the error during loading {}'.format(target_name))
            img = Image.fromarray(np.zeros((768, 768, 3), dtype=np.uint8))

        if self.have_coord:
            found_coord = self.coord[self.coord.ImageID == target_name]
        else:
            found_coord = None

        if self.use_augmentation:

            try:

                cx = 768/2
                cy = 768/2
                theta = random.random() * 2 * math.pi

                affine_param = (math.cos(theta), -math.sin(theta), cx - cx * math.cos(theta) + cy * math.sin(theta),
                                math.sin(theta), math.cos(theta), cy - cx * math.sin(theta) - cy * math.cos(theta))

                img = img.transform(img.size, Image.AFFINE, affine_param, Image.BILINEAR)

                left = np.array([[math.cos(theta), -math.sin(theta), cx - cx * math.cos(theta) + cy * math.sin(theta)],
                                 [math.sin(theta), math.cos(theta), cy - cx * math.sin(theta) - cy * math.cos(theta)],
                                 [0, 0, 1]])
                right = np.vstack([np.stack([found_coord.x, found_coord.y]), np.ones(len(found_coord))])

                found_coord = pd.DataFrame(found_coord)

                lr = left @ right
                found_coord.x = lr[0, :]
                found_coord.y = lr[1, :]

                found_coord.loc[:, 'rotate'] -= theta
                found_coord.loc[found_coord.rotate < -math.pi / 2, 'rotate'] += math.pi
                found_coord.loc[found_coord.rotate < -math.pi / 2, 'rotate'] += math.pi

            except:

                warnings.warn('catch the error during augmentation {}'.format(target_name))

                img = Image.fromarray(np.zeros((768, 768, 3), dtype=np.uint8))
                found_coord = found_coord.iloc[[0], :]
                found_coord.x = np.nan
                found_coord.y = np.nan
                found_coord.height = np.nan
                found_coord.ratio = np.nan
                found_coord.rotate = np.nan

        # -----

        img = np.array(img)

        try:
            img_input = self.normalizer(transforms.ToTensor()(img.astype(np.float)))
        except TypeError:
            warnings.warn('catch the error during normalization {}'.format(target_name))
            img_input = torch.zeros((3, 768, 768))

        if img.dtype != np.uint8:
            warnings.warn('catch the error for data type {}'.format(target_name))
            img = np.zeros((769, 768), dtype=np.uint8)

        return img_input, found_coord, img
