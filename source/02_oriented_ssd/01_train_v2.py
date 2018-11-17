import os
import argparse
import random
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange
from multiprocessing import Pool

import torch
from torch.autograd import Variable
import torch.optim as optim
from torch import nn

from lib.load_img import SampleLoader
from lib.default_box import dbox_params
from lib.model import build_model, set_batch_norm_eval
from lib.search_best_box import search_boxes
from lib.visualize import draw_predicted_boxes, draw_mask_from_coords
from lib.calc_localization_loss import calc_localization_loss
from lib.rle import rle_decode_color_multi, make_encoded_pixels
from lib.non_maximum_suppression import non_maximum_suppression
from lib.augment import augment_input, aggregate_output

from tensorboardX import SummaryWriter
writer = SummaryWriter()

from lib.log import Logger
logger = Logger('_train')

random.seed(1048)
np.random.seed(1048)


def search_boxes_wrap(tp):
    return search_boxes(tp[0], tp[1], tp[2], dbox_params)


def define_negative_threshold(boxes, batch_size, batch_targets):

    c = 0
    negs_values = np.array([])

    for b in range(batch_size):

        num_target = len(batch_targets[b])

        poss_all = list()
        negs_all = dict()

        for j in range(c, c + num_target):
            poss, negs = boxes[j]

            poss_all.extend(poss)
            negs_all.update(negs)

        poss_unique = set(poss_all)

        negs = dict(negs_all)
        for pos in poss_unique:
            negs.pop(pos, None)

        negs_values = np.append(negs_values, list(negs.values()))

        c += num_target

    negative_threshold = np.sort(negs_values)[-1024]

    return negative_threshold


def train_main(model, optimizer, list_train_img_with_ship, list_train_img_no_ship, num_iter, epoch):

    model.train()
    set_batch_norm_eval(model)

    # load target coord

    loader = SampleLoader(
        dir_img='../../dataset/train_v2',
        coord_path='../../input/coordinates.csv',
        use_augmentation=epoch < 25
    )

    ones = Variable(torch.ones(1).cuda())
    zeros = Variable(torch.zeros(1).cuda())

    # batch size

    batch_size = 12

    loss_loc_ep = 0.0
    loss_cls_ep = 0.0

    sum_threshold = 0

    # start iteration

    for _ in trange(num_iter):

        target_names = random.sample(list_train_img_with_ship, batch_size - batch_size // 2) + \
                       random.sample(list_train_img_no_ship, batch_size // 2)

        # load input and target

        list_input = list()
        batch_targets = list()

        for target_name in target_names:

            img_input, found_coord, _ = loader(target_name)

            list_input.append(img_input)
            batch_targets.append(found_coord)

        batch_input = torch.stack(list_input)

        # forward

        optimizer.zero_grad()
        net_out = model.forward(Variable(batch_input).cuda())

        net_out_numpy = [tensor.cpu().data.numpy() for tensor in net_out]
        net_out_numpy_batch = [[tensor[b, :, :, :] for tensor in net_out_numpy] for b in range(batch_size)]

        inputs = list()

        c = 0

        for targets, net_out_batch, in zip(batch_targets, net_out_numpy_batch):
            for _, target in targets.iterrows():
                inputs.append((net_out_batch, target, c))
                c += 1

        with Pool(8) as p:
            boxes = list(p.imap_unordered(search_boxes_wrap, inputs))

        boxes = [(pos, neg) for pos, neg, _ in sorted(boxes, key=lambda x: x[2])]

        # define negative threshold

        negative_threshold = define_negative_threshold(boxes, batch_size, batch_targets)
        sum_threshold += negative_threshold

        # calc classification loss

        c = 0
        loss_cls = 0.0

        for b in range(batch_size):

            num_target = len(batch_targets[b])

            poss_all = list()
            negs_all = list()

            for j in range(c, c + num_target):
                poss, negs = boxes[j]

                poss_all.extend(poss)
                negs_all.extend([key for key, value in negs.items() if value > negative_threshold])

            poss_unique = set(poss_all)

            negs = list(set(negs_all) - poss_unique)
            poss = list(poss_unique)

            for pos in poss:
                v = nn.BCEWithLogitsLoss()(net_out[pos[0]][b:b + 1, 6 * pos[3], pos[1], pos[2]], ones)
                loss_cls = loss_cls + v * 10
                loss_cls_ep += float(v)

            for neg in negs:
                v = nn.BCEWithLogitsLoss()(net_out[neg[0]][b:b + 1, 6 * neg[3], neg[1], neg[2]], zeros)
                loss_cls = loss_cls + v
                loss_cls_ep += float(v)

            c += num_target

        # calc localization loss

        c = 0
        loss_loc = 0.0

        for b in range(batch_size):

            num_target = len(batch_targets[b])

            for i, j in enumerate(range(c, c + num_target)):
                poss, _ = boxes[j]
                target = batch_targets[b].iloc[i]
                for pos in poss:
                    v = calc_localization_loss(pos, net_out, target, b=b)
                    loss_loc = loss_loc + v
                    loss_loc_ep += float(v)

            c += num_target

        loss = loss_cls + loss_loc

        # backward

        loss.backward()
        optimizer.step()

    loss_cls_ep /= num_iter
    loss_loc_ep /= num_iter

    logger.info('average of threshold: {:.4f}'.format(sum_threshold / float(num_iter)))

    logger.info('loss: {:.3f} (loss_cls: {:.3f} loss_loc: {:.3f})'.format(
        float(loss_cls_ep + loss_loc_ep), float(loss_cls_ep), float(loss_loc_ep)))

    writer.add_scalar('loss/classification', float(loss_cls_ep), epoch)
    writer.add_scalar('loss/location', float(loss_loc_ep), epoch)

    return model


def validate(model, list_valid_img_with_ship, epoch):

    dir_save = './_valid/ep{}'.format(epoch)

    os.makedirs(dir_save, exist_ok=True)

    model.eval()

    loader = SampleLoader(
        dir_img='../../dataset/train_v2',
        coord_path='../../input/coordinates.csv',
        use_augmentation=epoch < 25
    )

    for target_name in list_valid_img_with_ship[:50]:

        img_input, found_coord, original = loader(target_name)

        Image.fromarray(original).save(os.path.join(dir_save, '{}_original.png'.format(target_name[:-4])))

        # input_tensor = torch.unsqueeze(img_input, 0)
        input_tensor = augment_input(img_input)

        net_out = model.forward(Variable(input_tensor).cuda())

        net_out_numpy = [tensor.cpu().data.numpy() for tensor in net_out]
        # net_out_numpy_batch = [tensor[0, :, :, :] for tensor in net_out_numpy]
        net_out_numpy_batch = aggregate_output(net_out_numpy)

        img_predicted = draw_predicted_boxes(net_out_numpy_batch, dbox_params, rate=1.0)
        Image.fromarray(img_predicted).save(os.path.join(dir_save, '{}_predited.png'.format(target_name[:-4])))

        mask = draw_mask_from_coords(found_coord)
        Image.fromarray(mask).save(os.path.join(dir_save, '{}_mask.png'.format(target_name[:-4])))

        _, list_predicted_img = non_maximum_suppression(net_out_numpy_batch,
                                                        dbox_params, 0.4, 0.7, 0.1)

        encoded_pixels = make_encoded_pixels(list_predicted_img)

        img_submit = (rle_decode_color_multi(encoded_pixels, (768, 768)) * 255).astype(np.uint8)
        Image.fromarray(img_submit).save(os.path.join(dir_save, '{}_submit_ossd.png'.format(target_name[:-4])))


def main():

    tic = time.time()

    # parse arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iter', '-n', type=int, default=1000,
                        help='number of iteration')
    parser.add_argument('--num_epoch', '-e', type=int, default=30,
                        help='number of epoch')

    params = parser.parse_args()

    num_iter = params.num_iter

    list_train_img_all = os.listdir('../../dataset/train_v2')
    random.shuffle(list_train_img_all)

    rate_valid = 0.1

    list_train_img = list_train_img_all[:-int(len(list_train_img_all) * rate_valid)]
    list_valid_img = list_train_img_all[-int(len(list_train_img_all) * rate_valid):]

    assert len(set(list_train_img) & set(list_valid_img)) == 0

    segmentations = pd.read_csv('../../dataset/train_ship_segmentations_v2.csv')
    segmentations = segmentations.fillna('')

    no_ship_imgs = segmentations.ImageId[segmentations.EncodedPixels == ''].tolist()

    list_train_img_with_ship = list(set(list_train_img) - set(no_ship_imgs))
    list_train_img_no_ship = list(set(list_train_img) & set(no_ship_imgs))

    logger.info('num train img: {} (with ship: {} no shop: {})'.format(
        len(list_train_img), len(list_train_img_with_ship), len(list_train_img_no_ship)))

    list_valid_img_with_ship = list(set(list_valid_img) - set(no_ship_imgs))
    list_valid_img_no_ship = list(set(list_valid_img) & set(no_ship_imgs))

    logger.info('num valid img: {} (with ship: {} no shop: {})'.format(
        len(list_valid_img), len(list_valid_img_with_ship), len(list_valid_img_no_ship)))

    # build model
    model = build_model()

    # optimizer

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    os.makedirs('./_models', exist_ok=True)

    # train for each epoch

    for ep in range(params.num_epoch):

        logger.info('')
        logger.info('==> start epoch {}'.format(ep))

        # train
        model = train_main(model, optimizer, list_train_img_with_ship, list_train_img_no_ship, num_iter, epoch=ep)

        # validate
        validate(model, list_valid_img_with_ship, epoch=ep)

        # change learning rate

        if ep == 20 or ep == 30 or ep == 31:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                print('change learning rate into: {:.6f}'.format(param_group['lr']))

        # save model
        torch.save(model, '_models/model_ep{}.pt'.format(ep))

    # save model
    torch.save(model, '_models/model.pt')

    # show elapsed time

    toc = time.time() - tic
    logger.info('Elapsed time: {:.1f} [min]'.format(toc / 60.0))


if __name__ == '__main__':
    main()
