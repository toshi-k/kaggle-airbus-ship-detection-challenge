import os
import time
from functools import partial
import numpy as np
from PIL import Image
from multiprocessing import Pool

import torch
from torch.autograd import Variable
import torch.optim as optim
from torch import nn

from tensorboardX import SummaryWriter
writer = SummaryWriter()

from lib.load_img import SampleLoader
from lib.model import build_model, set_batch_norm_eval
from lib.default_box import dbox_params
from lib.search_best_box import coord2_img, search_boxes
from lib.calc_localization_loss import calc_localization_loss
from lib.visualize import draw_predicted_boxes


tic = time.time()

# load image

# target_name = '0a2e15e29.jpg'

target_names = [
    '000155de5.jpg',
    '0a1d87989.jpg',
    '0ace4520c.jpg',
    '0b1b3d75b.jpg',
    '1a7261884.jpg',
    '00003e153.jpg',
    '8edf5a0b0.jpg'
] # '7b2da4668.jpg',

os.makedirs('_sample', exist_ok=True)

for target_name in target_names:
    os.makedirs('_sample/{0:s}/'.format(target_name[:-4]), exist_ok=True)

loader = SampleLoader(
    dir_img='../../dataset/train',
    coord_path='../../input/coord_ica_05.csv',
    use_augmentation=False
)

list_input = list()
batch_targets = list()
for target_name in target_names:
    img_input, found_coord, original = loader(target_name)

    list_input.append(img_input)
    batch_targets.append(found_coord)

    Image.fromarray(original).save('_sample/{0:s}/img_original.png'.format(target_name[:-4]))

batch_input = torch.stack(list_input)

# build model

model = build_model()
model.train()
set_batch_norm_eval(model)

# load target_coord

# batch_targets = list()

for targets, target_name in zip(batch_targets, target_names):

    img_target = None
    for _, target in targets.iterrows():
        img_target = coord2_img(target.x, target.y, target.height, target.width, target.rotate, img_base=img_target)

    Image.fromarray(img_target).save('_sample/{0:s}/img_target.png'.format(target_name[:-4]))

# save default predicted

os.makedirs('default_predicted', exist_ok=True)

debug_pred = [np.zeros((24, 48, 48)), np.zeros((24, 24, 24)), np.zeros((24, 12, 12)), np.zeros((24, 6, 6))]

for l, temp in enumerate(debug_pred):

    debug_pred_l = list(debug_pred)

    for j in range(len(debug_pred)):

        debug_layer = np.copy(debug_pred[j])

        debug_layer[0:24:6] = -5

        if l == j:
            debug_layer[0, :, :] = 1

        debug_pred_l[j] = debug_layer

    default_predicted = draw_predicted_boxes(debug_pred_l, dbox_params, rate=0.8)
    Image.fromarray(default_predicted).save('default_predicted/layer_{}.png'.format(l))

# optimizer

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

ones = Variable(torch.ones(1).cuda())
zeros = Variable(torch.zeros(1).cuda())

batch_size = len(target_names)

for iter in range(101):

    print('-----')
    print('iter: {}'.format(iter))

    # forward

    optimizer.zero_grad()
    net_out = model.forward(Variable(batch_input).cuda())

    net_out_numpy = [tensor.cpu().data.numpy() for tensor in net_out]
    net_out_numpy_batch = [[tensor[b, :, :, :] for tensor in net_out_numpy] for b in range(batch_size)]

    search_boxes_dp = partial(search_boxes, dbox_params=dbox_params)

    def search_boxes_wrap(tp):
        return search_boxes_dp(*tp)

    inputs = list()

    c = 0

    for targets, net_out_batch, in zip(batch_targets, net_out_numpy_batch):
        for _, target in targets.iterrows():
            inputs.append((net_out_batch, target, c))
            c += 1

    with Pool(8) as p:
        boxes = list(p.imap_unordered(search_boxes_wrap, inputs))

    boxes = [(pos, neg) for pos, neg, _ in sorted(boxes, key=lambda x:x[2])]

    c = 0

    loss_cls = 0.0
    loss_loc = 0.0
    coef_loc = 0.5

    for b in range(batch_size):

        # calc classification loss

        loss_cls_b = 0.0

        num_target = len(batch_targets[b])

        poss_all = list()
        negs_all = list()

        for j in range(c, c+num_target):
            poss, negs = boxes[j]

            poss_all.extend(poss)
            negs_all.extend(negs)

        poss_unique = set(poss_all)

        negs = list(set(negs_all) - poss_unique)
        poss = list(poss_unique)

        print('poss')
        print(poss)

        print('negs')
        print(negs)

        # visualize positive boxes

        if (iter < 100 and iter % 10 == 0) or iter % 100 == 0:
            list_tensors = net_out_numpy_batch[b]
            # # list_tensors = [np.swapaxes(tensor, 1, 2) for tensor in list_tensors]
            img_predicted = draw_predicted_boxes(list_tensors, dbox_params, rate=1.0, list_tuple=poss)
            Image.fromarray(img_predicted).save('_sample/{0:s}/positive_boxes_{1:d}.png'.format(target_names[b][:-4], iter))

        for pos in poss:
            v = nn.BCEWithLogitsLoss()(net_out[pos[0]][b:b+1, 6*pos[3], pos[1], pos[2]], ones)
            loss_cls = loss_cls + v * 10
            loss_cls_b += float(v)

        for neg in negs:
            v = nn.BCEWithLogitsLoss()(net_out[neg[0]][b:b+1, 6*neg[3], neg[1], neg[2]], zeros)
            loss_cls = loss_cls + v
            loss_cls_b += float(v)

        # calc localization loss

        loss_loc_b = 0.0

        for i, j in enumerate(range(c, c+num_target)):
            poss, _ = boxes[j]
            target = batch_targets[b].iloc[i]
            for pos in poss:
                v = calc_localization_loss(pos, net_out, target, b=b)
                loss_loc = loss_loc + v
                loss_loc_b += float(v)

        c += num_target

        print('loss ({}): {:.3f} (loss_cls: {:.3f} loss_loc: {:.3f})'.format(
            target_names[b], float(loss_cls_b + loss_loc_b), float(loss_cls_b), float(loss_loc_b)))

    loss = loss_cls + loss_loc * coef_loc

    print('loss: {:.3f} (loss_cls: {:.3f} loss_loc: {:.3f})'.format(float(loss), float(loss_cls), float(loss_loc)))

    writer.add_scalar('loss/classification', float(loss_cls), iter)
    writer.add_scalar('loss/location', float(loss_loc), iter)

    # backward

    loss.backward()
    optimizer.step()

    # save predicted

    if (iter < 100 and iter % 10 == 0) or iter % 100 == 0:
        for b, target_name in enumerate(target_names):

            img_predicted = draw_predicted_boxes(net_out_numpy_batch[b], dbox_params, rate=1.0)
            Image.fromarray(img_predicted).save('_sample/{0:s}/predicted_boxes_{1:d}.png'.format(target_name[:-4], iter))

    # change learning rate

    if iter == 1000:
        coef_loc = 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
            print('change learning rate into: {:.6f}'.format(param_group['lr']))

    if iter == 100:
        coef_loc = 0.2
        print('change weight of location loss: {}'.format(coef_loc))


toc = time.time() - tic
print('Elapsed time: {:.1f}'.format(toc))

writer.close()
