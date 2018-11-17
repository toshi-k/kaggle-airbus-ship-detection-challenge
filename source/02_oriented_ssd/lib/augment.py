import torch
import numpy as np


def augment_input(img_input):
    img_input_numpy = img_input.numpy()
    img_input2 = torch.from_numpy(np.rot90(img_input_numpy, k=1, axes=(1, 2)).copy())
    img_input3 = torch.from_numpy(np.rot90(img_input_numpy, k=2, axes=(1, 2)).copy())
    img_input4 = torch.from_numpy(np.rot90(img_input_numpy, k=3, axes=(1, 2)).copy())
    img_input5 = torch.from_numpy(img_input_numpy[:, :, ::-1].copy())
    img_input6 = torch.from_numpy(np.rot90(img_input_numpy[:, :, ::-1], k=1, axes=(1, 2)).copy())
    img_input7 = torch.from_numpy(np.rot90(img_input_numpy[:, :, ::-1], k=2, axes=(1, 2)).copy())
    img_input8 = torch.from_numpy(np.rot90(img_input_numpy[:, :, ::-1], k=3, axes=(1, 2)).copy())

    input_tensor = torch.stack([img_input, img_input2, img_input3, img_input4,
                                img_input5, img_input6, img_input7, img_input8])

    return input_tensor


def aggregate_output(net_out_numpy):
    net_out_numpy_batch1 = [tensor[0, :, :, :] for tensor in net_out_numpy]

    def flip_box_02(t):
        t = np.rot90(t, -1, (1, 2))  # rotate grid
        temp = t[1:24:6].copy()
        t[1:24:6] = t[2:24:6]
        t[2:24:6] = -temp
        # rotate
        temp1 = t[0:6]
        temp2 = t[6:12]
        temp3 = t[12:18]
        temp4 = t[18:24]
        return np.concatenate([temp3, temp4, temp1, temp2], 0)

    def flip_box_03(t):
        t = np.rot90(t, -2, (1, 2))  # rotate grid
        t[1:24:6] = -t[1:24:6]
        t[2:24:6] = -t[2:24:6]
        return t

    def flip_box_04(t):
        t = np.rot90(t, -3, (1, 2))  # rotate grid
        temp = t[1:24:6].copy()
        t[1:24:6] = -t[2:24:6]
        t[2:24:6] = temp
        # rotate
        temp1 = t[0:6]
        temp2 = t[6:12]
        temp3 = t[12:18]
        temp4 = t[18:24]
        return np.concatenate([temp3, temp4, temp1, temp2], 0)

    def flip_box_05(t):
        t = t[:, :, ::-1]  # flip grid
        t[2:24:6] = -t[2:24:6]  # x-shift
        # rotate
        t[5:24:6] = -t[5:24:6]
        temp1 = t[0:6]
        temp2 = t[6:12]
        temp3 = t[12:18]
        temp4 = t[18:24]
        return np.concatenate([temp4, temp3, temp2, temp1], 0)

    net_out_numpy_batch2 = [flip_box_02(tensor[1, :, :, :]) for tensor in net_out_numpy]
    net_out_numpy_batch3 = [flip_box_03(tensor[2, :, :, :]) for tensor in net_out_numpy]
    net_out_numpy_batch4 = [flip_box_04(tensor[3, :, :, :]) for tensor in net_out_numpy]
    net_out_numpy_batch5 = [flip_box_05(tensor[4, :, :, :]) for tensor in net_out_numpy]
    net_out_numpy_batch6 = [flip_box_05(flip_box_02(tensor[5, :, :, :])) for tensor in net_out_numpy]
    net_out_numpy_batch7 = [flip_box_05(flip_box_03(tensor[6, :, :, :])) for tensor in net_out_numpy]
    net_out_numpy_batch8 = [flip_box_05(flip_box_04(tensor[7, :, :, :])) for tensor in net_out_numpy]

    net_out_numpy_batch_ave = [sum(t_all) / 8.0 for t_all in
                               zip(net_out_numpy_batch1, net_out_numpy_batch2,
                                   net_out_numpy_batch3, net_out_numpy_batch4,
                                   net_out_numpy_batch5, net_out_numpy_batch6,
                                   net_out_numpy_batch7, net_out_numpy_batch8)]

    return net_out_numpy_batch_ave
