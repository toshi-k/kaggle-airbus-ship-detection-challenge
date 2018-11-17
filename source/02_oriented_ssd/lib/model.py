import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


def set_batch_norm_eval(model):

    bn_count = 0
    bn_training = 0

    for module in model.modules():

        if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
            if module.training:
                bn_training += 1
            module.eval()
            bn_count += 1

            module.weight.requires_grad = False
            module.bias.requires_grad = False

    print('{} BN modules are set to eval'.format(bn_count))


class Model(nn.Module):

    def __init__(self):
        super().__init__()

        resnet34 = models.resnet34(pretrained=True)

        self.resnet34_main = nn.Sequential(
            resnet34.conv1,
            resnet34.bn1,
            resnet34.relu,
            resnet34.maxpool,
            resnet34.layer1,
            resnet34.layer2,
            resnet34.layer3
        )

        self.conv_ex1 = resnet34.layer4

        self.conv_ex2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
                                      nn.ReLU(inplace=True)
                                      )

        self.conv_ex3 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, padding=0, stride=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
                                      nn.ReLU(inplace=True)
                                      )

        self.ex0_intermediate = nn.Conv2d(256, 24, kernel_size=3, padding=1, stride=1)
        self.ex1_intermediate = nn.Conv2d(512, 24, kernel_size=3, padding=1, stride=1)
        self.ex2_intermediate = nn.Conv2d(512, 24, kernel_size=3, padding=1, stride=1)
        self.ex3_intermediate = nn.Conv2d(256, 24, kernel_size=3, padding=1, stride=1)

    def forward(self, x):

        list_output = list()

        main_out = self.resnet34_main.forward(x)
        list_output.append(self.ex0_intermediate(F.relu(main_out)))  # 48x48

        ex1_out = F.relu(self.conv_ex1(main_out))
        list_output.append(self.ex1_intermediate(ex1_out))  # 24x24

        ex2_out = self.conv_ex2(ex1_out)
        list_output.append(self.ex2_intermediate(ex2_out))  # 12x12

        ex3_out = self.conv_ex3(ex2_out)
        list_output.append(self.ex3_intermediate(ex3_out))  # 6x6

        return list_output


def build_model():

    model = Model()
    model.cuda()
    return model


def build_model_sample():

    model = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                          nn.LeakyReLU(),
                          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                          nn.LeakyReLU(),
                          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                          nn.LeakyReLU(),
                          nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1),
                          nn.Sigmoid()
                          )

    model.cuda()
    return model


if __name__ == '__main__':

    model = build_model()

    in_arr = np.zeros((3, 3, 768, 768), dtype=np.float32)
    in_tensor = torch.from_numpy(in_arr)
    in_var = Variable(in_tensor).cuda()

    out_vars = model.forward(in_var)

    print(model)
    [print(out_var.shape) for out_var in out_vars]
