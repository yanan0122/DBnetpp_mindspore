# test file for pytorch/mindspore APIs.
import sys

import mindspore
import mindspore as ms
from mindspore import ops,Tensor
from mindspore import context
from mindspore.common.initializer import initializer, Normal
from mindspore import load_checkpoint, load_param_into_net

import numpy as np
from numpy import random
import math
from collections import OrderedDict

import torch
from torch import nn


def ConvTranspose2d():

    inp = np.ones((1, 256//4, 16, 50))

    convtranspose2d_torch = nn.ConvTranspose2d(256//4, 256//4, 2, stride=2)

    output = convtranspose2d_torch(torch.from_numpy(inp))

    print(inp)


def test_upsample():

    random.seed(0)
    inp = random.rand(1,512,23,40)

    inp_torch = torch.from_numpy(inp)
    inp_ms = Tensor(inp,dtype=ms.float32)

    up_torch = nn.Upsample(scale_factor=2, mode='nearest')
    up_ms = ops.ResizeNearestNeighbor((46, 80))

    out_torch = up_torch(inp_torch).numpy()
    out_ms = up_ms(inp_ms).asnumpy()

    print(np.abs(out_ms-out_torch))

    # print(out_torch.shape)
    # print(out_ms.shape)

def test_bn():

    # random.seed(8)
    # inp = random.rand(1,64,184,320)

    inp = np.load("/opt/nvme1n1/wz/dbnet_torch/bninput.npy")

    inp_torch = torch.from_numpy(inp).type(torch.float32)
    inp_ms = Tensor(inp, dtype=ms.float32)

    bn_torch = torch.nn.BatchNorm2d(64)
    bn_ms = ms.nn.BatchNorm2d(64, momentum=0.1, use_batch_statistics=True)

    out_torch = bn_torch(inp_torch).detach().numpy()
    out_ms = bn_ms(inp_ms).asnumpy()

    # print(out_torch.shape, out_torch[0][10][10][:5])
    # print(out_ms.shape, out_ms[0][10][10][:5])

    print(np.abs(out_ms-out_torch))

    # print(out_torch.shape)
    # print(out_ms.shape)

def test_conv():

    # random.seed(3)
    # inp = random.rand(1,256,23,40)

    inp = np.ones((1,256,23,40))*512
    # print(inp[0][0][0][:5])

    inp_torch = torch.from_numpy(inp).type(torch.float32)
    inp_ms = Tensor(inp, dtype=ms.float32)

    conv_torch = torch.nn.Conv2d(256, 256//4, 3, padding=1, bias=False)
    conv_ms = ms.nn.Conv2d(256, 256//4, 3, pad_mode="pad", padding=1, has_bias=False)
    # print(conv_ms.weight.asnumpy().shape)

    nn.init.constant_(conv_torch.weight.data,1)
    # print(conv_torch.weight, conv_torch.bias)

    out_torch = conv_torch(inp_torch).detach().numpy()
    out_ms = conv_ms(inp_ms).asnumpy()

    # print(out_torch.shape, out_torch[0][10][10][:5])
    # print(out_ms.shape, out_ms[0][10][10][:5])

    print(np.abs(out_ms-out_torch))

# convert ckpt to mindspore format.
def convert_resnet18():

        par_dict = torch.load('/opt/nvme1n1/wz/dbnet_torch/path-to-model-directory/resnet18-5c106cde.pth')

        new_params_list = []

        for name in par_dict:
            param_dict = {}
            parameter = par_dict[name]

            print('========================py_name',name)

            if name.endswith('bn1.bias'):
                name = name[:name.rfind('bn1.bias')]
                name = name + 'bn1.beta'

            elif name.endswith('bn1.weight'):
                name = name[:name.rfind('bn1.weight')]
                name = name + 'bn1.gamma'

            elif name.endswith('bn2.bias'):
                name = name[:name.rfind('bn2.bias')]
                name = name + 'bn2.beta'

            elif name.endswith('bn2.weight'):
                name = name[:name.rfind('bn2.weight')]
                name = name + 'bn2.gamma'

            elif name.endswith('.running_mean'):
                name = name[:name.rfind('.running_mean')]
                name = name + '.moving_mean'

            elif name.endswith('.running_var'):
                name = name[:name.rfind('.running_var')]
                name = name + '.moving_variance'

            print('========================ms_name',name)

            param_dict['name'] = name
            param_dict['data'] = Tensor(parameter.detach().numpy())
            new_params_list.append(param_dict)

        save_checkpoint(new_params_list,  '/opt/nvme1n1/wz/dbnet_torch/path-to-model-directory/res18_ms.ckpt')

def test_conv_ms():

    # scale = 0.01

    data = Tensor(np.ones((1,24,64,64)),dtype=ms.float32)

    conv = ms.nn.Conv2d(24, 48, 4, has_bias=False, weight_init="ones")

    return conv(data)

def test_conv_torch():

    data = torch.ones(1,256,23,40)

    conv = nn.Conv2d(256, 256//4, 3, padding=1, bias=False)

    nn.init.constant_(conv.weight.data,1)

    return conv(data)

if __name__ == "__main__":

    context.set_context(mode=context.PYNATIVE_MODE)
    context.set_context(device_id=3)
    # test_upsample()
    # test_conv()
    # test_bn()
    # ConvTranspose2d()
    test_conv_ms()

# ones = ops.Ones()
# data = ones((1,3,736,1280),mindspore.float32)

# conv2d = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode="pad",
#                                has_bias=False)

# output = conv2d(data)

# print(output.shape)

# loss = nn.BCELoss()
# logits = Tensor(np.array([[0.1, 0.2, 0.3], [0.5, 0.7, 0.9]]), mindspore.float32)
# labels = Tensor(np.array([[0, 1, 0], [0, 0, 1]]), mindspore.float32)
# sig = mindspore.nn.Sigmoid()
# output = loss(sig(logits), labels)
# print(output)

# import torch
# from torch import nn

# m = nn.Sigmoid()
# loss = nn.BCELoss(reduction='none')
# input = torch.tensor(np.array([[0.1, 0.2, 0.3], [0.5, 0.7, 0.9]]),dtype=torch.float64)
# target = torch.tensor(np.array([[0, 1, 0], [0, 0, 1]]),dtype=torch.float64)
# output = loss(m(input), target)
# print(output)