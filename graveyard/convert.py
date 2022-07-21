# convert pytorch model to mindspore.
import os
import argparse

import torch
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint

parser = argparse.ArgumentParser(description="transform pytorch checkpoint to mindspore checkpoint")
parser.add_argument("--torch_file", type=str, required=True, help="input pytorch checkpoint filename")
parser.add_argument("--output_path", type=str, required=True, help="output mindspore checkpoint path")
args = parser.parse_args()

torch_param_dict = torch.load(args.torch_file, map_location=torch.device('cpu'))
ms_params = []

for name in torch_param_dict:

    param_dict = {}
    parameter = torch_param_dict[name]

    print(name)

    name = name.replace("model.module.backbone","resnet")
    name = name.replace("model.module.decoder","segdetector")
    name = name.replace("bn1.bias","bn1.beta")
    name = name.replace("bn2.bias","bn2.beta")
    name = name.replace("bn1.weight","bn1.gamma")
    name = name.replace("bn2.weight","bn2.gamma")
    name = name.replace(".running_mean",".moving_mean")
    name = name.replace(".running_var",".moving_variance")
    name = name.replace("out5.0.weight","out5.weight")
    name = name.replace("out4.0.weight","out4.weight")
    name = name.replace("out3.0.weight","out3.weight")
    name = name.replace("binarize.1.weight","binarize.1.gamma")
    name = name.replace("binarize.1.bias","binarize.1.beta")
    name = name.replace("binarize.4.weight","binarize.4.gamma")
    name = name.replace("binarize.4.bias","binarize.4.beta")

    # if name.endswith('bn1.bias'):
    #     name = name[:name.rfind('bn1.bias')]
    #     name = name + 'bn1.beta'

    # elif name.endswith('bn1.weight'):
    #     name = name[:name.rfind('bn1.weight')]
    #     name = name + 'bn1.gamma'

    # elif name.endswith('bn2.bias'):
    #     name = name[:name.rfind('bn2.bias')]
    #     name = name + 'bn2.beta'

    # elif name.endswith('bn2.weight'):
    #     name = name[:name.rfind('bn2.weight')]
    #     name = name + 'bn2.gamma'

    # elif name.endswith('.running_mean'):
    #     name = name[:name.rfind('.running_mean')]
    #     name = name + '.moving_mean'

    # elif name.endswith('.running_var'):
    #     name = name[:name.rfind('.running_var')]
    #     name = name + '.moving_variance'

    print(name)
    print(" ")

    param_dict['name'] = name
    param_dict['data'] = Tensor(parameter.detach().numpy())
    ms_params.append(param_dict)

save_checkpoint(ms_params, os.path.join(args.output_path, "pretrained_synthtext_resnet18.ckpt"))