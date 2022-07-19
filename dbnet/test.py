import os
import sys

import mindspore
import mindspore as ms
from mindspore import ops,Tensor
from mindspore import context,set_seed
import mindspore.nn as nn
from mindspore.common.initializer import initializer, Normal, Constant
from mindspore import load_checkpoint, load_param_into_net

import numpy as np
import math
from collections import OrderedDict

import DBnetpp_mindspore.dbnet.modules.backbone as backbone
import DBnetpp_mindspore.dbnet.modules.detector as detector
import DBnetpp_mindspore.dbnet.modules.loss as loss

def model(input_data):

	inp_tensor = Tensor(input_data,dtype=ms.float32)

	resnet = backbone.resnet18()
	segdetector = detector.SegDetector()
	pred = segdetector(resnet(inp_tensor))

	print(pred)

	BCEloss = loss.L1BalanceCELoss()

	loss,metrics = BCEloss(pred)

if __name__ == "__main__":

	print("test.py")
	context.set_context(device_id=7)

	input_data = np.load("test.npy")
