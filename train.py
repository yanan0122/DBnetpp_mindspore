import sys
import os

import yaml
import argparse
import numpy as np

import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import LearningRateScheduler, CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore.train.model import Model
from mindspore import context, Tensor

from dataloader.load import DataLoader
import modules.backbone as backbone
import modules.detector as detector
import modules.loss as loss
from modules.model import DBnet, WithLossCell, LossCallBack


def learning_rate_function(lr, cur_epoch_num):
    lr = 0.007
    epochs = 1200
    factor = 0.9

    rate = np.power(1.0 - cur_epoch_num / float(epochs + 1), factor)

    return rate * lr


def train():
    # Config
    stream = open('/home/group1/wjf_dbnet/dbnet_ms/dbnet/config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()

    # Dataset
    data_loader = DataLoader(config, func="train")
    train_dataset = ds.GeneratorDataset(data_loader, ['img', 'gts', 'gt_masks', 'thresh_maps', 'thresh_masks'])
    train_dataset = train_dataset.batch(config['train']['batch_size'])  # default batch size 16. dataset size 63

    # Network
    network = DBnet()
    pretrained_weights = load_checkpoint(config['train']['resume'])
    load_param_into_net(network, pretrained_weights)

    # Model: Loss & Optimizer
    opt = nn.SGD(params=network.trainable_params(), learning_rate=0.007, momentum=0.9, weight_decay=5e-4)
    criterion = loss.L1BalanceCELoss()
    network_with_loss = WithLossCell(network, criterion)
    model = Model(network_with_loss, optimizer=opt)

    # Train & Save
    config_ck = CheckpointConfig(save_checkpoint_steps=4, keep_checkpoint_max=10)
    ckpoint = ModelCheckpoint(prefix="DBNetpp", directory="./checkpoint/1/", config=config_ck)
    model.train(config['train']['n_epoch'], train_dataset, dataset_sink_mode=False,
                callbacks=[LossMonitor(), LearningRateScheduler(learning_rate_function), ckpoint])


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=[4, 5, 6, 7])
    train()
    print("Train has completed.")


# def feed(train_dataset):
# 	# debug function
# 	for item in train_dataset.create_dict_iterator(output_numpy=True):

# 		# print(item)
# 		img, gts, gt_masks, thresh_maps, thresh_masks = item['img'], item['gts'], item['gt_masks'], item['thresh_maps'], item['thresh_masks']

# 		np.save("/opt/nvme1n1/wz/dbnet_torch/gts.npy", gts)
# 		np.save("/opt/nvme1n1/wz/dbnet_torch/gt_masks.npy", gt_masks)
# 		np.save("/opt/nvme1n1/wz/dbnet_torch/thresh_maps.npy", thresh_maps)
# 		np.save("/opt/nvme1n1/wz/dbnet_torch/thresh_masks.npy", thresh_masks)

# img = Tensor(img, dtype=ms.float32)
# gts = Tensor(gts, dtype=ms.float32)
# gt_masks = Tensor(gt_masks, dtype=ms.float32)
# thresh_masks = Tensor(thresh_masks, dtype=ms.float32)
# thresh_maps = Tensor(thresh_maps, dtype=ms.float32)

# output_data = network_with_loss(img, gts, gt_masks, thresh_maps, thresh_masks)

# print(output_data)
