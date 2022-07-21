import time
import numpy as np
import yaml
import tqdm

from mindspore.train.callback import Callback
import mindspore.nn as nn
import mindspore.dataset as ds

import modules.backbone as backbone
import modules.detector as detector
import modules.loss as loss
from utils.SegDetectorRepresenter import SegDetectorRepresenter
from utils.quad_measurer import QuadMeasurer
from dataloader.load import DataLoader


class DBnet(nn.Cell):

    def __init__(self):
        super(DBnet, self).__init__()

        self.resnet = backbone.deformable_resnet18()
        self.segdetector = detector.SegDetector()

    def construct(self, img):
        pred = self.resnet(img)
        pred = self.segdetector(pred)

        return pred


class DBnetPP(nn.Cell):
    def __init__(self):
        super(DBnetPP, self).__init__(auto_prefix=False)

        self.resnet = backbone.resnet18()
        self.segdetector = detector.SegDetectorPP()

    def construct(self, img):
        pred = self.resnet(img)
        pred = self.segdetector(pred)

        return pred


class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to compute loss.

    Args:
        backbone (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
    """

    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)

        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, img, gt, gt_mask, thresh_map, thresh_mask):
        pred = self._backbone(img)

        loss = self._loss_fn(pred, gt, gt_mask, thresh_map, thresh_mask)

        return loss

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, return backbone network.
        """
        return self._backbone


class LossCallBack(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF terminating training.

    Note:
        If per_print_times is 0 do not print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1):

        super(LossCallBack, self).__init__()

        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")

        self._per_print_times = per_print_times
        self.loss_avg = AverageMeter()

    def step_end(self, run_context):

        cb_params = run_context.original_args()

        if cb_params.net_outputs is not None:
            loss = cb_params.net_outputs.asnumpy()
        else:
            print("custom loss callback class loss is None.")
            return

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        cur_num = cb_params.cur_step_num

        if cur_step_in_epoch == 1:
            self.loss_avg = AverageMeter()

        self.loss_avg.update(loss)

        if self._per_print_times != 0 and cur_num % self._per_print_times == 0:
            loss_log = "time: %s, epoch: %s, step: %s, loss is %s" % (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                cb_params.cur_epoch_num,
                cur_step_in_epoch,
                np.mean(self.loss_avg.avg))
            print(loss_log)
            loss_file = open("./loss.log", "a+")
            loss_file.write(loss_log)
            loss_file.write("\n")
            loss_file.close()


class LossCallBack_new(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF terminating training.

    Note:
        If per_print_times is 0 do not print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1):

        super(LossCallBack_new, self).__init__()

        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")

        self._per_print_times = per_print_times
        self.loss_avg = AverageMeter()
        stream = open('/home/group1/wjf_dbnet/dbnet_ms/dbnet/config.yaml', 'r', encoding='utf-8')
        self.arg = yaml.load(stream, Loader=yaml.FullLoader)
        stream.close()

        data_loader = DataLoader(self.arg, func="val")

        test_datasets = ds.GeneratorDataset(data_loader, ['img', 'gts', 'gt_masks', 'thresh_maps', 'thresh_masks'])
        self.test_datasets = test_datasets.batch(self.arg['train']['batch_size'])

        self.SegDetectorRepresenter = SegDetectorRepresenter(thresh=self.arg['train']['thresh'],
                                                             box_thresh=self.arg['train']['box_thresh'],
                                                             max_candidates=self.arg['train']['max_candidates'],
                                                             dest=self.arg['train']['dest'])
        self.QuadMeasurer = QuadMeasurer()

    def step_end(self, run_context):

        cb_params = run_context.original_args()

        if cb_params.net_outputs is not None:
            loss = cb_params.net_outputs.asnumpy()
        else:
            print("custom loss callback class loss is None.")
            return

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        cur_num = cb_params.cur_step_num

        # if cur_step_in_epoch == 1:
        #     self.loss_avg = AverageMeter()
        #
        # self.loss_avg.update(loss)

        if self._per_print_times != 0 and cur_num % self._per_print_times == 0:
            loss_log = "time: %s, epoch: %s, step: %s, loss is %s" % (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                cb_params.cur_epoch_num,
                cur_step_in_epoch,
                np.mean(loss))
            print(loss_log)
            # loss_file = open("./loss.log", "a+")
            # loss_file.write(loss_log)
            # loss_file.write("\n")
            # loss_file.close()

    def test(self, run_context):
        print("test datasets:{}".format(self.test_datasets))
        print("len(test datasets):{}".format(len(self.test_datasets)))

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        model = cb_params.train_network
        for i, batch in tqdm(enumerate(self.test_datasets), total=len(self.test_datasets)):
            pred = model(batch)
            output = self.SegDetectorRepresenter.represent(batch, pred, is_output_polygon=self.args['train']['polygon'])
            raw_metric = self.QuadMeasurer.validate_measure(batch, output,
                                                            is_output_polygon=self.args['train']['polygon'],
                                                            box_thresh=self.args['train']['box_thresh'])
            raw_metrics.append(raw_metric)


class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
