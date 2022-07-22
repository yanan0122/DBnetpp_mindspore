import time
import yaml

import mindspore
from mindspore import Tensor, context
import mindspore.dataset as ds
import mindspore.nn as nn

import sys
sys.path.insert(0, '.')
from datasets.load import DataLoader
from utils.metric import QuadMetric
from utils.post_process import SegDetectorRepresenter
from modules.model import DBnet, DBnetPP


class WithEvalCell(nn.Cell):
    def __init__(self, model, dataset):
        super(WithEvalCell, self).__init__(auto_prefix=False)
        self.model = model
        self.dataset = dataset
        self.metric_cls = QuadMetric()
        self.post_process = SegDetectorRepresenter()

        self.total_frame = 0.0
        self.total_time = 0.0
        self.raw_metrics = []

    def construct(self, batch):
        start = time.time()
        preds = self.model(batch['img'])
        boxes, scores = self.post_process(batch, preds, self.metric_cls.is_output_polygon)
        total_frame += batch['img'].size()[0]
        total_time += time.time() - start

        raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores))
        return raw_metric


def eval():
    ## Config
    stream = open('config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()

    ## Dataset
    data_loader = DataLoader(config, isTrain=False)
    dataset = ds.GeneratorDataset(data_loader, ['img', 'gts', 'gt_masks', 'thresh_maps', 'thresh_masks'])
    val_dataset = dataset.create_dict_iterator()

    ## Model
    model = DBnet()
    model_dict = mindspore.load_checkpoint('./checkpoints/DBnet/DBnet-1_63.ckpt')
    mindspore.load_param_into_net(model, model_dict)

    ## Eval Network
    eval_net = WithEvalCell(model, val_dataset)
    eval_net.set_train(False)

    ## eval
    raw_metrics = []
    for batch in val_dataset:
        raw_metric = eval_net(batch)
        raw_metrics.append(raw_metric)
    metrics = eval_net.metric_cls.gather_measure(eval_net.raw_metrics)
    print(f'FPS: {eval_net.total_frame / eval_net.total_time}')
    print(metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg)


if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=7)
    eval()
