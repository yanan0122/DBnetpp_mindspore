import time
import yaml
from tqdm.auto import tqdm

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

    def construct(self, batch,verbose=True):
        start = time.time()

        preds = self.model(batch['img'])
        boxes, scores = self.post_process(preds, False)
        raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores))

        cur_frame = batch['img'].shape[0]
        cur_time = time.time() - start

        if verbose:
            return raw_metric, (cur_frame, cur_time)
        else:
            return raw_metric

    def eval(self, dataset):
        total_frame = 0.0
        total_time = 0.0
        raw_metrics = []

        for batch in tqdm(dataset):
            raw_metric, (cur_frame, cur_time) = self(batch)
            self.raw_metrics.append(raw_metric)
            total_frame += cur_frame
            total_time += cur_time
        metrics = self.metric_cls.gather_measure(raw_metrics)
        print(f'FPS: {total_frame / total_time}')
        print(metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg)

def eval(model: nn.Cell, path: str):
    ## Config
    stream = open('config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()

    ## Dataset
    data_loader = DataLoader(config, isTrain=False)
    val_dataset = ds.GeneratorDataset(data_loader, ['img', 'polys', 'dontcare'])
    val_dataset = val_dataset.batch(1)
    dataset = val_dataset.create_dict_iterator()

    ## Model
    model_dict = mindspore.load_checkpoint(path)
    mindspore.load_param_into_net(model, model_dict)

    ## Eval
    eval_net = WithEvalCell(model, val_dataset)
    eval_net.set_train(False)
    eval_net.eval(dataset)


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=5)
    eval(DBnet(False), './checkpoints/DBnet/DBnet-19_63.ckpt')
