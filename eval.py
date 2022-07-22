from tqdm.auto import tqdm
import copy
import time
import yaml

import mindspore
import mindspore.dataset as ds
import mindspore.nn as nn

from datasets.load import DataLoader
from utils.metric import QuadMetric
from utils.post_process import SegDetectorRepresenter


class WithEvalCell(nn.Cell):
    def __init__(self, model, dataset):
        super(WithEvalCell, self).__init__(auto_prefix=False)
        self.model = model
        self.dataset = dataset

        self.post_process = SegDetectorRepresenter()
        self.metric_cls = QuadMetric()

    def eval(self):
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0
        for i, batch in enumerate(self.validate_loader):
            start = time.time()
            preds = self.model(batch['img'])
            boxes, scores = self.post_process(batch, preds, self.metric_cls.is_output_polygon)
            total_frame += batch['img'].size()[0]
            total_time += time.time() - start
            raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores))
            raw_metrics.append(raw_metric)
        metrics = self.metric_cls.gather_measure(raw_metrics)
        print('FPS:{}'.format(total_frame / total_time))
        return metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg


def eval():
    ## Config
    stream = open('config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()

    ## Dataset
    data_loader = DataLoader(config, func="test")
    val_dataset = ds.GeneratorDataset(data_loader, ['img', 'gts', 'gt_masks', 'thresh_maps', 'thresh_masks'])
    # dl = dataset.create_dict_iterator()

    ## Model
    model = mindspore.load_checkpoint('./checkpoints/DBNetPP-19_63.ckpt')

    ## Eval
    eval_net = WithEvalCell(model, val_dataset)
    eval_net.set_train(False)

if __name__ == '__main__':
    eval()
