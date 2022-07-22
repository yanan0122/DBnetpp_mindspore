import copy
import time
import yaml
from tqdm.auto import tqdm

import mindspore
import mindspore.dataset as ds
import mindspore.nn as nn

from DBnetpp_mindspore.dataloader.load import DataLoader


def get_post_processing(config):
    try:
        cls = eval(config['type'])(**config['args'])
        return cls
    except:
        return None


def get_metric(config):
    try:
        if 'args' not in config:
            args = {}
        else:
            args = config['args']
        if isinstance(args, dict):
            cls = eval(config['type'])(**args)
        else:
            cls = eval(config['type'])(args)
        return cls
    except:
        return None


def build_model(config):
    """
    get architecture model class
    """
    copy_config = copy.deepcopy(config)
    arch_type = copy_config.pop('type')
    arch_model = eval(arch_type)(copy_config)
    return arch_model


class WithEvalCell(nn.Cell):
    def __init__(self, model, dataset):
        super(WithEvalCell, self).__init__(auto_prefix=False)
        self.model = model
        self.dataset = dataset

        self.post_process = get_post_processing(config['post_processing'])
        self.metric_cls = get_metric(config['metric'])

    def eval(self):
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0
        for i, batch in enumerate(self.validate_loader):
            with torch.no_grad():
                start = time.time()
                preds = self.model(batch['img'])
                boxes, scores = self.post_process(batch, preds,is_output_polygon=self.metric_cls.is_output_polygon)
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
