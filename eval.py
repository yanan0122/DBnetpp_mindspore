import os
import sys
import pathlib

import mindspore
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))
# project = 'DBNet.pytorch'  # 工作项目根目录
# sys.path.append(os.getcwd().split(project)[0] + project)

import argparse
import time
import copy
from tqdm.auto import tqdm
import torch


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


class Eval():
    def __init__(self, model_path, gpu_id=0):

        # self.gpu_id = gpu_id
        # if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
        #     self.device = torch.device("cuda:%s" % self.gpu_id)
        #     torch.backends.cudnn.benchmark = True
        # else:
        #     self.device = torch.device("cpu")

        checkpoint = torch.load(model_path)
        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False

        self.validate_loader = get_dataloader(config['dataset']['validate'], config['distributed'])

        self.model = build_model(config['arch'])
        self.model.load_state_dict(checkpoint['state_dict'])
        # self.model.to(self.device)

        self.post_process = get_post_processing(config['post_processing'])
        self.metric_cls = get_metric(config['metric'])

    def eval(self):
        self.model.eval()
        # torch.cuda.empty_cache()  # speed up evaluating after training finished
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0
        for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            with torch.no_grad():
                # 数据进行转换和丢到gpu
                for key, value in batch.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                ###
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
    raw_metrics = []
    total_frame = 0.0
    total_time = 0.0
    for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
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


if __name__ == '__main__':
    ckpt = mindspore.load_checkpoint('./checkpoints/DBNetPP-19_63.ckpt')
    print(len(ckpt))