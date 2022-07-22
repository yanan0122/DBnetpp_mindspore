import numpy as np
from PIL import Image
import glob
import cv2
import os
import yaml
import copy
import pathlib

from mindspore.dataset.vision.py_transforms import RandomColorAdjust, ToTensor, Normalize

from .random_thansform import RandomAugment
from .make_seg_map import MakeSegDetectionData
from .make_border_map import MakeBorderMap


def get_img(img_path):
    img = cv2.imread(img_path)
    return img


def get_bboxes(gt_path, config):
    with open(gt_path, 'r', encoding='utf-8') as fid:
        lines = fid.readlines()
    polys = []
    tags = []
    for line in lines:
        line = line.replace('\ufeff', '').replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        if "#" in gt[-1]:
            tags.append(True)
        else:
            tags.append(False)
        if (config['train']['is_icdar2015']):
            box = [int(gt[i]) for i in range(8)]
        else:
            box = [int(gt[i]) for i in range(len(gt) - 1)]
        polys.append(box)
    return np.array(polys), tags


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_datalist(train_data_path):
    """
    获取训练和验证的数据list
    :param train_data_path: 训练的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :return:
    """
    train_data = []
    for p in train_data_path:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                if len(line) > 1:
                    img_path = pathlib.Path(line[0].strip(' '))
                    label_path = pathlib.Path(line[1].strip(' '))
                    if img_path.exists() and img_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
                        train_data.append((str(img_path), str(label_path)))
    return train_data


class DataLoader():

    def __init__(self, config, isTrain=True):

        self.config = config
        self.ra = RandomAugment()
        self.ms = MakeSegDetectionData()
        self.mb = MakeBorderMap()

        if isTrain:
            img_paths = glob.glob(os.path.join(config['train']['train_img_dir'],'*'
                                               + config['train']['train_img_format']))
        else:
            img_paths = glob.glob(os.path.join(config['test']['test_img_dir'],'*'
                                               + config['test']['test_img_format']))
        gt_paths = []

        if isTrain:
            for img_path in img_paths:
                im_name = img_path.split('/')[-1].split('.')[0]
                if(config['train']['is_icdar2015']):
                    gt_file_name = im_name + '.jpg.txt'
                else:
                    gt_file_name = im_name + '.txt'
                gt_paths.append(os.path.join(config['train']['train_gt_dir'], gt_file_name))
        else:
            for img_path in img_paths:
                im_name = img_path.split('/')[-1].split('.')[0]
                if(config['test']['is_icdar2015']):
                    gt_file_name = 'gt_' + im_name + '.jpg.txt'
                else:
                    gt_file_name = im_name + '.txt'
                gt_paths.append(os.path.join(config['test']['test_gt_dir'], gt_file_name))

        self.img_paths = img_paths
        self.gt_paths = gt_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):

        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = get_img(img_path)
        polys, dontcare = get_bboxes(gt_path,self.config)

        if self.config['test']['is_transform']:
            img, polys = self.ra.random_scale(img, polys, 640)
            img, polys = self.ra.random_rotate(img, polys, self.config['test']['radom_angle'])
            img, polys = self.ra.random_flip(img, polys)
            img, polys, dontcare = self.ra.random_crop_db(img, polys, dontcare)

        img, gt, gt_mask = self.ms.process(img, polys, dontcare)
        img, thresh_map, thresh_mask = self.mb.process(img, polys, dontcare)

        if self.config['test']['is_show']:
            cv2.imwrite('img.jpg',img)
            cv2.imwrite('gt.jpg',gt[0]*255)
            cv2.imwrite('gt_mask.jpg',gt_mask[0]*255)
            cv2.imwrite('thresh_map.jpg',thresh_map*255)
            cv2.imwrite('thresh_mask.jpg',thresh_mask*255)

        if self.config['test']['is_transform']:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            colorjitter = RandomColorAdjust(brightness=32.0 / 255, saturation=0.5)
            img = colorjitter(img)

        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = ToTensor()(img)
        img = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)

        # img = img.reshape((1,3,640,640))
        # print("load img shape ", img.shape)

        return img, gt, gt_mask, thresh_map, thresh_mask


if __name__ == '__main__':
    stream = open('./config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    data_loader = DataLoader(config)
    print(data_loader[1][0].shape)
