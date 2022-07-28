import numpy as np
from PIL import Image
import glob
import cv2
import os
import yaml

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
    dontcare = []
    for line in lines:
        line = line.replace('\ufeff', '').replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        if "#" in gt[-1]:
            dontcare.append(True)
        else:
            dontcare.append(False)
        if config['general']['is_icdar2015']:
            box = [int(gt[i]) for i in range(8)]
        else:
            box = [int(gt[i]) for i in range(len(gt) - 1)]
        polys.append(box)
    return np.array(polys), dontcare


class DataLoader():
    def __init__(self, config, isTrain=True):
        self.config = config
        self.isTrain = isTrain

        self.ra = RandomAugment()
        self.ms = MakeSegDetectionData()
        self.mb = MakeBorderMap()

        if isTrain:
            img_paths = glob.glob(os.path.join(config['train']['train_img_dir'],
                                               '*' + config['train']['train_img_format']))
        else:
            img_paths = glob.glob(os.path.join(config['test']['test_img_dir'],
                                               '*' + config['test']['test_img_format']))
        gt_paths = []

        if isTrain:
            for img_path in img_paths:
                im_name = img_path.split('/')[-1].split('.')[0]
                if(config['general']['is_icdar2015']):
                    gt_file_name = im_name + '.jpg.txt'
                else:
                    gt_file_name = im_name + '.txt'
                gt_paths.append(os.path.join(config['train']['train_gt_dir'], gt_file_name))
        else:
            for img_path in img_paths:
                im_name = img_path.split('/')[-1].split('.')[0]
                if(config['general']['is_icdar2015']):
                    gt_file_name = 'gt_' + im_name + '.txt'
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
        polys, dontcare = get_bboxes(gt_path, self.config)

        if self.isTrain and self.config['train']['is_transform']:
            img, polys = self.ra.random_scale(img, polys, 640)
            img, polys = self.ra.random_rotate(img, polys, self.config['train']['random_angle'])
            img, polys = self.ra.random_flip(img, polys)
            img, polys, dontcare = self.ra.random_crop_db(img, polys, dontcare)
        else:
            img, polys = self.ra.rescale(img, polys)

        if self.isTrain:
            img, gt, gt_mask = self.ms.process(img, polys, dontcare)
            img, thresh_map, thresh_mask = self.mb.process(img, polys, dontcare)
        else:
            polys = np.stack(polys, 0)
            dontcare = np.array(dontcare, dtype=np.bool8)

        if self.config['general']['is_show']:
            cv2.imwrite('./img.jpg', img)
            cv2.imwrite('./gt.jpg', gt[0]*255)
            cv2.imwrite('./gt_mask.jpg', gt_mask*255)
            cv2.imwrite('./thresh_map.jpg', thresh_map*255)
            cv2.imwrite('./thresh_mask.jpg', thresh_mask*255)

        if self.isTrain and self.config['train']['is_transform']:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            colorjitter = RandomColorAdjust(brightness=32.0 / 255, saturation=0.5)
            img = colorjitter(img)

        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = ToTensor()(img)
        img = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)

        if self.isTrain:
            return img, gt, gt_mask, thresh_map, thresh_mask
        else:
            return img, polys, dontcare

    def get_tags(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = get_img(img_path)
        polys, dontcare = get_bboxes(gt_path, self.config)

        if self.isTrain and self.config['train']['is_transform']:
            img, polys = self.ra.random_scale(img, polys, 640)
            img, polys = self.ra.random_rotate(img, polys, self.config['train']['random_angle'])
            img, polys = self.ra.random_flip(img, polys)
            img, polys, dontcare = self.ra.random_crop_db(img, polys, dontcare)
        else:
            img, polys = self.ra.rescale(img, polys)

        return img, polys, dontcare


if __name__ == '__main__':
    stream = open('./config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    data_loader = DataLoader(config, isTrain=False)
    import mindspore.dataset as ds
    train_dataset = ds.GeneratorDataset(data_loader, ['img', 'polys', 'dontcare'])
    train_dataset = train_dataset.batch(1)
    it = train_dataset.create_dict_iterator()
    test = next(it)
    print(test['img'].shape, test['polys'].shape, test['dontcare'].shape)
    # sam = data_loader[19]
    # print(sam[0].shape, len(sam[1]), sam[2])