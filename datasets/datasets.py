import os
import random

import cv2
import numpy as np
from PIL import ImageEnhance, Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def filename_without_ext(file_path):
    basename = os.path.basename(file_path)
    filename = os.path.splitext(basename)[0]
    return filename

def cv_random_flip(img, label, depth):
    if random.randint(0, 1) == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if depth is not None:
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

    return img, label, depth

def randomCrop(img, label, depth):
    border = 30
    img_width = img.size[0]
    img_height = img.size[1]
    crop_width = np.random.randint(img_width - border, img_width)
    crop_height = np.random.randint(img_height - border, img_height)

    regions = ((img_width - crop_width) >> 1, (img_height - crop_height) >> 1,
               (img_width + crop_width) >> 1, (img_height + crop_height) >> 1)
    if depth is not None:
        return img.crop(regions), label.crop(regions), depth.crop(regions)
    else:
        return img.crop(regions), label.crop(regions), None

def randomRotation(img, label, depth, mode=Image.BICUBIC):
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        img = img.rotate(random_angle, mode)
        label = label.rotate(random_angle, Image.NEAREST)
        if depth is not None:
            depth = depth.rotate(random_angle, mode)

    return img, label, depth

def ColorEnhance(img):
    brightness = random.randint(5, 15) / 10.0
    contrast = random.randint(5, 15) / 10.0
    colorness = random.randint(0, 20) / 10.0
    sharpness = random.randint(0, 30) / 10.0

    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(colorness)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)

    return img

def randomGaussian(img, mean=0.1, sigma=0.35):
    #img : numpy array
    gauss = np.random.normal(mean, sigma, img.shape)
    gauss = gauss.reshape(img.shape)
    img = img + gauss

    return img

class SalObjDataset(Dataset):
    def __init__(self, img_size=256, RGBD=True, mode=0):
        '''
        /KSC2021
            ㄴ datasets
                ㄴ NJU2K
                    ㄴ depth
                    ㄴ GT
                    ㄴ RGB_left
                    ㄴ ...
                ㄴ SIP
                    ㄴ depth
                    ㄴ GT
                    ㄴ RGB
                    ㄴ ...
                ㄴ DUT-RGBD
                    ㄴ train_data
                        ㄴ train_depth (.png)
                        ㄴ train_images (.jpg)
                        ㄴ train_masks (.png)
                    ㄴ test_data
                        ㄴ test_depth
                        ㄴ test_images
                        ㄴ test_masks
                ㄴ LFSD
                    ㄴ depth
                    ㄴ GT
                    ㄴ RGB
                    ㄴ ...
                ㄴ SSD
                    ㄴ depth
                    ㄴ GT
                    ㄴ RGB
                    ㄴ ...
            ㄴ models
                ㄴ DCF, BASNet, PoolNet, u2net
            ㄴ utils.py
            ㄴ train_rgbd.py

        :param img_size: image size
        :param RGBD: If true, RGBD. else, RGB dataset.
        :param mode: train : 0 / val : 1 / test : 2
        '''

        #root = os.path.join('datasets', dataset)
        self.img_size = img_size
        self.RGBD = RGBD
        self.mode = mode

        if self.mode == 0:
            datasets = ['NJU2K', 'SIP', 'DUT-RGBD/train_data']
        elif self.mode == 1:
            datasets = ['DUT-RGBD/test_data']
        else:
            AssertionError('mode must be in 0, 1')

        self.file_list = []
        self.img_list = []
        self.gt_list = []
        self.depth_list = []

        # Make file lists respectively.
        for dataset in datasets:
            img_path, gt_path, depth_path = self.make_paths(dataset)
            file_list = sorted(os.listdir(img_path))
            self.file_list.extend(file_list)
            self.img_list.extend([os.path.join(img_path, i) for i in file_list])
            self.gt_list.extend([os.path.join(gt_path, filename_without_ext(i)+'.png') for i in file_list])

            if self.RGBD:
                self.depth_list.extend([os.path.join(depth_path, filename_without_ext(i) + '.png') for i in file_list])

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if self.RGBD:
            self.depths_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        gt = Image.open(self.gt_list[idx]).convert('L')
        if self.RGBD:
            depth = cv2.imread(self.depth_list[idx])
            depth = Image.fromarray(depth).convert('L')
        else:
            depth = None

        if self.mode == 0:
            # data augmentation
            img, gt, depth = cv_random_flip(img, gt, depth)
            img, gt, depth = randomCrop(img, gt, depth)
            img, gt, depth = randomRotation(img, gt, depth)
            img = ColorEnhance(img)

        img = self.img_transform(img)
        gt = transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST)(gt)
        gt = torch.FloatTensor(np.array(gt) / 255).unsqueeze(0)

        if self.RGBD:
            depth = self.depths_transform(depth)
            return img, gt, depth

        return img, gt

    def make_paths(self, dataset):
        root = os.path.join('datasets', dataset)
        if dataset == 'NJU2K':
            return os.path.join(root, 'RGB_left'), os.path.join(root, 'GT'), os.path.join(root, 'depth')
        elif dataset == 'SIP':
            return os.path.join(root, 'RGB'), os.path.join(root, 'GT'), os.path.join(root, 'depth')
        elif dataset == 'DUT-RGBD/train_data':
            return os.path.join(root, 'train_images'), os.path.join(root, 'train_gts'), os.path.join(root, 'train_depth')
        elif dataset == 'DUT-RGBD/test_data':
            return os.path.join(root, 'test_images'), os.path.join(root, 'test_gts'), os.path.join(root, 'test_depth')


class TestDataset(Dataset):
    def __init__(self, img_size=256, RGBD=True, dataset='DUT-RGBD/test_data'):
        '''
        /KSC2021
            ㄴ datasets
                ㄴ NJU2K
                    ㄴ depth
                    ㄴ GT
                    ㄴ RGB_left
                    ㄴ ...
                ㄴ SIP
                    ㄴ depth
                    ㄴ GT
                    ㄴ RGB
                    ㄴ ...
                ㄴ DUT-RGBD
                    ㄴ train_data
                        ㄴ train_depth (.png)
                        ㄴ train_images (.jpg)
                        ㄴ train_masks (.png)
                    ㄴ test_data
                        ㄴ test_depth
                        ㄴ test_images
                        ㄴ test_masks
                ㄴ LFSD
                    ㄴ depth
                    ㄴ GT
                    ㄴ RGB
                    ㄴ ...
                ㄴ SSD
                    ㄴ depth
                    ㄴ GT
                    ㄴ RGB
                    ㄴ ...
            ㄴ models
                ㄴ DCF, BASNet, PoolNet, u2net
            ㄴ utils.py
            ㄴ train_rgbd.py

        :param img_size: image size
        :param RGBD: If true, RGBD. else, RGB dataset.
        :param mode: train : 0 / val : 1 / test : 2
        '''

        #root = os.path.join('datasets', dataset)
        self.img_size = img_size
        self.RGBD = RGBD
        self.dataset = dataset

        self.file_list = []
        self.img_list = []
        self.gt_list = []
        self.depth_list = []

        # Make file lists respectively.
        img_path, gt_path, depth_path = self.make_paths(dataset)
        file_list = sorted(os.listdir(img_path))
        self.file_list.extend(file_list)
        self.img_list.extend([os.path.join(img_path, i) for i in file_list])
        self.gt_list.extend([os.path.join(gt_path, filename_without_ext(i)+'.png') for i in file_list])

        if self.RGBD:
            self.depth_list.extend([os.path.join(depth_path, filename_without_ext(i) + '.png') for i in file_list])

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if self.RGBD:
            self.depths_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        gt = Image.open(self.gt_list[idx]).convert('L')
        if self.RGBD:
            depth = cv2.imread(self.depth_list[idx])
            depth = Image.fromarray(depth).convert('L')
        else:
            depth = None

        img = self.img_transform(img)
        gt = transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST)(gt)
        gt = torch.FloatTensor(np.array(gt) / 255).unsqueeze(0)

        if self.RGBD:
            depth = self.depths_transform(depth)
            return img, gt, depth

        return img, gt

    def make_paths(self, dataset):
        root = os.path.join('datasets', dataset)
        if dataset == 'DUT-RGBD/test_data':
            return os.path.join(root, 'test_images'), os.path.join(root, 'test_gts'), os.path.join(root, 'test_depth')
        elif dataset == 'LFSD' or dataset == 'SSD' or dataset == 'STEREO-1000':
            return os.path.join(root, 'RGB'), os.path.join(root, 'GT'), os.path.join(root, 'depth')
