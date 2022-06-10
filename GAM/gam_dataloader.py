import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import cv2
import random


class SalObjDataset(data.Dataset):
    def __init__(self, image_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        # self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
        #             or f.endswith('.png')]
        self.images = sorted(self.images)
        # self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
    def my_transform(self, img_bg, img, gt):
        img_bg = transforms.Resize((self.trainsize, self.trainsize))(img_bg)
        img = transforms.Resize((self.trainsize, self.trainsize))(img)
        gt = transforms.Resize((self.trainsize, self.trainsize))(gt)

        if random.random()>0.5:
            img_bg = transforms.functional.hflip(img_bg)
            img = transforms.functional.hflip(img)
            gt = transforms.functional.hflip(gt)

        img_bg = transforms.ToTensor()(img_bg)
        img = transforms.ToTensor()(img)
        gt = transforms.ToTensor()(gt)

        img_bg = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_bg)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img_bg, img, gt

    def __getitem__(self, index):
        path_img = self.images[index]
        path_bg = "E:/Light-Field-new/train_data_gan/inpainting_images/" + path_img.split('/')[-1][:-4] + ".jpg"
        path_sal = "E:/Light-Field-new/train_data_gan/train_masks/" + path_img.split('/')[-1][:-4] + ".png"
        img_bg = self.rgb_loader(path_bg)
        img = self.rgb_loader(path_img)
        gt = self.binary_loader(path_sal)
        img_bg, img, gt = self.my_transform(img_bg, img, gt)

        # image = self.img_transform(image)
        # gt = self.gt_transform(gt)
        return img, img_bg, gt

    def filter_files(self):
        # assert len(self.images) == len(self.gts)
        images = []
        for img_path in self.images:
            images.append(img_path)
        self.images = images

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):

    dataset = SalObjDataset(image_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory, drop_last=True)
    return data_loader
