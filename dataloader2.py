import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import numpy as np
import scipy.io as sio
class SalObjDataset(data.Dataset):
    mean_rgb = np.array([0.485, 0.456, 0.406])
    std_rgb = np.array([0.229, 0.224, 0.225])
    mean_focal = np.tile(mean_rgb, 12)
    std_focal = np.tile(std_rgb, 12)
    def __init__(self, image_root, gt_root, focal_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.focals = [focal_root + f for f in os.listdir(focal_root) if f.endswith('.mat')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.focals = sorted(self.focals) # 排序
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
    #   return image, mask
    def focal_transform(self, x):
        x = np.array(x, dtype=np.float) / 255.0
        x -= self.mean_focal
        x /= self.std_focal
        x = x.transpose(2, 0, 1)
        x = torch.from_numpy(x).float()
        return x
    def __getitem__(self, index):
        name_img = self.images[index].split('/')[-1][:-4]
        pathimg_bg = 'E:/Light-Field-new/train_data_gan/inpainting_images/' + name_img + '.jpg'
        pathfocal_bg = 'E:/Light-Field-new/train_data_gan/inpainting_focal/' + name_img + '.mat'
        image = self.img_transform(self.rgb_loader(self.images[index]))
        img_bg = self.img_transform(self.rgb_loader(pathimg_bg))
        gt = self.gt_transform(self.binary_loader(self.gts[index]))
        focal = self.focal_transform(self.focal_loader(self.focals[index]))
        focal_bg = self.focal_transform(self.focal_loader(pathfocal_bg))
        return image, img_bg, gt, focal, focal_bg
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def focal_loader(self, path):
        with open(path, 'rb') as f:
            focal = sio.loadmat(f)
            focal = focal['img']
            return focal

    def resize(self, img, gt, focal):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), focal
        else:
            return img, gt, focal

    def __len__(self):
        return self.size

def get_loader(image_root, gt_root, focal_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root, focal_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory, drop_last=True)
    return data_loader


