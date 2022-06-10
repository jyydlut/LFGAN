import torch
from GAM.gam_model import STN, NLayerDiscriminator, GANLoss
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from GAM.gam_dataloader import get_loader
import os
from PIL import ImageFile
import argparse
def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.04, help='initial learning rate for adam')
    parser.add_argument('--ganmodel', type=str, default='lsgan', help='the type of GAN objective. \
    [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--input_nc', type=int, default=3,
                        help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--dim', type=int, default=128)
    return parser.parse_args()


class trainer(object):
    def __init__(self, data, config):
        self.data = data
        self.lr = config.lr
        self.dim = config.dim
        self.beta1 = config.beta1
        self.build_model()
    def build_model(self):
        self.gam = STN().cuda()
        self.gam.train()
        self.optimizer = torch.optim.Adam(self.gam.parameters(), lr=self.lr)
        self.loss_mse = torch.nn.SmoothL1Loss()
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train(self):
        data_iter = iter(self.data)
        step_per_epoch = len(self.data)
        save_path = './ckpt/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(step_per_epoch*100):
            try:
                data_item = next(data_iter)
            except:
                data_iter = iter(self.data)
                data_item = next(data_iter)
            img, bg, mask = data_item
            img, bg, mask = img.cuda(), bg.cuda(), mask.cuda()

            """set a random theat to generate the fake img"""
            batch_size, _, w, h = img.size()
            theat_init = torch.FloatTensor([1, 0, 0, 0, 1, 0, 0, 0, 1]).view(1, 3, 3).cuda()
            theat_init = theat_init.repeat(batch_size, 1, 1)
            theat_add = ((torch.rand(batch_size, 2, 3) * 2 - 1)*0.5).cuda()
            theat_init[:, 0:2, :] = theat_init[:, 0:2, :] + theat_add
            theat = theat_init[:, 0:2, :]
            fg = img*mask
            grid = F.affine_grid(theat, fg.size())
            fg_fake= F.grid_sample(fg, grid)
            grid = F.affine_grid(theat, mask.size())
            mask_fake = F.grid_sample(mask, grid)

            """solve the inv of theat and calculate the gt"""
            theat_inv = torch.inverse(theat_init)
            theat_gt = theat_inv[:, 0:2, :]
            grid = F.affine_grid(theat_gt, fg_fake.size())
            fg_gt = F.grid_sample(fg_fake, grid)
            grid = F.affine_grid(theat_gt, mask_fake.size())
            mask_gt = F.grid_sample(mask_fake, grid)

            """feed fg_fake and bg into STN_model for obtaining img_pre"""
            theat_pre = self.gam(fg_fake, bg)
            grid = F.affine_grid(theat_pre, fg_fake.size())
            fg_pre = F.grid_sample(fg_fake, grid)
            grid = F.affine_grid(theat_pre, mask_fake.size())
            mask_pre = F.grid_sample(mask_fake, grid)
            img_gt = img
            img_pre = fg_pre + bg * (1 - mask_pre)

            """Calculate loss"""
            self.optimizer.zero_grad()
            loss = self.loss_mse(theat_pre, theat_gt)
            loss.backward()
            self.optimizer.step()


            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f'%
                      (i, step_per_epoch, i % step_per_epoch, len(self.data), loss.item()))
            if i % (len(self.data)*5) == 0:
                torch.save(self.gam.state_dict(), save_path + 'GAM.pth' + '.%d' % i)

if __name__ == '__main__':
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    image_root = 'E:/Light-Field-new/train_data_gan/train_images/'
    data = get_loader(image_root, batchsize=64, trainsize=256)
    cudnn.benchmark = True
    config = get_parameters()
    train = trainer(data, config)
    train.train()