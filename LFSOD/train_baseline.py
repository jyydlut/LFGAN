import os
from torch.autograd import Variable
import torch
from LFSOD.models import model, modelfocal
from LFSOD import utils
from LFSOD.config import opt
from dataloader1 import get_loader
import random
class trainer(object):

    def __init__(self, data, config):
        self.data = data
        self.lr = config.lr[0]/16
        self.max_epoch = config.epoch
        self.triansize = config.trainsize
        self.decay_rate = config.decay_rate
        self.decay_epoch = config.decay_epoch
        self.clip = config.clip
        self.build_model()

    def build_model(self):
        self.model = model().cuda()
        self.model_focal = modelfocal().cuda()
        self.optim = torch.optim.Adam(self.model.parameters(), self.lr)
        self.optim_focal = torch.optim.Adam(self.model_focal.parameters(), self.lr)
        self.loss_bce = torch.nn.BCELoss()
        self.epoch = 0
        self.iteration = 0

    def train(self):
        save_path = './ckpt/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for epoch in range(self.max_epoch):
            for i, pack in enumerate(self.data):
                self.model.train()
                self.model_focal.train()
                total_step = len(self.data)
                images, gts, focal = pack
                basize, dime, height, width = focal.size()  # 2*36*256*256
                images, gts, focal = images.cuda(), gts.cuda(), focal.cuda()
                images, gts, focal = Variable(images), Variable(gts), Variable(focal)
                gts_focal = gts.repeat(1, 12, 1, 1)
                focal = focal.view(1, basize, dime, height, width).transpose(0, 1)
                gts_focal = gts_focal.view(1, basize, 12, height, width).transpose(0, 1)
                focal = torch.cat(torch.chunk(focal, 12, dim=2), dim=1)
                gts_focal = torch.cat(torch.chunk(gts_focal, 12, dim=2), dim=1)
                focal = torch.cat(torch.chunk(focal, basize, dim=0), dim=1).squeeze()
                gts_focal = torch.cat(torch.chunk(gts_focal, basize, dim=0), dim=1).squeeze(0)
                if random.random()>0.5:
                    focal = torch.flip(focal, [3])
                    images = torch.flip(images, [3])
                    gts = torch.flip(gts, [3])
                    gts_focal = torch.flip(gts_focal, [3])
                self.optim.zero_grad()
                self.optim_focal.zero_grad()

                rgbout, rgbfeat = self.model(images)  # RGBNet's output
                loss_rgb = self.loss_bce(rgbout, gts)
                focalout, finalout = self.model_focal(focal, rgbfeat)
                loss_focal = self.loss_bce(focalout, gts_focal)
                loss_final = self.loss_bce(finalout, gts)
                loss = loss_final + loss_focal + loss_rgb
                loss.backward()

                self.optim.step()
                self.optim_focal.step()
                if i % 10 == 0 or i == total_step:
                    print('epoch {:03d}, step {:04d}, lossrgb: {:.4f}, lossfocal: {:0.4f}, lossfinal: {:0.4f}'
                          .format(epoch, i, loss_rgb.item(), loss_focal.item(), loss_final.item()))
            utils.adjust_lr(self.optim, self.lr, epoch, self.decay_rate, self.decay_epoch)
            utils.adjust_lr(self.optim_focal, self.lr, epoch, self.decay_rate, self.decay_epoch)
            if (epoch + 1) % 5 == 0:
                torch.save(self.model.state_dict(), save_path + 'modelbase.pth' + '.%d' % epoch)
                torch.save(self.model_focal.state_dict(), save_path + 'modelfocalbase.pth' + '.%d' % epoch)


if __name__ == '__main__':
    config = opt
    train_loader = get_loader(opt.img_root, opt.gt_root, opt.focal_root, batchsize=1, trainsize=opt.trainsize)
    train = trainer(train_loader, config)
    train.train()