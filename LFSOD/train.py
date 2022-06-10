from torch.autograd import Variable
import torch.nn.functional as F
import torch
from LFSOD.models import model, modelfocal
from GAM.gam_model import STN, NLayerDiscriminator, GANLoss
from LFSOD import utils
from LFSOD.config import opt
from dataloader2 import get_loader
from FCM.fcm_model import Unet
from scipy import misc
import itertools
import numpy as np
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
        self.model = model().cuda().train()
        self.model_focal = modelfocal().cuda().train()
        self.gam = STN().cuda().train()
        self.gamD = NLayerDiscriminator(3).cuda().train()
        self.fcm = Unet(4,3).cuda().train()
        self.fcmD = NLayerDiscriminator(3).cuda().train()

        self.gam.load_state_dict(torch.load('../GAM/ckpt/GAM.pth'))
        self.fcm.load_state_dict(torch.load('../FCM/ckpt/FCM.pth'))

        self.optim_rgb = torch.optim.Adam(self.model.parameters(), self.lr)
        self.optim_focal = torch.optim.Adam(self.model_focal.parameters(), self.lr)
        self.optim_gam = torch.optim.Adam(self.gam.parameters(), self.lr)
        self.optim_fcm = torch.optim.Adam(self.fcm.parameters(), self.lr)
        self.optim_dis = torch.optim.Adam(itertools.chain(self.gamD.parameters(),
                                self.fcmD.parameters()), lr=self.lr, betas=(0.5, 0.999))
        self.loss_bce = torch.nn.BCELoss()
        self.loss_mse = torch.nn.SmoothL1Loss()
        self.criterionGAN = GANLoss(gan_mode='vanilla').cuda()

        self.epoch = 0
        self.iteration = 0
    def process_focal(self, x, shp):
        basize, dime, height, width = shp
        x = x.view(1, basize, dime, height, width).transpose(0, 1)
        x = torch.cat(torch.chunk(x, 12, dim=2), dim=1)
        x = torch.cat(torch.chunk(x, basize, dim=0), dim=1).squeeze()
        return x
    def process_gt(self, x, shp):
        basize, dime, height, width = shp
        x = x.repeat(1, 12, 1, 1)
        x = x.view(1, basize, 12, height, width).transpose(0, 1)
        x = torch.cat(torch.chunk(x, 12, dim=2), dim=1)
        x = torch.cat(torch.chunk(x, basize, dim=0), dim=1).squeeze(0)
        return x
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    def normal_w(self, w):
        w = (w - w.min() + 0.00001) / (w.max() - w.min() + 0.00001)
        w = (1 - w).detach()
        return w
    def train(self):
        data_iter = iter(self.data)
        step_per_epoch = len(self.data)
        save_path = './ckpt/'
        for epoch in range(self.max_epoch):
            data_iter = iter(self.data)
            step_per_epoch = len(self.data)
            for i in range(step_per_epoch * 2):
                try:
                    data_item = next(data_iter)
                except:
                    data_iter = iter(self.data)
                    data_item = next(data_iter)
                total_step = step_per_epoch
                images, bgs, gts, focal, bgf = data_item
                shp = focal.size()
                images, bgs, gts = Variable(images.cuda()), Variable(bgs.cuda()), Variable(gts.cuda())
                focal, bgf = Variable(focal.cuda()), Variable(bgf.cuda())
                gts_focal = self.process_gt(gts, shp)
                focal, bgf = self.process_focal(focal, shp), self.process_focal(bgf, shp)

                self.optim_rgb.zero_grad()
                self.optim_focal.zero_grad()
                self.optim_gam.zero_grad()
                self.optim_fcm.zero_grad()
                self.optim_dis.zero_grad()

                theat_e = torch.FloatTensor([1, 0, 0, 0, 1, 0, 0, 0, 1]).view(1, 3, 3).cuda()
                fgs = images*gts
                theat_k = theat_e
                theat_s = self.gam(bgs, fgs)
                theat_k[:, 0:2, :] = theat_e[:, 0:2, :]+theat_s
                theat = theat_k[:, 0:2, :]
                grid = F.affine_grid(theat, fgs.size())
                fgs_new = F.grid_sample(fgs, grid)
                grid = F.affine_grid(theat, gts.size())
                gts_new = F.grid_sample(gts, grid)
                images_new = bgs*(1-gts_new)+fgs_new

                focalfgs = focal*gts_focal
                theatf = theat.repeat(12, 1, 1)
                theatf = theatf.detach()
                grid = F.affine_grid(theatf, focal.size())
                focalfgs_new = F.grid_sample(focalfgs, grid)
                grid = F.affine_grid(theatf, gts_focal.size())
                gts_focal_new = F.grid_sample(gts_focal, grid)
                focal_new = bgf*(1-gts_focal_new)+focalfgs_new
                focal_newh = self.fcm(torch.cat((focal_new.detach(), gts_focal_new), dim=1))

                self.set_requires_grad(self.gamD, False)
                self.set_requires_grad(self.fcmD, False)

                if ((int)(i / 202) % 2 == 0):
                    rgbout, rgbfeat = self.model(images)
                    focalout, finalout = self.model_focal(focal, rgbfeat)
                    loss_rgb = self.loss_bce(rgbout, gts)
                    loss_focal = self.loss_bce(focalout, gts_focal)
                    loss_final = self.loss_bce(finalout, gts)
                else:

                    with torch.no_grad():
                        out_ori, _ = self.model(images)
                        grid = F.affine_grid(theat, out_ori.size())
                        out_ori = F.grid_sample(out_ori, grid)
                    rgbout_new, rgbfeat_new = self.model(images_new.detach())
                    focalout_new, finalout_new = self.model_focal(focal_newh, rgbfeat_new)
                    w = (out_ori - rgbout_new) * (out_ori - rgbout_new)
                    w = w / (w.max()+0.00001)
                    w = (1 - w).detach()
                    wf = w.repeat(12, 1, 1, 1)
                    loss_rgb = self.loss_bce(w*rgbout_new, w*gts.detach())
                    loss_focal = self.loss_bce(wf*focalout_new, wf*gts_focal.detach())
                    loss_final = self.loss_bce(w*finalout_new, w*gts.detach())
                pred_fake = self.gamD(focal_new)
                pred_fakeh = self.fcmD(focal_newh)
                loss_g = 0.05*(self.criterionGAN(pred_fake, True)+self.criterionGAN(pred_fakeh, True))
                loss_n = self.loss_mse(torch.norm(theat_s, p=2), (1 * torch.tensor(0.3, dtype=torch.float).cuda()))
                loss = loss_final + loss_focal + loss_rgb + loss_g + loss_n
                loss.backward()
                utils.clip_gradient(self.optim_rgb, 0.5)
                utils.clip_gradient(self.optim_focal, 0.5)
                utils.clip_gradient(self.optim_gam, 0.5)
                utils.clip_gradient(self.optim_fcm, 0.5)
                self.optim_rgb.step()
                self.optim_focal.step()
                self.optim_gam.step()
                self.optim_fcm.step()

                self.set_requires_grad(self.gamD, True)
                self.set_requires_grad(self.fcmD, True)
                pred_fake = self.gamD(focal_new.detach())
                pred_fakeh = self.fcmD(focal_newh.detach())
                pred_real = self.gamD(focal)
                pred_realh = self.fcmD(focal)
                loss_d_real = self.criterionGAN(pred_real, True)
                loss_d_realh = self.criterionGAN(pred_realh, True)
                loss_d_fake = self.criterionGAN(pred_fake, False)
                loss_d_fakeh = self.criterionGAN(pred_fakeh, False)
                loss_d = 0.025 * (loss_d_real + loss_d_fake + loss_d_realh + loss_d_fakeh)
                loss_d.backward()
                self.optim_dis.step()

                if i % 10 == 0 or i == total_step:
                    print('epoch {:03d}, step {:04d}, lossrgb: {:.4f}, lossfocal: {:0.4f}, lossfinal: {:0.4f},lossg: {:.4f}, lossd: {:.4f}, lossn: {:.4f}'
                          .format(epoch, i, loss_rgb.item(), loss_focal.item(), loss_final.item(), loss_g.item(), loss_d.item(), loss_n.item()))
            utils.adjust_lr(self.optim_rgb, self.lr, epoch, self.decay_rate, self.decay_epoch)
            utils.adjust_lr(self.optim_focal, self.lr, epoch, self.decay_rate, self.decay_epoch)
            utils.adjust_lr(self.optim_gam, self.lr, epoch, self.decay_rate, self.decay_epoch)
            utils.adjust_lr(self.optim_fcm, self.lr, epoch, self.decay_rate, self.decay_epoch)
            utils.adjust_lr(self.optim_dis, self.lr, epoch, self.decay_rate, self.decay_epoch)
            if (epoch + 1) % 5 == 0:
                torch.save(self.model.state_dict(), save_path + 'model.pth' + '.%d' % epoch)
                torch.save(self.model_focal.state_dict(), save_path + 'modelfocal.pth' + '.%d' % epoch)
                torch.save(self.fcm.state_dict(), save_path + 'FCM.pth' + '.%d' % epoch)
                torch.save(self.gam.state_dict(), save_path + 'GAM.pth' + '.%d' % epoch)

if __name__ == '__main__':
    config = opt
    train_loader = get_loader(opt.img_root, opt.gt_root, opt.focal_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    train = trainer(train_loader, config)
    train.train()