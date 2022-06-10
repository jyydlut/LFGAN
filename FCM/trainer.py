from FCM.fcm_model import Unet
import torch.optim
from LFSOD.config import opt
from dataloader2 import get_loader
import random, argparse, os, torch
import numpy as np
def get_parameters():

    parser = argparse.ArgumentParser()

    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
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
        self.build_model()
    def build_model(self):
        self.net = Unet(4, 3).cuda()
        self.net.train()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0002)
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
            img, bgs, gts, fcl, fclbg = data_item
            img, bgs, gts, fcl, fclbg = img.cuda(), bgs.cuda(), gts.cuda(), fcl.cuda(), fclbg.cuda()
            gtsf = self.process_gt(gts, fcl.size())
            fcl, fclbg = self.process_focal(fcl, fcl.size()), self.process_focal(fclbg, fcl.size())
            fgf = gtsf*fcl
            rand = np.array(range(0, 12))
            random.shuffle(rand)
            randr = torch.from_numpy(rand).cuda()
            fcl_gt = fcl[randr.long(), :, :, :]
            fg_gt = fgf[randr.long(), :, :, :]
            fcl_fake = fg_gt + fclbg*(1-gtsf)
            x = torch.cat((fcl_fake, gtsf), dim=1)
            output = self.net(x)
            loss = torch.nn.L1Loss()(output, fcl_gt)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f' % (i, step_per_epoch, i % step_per_epoch, len(self.data), loss.item()))
            if i % (len(self.data)*5) == 0:
                torch.save(self.net.state_dict(), save_path + 'FCM.pth' + '.%d' % i)

if __name__ == '__main__':
    image_root = 'E:/Light-Field-new/train_data_gan/train_images/'
    focal_root = 'E:/Light-Field-new/train_data_gan/train_focal/'
    gt_root = 'E:/Light-Field-new/train_data_gan/train_masks/'
    train_loader = get_loader(image_root, gt_root, focal_root, batchsize=1, trainsize=256)
    config = opt
    train = trainer(train_loader, config)
    train.train()