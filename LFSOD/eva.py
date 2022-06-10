import torch
import torch.nn.functional as F
import numpy as np
import os
from scipy import misc
from dataloader1 import test_dataset
from LFSOD.models import model, modelfocal
from LFSOD.config import opt
config=opt
model = model()
model.load_state_dict(torch.load('./ckpt/model.pth'))
model.cuda()
model.eval()
modelfocal = modelfocal()
modelfocal.load_state_dict(torch.load('./ckpt/modelfocal.pth'))
modelfocal.cuda()
modelfocal.eval()
test_datasets = ['DUTS', 'HFUT', 'LFSD']
testimage = ['E:/Light-Field-new/test_data/test_images/', 'D:/datasets/HFUT-LFSD/HFUT-LFSD/test_images/', 'D:/CPD+LF1/results/VGG16/LFSD/LFSD/test_images/']
testgt = ['E:/Light-Field-new/test_data/test_masks/', 'D:/datasets/HFUT-LFSD/HFUT-LFSD/test_masks/', 'D:/CPD+LF1/results/VGG16/LFSD/LFSD/test_masks/']
testfocal = ['E:/Light-Field-new/test_data/test_focal/', 'D:/datasets/HFUT-LFSD/HFUT-LFSD/test_focal/', 'D:/CPD+LF1/results/VGG16/LFSD/LFSD/test_focal/']
for d in range(3):
    save_path = './results/' + test_datasets[d] + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    test_loader = test_dataset(testimage[d], testgt[d], testfocal[d], config.testsize)
    for i in range(test_loader.size):
        image, focal, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        dim, height, width = focal.size()
        basize = 1
        focal = focal.view(1, basize, dim, height, width).transpose(0, 1)
        focal = torch.cat(torch.chunk(focal, 12, dim=2), dim=1)
        focal = torch.cat(torch.chunk(focal, 12, dim=1), dim=0).squeeze()
        focal = F.interpolate(focal, size=(256, 256), mode='bilinear')
        image, focal = image.cuda(), focal.cuda()
        with torch.no_grad():
            rgbout, rgbfeat = model(image)
            focalout, finalout = modelfocal(focal, rgbfeat, mode='test')
        res = F.upsample(finalout, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        misc.imsave(save_path+name, res)