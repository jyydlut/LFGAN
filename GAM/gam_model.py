import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.encoder = nn.ModuleList()
        in_channels = 6
        conv_dim = 64
        self.n_layers = 5
        for i in range(self.n_layers):
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_channels, conv_dim * 2 ** i, 3, padding=1, bias=False),
                nn.BatchNorm2d(conv_dim * 2 ** i),
                nn.MaxPool2d(2, 2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))
            in_channels = conv_dim * 2 ** i
        self.max_pool = nn.MaxPool2d(2, 2)
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[3].weight.data.fill_(0)
        self.fc_loc[3].bias.data = torch.FloatTensor([0, 0, 0, 0, 0, 0])

    def forward(self, xf, xb):
        input = torch.cat((xf, xb), dim=1)
        input = nn.functional.upsample(input, size=(64, 64), mode='bilinear', align_corners=False)
        for layer in self.encoder:
            input = layer(input)
        xs = self.max_pool(input).view(-1, 1024)
        theta = self.fc_loc(xs)

        theta = theta.view(-1, 2, 3)
        theta = theta
        return theta
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        use_bias = False
        kw = 4
        padw = 1
        self.feature_img = self.get_feature_extractor(input_nc, ndf, n_layers, kw, padw)
        self.classifier = self.get_classifier(ndf, n_layers, kw, padw)  # 2*ndf
        convc = []
        convc += [nn.Upsample(scale_factor=4)]
        convc += [nn.Conv2d(1, 1, 3, padding=1)]
        convc += [nn.Conv2d(1, 1, 3, padding=1)]
        convc += [nn.Upsample(scale_factor=2)]
        convc += [nn.Conv2d(1, 1, 3, padding=1)]
        convc += [nn.Conv2d(1, 1, 3, padding=1)]
        self.convc = nn.Sequential(*convc)
    def get_feature_extractor(self, input_nc, ndf, n_layers, kw, padw):
        model = [SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
                 nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            model += [
                SpectralNorm(nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=kw, stride=2, padding=padw)),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        return nn.Sequential(*model)

    def get_classifier(self, ndf, n_layers, kw, padw):
        nf_mult_prev = min(2**(n_layers-1), 8)
        nf_mult = min(2**n_layers, 8)
        model = [
            SpectralNorm(nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=kw, stride=1, padding=padw)),
            nn.BatchNorm2d(ndf*nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        model += [SpectralNorm(nn.Conv2d(ndf*nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]
        return nn.Sequential(*model)

    def forward(self, image_in):
        feat = self.feature_img(image_in)
        out = self.classifier(feat)
        return out
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)
class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

