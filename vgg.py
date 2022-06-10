import torch.nn as nn
import torchvision

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.gn1_1 = nn.GroupNorm(num_groups=4, num_channels=64, eps=1e-05, affine=False)
        self.gn1_2 = nn.GroupNorm(num_groups=4, num_channels=64, eps=1e-05, affine=False)
        self.gn2_1 = nn.GroupNorm(num_groups=4, num_channels=128, eps=1e-05, affine=False)
        self.gn2_2 = nn.GroupNorm(num_groups=4, num_channels=128, eps=1e-05, affine=False)
        self.gn3_1 = nn.GroupNorm(num_groups=8, num_channels=256, eps=1e-05, affine=False)
        self.gn3_2 = nn.GroupNorm(num_groups=8, num_channels=256, eps=1e-05, affine=False)
        self.gn3_3 = nn.GroupNorm(num_groups=8, num_channels=256, eps=1e-05, affine=False)
        self.gn4_1 = nn.GroupNorm(num_groups=16, num_channels=512, eps=1e-05, affine=False)
        self.gn4_2 = nn.GroupNorm(num_groups=16, num_channels=512, eps=1e-05, affine=False)
        self.gn4_3 = nn.GroupNorm(num_groups=16, num_channels=512, eps=1e-05, affine=False)
        self.gn5_1 = nn.GroupNorm(num_groups=16, num_channels=512, eps=1e-05, affine=False)
        self.gn5_2 = nn.GroupNorm(num_groups=16, num_channels=512, eps=1e-05, affine=False)
        self.gn5_3 = nn.GroupNorm(num_groups=16, num_channels=512, eps=1e-05, affine=False)
        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(3, 64, 3, 1, 1))
        conv1.add_module('gn1_1', self.gn1_1)
        conv1.add_module('relu1_1', nn.ReLU(inplace=True))
        conv1.add_module('conv1_2', nn.Conv2d(64, 64, 3, 1, 1))
        conv1.add_module('gn1_2', self.gn1_2)
        conv1.add_module('relu1_2', nn.ReLU(inplace=True))
        self.conv1 = conv1
        conv2 = nn.Sequential()
        conv2.add_module('pool1', nn.AvgPool2d(2, stride=2))
        conv2.add_module('conv2_1', nn.Conv2d(64, 128, 3, 1, 1))
        conv2.add_module('gn2_1', self.gn2_1)
        conv2.add_module('relu2_1', nn.ReLU())
        conv2.add_module('conv2_2', nn.Conv2d(128, 128, 3, 1, 1))
        conv2.add_module('gn2_2', self.gn2_2)
        conv2.add_module('relu2_2', nn.ReLU())
        self.conv2 = conv2

        conv3 = nn.Sequential()
        conv3.add_module('pool2', nn.AvgPool2d(2, stride=2))
        conv3.add_module('conv3_1', nn.Conv2d(128, 256, 3, 1, 1))
        conv3.add_module('gn3_1', self.gn3_1)
        conv3.add_module('relu3_1', nn.ReLU())
        conv3.add_module('conv3_2', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('gn3_2', self.gn3_2)
        conv3.add_module('relu3_2', nn.ReLU())
        conv3.add_module('conv3_3', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('gn3_3', self.gn3_3)
        conv3.add_module('relu3_3', nn.ReLU())
        self.conv3 = conv3

        conv4_1 = nn.Sequential()
        conv4_1.add_module('pool3_1', nn.AvgPool2d(2, stride=2))
        conv4_1.add_module('conv4_1_1', nn.Conv2d(256, 512, 3, 1, 1))
        conv4_1.add_module('gn4_1', self.gn4_1)
        conv4_1.add_module('relu4_1_1', nn.ReLU())
        conv4_1.add_module('conv4_2_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv4_1.add_module('gn4_2', self.gn4_2)
        conv4_1.add_module('relu4_2_1', nn.ReLU())
        conv4_1.add_module('conv4_3_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv4_1.add_module('gn4_3', self.gn4_3)
        conv4_1.add_module('relu4_3_1', nn.ReLU())
        self.conv4_1 = conv4_1

        conv5_1 = nn.Sequential()
        conv5_1.add_module('pool4_1', nn.AvgPool2d(2, stride=2))
        conv5_1.add_module('conv5_1_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv5_1.add_module('gn5_1', self.gn5_1)
        conv5_1.add_module('relu5_1_1', nn.ReLU())
        conv5_1.add_module('conv5_2_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv5_1.add_module('gn5_2', self.gn5_2)
        conv5_1.add_module('relu5_2_1', nn.ReLU())
        conv5_1.add_module('conv5_3_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv5_1.add_module('gn5_3', self.gn5_3)
        conv5_1.add_module('relu5_3_1', nn.ReLU())
        self.conv5_1 = conv5_1

        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.copy_params_from_vgg16(vgg16)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight.data.zero_()
                nn.init.normal(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4_1(x)
        x = self.conv5_1(x)
        return x
    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1.conv1_1, self.conv1.relu1_1,
            self.conv1.conv1_2, self.conv1.relu1_2,
            self.conv2.pool1,
            self.conv2.conv2_1, self.conv2.relu2_1,
            self.conv2.conv2_2, self.conv2.relu2_2,
            self.conv3.pool2,
            self.conv3.conv3_1, self.conv3.relu3_1,
            self.conv3.conv3_2, self.conv3.relu3_2,
            self.conv3.conv3_3, self.conv3.relu3_3,
            #self.conv3_4, self.bn3_4, self.relu3_4,
            self.conv4_1.pool3_1,
            self.conv4_1.conv4_1_1, self.conv4_1.relu4_1_1,
            self.conv4_1.conv4_2_1, self.conv4_1.relu4_2_1,
            self.conv4_1.conv4_3_1, self.conv4_1.relu4_3_1,
            #self.conv4_4, self.bn4_4, self.relu4_4,
            self.conv5_1.pool4_1,
            self.conv5_1.conv5_1_1, self.conv5_1.relu5_1_1,
            self.conv5_1.conv5_2_1, self.conv5_1.relu5_2_1,
            self.conv5_1.conv5_3_1, self.conv5_1.relu5_3_1,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data