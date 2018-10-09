import torch
import torch.nn as nn
import torch.nn.functional as F


class Base3DNet(nn.Module):
    def __init__(self, im_channel, dp_channel, num_class):
        super(Base3DNet, self).__init__()
        num_hidden = 8
        self.bn0 = nn.BatchNorm2d(im_channel + dp_channel)
        self.conv1 = nn.Conv2d(im_channel + dp_channel, num_hidden, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden * 2, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_hidden * 2)
        self.conv3 = nn.Conv2d(num_hidden * 2, num_hidden * 4, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(num_hidden * 4)
        self.conv4 = nn.Conv2d(num_hidden * 4, num_hidden * 8, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(num_hidden * 8)
        self.conv5 = nn.Conv2d(num_hidden * 8, num_hidden * 16, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(num_hidden * 16)
        self.conv6 = nn.Conv2d(num_hidden * 16, num_hidden * 16, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(num_hidden * 16)
        self.conv7 = nn.Conv2d(num_hidden * 24, num_hidden * 16, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(num_hidden * 16)
        self.conv8 = nn.Conv2d(num_hidden * 20, num_hidden * 16, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(num_hidden * 16)
        self.conv9 = nn.Conv2d(num_hidden * 18, num_hidden * 16, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(num_hidden * 16)
        self.conv10 = nn.Conv2d(num_hidden * 17, num_hidden * 16, 3, 1, 1)
        self.bn10 = nn.BatchNorm2d(num_hidden * 16)
        self.conv = nn.Conv2d(num_hidden * 16, num_class, 3, 1, 1)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

        self.im_channel = im_channel
        self.dp_channel = dp_channel
        self.n_class = num_class

    def forward(self, images, depths, flows):
        for i in range(len(images)):
            x0 = torch.cat((images[i], depths[i]), 1)
            x0 = self.bn0(x0)
            x1 = F.relu(self.bn1(self.conv1(x0)))
            x2 = self.maxpool(x1)
            x2 = F.relu(self.bn2(self.conv2(x2)))
            x3 = self.maxpool(x2)
            x3 = F.relu(self.bn3(self.conv3(x3)))
            x4 = self.maxpool(x3)
            x4 = F.relu(self.bn4(self.conv4(x4)))
            x5 = self.maxpool(x4)
            x5 = F.relu(self.bn5(self.conv5(x5)))
            # x6 = self.maxpool(x5)
            x6 = x5
            x6 = F.relu(self.bn6(self.conv6(x6)))
            # pred = self.conv(x6)
            x7 = F.interpolate(x6, scale_factor=2, mode='bilinear')
            x7 = torch.cat((x7, x4), 1)
            x7 = F.relu(self.bn7(self.conv7(x7)))
            x8 = F.interpolate(x7, scale_factor=2, mode='bilinear')
            x8 = torch.cat((x8, x3), 1)
            x8 = F.relu(self.bn8(self.conv8(x8)))
            x9 = F.interpolate(x8, scale_factor=2, mode='bilinear')
            x9 = torch.cat((x9, x2), 1)
            x9 = F.relu(self.bn9(self.conv9(x9)))
            x10 = F.interpolate(x9, scale_factor=2, mode='bilinear')
            x10 = torch.cat((x10, x1), 1)
            x10 = F.relu(self.bn10(self.conv10(x10)))
            pred = self.conv(x10)
        return pred
