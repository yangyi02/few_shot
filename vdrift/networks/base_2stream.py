import torch
import torch.nn as nn
import torch.nn.functional as F


class Base2StreamNet(nn.Module):
    def __init__(self, im_channel, dp_channel, num_class):
        super(Base2StreamNet, self).__init__()
        num_hidden_i = 8
        self.bn_i0 = nn.BatchNorm2d(im_channel)
        self.conv_i1 = nn.Conv2d(im_channel, num_hidden_i, 3, 1, 1)
        self.bn_i1 = nn.BatchNorm2d(num_hidden_i)
        self.conv_i2 = nn.Conv2d(num_hidden_i, num_hidden_i * 2, 3, 1, 1)
        self.bn_i2 = nn.BatchNorm2d(num_hidden_i * 2)
        self.conv_i3 = nn.Conv2d(num_hidden_i * 2, num_hidden_i * 4, 3, 1, 1)
        self.bn_i3 = nn.BatchNorm2d(num_hidden_i * 4)
        self.conv_i4 = nn.Conv2d(num_hidden_i * 4, num_hidden_i * 8, 3, 1, 1)
        self.bn_i4 = nn.BatchNorm2d(num_hidden_i * 8)
        self.conv_i5 = nn.Conv2d(num_hidden_i * 8, num_hidden_i * 16, 3, 1, 1)
        self.bn_i5 = nn.BatchNorm2d(num_hidden_i * 16)
        self.conv_i6 = nn.Conv2d(num_hidden_i * 16, num_hidden_i * 32, 3, 1, 1)
        self.bn_i6 = nn.BatchNorm2d(num_hidden_i * 32)
        self.conv_i7 = nn.Conv2d(num_hidden * 24, num_hidden * 16, 3, 1, 1)
        self.bn_i7 = nn.BatchNorm2d(num_hidden * 16)
        self.conv_i8 = nn.Conv2d(num_hidden * 20, num_hidden * 16, 3, 1, 1)
        self.bn_i8 = nn.BatchNorm2d(num_hidden * 16)
        self.conv_i9 = nn.Conv2d(num_hidden * 18, num_hidden * 16, 3, 1, 1)
        self.bn_i9 = nn.BatchNorm2d(num_hidden * 16)
        self.conv_i10 = nn.Conv2d(num_hidden * 17, num_hidden * 16, 3, 1, 1)
        self.bn_i10 = nn.BatchNorm2d(num_hidden * 16)
        # self.conv_i = nn.Conv2d(num_hidden_i * 32, num_class, 3, 1, 1)

        num_hidden_d = 8
        self.bn_d0 = nn.BatchNorm2d(dp_channel)
        self.conv_d1 = nn.Conv2d(dp_channel, num_hidden_d, 3, 1, 1)
        self.bn_d1 = nn.BatchNorm2d(num_hidden_d)
        self.conv_d2 = nn.Conv2d(num_hidden_d, num_hidden_d * 2, 3, 1, 1)
        self.bn_d2 = nn.BatchNorm2d(num_hidden_d * 2)
        self.conv_d3 = nn.Conv2d(num_hidden_d * 2, num_hidden_d * 4, 3, 1, 1)
        self.bn_d3 = nn.BatchNorm2d(num_hidden_d * 4)
        self.conv_d4 = nn.Conv2d(num_hidden_d * 4, num_hidden_d * 8, 3, 1, 1)
        self.bn_d4 = nn.BatchNorm2d(num_hidden_d * 8)
        self.conv_d5 = nn.Conv2d(num_hidden_d * 8, num_hidden_d * 16, 3, 1, 1)
        self.bn_d5 = nn.BatchNorm2d(num_hidden_d * 16)
        self.conv_d6 = nn.Conv2d(num_hidden_d * 16, num_hidden_d * 32, 3, 1, 1)
        self.bn_d6 = nn.BatchNorm2d(num_hidden_d * 32)
        self.conv_d7 = nn.Conv2d(num_hidden * 24, num_hidden * 16, 3, 1, 1)
        self.bn_d7 = nn.BatchNorm2d(num_hidden * 16)
        self.conv_d8 = nn.Conv2d(num_hidden * 20, num_hidden * 16, 3, 1, 1)
        self.bn_d8 = nn.BatchNorm2d(num_hidden * 16)
        self.conv_d9 = nn.Conv2d(num_hidden * 18, num_hidden * 16, 3, 1, 1)
        self.bn_d9 = nn.BatchNorm2d(num_hidden * 16)
        self.conv_d10 = nn.Conv2d(num_hidden * 17, num_hidden * 16, 3, 1, 1)
        self.bn_d10 = nn.BatchNorm2d(num_hidden * 16)
        # self.conv_d = nn.Conv2d(num_hidden_d * 32, num_class, 3, 1, 1)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.dropout = nn.Dropout2d()
        self.conv = nn.Conv2d(num_hidden_i * 16 + num_hidden_d * 16, num_class, 3, 1, 1)

        self.im_channel = im_channel
        self.dp_channel = dp_channel
        self.n_class = num_class

    def forward(self, images, depths, flows):
        for i in range(len(images)):
            x0 = images[i]
            x0 = self.bn_i0(x0)
            x1 = F.relu(self.bn_i1(self.conv_i1(x0)))
            x2 = self.maxpool(x1)
            x2 = F.relu(self.bn_i2(self.conv_i2(x2)))
            x3 = self.maxpool(x2)
            x3 = F.relu(self.bn_i3(self.conv_i3(x3)))
            x4 = self.maxpool(x3)
            x4 = F.relu(self.bn_i4(self.conv_i4(x4)))
            x5 = self.maxpool(x4)
            x5 = F.relu(self.bn_i5(self.conv_i5(x5)))
            # x6 = self.maxpool(x5)
            x6 = x5
            x6 = F.relu(self.bn_i6(self.conv_i6(x6)))
            x7 = F.interpolate(x6, scale_factor=2, mode='bilinear')
            x7 = torch.cat((x7, x4), 1)
            x7 = F.relu(self.bn_i7(self.conv_i7(x7)))
            x8 = F.interpolate(x7, scale_factor=2, mode='bilinear')
            x8 = torch.cat((x8, x3), 1)
            x8 = F.relu(self.bn_i8(self.conv_i8(x8)))
            x9 = F.interpolate(x8, scale_factor=2, mode='bilinear')
            x9 = torch.cat((x9, x2), 1)
            x9 = F.relu(self.bn_i9(self.conv_i9(x9)))
            x10 = F.interpolate(x9, scale_factor=2, mode='bilinear')
            x10 = torch.cat((x10, x1), 1)
            x10 = F.relu(self.bn_i10(self.conv_i10(x10)))
            # pred_i = self.conv_i(x10)
            im_feat = x10

        for i in range(len(depths)):
            x0 = depths[i]
            x0 = self.bn_d0(x0)
            x1 = F.relu(self.bn_d1(self.conv_d1(x0)))
            x2 = self.maxpool(x1)
            x2 = F.relu(self.bn_d2(self.conv_d2(x2)))
            x3 = self.maxpool(x2)
            x3 = F.relu(self.bn_d3(self.conv_d3(x3)))
            x4 = self.maxpool(x3)
            x4 = F.relu(self.bn_d4(self.conv_d4(x4)))
            x5 = self.maxpool(x4)
            x5 = F.relu(self.bn_d5(self.conv_d5(x5)))
            # x6 = self.maxpool(x5)
            x6 = x5
            x6 = F.relu(self.bn_d6(self.conv_d6(x6)))
            x7 = F.interpolate(x6, scale_factor=2, mode='bilinear')
            x7 = torch.cat((x7, x4), 1)
            x7 = F.relu(self.bn_d7(self.conv_d7(x7)))
            x8 = F.interpolate(x7, scale_factor=2, mode='bilinear')
            x8 = torch.cat((x8, x3), 1)
            x8 = F.relu(self.bn_d8(self.conv_d8(x8)))
            x9 = F.interpolate(x8, scale_factor=2, mode='bilinear')
            x9 = torch.cat((x9, x2), 1)
            x9 = F.relu(self.bn_d9(self.conv_d9(x9)))
            x10 = F.interpolate(x9, scale_factor=2, mode='bilinear')
            x10 = torch.cat((x10, x1), 1)
            x10 = F.relu(self.bn_d10(self.conv_d10(x10)))
            # pred_d = self.conv_d(x10)
            dp_feat = x10

        for i in range(len(images)):
            x = torch.cat((im_feat, dp_feat), 1)
            # x = self.dropout(x)
            pred = self.conv(x)
        return pred
