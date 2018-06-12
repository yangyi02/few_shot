import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftAttn3DNet(nn.Module):
    def __init__(self, attention_size, im_channel, dp_channel, num_class):
        super(SoftAttn3DNet, self).__init__()
        num_hidden = 24
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
        self.conv6 = nn.Conv2d(num_hidden * 16, num_hidden * 32, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(num_hidden * 32)

        self.conv_attn = nn.Conv2d(num_hidden * 32, 1, 3, 1, 1)

        self.fc = nn.Linear(num_hidden * 32, num_class)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

        self.attn_size = attention_size
        self.im_channel = im_channel
        self.dp_channel = dp_channel
        self.n_class = num_class

    def soft_attn(self, attn_feat):
        attn_score = []
        for i in range(len(attn_feat)):
            x = self.conv_attn(attn_feat[i])
            attn_score.append(x.view(x.size(0), -1))
        x_a = torch.cat(attn_score, 1)
        attn = F.softmax(x_a, 1)
        return attn

    def attn_pool(self, im_feat, attn):
        for i in range(len(im_feat)):
            im_feat[i] = im_feat[i].view(im_feat[i].size(0), im_feat[i].size(1), -1)
        im_feat = torch.cat(im_feat, 2)
        final_feat = torch.sum(im_feat * attn[:, None, :], 2)
        return final_feat

    def forward(self, images, depths, direction):
        im_feat = []
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
            x6 = self.maxpool(x5)
            x6 = F.relu(self.bn6(self.conv6(x6)))
            im_feat.append(x6)
        attn = self.soft_attn(im_feat)
        x = self.attn_pool(im_feat, attn)
        x = x.view(x.size(0), -1)
        pred = self.fc(x)
        return pred, attn
