import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HardDirectNet(nn.Module):
    def __init__(self, im_size, im_channel, direction_dim, num_class):
        super(HardDirectNet, self).__init__()
        num_hidden = 24
        self.bn0 = nn.BatchNorm2d(im_channel)
        self.conv1 = nn.Conv2d(im_channel, num_hidden, 3, 1, 1)
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
        self.adapt_avgpool = nn.AdaptiveAvgPool2d(1)

        self.bn_di0 = nn.BatchNorm1d(direction_dim)
        self.fc_box = nn.Linear(direction_dim, 4)

        self.fc = nn.Linear(num_hidden * 32, num_class)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

        self.im_size = im_size
        self.im_channel = im_channel
        self.direction_dim = direction_dim
        self.n_class = num_class

    def generate_grid(self, batch_size):
        grid = np.mgrid[0:self.im_size, 0:self.im_size]
        grid = np.tile(grid, [batch_size, 1, 1, 1])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        grid = torch.from_numpy(grid).float().to(device)
        grid.requires_grad = False
        return grid

    def stn(self, direction):
        di0 = self.bn_di0(direction)
        box = self.fc_box(di0)
        return box

    def crop_image(self, orig_im, box):
        # Adjust box w.r.t. the coordinate system [-1, 1] for grid_sample usage
        box = F.sigmoid(box)
        boxes = torch.zeros_like(box)
        boxes[:, 0] = box[:, 0] - box[:, 2] / 2.0
        boxes[:, 1] = box[:, 1] - box[:, 3] / 2.0
        boxes[:, 2] = box[:, 0] + box[:, 2] / 2.0
        boxes[:, 3] = box[:, 1] + box[:, 3] / 2.0
        boxes = F.relu(boxes)
        boxes = 1 - F.relu(1 - boxes)
        bb = boxes * 2 - 1
        # Compute the sampling distance for every pixel in the box
        sample_dist_x = (bb[:, 2] - bb[:, 0]) / self.im_size
        sample_dist_y = (bb[:, 3] - bb[:, 1]) / self.im_size
        start_x = bb[:, 0]
        start_y = bb[:, 1]

        grid = self.generate_grid(bb.size(0))
        grid_x = grid[:, 1, :, :] * sample_dist_x[:, None, None] + start_x[:, None, None]
        grid_y = grid[:, 0, :, :] * sample_dist_y[:, None, None] + start_y[:, None, None]
        grid = torch.stack((grid_x, grid_y), 3)
        crop_im = F.grid_sample(orig_im, grid)
        return crop_im, boxes

    def forward(self, image, depth, box, direction):
        box = self.stn(direction)
        crop_im, box = self.crop_image(image, box)
        x0 = crop_im
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
        x = self.adapt_avgpool(x6)
        x = x.view(x.size(0), -1)
        pred = self.fc(x)
        return pred, box
