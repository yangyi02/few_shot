import torch
import torch.nn.functional as F
import torch.nn as nn


# Recommend
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


# this may be unstable sometimes.Notice set the size_average
def CrossEntropy2d(input, target, weight=None, size_average=False):
    # input:(n, c, h, w) target:(n, h, w)
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

    target_mask = target >= 0
    target = target[target_mask]
    #loss = F.nll_loss(F.log_softmax(input), target, weight=weight, size_average=False)
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= target_mask.sum().data[0]

    return loss


# Define the Binary Cross Entropy loss function
class BCELoss2d(nn.Module):
    """
    Code taken from:
    https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208
    """

    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


# Define the Mean Square Error loss function with mask
class MSELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=False):
        super(MSELoss2d, self).__init__()
        self.mse_loss = nn.MSELoss(weight, size_average)

    def forward(self, inputs, targets, mask):
        """
            inputs: n * c * h * w
            targets: n * c * h * w
            mask: n * 1 * h * w
        """
        n, c, h, w = inputs.size()
        mask = mask.repeat(1, c, 1, 1)
        num_pos = mask.sum()
        inputs = inputs * mask
        targets = targets * mask
        return self.mse_loss(inputs, targets).sum() / num_pos
