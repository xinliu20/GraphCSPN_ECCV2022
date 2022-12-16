import torch
import torch.nn as nn


class L1Loss(nn.Module):
    def __init__(self, max_depth=10.0):
        super(L1Loss, self).__init__()

        self.max_depth = max_depth
        self.t_valid = 0.00001

    def forward(self, pred, gt):
        gt = torch.clamp(gt, min=0, max=self.max_depth)
        pred = torch.clamp(pred, min=0, max=self.max_depth)

        mask = (gt > self.t_valid).type_as(pred).detach()

        d = torch.abs(pred - gt) * mask

        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask, dim=[1, 2, 3])

        loss = d / (num_valid + 1e-8)

        loss = loss.mean()

        return loss
