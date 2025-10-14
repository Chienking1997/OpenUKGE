import torch
import torch.nn as nn
import torch.nn.functional as F


class GtransELoss(nn.Module):
    def __init__(self, alpha, margin):
        super(GtransELoss, self).__init__()
        self.alpha = alpha
        self.margin = margin


    def forward(self, pos_score, neg_score, pro):
        modified_margin = self.margin * (pro ** self.alpha)

        diff = neg_score.view(-1, pos_score.shape[0]) - pos_score
        loss = F.relu(modified_margin + diff)

        return loss.mean()