import torch
import torch.nn as nn
import torch.nn.functional as F


class FocusELoss(nn.Module):
    def __init__(self):
        super(FocusELoss, self).__init__()


    def forward(self, pos_score, neg_score):
        neg_logsumexp = torch.logsumexp(neg_score, dim=1)
        loss = -F.logsigmoid(pos_score - neg_logsumexp)
        return loss.sum()

