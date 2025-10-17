import torch
import torch.nn as nn


class FocusELoss(nn.Module):
    def __init__(self):
        super(FocusELoss, self).__init__()


    def forward(self, pos_score, neg_score):
        neg_score = torch.sum(torch.exp(neg_score).view(-1, pos_score.shape[0]), dim=0)
        loss = torch.log(torch.exp(pos_score) / (torch.exp(pos_score) + neg_score))
        loss = torch.sum(loss)
        return -loss

