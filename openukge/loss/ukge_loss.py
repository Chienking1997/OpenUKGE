import torch
import torch.nn as nn


class UKGELoss(nn.Module):
    def __init__(self):
        super(UKGELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pos_score, neg_score, pro):
        loss_1 = self.criterion(pos_score, pro)  # l_pos
        loss_2 = torch.mean(torch.square(neg_score))  # l_neg

        loss = loss_1 + loss_2
        return loss

