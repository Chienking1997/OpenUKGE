import torch
import torch.nn as nn


class PASSLEAFLoss(nn.Module):
    def __init__(self):
        super(PASSLEAFLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pos_score, semi_score, pro, semi_pro):
        loss_1 = self.criterion(pos_score, pro)  # l_pos
        loss_2 = self.criterion(semi_score, semi_pro)  # l_neg

        loss = loss_1 + loss_2
        return loss
