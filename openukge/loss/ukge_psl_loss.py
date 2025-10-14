import torch
import torch.nn as nn


class UKGE_PSL_Loss(nn.Module):
    def __init__(self):
        super(UKGE_PSL_Loss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pos_score, neg_score, pro, psl_score, psl_pro):
        loss_1 = self.criterion(pos_score, pro)  # l_pos
        loss_2 = torch.mean(torch.square(neg_score))  # l_neg
        tmp = torch.clamp((psl_pro - psl_score), min=0)
        loss_3 = 0.2 * torch.mean(torch.square(tmp))
        loss = loss_1 + loss_2 + loss_3
        return loss

