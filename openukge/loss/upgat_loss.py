import torch
import torch.nn as nn


class UPGATLoss(nn.Module):
    def __init__(self):
        super(UPGATLoss, self).__init__()
        self.criterion = nn.MSELoss()


    def forward(self, pos_score, neg_score, pro, pseudo_score=None, pseudo_pro=None):
        """Calculating the loss score of UPGAT model.

        Args:
            pos_score: The score of positive samples.
            neg_score: The score of negative samples.
            pos_sample: The positive samples.
            pseudo_score: The score of pseudo samples, defaults to None.
            pseudo_sample: The pseudo samples, defaults to None.

        Returns:
            loss: The training loss for back propagation.
        """
        loss_1 = self.criterion(pos_score, pro)
        loss_2 = torch.mean(neg_score ** 2)
        loss = loss_1 + loss_2
        if pseudo_score is not None:
            loss_3 = self.criterion(pseudo_score, pseudo_pro)
            loss = loss + loss_3

        return loss



