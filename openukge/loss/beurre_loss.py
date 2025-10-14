import torch
import torch.nn as nn


class BEUrRELoss(nn.Module):
    def __init__(self, rule_config=None, regularization=None):
        super(BEUrRELoss, self).__init__()
        self.rule_config = rule_config
        self.regularization = regularization
        self.criterion = nn.MSELoss()

    def forward(self, pos_sample, neg_sample, pro, model, device):
        neg_ratio = 1
        pos_score = model(pos_sample)
        neg_score = model(neg_sample)
        pos_loss = self.criterion(pos_score, pro)  # l_pos
        neg_loss = torch.mean(torch.square(neg_score))  # l_neg
        main_loss = pos_loss + neg_ratio * neg_loss
        if self.rule_config is not None:
            logic_loss = self.get_logic_loss(model, pos_sample, device)
        else:
            logic_loss = 0
        if self.regularization is not None:
            l2_reg = self.l2_regularization(model, pos_sample, self.regularization, device)
        else:
            l2_reg = 0
        loss = main_loss + l2_reg + logic_loss
        return loss

    def get_logic_loss(self, model, triples, device):
        # transitive rule loss regularization
        transitive_coff = torch.tensor(self.regularization['transitive']).to(device)
        if transitive_coff > 0:
            transitive_rule_reg = transitive_coff * model.transitive_rule_loss(triples, self.rule_config)
        else:
            transitive_rule_reg = 0

        # composite rule loss regularization
        composite_coff = torch.tensor(self.regularization['composite']).to(device)
        if composite_coff > 0:
            composition_rule_reg = composite_coff * model.composition_rule_loss(triples, self.rule_config, device)
        else:
            composition_rule_reg = 0

        return (transitive_rule_reg + composition_rule_reg) / len(triples)

    @staticmethod
    def l2_regularization(model, triples, regularization, device):
        """
        Computes the L2 regularization loss for the model.

        Args:


        Returns:
            L2_reg: The L2 regularization loss.
        """

        # regularization on delta
        if regularization['delta'] > 0 or regularization['min'] > 0:
            delta_coff, min_coff = (torch.tensor(regularization['delta']).to(device),
                                    torch.tensor(regularization['min']).to(device))
            delta_reg1 = delta_coff * torch.norm(model.delta_embedding(triples[:, 0]), dim=1).mean()
            delta_reg2 = delta_coff * torch.norm(model.delta_embedding(triples[:, 2]), dim=1).mean()

            min_reg1 = min_coff * torch.norm(model.min_embedding(triples[:, 0]), dim=1).mean()
            min_reg2 = min_coff * torch.norm(model.min_embedding(triples[:, 2]), dim=1).mean()
        else:
            delta_reg1, delta_reg2, min_reg1, min_reg2 = 0, 0, 0, 0

        if regularization['rel_trans'] > 0 or regularization['rel_scale'] > 0:
            rel_trans_coff = torch.tensor(regularization['rel_trans']).to(device)
            rel_trans_reg = rel_trans_coff * (
                    torch.norm(model.rel_trans_for_head(triples[:, 1]), dim=1).mean() +
                    torch.norm(model.rel_trans_for_tail(triples[:, 1]), dim=1).mean()
            )

            rel_scale_coff = torch.tensor(regularization['rel_scale']).to(device)
            rel_scale_reg = rel_scale_coff * (
                    torch.norm(model.rel_scale_for_head(triples[:, 1]), dim=1).mean() +
                    torch.norm(model.rel_scale_for_tail(triples[:, 1]), dim=1).mean()

            )
        else:
            rel_trans_reg, rel_scale_reg = 0, 0

        l2_reg = delta_reg1 + delta_reg2 + min_reg1 + min_reg2 + rel_trans_reg + rel_scale_reg

        return l2_reg
