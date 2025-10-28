import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GMUCLoss(nn.Module):
    """
    GMUC Loss Function

    This loss combines three components:
        1. Rank loss: margin-based ranking loss between positive and negative samples.
        2. MSE loss: measures the difference between predicted and true confidence values.
        3. AE loss: reconstruction loss from the autoencoder part of the model.

    Args:
        args: An argument/config object containing hyperparameters such as margin and loss weights.
        model: The main model using this loss (for consistent device handling, etc.).
    """

    def __init__(self, num_neg, mae_weight, margin, if_conf, rank_weight, ae_weight):
        super().__init__()
        self.num_neg = num_neg
        self.mae_weight = mae_weight
        self.margin = margin
        self.if_conf = if_conf
        self.rank_weight = rank_weight
        self.ae_weight = ae_weight


    def forward(
        self,
        query_scores: Tensor,
        query_scores_var: Tensor,
        query_ae_loss: Tensor,
        false_scores: Tensor,
        query_confidence: Tensor,
    ) -> Tensor:
        """
        Compute the GMUC total loss.

        Args:
            query_scores: Scores for positive (true) triples, shape [batch_size].
            query_scores_var: Predicted confidence values, shape [batch_size].
            query_ae_loss: Autoencoder reconstruction loss (scalar).
            false_scores: Scores for negative (false) triples, shape [batch_size * num_neg].
            query_confidence: Ground-truth confidence values, shape [batch_size].

        Returns:
            total_loss: Combined scalar loss for backpropagation.
        """
        device = query_scores.device

        # === 1. Handle negative samples ===
        # If multiple negative samples exist per positive triple, average them.
        if self.num_neg != 1:
            false_scores = false_scores.view(query_scores.size(0), self.num_neg)
            false_scores = false_scores.mean(dim=1)

        # === 2. Confidence mask ===
        # For samples with confidence < 0.5, set their contribution to zero in rank loss.
        zero_mask = torch.zeros_like(query_confidence, device=query_scores.device)
        query_conf_mask = torch.where(query_confidence < 0.5, zero_mask, query_confidence)

        # === 3. MSE loss (confidence regression) ===
        # Measures how close the predicted confidence (query_scores_var) is to the true confidence.
        mse_loss = F.mse_loss(query_scores_var, query_confidence, reduction="sum")
        mse_loss = self.mae_weight * mse_loss  # "mae_weight" is a hyperparameter name; kept for compatibility.

        # === 4. Rank loss (margin-based) ===
        # Encourages the score of a true triple to be higher than that of a false one by at least `margin`.
        rank_diff = self.margin - (query_scores - false_scores)
        rank_loss = F.relu(rank_diff)  # Only penalize when margin is violated
        if self.if_conf:
            # Weight rank loss by confidence (higher confidence = stronger contribution)
            rank_loss = (rank_loss * query_conf_mask).mean()
        else:
            rank_loss = rank_loss.mean()
        rank_loss = self.rank_weight * rank_loss

        # === 5. AE loss (autoencoder reconstruction) ===
        # Regularization term encouraging consistent latent representations.
        ae_loss = self.ae_weight * query_ae_loss

        # === 6. Total loss ===
        total_loss = rank_loss + mse_loss + ae_loss

        return total_loss
