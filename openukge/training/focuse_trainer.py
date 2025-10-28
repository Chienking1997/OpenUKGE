import torch
from tqdm.auto import tqdm, trange

# === Local project imports ===
from ..utils import prepare_device
from ..evaluation import (
    high_link_predict,
    weight_link_predict,
    val_link_predict,
    val_high_link_predict,
    val_weight_link_predict,
    mean_ndcg,
)


class FocusETrainer:
    """
    Trainer for FocusE Knowledge Graph Embeddings with weighted scoring.

    Handles training, validation, and testing with uncertainty and weighted link predictions.

    Args:
        data: Dataset wrapper providing train/val/test dataloaders.
        model: PyTorch model instance.
        loss: Loss function callable.
        opt: Optimizer/scheduler builder (must implement `build_optimizer` and `build_scheduler`).
        early_stop: EarlyStopping instance.
        save_path (str): Path to save the best model checkpoint.
        config (dict, optional): Optional configuration dictionary.
    """

    def __init__(self, data=None, model=None, loss=None, opt=None,
                 early_stop=None, save_path=None, config=None):
        # === Data and device setup ===
        self.data = data
        self.train_loader = data.train_dataloader()
        self.valid_loader = data.val_dataloader()
        self.test_data = data.test_dataloader()
        self.device = prepare_device()
        self.num_neg = data.sampler.num_neg  # Number of negative samples for weighted scoring

        # === Model & components ===
        self.model = model.to(self.device)
        self.loss_fn = loss
        self.early_stop = early_stop
        self.save_path = save_path
        self.config = config

        # === Optimizer & scheduler ===
        self.optimizer = opt.build_optimizer(model=self.model)
        self.scheduler = opt.build_scheduler(optimizer=self.optimizer)

    # --------------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------------
    def fit(self, epochs: int, eval_freq: int = 5):
        """
        Run the training loop with periodic validation and early stopping.

        Args:
            epochs (int): Total number of epochs.
            eval_freq (int): Frequency of validation in epochs.
        """
        epochs_bar = trange(1, epochs + 1, desc="Training", leave=True)

        for epoch in epochs_bar:
            avg_loss = self._train_epoch(epoch - 1, epochs - 1)
            epochs_bar.set_description(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

            # === Periodic validation ===
            if self.early_stop and epoch % eval_freq == 0:
                monitor_metric = self.early_stop.get_monitor()
                mode = self.early_stop.get_monitor_mode()
                val_result = self._valid_epoch(monitor_metric, mode)

                stop_flag, best_val = self.early_stop(
                    val_result, model=self.model, save_path=self.save_path
                )
                epochs_bar.set_postfix(val=f"{val_result:>8.4f}", best=f"{best_val:>8.4f}")

                if stop_flag:
                    print(f"🛑 Early stopping triggered at epoch {epoch}.")
                    break

        print(f"✅ Training completed. Best model saved at: {self.save_path}")

    # --------------------------------------------------------------------------
    # Train One Epoch
    # --------------------------------------------------------------------------
    def _train_epoch(self, epoch: int, total_epochs: int) -> float:
        """
        Run one training epoch and return the average loss.

        Args:
            epoch (int): Current epoch index (0-based).
            total_epochs (int): Total number of epochs.

        Returns:
            float: Average loss over this epoch.
        """
        self.model.train()
        self.model.adjust_parameters(epoch, total_epochs)  # Adjust dynamic parameters
        total_loss = 0.0
        train_bar = tqdm(self.train_loader, desc="Train", leave=False)

        for batch in train_bar:
            # --- Move tensors to device ---
            pos_sample = batch["positive_sample"].to(self.device)
            neg_sample = batch["negative_sample"].to(self.device)
            pro = batch["probabilities"].to(self.device)
            pro_expand = pro.view(-1, 1).expand(-1, 2 * self.num_neg)  # Expand for negative sampling

            # --- Forward + Backward ---
            self.optimizer.zero_grad()
            pos_score = self.model.forward_weighted(pos_sample, pro)
            neg_score = self.model.forward_weighted(neg_sample, pro, pro_expand)
            regularization1 = self.model.regularization(pos_sample)
            regularization2 = self.model.regularization2()
            loss = self.loss_fn(pos_score, neg_score) + regularization1 + regularization2
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / len(train_bar)

    # --------------------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------------------
    def _valid_epoch(self, monitor: str, mode: str) -> float:
        """
        Run a validation epoch and return the metric specified by `monitor`.

        Args:
            monitor (str): Metric to monitor, one of {"umrr", "umr", "mrr", "mr", "wmrr", "wmr"}.
            mode (str): Prediction mode, e.g., "head", "tail", "average".

        Returns:
            float: Validation score for the monitored metric.
        """
        self.model.eval()
        with torch.no_grad():
            val_triples = self.valid_loader["triples"].to(self.device)
            val_probs = self.valid_loader["probabilities"].to(self.device)

            # --- Uncertainty-aware link prediction ---
            if monitor in {"umrr", "umr"}:
                result = val_link_predict(val_triples, val_probs, self.model,
                                          self.valid_loader, prediction_mode=mode)
                mr_raw, mrr_raw, *_ = result
                return mrr_raw.item() if monitor == "umrr" else mr_raw.item()

            # --- High-confidence link prediction ---
            elif monitor in {"mrr", "mr"}:
                high_triples = self.valid_loader["high_triples"].to(self.device)
                high_probs = self.valid_loader["high_probabilities"].to(self.device)
                result = val_high_link_predict(high_triples, high_probs, self.model,
                                               self.valid_loader, prediction_mode=mode)
                mr_raw, mrr_raw, *_ = result
                return mrr_raw.item() if monitor == "mrr" else mr_raw.item()

            # --- Weighted link prediction ---
            elif monitor in {"wmrr", "wmr"}:
                result = val_weight_link_predict(val_triples, val_probs, self.model,
                                                 self.valid_loader, prediction_mode=mode)
                mr_raw, mrr_raw, *_ = result
                return mrr_raw.item() if monitor == "wmrr" else mr_raw.item()

            else:
                raise ValueError(f"Unsupported monitor type: {monitor}")

    # --------------------------------------------------------------------------
    # Testing
    # --------------------------------------------------------------------------
    def test(self):
        """
        Evaluate the best saved model on the test set.
        Includes weighted and high-confidence link predictions and ranking metrics.
        """
        self.model.eval()
        with torch.no_grad():
            # --- Load best model ---
            self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))

            test_triples = self.test_data["triples"].to(self.device)
            test_probs = self.test_data["probabilities"].to(self.device)
            test_high_triples = self.test_data["high_triples"].to(self.device)
            test_high_probs = self.test_data["high_probabilities"].to(self.device)

            # --- Link prediction ---
            high_link_predict(test_high_triples, test_high_probs, self.model, self.test_data)
            weight_link_predict(test_triples, test_probs, self.model, self.test_data)

            # --- Ranking metrics ---
            linear_ndcg, exp_ndcg = mean_ndcg(self.test_data["hr_map"], self.model, self.device)
            print(f"nDCG_linear: {linear_ndcg:.4f} | nDCG_exp: {exp_ndcg:.4f}")

            print("✅ Test completed.")
