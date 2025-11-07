import torch
from tqdm.auto import tqdm, trange
from ..utils import prepare_device
from ..evaluation import conf_predict, print_results
from ..evaluation import link_predict, high_link_predict, weight_link_predict
from ..evaluation import val_link_predict, val_high_link_predict, val_weight_link_predict
from ..evaluation import mean_ndcg, ece_t


import torch
from tqdm.auto import tqdm, trange

# === Local project imports ===
from ..utils import prepare_device
from ..evaluation import (
    conf_predict,
    print_results,
    link_predict,
    high_link_predict,
    weight_link_predict,
    val_link_predict,
    val_high_link_predict,
    val_weight_link_predict,
    mean_ndcg,
    ece_t,
)


class BEUrRETrainer:
    """
    BEUrRE Trainer for Knowledge Graph Embeddings with uncertainty handling.

    Handles model training, validation, and testing across multiple evaluation modes.

    Args:
        data: Dataset wrapper that provides train/val/test dataloaders.
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

        # === Model & components ===
        self.model = model.to(self.device)
        self.loss_fn = loss
        self.early_stop = early_stop
        self.save_path = save_path
        self.config = config  # Reserved for extended configs

        # === Optimizer & scheduler ===
        self.optimizer = opt.build_optimizer(model=self.model)
        self.scheduler = opt.build_scheduler(optimizer=self.optimizer)

    # --------------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------------
    def fit(self, epochs: int, eval_freq: int = 5):
        """
        Run the main training loop with periodic validation and early stopping.

        Args:
            epochs (int): Total number of training epochs.
            eval_freq (int): Frequency (in epochs) of validation.
        """
        epochs_bar = trange(1, epochs + 1, desc="Training", leave=True)

        for epoch in epochs_bar:
            avg_loss = self._train_epoch()
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
    def _train_epoch(self) -> float:
        """Run one training epoch and return the average loss."""
        self.model.train()
        total_loss = 0.0
        train_bar = tqdm(self.train_loader, desc="Train", leave=False)

        for batch in train_bar:
            # --- Move tensors to device ---
            pos_sample = batch["positive_sample"].to(self.device)
            neg_sample = batch["negative_sample"].to(self.device)
            pro = batch["probabilities"].to(self.device)

            # --- Forward + Backward ---
            self.optimizer.zero_grad()
            loss = self.loss_fn(pos_sample, neg_sample, pro, self.model, self.device)
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
            monitor (str): Metric to monitor.
            mode (str): Prediction mode, e.g., "head", "tail", "average".

        Returns:
            float: Validation score for the monitored metric.
        """
        self.model.eval()
        with torch.no_grad():
            val_triples = self.valid_loader["triples"].to(self.device)
            val_probs = self.valid_loader["probabilities"].to(self.device)

            # --- Confidence metrics ---
            if monitor in {"mse", "mae"}:
                mse, mae = conf_predict(val_triples, val_probs, self.model)
                return {"mse": mse.item(), "mae": mae.item()}[monitor]

            # --- Uncertainty-aware link prediction ---
            elif monitor in {"umrr", "umr"}:
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

            # --- Ranking quality (NDCG) ---
            elif monitor == "ndcg":
                linear_ndcg, _ = mean_ndcg(self.valid_loader["hr_map"], self.model, self.device)
                return linear_ndcg

            else:
                raise ValueError(f"Unsupported monitor type: {monitor}")

    # --------------------------------------------------------------------------
    # Testing
    # --------------------------------------------------------------------------
    def test(self):
        """
        Evaluate the best saved model on the test set.
        Includes confidence, link prediction, ranking, and calibration metrics.
        """
        self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))
        self.model.eval()
        with torch.no_grad():
            # --- Load best model --- 
            test_triples = self.test_data["triples"].to(self.device)
            test_probs = self.test_data["probabilities"].to(self.device)
            test_high_triples = self.test_data["high_triples"].to(self.device)
            test_high_probs = self.test_data["high_probabilities"].to(self.device)
            test_neg = self.test_data["test_neg"].to(self.device)
            test_neg_pro = self.test_data["test_neg_pro"].to(self.device)

            # --- Confidence prediction ---
            mse, mae = conf_predict(test_triples, test_probs, self.model)
            print("Only Test Data")
            print_results(mse, mae)

            mse_neg, mae_neg = conf_predict(test_neg, test_neg_pro, self.model)
            print("Test and Negative Data")
            print_results(mse_neg, mae_neg)

            # --- Link prediction (three types) ---
            link_predict(test_triples, test_probs, self.model, self.test_data)
            high_link_predict(test_high_triples, test_high_probs, self.model, self.test_data)
            weight_link_predict(test_triples, test_probs, self.model, self.test_data)

            # --- Ranking metrics ---
            linear_ndcg, exp_ndcg = mean_ndcg(self.test_data["hr_map"], self.model, self.device)
            print(f"nDCG_linear: {linear_ndcg:.4f} | nDCG_exp: {exp_ndcg:.4f}")

            # --- Calibration (ECE) ---
            ece = ece_t(test_neg, test_neg_pro, self.model)
            print(f"ECE with neg: {ece:.4f}")

            ece = ece_t(test_triples, test_probs, self.model)
            print(f"ECE: {ece:.4f}")

            print("✅ Test completed.")

