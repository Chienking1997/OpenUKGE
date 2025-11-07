import torch
from tqdm.auto import tqdm, trange

# === Local imports ===
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
    ece_t
)


class UKGsETrainer:
    """
    Uncertain Knowledge Graph simple-but-effective Embedding Trainer

    Handles training, validation, and final testing of uncertain knowledge
    graph embedding models.

    Args:
        data: Dataset wrapper providing train/val/test dataloaders.
        model: PyTorch model instance.
        loss: Loss function callable.
        opt: Optimizer and scheduler constructor.
        early_stop: EarlyStopping instance.
        save_path (str): Path to save best model checkpoint.
        config (dict): Optional configuration parameters.
    """

    def __init__(self, data=None, model=None, loss=None, opt=None,
                 early_stop=None, save_path=None, config=None):

        # === Data and device setup ===
        self.data = data
        self.train_loader = data.train_dataloader()
        self.valid = data.val_dataloader()
        self.test_data = data.test_dataloader()
        self.device = prepare_device()

        # === Model & components ===
        self.model = model.to(self.device)
        self.loss_fn = loss
        self.early_stop = early_stop
        self.save_path = save_path
        self.config = config

        # === Optimizer & Scheduler ===
        self.optimizer = opt.build_optimizer(model=self.model)
        self.scheduler = opt.build_scheduler(optimizer=self.optimizer)

    # ----------------------------------------------------------------------
    # Training Loop
    # ----------------------------------------------------------------------
    def fit(self, epochs: int = None, eval_freq: int = 5):
        """
        Main training loop with periodic validation and early stopping.

        Args:
            epochs (int): Total training epochs.
            eval_freq (int): Validation frequency.
        """
        epochs_bar = trange(1, epochs + 1, desc="Training", leave=True)

        for epoch in epochs_bar:
            avg_loss = self._train_epoch()
            epochs_bar.set_description(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

            # --- Validate periodically ---
            if epoch % eval_freq == 0:
                monitor_metric = self.early_stop.get_monitor()
                mode = self.early_stop.get_monitor_mode()

                val_score = self._valid_epoch(monitor_metric, mode)

                stop_flag, best_val = self.early_stop(
                    val_score, model=self.model, save_path=self.save_path
                )

                epochs_bar.set_postfix(val=f"{val_score:.4f}", best=f"{best_val:.4f}")

                if stop_flag:
                    print(f"🛑 Early stopping triggered at epoch {epoch}")
                    break

        print(f"✅ Training completed. Best model saved at: {self.save_path}")

    # ----------------------------------------------------------------------
    # Train One Epoch
    # ----------------------------------------------------------------------
    def _train_epoch(self) -> float:
        """
        Run one training epoch and return average loss.
        """
        self.model.train()
        total_loss = 0.0
        train_bar = tqdm(self.train_loader, desc="Train", leave=False)

        for train_batch in train_bar:
            pos = train_batch["positive_sample"].to(self.device)
            neg = train_batch["negative_sample"].to(self.device)
            pro = train_batch["probabilities"].to(self.device)

            self.optimizer.zero_grad()

            pos_score = self.model(pos)
            neg_score = self.model(neg)

            # If model provides regularization term
            # if hasattr(self.model, "regularization"):
            #     reg = self.model.regularization(pos)
            # else:
            #     reg = 0.0

            loss = self.loss_fn(pos_score, neg_score, pro)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / len(train_bar)

    # ----------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------
    def _valid_epoch(self, monitor: str, mode: str) -> float:
        """
        Run validation and return value of metric specified by `monitor`.

        Valid metrics:
            {"mse", "mae", "umrr", "mrr", "wmrr"}
        """
        self.model.eval()
        with torch.no_grad():
            val_triples = self.valid["triples"].to(self.device)
            val_probs = self.valid["probabilities"].to(self.device)

            # --- (1) Confidence prediction metrics ---
            if monitor in {"mse", "mae"}:
                mse, mae = conf_predict(val_triples, val_probs, self.model)
                return {"mse": mse.item(), "mae": mae.item()}[monitor]

            # --- (2) Uncertainty-aware link prediction (UMRR/UMR) ---
            elif monitor in {"umrr", "umr"}:
                result = val_link_predict(val_triples, val_probs, self.model, self.valid, prediction_mode=mode)
                (
                    mr_raw, mrr_raw, hit1_raw, hit3_raw, hit10_raw,
                    mr_filter, mrr_filter, hit1_filter, hit3_filter, hit10_filter
                ) = result
                return mrr_raw.item() if monitor == "umrr" else mr_raw.item()

            # --- (3) High-confidence link prediction (MRR/MR) ---
            elif monitor in {"mrr", "mr"}:
                high_triples = self.valid["high_triples"].to(self.device)
                high_probs = self.valid["high_probabilities"].to(self.device)
                result = val_high_link_predict(high_triples, high_probs, self.model, self.valid, prediction_mode=mode)
                (
                    mr_raw, mrr_raw, hit1_raw, hit3_raw, hit10_raw,
                    mr_filter, mrr_filter, hit1_filter, hit3_filter, hit10_filter
                ) = result
                return mrr_raw.item() if monitor == "mrr" else mr_raw.item()

            # --- (4) Weighted link prediction (WMRR/WMR) ---
            elif monitor in {"wmrr", "wmr"}:
                result = val_weight_link_predict(val_triples, val_probs, self.model, self.valid, prediction_mode=mode)
                mr_raw, mrr_raw, hit20_raw, hit40_raw = result
                return mrr_raw.item() if monitor == "wmrr" else mr_raw.item()

            # --- (5) Ranking quality (NDCG) ---
            elif monitor == "ndcg":
                linear_ndcg, _ = mean_ndcg(self.valid["hr_map"], self.model, self.device)
                return linear_ndcg

            else:
                raise ValueError(f"Unsupported monitor type: {monitor}")

    # ----------------------------------------------------------------------
    # Testing
    # ----------------------------------------------------------------------
    def test(self):
        """Evaluate model on the test set using various metrics."""
        self.model.load_state_dict(
                torch.load(self.save_path, map_location=self.device)
            )
        self.model.eval()
        with torch.no_grad():         

            data = self.test_data
            device = self.device

            # === Confidence Metrics ===
            mse, mae = conf_predict(
                data["triples"].to(device),
                data["probabilities"].to(device),
                self.model
            )
            print("Only Test Data")
            print_results(mse, mae)

            mse_neg, mae_neg = conf_predict(
                data["test_neg"].to(device),
                data["test_neg_pro"].to(device),
                self.model
            )
            print("Test and Negative Data")
            print_results(mse_neg, mae_neg)

            # === Link Prediction Metrics ===
            link_predict(data["triples"].to(device),
                         data["probabilities"].to(device),
                         self.model, data)

            high_link_predict(data["high_triples"].to(device),
                              data["high_probabilities"].to(device),
                              self.model, data)

            weight_link_predict(data["triples"].to(device),
                                data["probabilities"].to(device),
                                self.model, data)

            # === Ranking (NDCG) ===
            ndcg_lin, ndcg_exp = mean_ndcg(data["hr_map"], self.model, device)
            print(f"nDCG_linear: {ndcg_lin:.4f} | nDCG_exp: {ndcg_exp:.4f}")

            # === Calibration (ECE) ===
            ece_neg = ece_t(data["test_neg"].to(device),
                            data["test_neg_pro"].to(device),
                            self.model)
            print(f"ECE (with negatives): {ece_neg:.4f}")

            ece = ece_t(data["triples"].to(device),
                        data["probabilities"].to(device),
                        self.model)
            print(f"ECE: {ece:.4f}")

            print("✅ Test completed.")
