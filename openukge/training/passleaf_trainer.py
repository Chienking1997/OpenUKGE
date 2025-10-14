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


class PASSLEAFTrainer:
    """
    PASSLEAF Trainer.

    Implements semi-supervised negative sampling for uncertainty-aware
    knowledge graph embedding models. Handles model training, validation,
    and testing with dynamic semi-sample updates.

    Args:
        data: Dataset wrapper with train/val/test dataloaders.
        model: PyTorch model instance.
        loss: Loss function callable.
        opt: Optimizer/scheduler builder (must implement `build_optimizer` and `build_scheduler`).
        early_stop: EarlyStopping instance.
        save_path (str): Path to save the best model checkpoint.
        t_new_semi (float): Threshold controlling new semi-sample generation.
        t_semi_train (float): Threshold controlling semi-sample use in training.
        config (dict, optional): Configuration dictionary.
    """

    def __init__(
        self,
        data=None,
        model=None,
        loss=None,
        opt=None,
        early_stop=None,
        save_path=None,
        t_new_semi=None,
        t_semi_train=None,
        config=None,
    ):
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

        # === Semi-supervised thresholds ===
        self.t_new_semi = t_new_semi
        self.t_semi_train = t_semi_train

        # === Optimizer & scheduler ===
        self.optimizer = opt.build_optimizer(model=self.model)
        self.scheduler = opt.build_scheduler(optimizer=self.optimizer)

    # --------------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------------
    def fit(self, epochs: int, eval_freq: int = 5, t_new_semi=20, t_semi_train=30):
        """
        Run the main training loop with periodic validation and early stopping.

        Args:
            epochs (int): Total training epochs.
            eval_freq (int): Validation frequency (every N epochs).
            t_new_semi (float, optional): Override for new semi-sample threshold.
            t_semi_train (float, optional): Override for semi-training threshold.
        """
        self.t_new_semi = t_new_semi
        self.t_semi_train = t_semi_train

        epochs_bar = trange(1, epochs + 1, desc="Training", leave=True)

        for epoch in epochs_bar:
            avg_loss = self._train_epoch(epoch)
            epochs_bar.set_description(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

            # === Periodic validation ===
            if epoch % eval_freq == 10:
                monitor = self.early_stop.get_monitor()
                mode = self.early_stop.get_monitor_mode()

                val_result = self._valid_epoch(monitor, mode)
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
    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch with semi-supervised negative sampling."""
        self.model.train()
        total_loss = 0.0
        train_bar = tqdm(self.train_loader, desc="Train", leave=False)

        for batch in train_bar:
            # --- Extract batch tensors ---
            pos_sample = batch["positive_sample"].to(self.device)
            pro = batch["probabilities"].to(self.device)
            neg_sample = batch["negative_sample"]

            # --- Semi-supervised sampling ---
            semi_sample, semi_pro = self.data.semi_dataloader(
                neg_sample, self.model, epoch, self.device, self.t_new_semi, self.t_semi_train
            )
            semi_sample = torch.LongTensor(semi_sample).to(self.device)
            semi_pro = torch.FloatTensor(semi_pro).to(self.device)

            # --- Forward pass ---
            self.optimizer.zero_grad()
            pos_score = self.model(pos_sample)
            semi_score = self.model(semi_sample)
            regularization = self.model.regularization(pos_sample)

            # --- Compute loss ---
            loss = self.loss_fn(pos_score, semi_score, pro, semi_pro) + regularization

            # --- Backward + Optimize ---
            loss.backward(retain_graph=True)
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
            monitor (str): Metric to monitor. One of:
                {"mse", "mae", "umrr", "umr", "mrr", "mr", "wmrr", "wmr", "ndcg"}.
            mode (str): Link prediction mode, one of:
                {"head", "tail", "average"}.
                - "head": Predict head entities only.
                - "tail": Predict tail entities only.
                - "average": Mean of head and tail predictions.

        Returns:
            float: Validation score for the monitored metric.
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
                linear_ndcg, _ = mean_ndcg(self.test_data["hr_map"], self.model, self.device)
                return linear_ndcg

            else:
                raise ValueError(f"Unsupported monitor type: {monitor}")

    # --------------------------------------------------------------------------
    # Testing
    # --------------------------------------------------------------------------
    def test(self):
        """
        Load the best model checkpoint and evaluate on the test set.
        Includes link prediction, calibration, and ranking metrics.
        """
        self.model.eval()
        with torch.no_grad():
            self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))
            test_data = self.test_data
            device = self.device

            # === Confidence prediction ===
            mse, mae = conf_predict(
                test_data["test_neg"].to(device),
                test_data["test_neg_pro"].to(device),
                self.model
            )
            print_results(mse, mae)

            # === Link prediction (three types) ===
            link_predict(
                test_data["triples"].to(device),
                test_data["probabilities"].to(device),
                self.model, test_data
            )

            high_link_predict(
                test_data["high_triples"].to(device),
                test_data["high_probabilities"].to(device),
                self.model, test_data
            )

            weight_link_predict(
                test_data["triples"].to(device),
                test_data["probabilities"].to(device),
                self.model, test_data
            )

            # === Ranking metrics ===
            linear_ndcg, exp_ndcg = mean_ndcg(test_data["hr_map"], self.model, device)
            print(f"nDCG_linear: {linear_ndcg:.4f} | nDCG_exp: {exp_ndcg:.4f}")

            # === Calibration (ECE) ===
            ece = ece_t(
                test_data["test_neg"].to(device),
                test_data["test_neg_pro"].to(device),
                self.model
            )
            print(f"ECE with neg: {ece:.4f}")
            ece = ece_t(
                test_data["triples"].to(device),
                test_data["probabilities"].to(device),
                self.model
            )
            print(f"ECE: {ece:.4f}")
            print("✅ Test completed.")
