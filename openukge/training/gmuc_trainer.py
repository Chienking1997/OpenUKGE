import torch
from tqdm.auto import tqdm, trange
from collections import defaultdict as ddict

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


class GMUCTrainer:
    """
    Uncertain Knowledge Graph Embedding (UKGE) Trainer.

    Handles model training, validation, and testing across multiple
    evaluation modes, including uncertainty-aware and confidence-based metrics.

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
                 early_stop=None, save_path=None, config=None, has_ont=False, if_ne=False):

        # === Data and device setup ===

        self.data = data
        self.train_loader = data.train_dataloader()
        self.valid = data.val_dataloader()
        self.test_data = data.test_dataloader()
        self.device = prepare_device()
        self.data.device_setter(self.device)

        # === Model & components ===
        self.model = model.to(self.device)
        self.loss_fn = loss
        self.early_stop = early_stop
        self.save_path = save_path
        self.config = config  # Reserved for extended configs
        self.has_ont = has_ont
        self.if_ne = if_ne

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
            epochs (int): Total training epochs.
            eval_freq (int): Frequency (in epochs) of validation.
        """
        epochs_bar = trange(1, epochs + 1, desc="Training", leave=True)

        for epoch in epochs_bar:
            avg_loss = self._train_epoch()
            epochs_bar.set_description(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

            # === Periodic validation ===
            if epoch % eval_freq == 0:
                monitor_metric = self.early_stop.get_monitor()
                mode = self.early_stop.get_monitor_mode()
                val_result = self._valid_epoch(monitor_metric)

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
            support = batch["support"]
            query = batch["query"]
            false = batch["false"]
            support_meta = batch["support_meta"]
            query_meta = batch["query_meta"]
            false_meta = batch["false_meta"]
            symbolid_ic = batch["symbolid_ic"]



            # --- Forward + Backward ---
            self.optimizer.zero_grad()
            if self.has_ont:
                query_scores, query_scores_var, false_scores, query_confidence = self.model(support, support_meta,
                                                                                            query,
                                                                                            query_meta, false,
                                                                                            false_meta,
                                                                                            self.if_ne)
                loss = self.loss_fn(query_scores, query_scores_var, false_scores, query_confidence, symbolid_ic)
            else:
                (query_scores, query_scores_var, query_ae_loss, false_scores, false_scores_var, false_ae_loss,
                 query_confidence) = self.model(support, support_meta, query, query_meta, false, false_meta)
                loss = self.loss_fn(query_scores, query_scores_var, query_ae_loss, false_scores, query_confidence)


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

    def _valid_epoch(self, monitor: str) -> float:
        """
        Run a validation epoch and return the metric specified by `monitor`.

        Args:
            monitor (str): Metric to monitor. One of:
                {"mse", "mae",  "mrr", "mr",  "wmr", "wmrr"}.

        Returns:
            float: Validation score for the monitored metric.
        """
        results = ddict(list)
        self.model.eval()
        with torch.no_grad():
            for batch in self.valid:
                result = self.val_test(batch)
                for metric, value in result.items():
                    results[metric].append(value)
        final_result = {}
        for metric, values in results.items():
            final_result[metric] = sum(values) / len(values)
        return final_result[monitor]
    # --------------------------------------------------------------------------
    # Testing
    # --------------------------------------------------------------------------
    def test(self):
        """
        Load the best model checkpoint and evaluate on the test set.
        Includes link prediction, calibration, and ranking metrics.
        """
        results = ddict(list)
        self.model.eval()
        with torch.no_grad():
            self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))
            for batch in self.test_data:
                result = self.val_test(batch)
                for metric, value in result.items():
                    results[metric].append(value)
            final_result = {}
            for metric, values in results.items():
                final_result[metric] = sum(values) / len(values)
            self.print_results_table(final_result)
            print("✅ Test completed.")


    def val_test(self, batch):
        task = batch["query"]
        test_tasks = batch["triples"]
        support_triples = test_tasks[:self.data.few]  # 支持集样本

        # 构建支持集数据（转换为tensor并指定设备）
        support_pairs = [[self.data.symbol2id[triple[0]], self.data.symbol2id[triple[2]], float(triple[3])]
                         for triple in support_triples]
        support = torch.tensor(support_pairs, dtype=torch.float32, device=self.device)

        support_left = [self.data.ent2id[triple[0]] for triple in support_triples]
        support_right = [self.data.ent2id[triple[2]] for triple in support_triples]
        support_meta = self.data.sampler.get_meta(support_left, support_right)

        # 确定候选实体
        if self.data.type_constrain:
            candidates = self.data.rel2candidates[task]
        else:
            candidates = list(self.data.ent2id.keys())[:1000]  # 取前1000个候选

        # 初始化评估指标存储（使用tensor累积以提高效率）
        all_conf = torch.tensor([], dtype=torch.float32, device=self.device)
        raw_conf_filter = torch.tensor([], dtype=torch.float32, device=self.device)
        filter_conf_filter = torch.tensor([], dtype=torch.float32, device=self.device)

        pos_mae = torch.tensor([], dtype=torch.float32, device=self.device)
        pos_mse = torch.tensor([], dtype=torch.float32, device=self.device)
        neg_mae = torch.tensor([], dtype=torch.float32, device=self.device)
        neg_mse = torch.tensor([], dtype=torch.float32, device=self.device)

        r_mr, r_mrr = [], []
        r_hits = {f'r_hits@{k}': [] for k in [1, 3, 5, 10]}
        mr, mrr = [], []
        hits = {f'hits@{k}': [] for k in [1, 3, 5, 10]}

        # 遍历每个头实体的查询
        for e1, e2s in self.data.rele1_e2[task].items():
            # 收集真实尾实体和置信度（排除支持集中的样本）
            true_e2, true_s = [], []
            for _ in e2s:
                if [e1, task, _[0], _[1]] not in support_triples:
                    true_e2.append(_[0])
                    true_s.append(float(_[1]))

            num_e2 = len(true_e2)
            if num_e2 == 0:
                continue  # 无真实样本则跳过

            # 构建查询对（正例+候选负例）
            query_pairs = []
            query_left, query_right = [], []
            # 添加正例
            for i in range(num_e2):
                query_pairs.append([self.data.symbol2id[e1], self.data.symbol2id[true_e2[i]]])
                query_left.append(self.data.ent2id[e1])
                query_right.append(self.data.ent2id[true_e2[i]])
            # 添加候选负例
            for ent in candidates:
                query_pairs.append([self.data.symbol2id[e1], self.data.symbol2id[ent]])
                query_left.append(self.data.ent2id[e1])
                query_right.append(self.data.ent2id[ent])

            # 转换为tensor并指定设备
            query = torch.tensor(query_pairs, dtype=torch.long, device=self.device)
            query_meta = self.data.sampler.get_meta(query_left, query_right)

            # 模型评分
            if self.has_ont:
                scores, scores_var = self.model.score_func(support, support_meta, query, query_meta)
            else:
                scores, scores_var, _ = self.model.score_func(support, support_meta, query, query_meta)

            # 分离计算图（无需转为numpy）
            scores = scores.detach()
            scores_var = scores_var.detach()

            # 构建过滤后的分数（排除已知正例）
            # 收集符合条件的负例索引
            neg_indices = []
            for idx, ent in enumerate(candidates):
                if ent not in self.data.e1rel_e2.get(e1 + task, set()) and ent not in true_e2:
                    neg_indices.append(idx + num_e2)  # 偏移量：前num_e2是正例
            # 拼接正例分数和符合条件的负例分数
            f_scores = torch.cat([scores[:num_e2], scores[neg_indices]], dim=0) if neg_indices else scores[:num_e2]

            # 累积所有置信度
            all_conf = torch.cat([all_conf, torch.tensor(true_s, dtype=torch.float32, device=self.device)], dim=0)

            # ====================== 原始尾实体预测评估 ======================
            # 筛选高置信度正例
            query_conf_tensor = torch.tensor(true_s, dtype=torch.float32, device=self.device)
            pos_mask = query_conf_tensor > 0
            pos_scores_raw = scores[:num_e2][pos_mask]
            raw_conf_filter = torch.cat([raw_conf_filter, query_conf_tensor[pos_mask]], dim=0)
            num_e2_raw = pos_scores_raw.numel()

            # 拼接筛选后的正例和所有负例
            scores_filter_raw = torch.cat([pos_scores_raw, scores[num_e2:]], dim=0)

            # 计算排名
            if num_e2_raw > 0 and scores_filter_raw.numel() > 0:
                score_sort_raw = torch.argsort(scores_filter_raw, descending=True)  # 降序排序索引
                ranks_raw = []
                for i in range(num_e2_raw):
                    # 找到正例在排序结果中的位置（+1为排名）
                    pos = torch.where(score_sort_raw == i)[0].item()
                    ranks_raw.append(pos + 1)

                # 修正排名（处理并列）
                ranks_sort_raw = torch.sort(torch.tensor(ranks_raw, device=self.device))[0]
                for i in range(ranks_sort_raw.numel()):
                    rank = ranks_sort_raw[i] - i
                    rank = 1.0 if rank <= 0 else rank  # 确保排名至少为1

                    # 累积指标
                    for k in [1, 3, 5, 10]:
                        r_hits[f'r_hits@{k}'].append(1.0 if rank <= k else 0.0)
                    r_mrr.append(1.0 / rank)
                    r_mr.append(rank)

            # ====================== 过滤后尾实体预测评估 ======================
            # 筛选高置信度正例（基于过滤后的分数）
            pos_scores_filt = f_scores[:num_e2][pos_mask]
            filter_conf_filter = torch.cat([filter_conf_filter, query_conf_tensor[pos_mask]], dim=0)
            num_e2_filt = pos_scores_filt.numel()

            # 拼接筛选后的正例和过滤后的负例
            scores_filter_filt = torch.cat([pos_scores_filt, f_scores[num_e2:]],
                                           dim=0) if f_scores.numel() > num_e2 else pos_scores_filt

            # 计算排名
            if num_e2_filt > 0 and scores_filter_filt.numel() > 0:
                score_sort_filt = torch.argsort(scores_filter_filt, descending=True)
                ranks_filt = []
                for i in range(num_e2_filt):
                    pos = torch.where(score_sort_filt == i)[0].item()
                    ranks_filt.append(pos + 1)

                # 修正排名
                ranks_sort_filt = torch.sort(torch.tensor(ranks_filt, device=self.device))[0]
                for i in range(ranks_sort_filt.numel()):
                    rank = ranks_sort_filt[i] - i
                    rank = 1.0 if rank <= 0 else rank

                    # 累积指标
                    for k in [1, 3, 5, 10]:
                        hits[f'hits@{k}'].append(1.0 if rank <= k else 0.0)
                    mrr.append(1.0 / rank)
                    mr.append(rank)

            # ====================== 置信度预测评估 ======================
            # 正例误差计算
            pos_true = torch.tensor(true_s, dtype=torch.float32, device=self.device)
            pos_pred = scores_var[:num_e2]
            pos_mae = torch.cat([pos_mae, torch.abs(pos_true - pos_pred)], dim=0)
            pos_mse = torch.cat([pos_mse, torch.square(pos_true - pos_pred)], dim=0)

            # 负例误差计算
            neg_pred = scores_var[num_e2:]
            neg_mae = torch.cat([neg_mae, torch.abs(neg_pred)], dim=0)
            neg_mse = torch.cat([neg_mse, torch.square(neg_pred)], dim=0)

        # ====================== 结果计算 ======================
        results = {}
        # 置信度预测指标
        results["MAE"] = torch.mean(pos_mae).item() if pos_mae.numel() > 0 else 0.0
        results["MSE"] = torch.mean(pos_mse).item() if pos_mse.numel() > 0 else 0.0

        # 原始预测结果
        r_mr_tensor = torch.tensor(r_mr, dtype=torch.float32, device=self.device) if r_mr else None
        results["raw_mr"] = torch.mean(r_mr_tensor).item() if r_mr_tensor is not None else 0.0

        r_mrr_tensor = torch.tensor(r_mrr, dtype=torch.float32, device=self.device) if r_mrr else None
        results["raw_mrr"] = torch.mean(r_mrr_tensor).item() if r_mrr_tensor is not None else 0.0

        # 原始加权指标
        if raw_conf_filter.numel() > 0 and r_mr_tensor is not None and r_mr_tensor.numel() == raw_conf_filter.numel():
            results["raw_wmr"] = (torch.sum(raw_conf_filter * r_mr_tensor) / torch.sum(raw_conf_filter)).item()
        else:
            results["raw_wmr"] = 0.0

        if raw_conf_filter.numel() > 0 and r_mrr_tensor is not None and r_mrr_tensor.numel() == raw_conf_filter.numel():
            results["raw_wmrr"] = (torch.sum(raw_conf_filter * r_mrr_tensor) / torch.sum(raw_conf_filter)).item()
        else:
            results["raw_wmrr"] = 0.0

        # 原始Hits@k
        for k in [1, 3, 5, 10]:
            hits_list = r_hits[f'r_hits@{k}']
            if hits_list:
                results[f'raw_hits@{k}'] = torch.mean(torch.tensor(hits_list, device=self.device)).item()
            else:
                results[f'raw_hits@{k}'] = 0.0

        # 过滤后预测结果
        mr_tensor = torch.tensor(mr, dtype=torch.float32, device=self.device) if mr else None
        results["mr"] = torch.mean(mr_tensor).item() if mr_tensor is not None else 0.0

        mrr_tensor = torch.tensor(mrr, dtype=torch.float32, device=self.device) if mrr else None
        results["mrr"] = torch.mean(mrr_tensor).item() if mrr_tensor is not None else 0.0

        # 过滤后加权指标
        if filter_conf_filter.numel() > 0 and mr_tensor is not None and mr_tensor.numel() == filter_conf_filter.numel():
            results["wmr"] = (torch.sum(filter_conf_filter * mr_tensor) / torch.sum(filter_conf_filter)).item()
        else:
            results["wmr"] = 0.0

        if filter_conf_filter.numel() > 0 and mrr_tensor is not None and mrr_tensor.numel() == filter_conf_filter.numel():
            results["wmrr"] = (torch.sum(filter_conf_filter * mrr_tensor) / torch.sum(filter_conf_filter)).item()
        else:
            results["wmrr"] = 0.0

        # 过滤后Hits@k
        for k in [1, 3, 5, 10]:
            hits_list = hits[f'hits@{k}']
            if hits_list:
                results[f'hits@{k}'] = torch.mean(torch.tensor(hits_list, device=self.device)).item()
            else:
                results[f'hits@{k}'] = 0.0

        return results

    @staticmethod
    def print_results_table(results: dict):
        """
        以控制台表格形式美观输出模型评估结果。
        自动对齐，分为三类：
          1. Error Metrics (MAE, MSE)
          2. Weighted Metrics (WMR, WMRR)
          3. Ranking Metrics (MR, MRR, Hits@k)
        """

        # ====== 分类定义 ======
        sections = {
            "Error Metrics": [
                ("MAE", "MAE", None),
                ("MSE", "MSE", None),
            ],
            "Weighted Metrics": [
                ("WMR", "raw_wmr", "wmr"),
                ("WMRR", "raw_wmrr", "wmrr"),
            ],
            "Ranking Metrics": [
                ("MR", "raw_mr", "mr"),
                ("MRR", "raw_mrr", "mrr"),
                ("Hits@1", "raw_hits@1", "hits@1"),
                ("Hits@3", "raw_hits@3", "hits@3"),
                ("Hits@5", "raw_hits@5", "hits@5"),
                ("Hits@10", "raw_hits@10", "hits@10"),
            ],
        }

        # ====== 打印表格函数 ======
        def print_section(title, rows, has_filtered=True):
            print(f"\n{'=' * 65}")
            print(f"{title.center(65)}")
            print(f"{'=' * 65}")
            if has_filtered:
                print(f"{'Metric':<15} {'Raw Value':>15} {'Filtered Value':>20}")
                print(f"{'-' * 65}")
            else:
                print(f"{'Metric':<15} {'Value':>15}")
                print(f"{'-' * 40}")

            for label, raw_key, filt_key in rows:
                raw_val = results.get(raw_key, "-") if raw_key else "-"
                filt_val = results.get(filt_key, "-") if filt_key else None

                if filt_key is None:  # Error Metrics 无 Filtered 列
                    val_str = f"{raw_val:>15.6f}" if isinstance(raw_val, (float, int)) else f"{raw_val:>15}"
                    print(f"{label:<15}{val_str}")
                else:
                    raw_str = f"{raw_val:>15.6f}" if isinstance(raw_val, (float, int)) else f"{raw_val:>15}"
                    filt_str = f"{filt_val:>20.6f}" if isinstance(filt_val, (float, int)) else f"{filt_val:>20}"
                    print(f"{label:<15}{raw_str}{filt_str}")
            print(f"{'=' * 65}")

        # ====== 打印三个部分 ======
        print_section("Error Metrics", sections["Error Metrics"], has_filtered=False)
        print_section("Weighted Metrics", sections["Weighted Metrics"], has_filtered=True)
        print_section("Ranking Metrics", sections["Ranking Metrics"], has_filtered=True)


