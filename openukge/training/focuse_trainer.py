import torch
from tqdm.auto import tqdm, trange
from ..utils import prepare_device
from ..evaluation import conf_predict, print_results
from ..evaluation import link_predict, high_link_predict, weight_link_predict
from ..evaluation import val_link_predict, val_high_link_predict, val_weight_link_predict
from ..evaluation import mean_ndcg, ece_t


class FocusETrainer:
    def __init__(self, data=None, model=None, loss=None, opt=None, early_stop=None, save_path=None, config=None):
        self.data = data
        self.train = data.train_dataloader()
        self.valid = data.val_dataloader()
        self.test_data = data.test_dataloader()
        self.device = prepare_device()
        self.num_neg = data.sampler.num_neg
        self.model = model.to(self.device)

        self.loss = loss
        self.early_stop = early_stop
        self.save_path = save_path
        self.config = config
        self.optimizer = opt.build_optimizer(model=self.model)
        self.scheduler = opt.build_scheduler(optimizer=self.optimizer)

    def fit(self, epochs=None, eval_freq=5):
        epochs_bar = trange(1, epochs + 1, leave=True)
        for epoch in epochs_bar:
            avg_loss = self.train_epoch(epoch-1, epochs-1)
            epochs_bar.set_description("Epoch %d | loss: %.4f" % (epoch, avg_loss))
            if epoch % eval_freq == 0:
                result = self.valid_epoch(self.early_stop.get_monitor(), self.early_stop.get_monitor_mode())
                stop_flag, best_valid_result = self.early_stop(result, model=self.model,
                                                               save_path=self.save_path)
                epochs_bar.set_postfix_str("val_result: %.4f, best_val_result: %.4f" %
                                           (result, best_valid_result))

                if stop_flag:
                    print("Early Stop!")
                    break
        print("Training Finished! The model saved at %s" % self.save_path)

    def train_epoch(self, epoch, epochs):
        self.model.train()
        self.model.adjust_parameters(epoch, epochs)
        total_loss = 0  # Initialize total loss for averaging

        train_bar = tqdm(self.train, desc=f"Training", leave=False)
        num_batches = len(train_bar)

        for train_batch in train_bar:
            pos_sample = train_batch["positive_sample"].to(self.device)
            neg_sample = train_batch["negative_sample"].to(self.device)
            pro = train_batch["probabilities"].to(self.device)
            self.optimizer.zero_grad()

            pos_score = self.model(pos_sample, pro)
            neg_score = self.model(neg_sample, pro, num_neg=self.num_neg)
            regularization1 = self.model.regularization(pos_sample)
            regularization2 = self.model.regularization2()
            loss = self.loss(pos_score, neg_score) + regularization1 + regularization2
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / num_batches

    def valid_epoch(self, monitor, mode):
        self.model.eval()
        with torch.no_grad():
            val_triples = self.valid["triples"].to(self.device)
            val_pro = self.valid["probabilities"].to(self.device)
            if monitor == 'mse' or monitor == 'mae':
                mse, mae = conf_predict(val_triples, val_pro, self.model)
                result = {'mse': mse.item(), 'mae': mae.item()}
                return result[monitor]
            elif monitor == 'umrr':
                result = val_link_predict(val_triples, val_pro, self.model, self.valid, prediction_mode=mode)
                (mr_raw, mrr_raw, hit1_raw, hit3_raw, hit10_raw,
                 mr_filter, mrr_filter, hit1_filter, hit3_filter, hit10_filter) = result
                return mrr_raw
            elif monitor == 'mrr':
                high_triples = self.valid["high_triples"].to(self.device)
                high_pro = self.valid["high_probabilities"].to(self.device)
                result = val_high_link_predict(high_triples, high_pro, self.model, self.valid, prediction_mode=mode)
                (mr_raw, mrr_raw, hit1_raw, hit3_raw, hit10_raw,
                 mr_filter, mrr_filter, hit1_filter, hit3_filter, hit10_filter) = result
                return mrr_raw
            elif monitor == 'wmrr':
                result = val_weight_link_predict(val_triples, val_pro, self.model, self.valid, prediction_mode=mode)
                mr_raw, mrr_raw, hit20_raw, hit40_raw = result
                return mrr_raw

    def test(self):
        self.model.eval()
        with torch.no_grad():
            self.model.load_state_dict(torch.load(self.save_path))
            test_triples = self.test_data["triples"].to(self.device)
            test_pro = self.test_data["probabilities"].to(self.device)
            test_high_triples = self.test_data["high_triples"].to(self.device)
            test_high_pro = self.test_data["high_probabilities"].to(self.device)
            # mse, mae = conf_predict(test_triples, test_pro, self.model)
            # print_results(mse, mae)
            # link_predict(test_triples, test_pro, self.model, self.test_data)
            high_link_predict(test_high_triples, test_high_pro, self.model, self.test_data)
            weight_link_predict(test_triples, test_pro, self.model, self.test_data)
            linear_ndcg, exp_ndcg = mean_ndcg(self.test_data["hr_map"], self.model, self.device)
            print(linear_ndcg, exp_ndcg)
            # === Calibration (ECE) ===
            ece = ece_t(
                self.test_data["test_neg"].to(self.device),
                self.test_data["test_neg_pro"].to(self.device),
                self.model
            )
            print(f"ECE with neg: {ece:.4f}")
            ece = ece_t(
                self.test_data["triples"].to(self.device),
                self.test_data["probabilities"].to(self.device),
                self.model
            )
            print(f"ECE: {ece:.4f}")
            print("✅ Test completed.")

