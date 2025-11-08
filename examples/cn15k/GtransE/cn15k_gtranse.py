from openukge.data import cn15k
from openukge.models import GtransE
from openukge.training import GtransETrainer, OptimBuilder, EarlyStop
from openukge.loss import GtransELoss
from openukge.utils import seed_everything


def main():
    seed_everything()
    data = cn15k.load_data('data', num_neg=10, batch_size=256)
    model = GtransE(data.num_ent, data.num_rel, emb_dim=200, margin=10)
    loss = GtransELoss(alpha=4, margin=10)
    opt = {'optimizer': {'type': 'Adam', 'lr': 0.0001, 'betas': (0.9, 0.99), 'weight_decay': 0.001}}
    # 'scheduler': {'type': 'MultiStepLR', 'milestones': [12], 'gamma': 0.1}}
    optimizer = OptimBuilder(opt)
    early_stop = EarlyStop(patience=2, monitor="mrr", mode="max", monitor_mode="tail")
    trainer = GtransETrainer(data, model, loss, optimizer, early_stop,
                             save_path="./out_pt/GtransE/cn15k/cn15k-gtranse-mrr2.pt")
    trainer.fit(epochs=1000, eval_freq=50)
    trainer.test()


if __name__ == '__main__':
    main()
