from openukge.data import ppi5k
from openukge.models import GtransE
from openukge.training import GtransETrainer, OptimBuilder, EarlyStop
from openukge.loss import GtransELoss
from openukge.utils import seed_everything


def main(test_only=False):
    seed_everything()
    data = ppi5k.load_data('data', num_neg=10, batch_size=256)
    model = GtransE(data.num_ent, data.num_rel, emb_dim=100, margin=10)
    loss = GtransELoss(alpha=2, margin=10)
    opt = {'optimizer': {'type': 'Adam', 'lr': 1e-4, 'weight_decay': 0.001}}
    # 'scheduler': {'type': 'MultiStepLR', 'milestones': [12], 'gamma': 0.1}}
    optimizer = OptimBuilder(opt)
    early_stop = EarlyStop(patience=2, monitor="mrr", mode="max", monitor_mode="tail")
    trainer = GtransETrainer(data, model, loss, optimizer,early_stop,
                             save_path="./out_pt/GtransE/ppi5k/ppi5k-gtranse-mrr.pt")
    if not test_only:
        trainer.fit(epochs=1000, eval_freq=1)
        trainer.test()
    else:
        trainer.test()

if __name__ == '__main__':
    main(test_only=False)
