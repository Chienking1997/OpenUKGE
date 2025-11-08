from openukge.data import ppi5k
from openukge.models import FocusE
from openukge.training import FocusETrainer, OptimBuilder, EarlyStop
from openukge.loss import FocusELoss
from openukge.utils import seed_everything


def main(test_only=False):
    seed_everything(321)
    data = ppi5k.load_data('data', num_neg=30, batch_size=1024)  # neg30
    model = FocusE(data.num_ent, data.num_rel, margin=9.0,
                   emb_dim=400, base_model='DistMult', reg_scale1=0.005, reg_scale2=0.001)
    loss = FocusELoss()
    opt = {'optimizer': {'type': 'Adam', 'lr': 5e-4, 'betas': (0.9, 0.99)}}
     # 'scheduler': {'type': 'MultiStepLR', 'milestones': [100], 'gamma': 0.1}}
    optimizer = OptimBuilder(opt)
    early_stop = EarlyStop(patience=4, monitor="wmrr", mode="max", monitor_mode="tail")
    trainer = FocusETrainer(data, model, loss, optimizer, early_stop,
                            save_path="./out_pt/FocusE/ppi5k/ppi5k-focus-wmrr.pt")
    if not test_only:
        trainer.fit(epochs=200, eval_freq=10)
        trainer.test()
    else:
        trainer.test()

if __name__ == '__main__':
    main(test_only=False)
