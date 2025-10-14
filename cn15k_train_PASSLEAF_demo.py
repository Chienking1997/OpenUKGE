from openukge.data import cn15k
from openukge.models import PASSLEAF
from openukge.training import PASSLEAFTrainer, OptimBuilder, EarlyStop
from openukge.loss import PASSLEAFLoss
from openukge.utils import seed_everything


def main():
    seed_everything()
    data = cn15k.load_data('data', num_neg=10, batch_size=512)
    model = PASSLEAF(data.num_ent, data.num_rel, emb_dim=512,
                     reg_scale=0.005, score_function='DistMult', margin=4)
    loss = PASSLEAFLoss()
    opt = {'optimizer': {'type': 'Adam', 'lr': 0.001, 'betas': (0.9, 0.999)}, }
    # 'scheduler': {'name': 'MultiStepLR', 'milestones': [12], 'gamma': 0.1}}
    optimizer = OptimBuilder(opt)
    early_stop = EarlyStop(patience=2, monitor="umrr", mode="max", monitor_mode="tail")
    trainer = PASSLEAFTrainer(data, model, loss, optimizer, early_stop,
                              save_path="./out_pt/passleaf/cn15k/cn15k-umrr.pt",
                              t_new_semi=20, t_semi_train=30)
    trainer.fit(epochs=2000, eval_freq=20)
    trainer.test()


if __name__ == '__main__':
    main()
