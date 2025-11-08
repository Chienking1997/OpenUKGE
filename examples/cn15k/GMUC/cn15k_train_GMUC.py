from openukge.data import cn15k
from openukge.models import GMUC
from openukge.training import GMUCTrainer, OptimBuilder, EarlyStop
from openukge.loss import GMUCLoss
from openukge.utils import seed_everything


def main(test_only=False):
    seed_everything(888)
    data = cn15k.load_few_shot_data('data', num_neg=1, batch_size=64,
                                    max_neighbor=30, few=3, type_constrain=True)
    model = GMUC(num_symbols=data.num_symbols, emb_dim=50, dropout=0.5, process_steps=2)
    loss = GMUCLoss(num_neg=1, mae_weight=1.0, margin=5.0,
                    if_conf=True, rank_weight=1.0, ae_weight=0.00001)
    opt = {'optimizer': {'type': 'Adam', 'lr': 0.001, 'betas': (0.9, 0.99)},
    'scheduler': {'type': 'MultiStepLR', 'milestones': [50], 'gamma': 0.1}}
    optimizer = OptimBuilder(opt)
    early_stop = EarlyStop(patience=1, monitor="hits@10", mode="max", monitor_mode="tail")
    trainer = GMUCTrainer(data, model, loss, optimizer, early_stop,
                          save_path="./out_pt/GMUC/cn15k/cn15k-GMUC-hits10.pt")
    if not test_only:
        trainer.fit(epochs=1000, eval_freq=50)
        trainer.test()
    else:
        trainer.test()

if __name__ == '__main__':
    main()
