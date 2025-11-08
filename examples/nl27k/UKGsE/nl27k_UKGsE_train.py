from openukge.data import nl27k
from openukge.models import UKGsE
from openukge.training import UKGsETrainer, Word2VecTrainer, OptimBuilder, EarlyStop
from openukge.loss import UKGsELoss
from openukge.utils import seed_everything


def main(test_only=False):
    seed_everything()
    data = nl27k.load_data('data', num_neg=1, batch_size=64)
    if test_only:
        w2v_embedding, word2idx = None, None
    else:
        trainer = Word2VecTrainer(data.train, emb_dim=128, window=2,
            sg=0, neg_num=5, lr=0.001, batch_size=128, epochs=8)
        w2v_embedding, word2idx = trainer.train()
    model = UKGsE(w2v_embedding, word2idx, data.num_ent, data.num_rel, emb_dim=128)
    loss = UKGsELoss()
    opt = {'optimizer': {'type': 'Adam', 'lr': 0.001, 'betas': (0.9, 0.99)}}
    optimizer = OptimBuilder(opt)
    early_stop = EarlyStop(patience=3, monitor="mse", mode="min", monitor_mode="tail")
    trainer = UKGsETrainer(data, model, loss, optimizer, early_stop,
                           save_path="./out_pt/UKGsE/nl27k/nl27k-UKGsE-mse.pt")
    if not test_only:
        trainer.fit(epochs=100, eval_freq=4)
        trainer.test()
    else:
        trainer.test()


if __name__ == '__main__':
    main(test_only=False)
