from openukge.data import ppi5k
from openukge.models import BEUrRE
from openukge.training import BEUrRETrainer, OptimBuilder, EarlyStop
from openukge.loss import BEUrRELoss
from openukge.utils import seed_everything


def main(test_only=False):
    """
        Main function for model training and testing.

        Args:
            test_only (bool): A flag to control the program's running mode.
                - When False (default), the program executes the full workflow: first performs model training
                  (including multi-epoch fitting), and automatically runs test evaluation after training completes,
                  outputting performance metrics on the test set.
                - When True, the program skips the training process and directly loads the saved model weights
                  for test evaluation. This is suitable for scenarios where you want to independently verify
                  model performance after training is completed.
    """
    seed_everything()
    data = ppi5k.load_data('data', num_neg=30, batch_size=256)
    rule_config = {
            'transitive': {
                'use': True,
                'relations': [0],
            }
        }
    regularization = {'delta': 0, 'min': 0, 'rel_trans': 0, 'rel_scale': 0,
                      'inverse': 0, 'transitive': 0.1, 'composite': 0}
    model = BEUrRE(data.num_ent, data.num_rel, emb_dim=200, gumbel_beta=0.001)
    loss = BEUrRELoss(rule_config=rule_config, regularization=regularization)
    opt = {'optimizer': {'type': 'Adam', 'lr': 1e-3}, }
    optimizer = OptimBuilder(opt)
    early_stop = EarlyStop(patience=2, min_delta=0, monitor="ndcg", mode="max", monitor_mode="tail")
    trainer = BEUrRETrainer(data, model, loss, optimizer, early_stop,
                            save_path="./out_pt/BEUrRE/ppi5k/ppi5k-beurre_ndcg-rule2.pt")
    if not test_only:
        trainer.fit(epochs=1000, eval_freq=5)
        trainer.test()
    else:
        trainer.test()

if __name__ == '__main__':
    main()