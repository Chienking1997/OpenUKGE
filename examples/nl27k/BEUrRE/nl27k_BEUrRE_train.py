from openukge.data import nl27k
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
    data = nl27k.load_data('data', num_neg=30, batch_size=2048)
    rule_config = {
        'transitive': {  # (a,r,b)^(b,r,c)=>(a,r,c)
            'use': True,
            'relations': [0, 47, 10],
        },
        'composite': {
            'use': True,
            'relations': [(195, 122, 58)],
        },
    }
    regularization = {'delta': 0.5, 'min': 0.001, 'rel_trans': 0.001, 'rel_scale': 0.001,
                      'inverse': 0, 'transitive': 0.1, 'composite': 0.1}
    model = BEUrRE(data.num_ent, data.num_rel, emb_dim=100, gumbel_beta=0.001)
    loss = BEUrRELoss(rule_config=rule_config, regularization=regularization)
    opt = {'optimizer': {'type': 'Adam', 'lr': 5e-4} }
    optimizer = OptimBuilder(opt)
    early_stop = EarlyStop(patience=5, monitor="ndcg", mode="max", monitor_mode="tail")
    trainer = BEUrRETrainer(data, model, loss, optimizer, early_stop,
                            save_path="./out_pt/BEUrRE/nl27k/nl27k-beurre-ndcg-rule1.pt")
    if not test_only:
        trainer.fit(epochs=1000, eval_freq=5)
        trainer.test()
    else:
        trainer.test()
if __name__ == '__main__':
    main(test_only=False)