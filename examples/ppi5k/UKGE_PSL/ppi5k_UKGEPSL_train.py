from openukge.data import ppi5k
from openukge.models import UKGEPSL
from openukge.training import UKGEPSLTrainer, OptimBuilder, EarlyStop
from openukge.loss import UKGEPSLLoss
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
    seed_everything(888)
    data = ppi5k.load_data('data', num_neg=10, batch_size=256, use_psl=True)
    model = UKGEPSL(data.num_ent, data.num_rel, emb_dim=128, reg_scale=0.005)
    loss = UKGEPSLLoss()
    opt = {'optimizer': {'type': 'Adam', 'lr': 0.001, 'betas': (0.9, 0.99)},
           'scheduler': {'type': 'MultiStepLR', 'milestones': [12], 'gamma': 0.1}}
    optimizer = OptimBuilder(opt)
    early_stop = EarlyStop(patience=4, monitor="mae", mode="min", monitor_mode="tail")
    trainer = UKGEPSLTrainer(data, model, loss, optimizer, early_stop, psl=True,
                             save_path="./out_pt/UKGE_PSL/ppi5k/ppi5k-ukgepsl-mse.pt")
    if not test_only:
        trainer.fit(epochs=100, eval_freq=2)
        trainer.test()  # Test after training
    else:
        trainer.test()  # Directly test the saved model


if __name__ == '__main__':
    main(test_only=False)
