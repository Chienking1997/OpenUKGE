from openukge.data import ppi5k
from openukge.models import UPGAT
from openukge.training import UPGATTrainer, OptimBuilder, EarlyStop
from openukge.loss import UPGATLoss
from openukge.utils import seed_everything


def main():
    seed_everything(888)
    data = ppi5k.load_data('data', num_neg=10, batch_size=256)
    model = UPGAT(data.num_ent, data.num_rel, emb_dim=256)
    loss = UPGATLoss()
    opt = {'optimizer': {'type': 'Adam', 'lr': 0.001, 'betas': (0.9, 0.99)},
    'scheduler': {'type': 'MultiStepLR', 'milestones': [12], 'gamma': 0.1}}
    optimizer = OptimBuilder(opt)
    early_stop = EarlyStop(patience=3, monitor="mae", mode="min", monitor_mode="tail")
    trainer = UPGATTrainer(data, model, loss, optimizer,early_stop,teacher_model=True,
                           save_path="./out_pt/UPGAT/ppi5k/teacher/ppi5k-upgat-mae-teacher.pt")
    trainer.fit(epochs=1000, eval_freq=1)
    trainer.test()
    trainer.pseudo_tail_predict(output_file_path='./data/ppi5k/extra/pseudo.tsv')

def main2(test_only=False):
    seed_everything(888)
    data = ppi5k.load_data('data', num_neg=10, batch_size=256,use_pseudo=True)
    model = UPGAT(data.num_ent, data.num_rel, emb_dim=256)
    loss = UPGATLoss()
    opt = {'optimizer': {'type': 'Adam', 'lr': 0.001, 'betas': (0.9, 0.99)},
    'scheduler': {'type': 'MultiStepLR', 'milestones': [12], 'gamma': 0.1}}
    optimizer = OptimBuilder(opt)
    early_stop = EarlyStop(patience=3, monitor="mae", mode="min", monitor_mode="tail")
    trainer = UPGATTrainer(data, model, loss, optimizer,early_stop, teacher_model=False,
                           save_path="./out_pt/upgat/ppi5k/student/ppi5k-upgat-mae-student.pt")
    if not test_only:
        trainer.fit(epochs=1000, eval_freq=1)
        trainer.test()
    else:
        trainer.test()


if __name__ == '__main__':
    test_only = False        #set test_only
    if test_only:
        main2(test_only)     #student
    else:
        main()               #teacher
        main2(test_only)     #student
