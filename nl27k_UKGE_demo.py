from openukge.data import nl27k
from openukge.models import UKGE
from openukge.training import UKGETrainer, OptimBuilder, EarlyStop
from openukge.loss import UKGELoss
from openukge.utils import seed_everything

seed_everything()
data = nl27k.load_data('data', num_neg=10, batch_size=512)
model = UKGE(data.num_ent, data.num_rel, emb_dim=128)
loss = UKGELoss()
opt = {'optimizer': {'type': 'Adam', 'lr': 0.001}}
optimizer = OptimBuilder(opt)
early_stop = EarlyStop(patience=2, min_delta=0, 
                       monitor="mse", mode="min", 
                       monitor_mode="tail")
trainer = UKGETrainer(data, model, loss, 
                      optimizer, early_stop, 
                      save_path="best_nl27k-mse.pt")
trainer.fit(epochs=100, eval_freq=2)
trainer.test()
