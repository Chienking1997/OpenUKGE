import torch
import torch.optim as optim
from ..models import CBOWModel, SkipGramModel
from ..data import Word2VecData
from tqdm import tqdm


class Word2Vec:
    def __init__(self, input_data, window_size=2, batch_size=64, min_count=1,
                 dim=100, lr=0.025, neg_count=5, epochs=5, device="cpu", sg=1):
        self.input_data = input_data
        self.window_size = window_size
        self.batch_size = batch_size
        self.min_count = min_count
        self.dim = dim
        self.lr = lr
        self.neg_count = neg_count
        self.epochs = epochs
        self.device = torch.device(device)
        self.sg = sg

        # Initialize data and model
        self.data = Word2VecData(self.input_data, self.min_count)
        if sg == 1:
            self.model = SkipGramModel(self.data.word_count, self.dim, self.device).to(self.device)
        else:
            self.model = CBOWModel(self.data.word_count, self.dim, self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        pairs_count = self.data.evaluate_pairs_count(self.window_size)
        batch_count = pairs_count // self.batch_size
        for epoch in range(self.epochs):
            process_bar = tqdm(range(int(batch_count)), desc=f"Word2Vec Epoch {epoch + 1}", leave=False)
            total_loss = 0
            for _ in process_bar:
                if self.sg == 1:
                    loss = self.train_epoch_sg()
                else:
                    loss = self.train_epoch_cb()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                process_bar.set_postfix(loss=total_loss / (process_bar.n + 1))

        return self.model.vec_embedding(), self.data.word2id_dict

    def train_epoch_cb(self):
        pos_pairs = self.data.get_batch_pairs(self.batch_size, self.window_size)
        pos_w = [pair[0] for pair in pos_pairs]
        pos_v = [pair[1] for pair in pos_pairs]
        neg_v = self.data.get_negative_sampling(len(pos_pairs), self.neg_count)
        loss = self.model(pos_w, pos_v, neg_v)
        return loss

    def train_epoch_sg(self):
        pos_pairs = self.data.get_batch_pairs_sg(self.batch_size, self.window_size)
        pos_u = [pair[0] for pair in pos_pairs]
        pos_w = [pair[1] for pair in pos_pairs]
        neg_w = self.data.get_negative_sampling(len(pos_pairs), self.neg_count)
        loss = self.model(pos_u, pos_w, neg_w)
        return loss
