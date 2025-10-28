import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from openukge.data import Word2VecUncertainDataset, collate_skipgram, collate_cbow
from openukge.models.word2vec import Word2VecSkipGramModel, Word2VecCBOWModel


class Word2VecTrainer:
    """Trainer for uncertain KG Word2Vec."""

    def __init__(self, quadruples, emb_dim=100, window=2, sg=1, neg_num=5, lr=0.01, batch_size=128, epochs=5, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sg = sg
        self.neg_num = neg_num
        self.epochs = epochs

        self.dataset = Word2VecUncertainDataset(quadruples, window_size=window, sg=sg, neg_sample_num=neg_num)
        self.collate_fn = collate_skipgram if sg else collate_cbow
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        if self.sg:
            self.model = Word2VecSkipGramModel(len(self.dataset.word2idx), emb_dim).to(self.device)
        else:
            self.model = Word2VecCBOWModel(len(self.dataset.word2idx), emb_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self):
        print(f"✅ Pairs: {len(self.dataset)} | Vocab: {len(self.dataset.word2idx)} | Device: {self.device}")

        # 外层 tqdm 显示 epoch 进度
        epoch_bar = tqdm(range(self.epochs), desc="Word2Vec Training", position=0)
        for epoch in epoch_bar:
            total_loss = 0.0

            # 内层 tqdm 显示 batch 进度（仅显示当前 batch loss）
            batch_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                position=1,
                leave=False
            )

            for batch in batch_bar:
                if self.sg:
                    centers, contexts, weights = [x.to(self.device) for x in batch]
                    batch_size = centers.size(0)
                else:
                    contexts, targets, weights = [x.to(self.device) for x in batch]
                    batch_size = targets.size(0)

                negatives = torch.multinomial(
                    self.dataset.neg_dist,
                    batch_size * self.neg_num,
                    replacement=True
                ).view(batch_size, self.neg_num).to(self.device)

                loss = (
                    self.model.forward_skipgram(centers, contexts, negatives, weights)
                    if self.sg else self.model.forward_cbow(contexts, targets, negatives, weights)
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                # batch_bar.set_postfix(loss=f"{loss.item():.4f}")  # 只显示当前 batch 的 loss

            # 外层进度条显示平均 loss
            avg_loss = total_loss / len(self.dataloader)
            epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")

        self.save_embeddings()
        return self.model.input_emb.weight, self.dataset.word2idx

    def save_embeddings(self, path="word2vec_uncertain.vec"):
        emb_matrix = self.model.input_emb.weight.data.cpu()
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{len(self.dataset.word2idx)} {emb_matrix.size(1)}\n")
            for i, w in self.dataset.idx2word.items():
                vec_str = " ".join(map(str, emb_matrix[i].tolist()))
                f.write(f"{w} {vec_str}\n")
        print(f"✅ Embeddings saved to {path}")
