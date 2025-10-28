import torch
from torch.utils.data import Dataset
from collections import defaultdict
from typing import List, Tuple


class Word2VecUncertainDataset(Dataset):
    """Word2Vec Dataset for uncertain knowledge graph: (h, r, t, confidence)."""

    def __init__(self, quadruples: List[List[float]], window_size: int = 2, sg: int = 1, neg_sample_num: int = 5):
        self.sg = sg
        self.window = window_size
        self.neg_sample_num = neg_sample_num
        self.quads = self._convert_triples(quadruples)
        self.word2idx, self.idx2word = self._build_vocab()
        self.neg_dist = self._build_neg_distribution()
        self.training_pairs = self._generate_pairs()

    @staticmethod
    def _convert_triples(quads):
        return [[str(h), f"r{r}", str(t), float(p)] for h, r, t, p in quads]

    def _build_vocab(self):
        vocab = {token for quad in self.quads for token in quad[:3]}
        word2idx = {w: i for i, w in enumerate(sorted(vocab, key=str))}
        idx2word = {i: w for w, i in word2idx.items()}
        return word2idx, idx2word

    def _build_neg_distribution(self):
        freq = defaultdict(int)
        for quad in self.quads:
            for token in quad[:3]:
                freq[token] += 1
        freqs = torch.tensor([freq[w] for w in self.word2idx.keys()], dtype=torch.float)
        return (freqs ** 0.75) / torch.sum(freqs ** 0.75)

    def _generate_pairs(self):
        pairs = []
        for quad in self.quads:
            tokens, p = quad[:3], quad[3]
            for i, center in enumerate(tokens):
                context_range = range(max(0, i - self.window), min(len(tokens), i + self.window + 1))
                context_words = [tokens[j] for j in context_range if j != i]
                if self.sg:
                    for ctx in context_words:
                        pairs.append((self.word2idx[center], self.word2idx[ctx], p))
                elif context_words:
                    pairs.append(([self.word2idx[w] for w in context_words], self.word2idx[center], p))
        return pairs

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, idx):
        return self.training_pairs[idx]


def collate_skipgram(batch: List[Tuple[int, int, float]]):
    centers = torch.tensor([b[0] for b in batch], dtype=torch.long)
    contexts = torch.tensor([b[1] for b in batch], dtype=torch.long)
    weights = torch.tensor([b[2] for b in batch], dtype=torch.float)

    return centers, contexts, weights


def collate_cbow(batch: List[Tuple[List[int], int, float]]):
    max_len = max(len(b[0]) for b in batch)
    padded_contexts = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, (ctx_list, _, _) in enumerate(batch):
        padded_contexts[i, :len(ctx_list)] = torch.tensor(ctx_list, dtype=torch.long)
    targets = torch.tensor([b[1] for b in batch], dtype=torch.long)
    weights = torch.tensor([b[2] for b in batch], dtype=torch.float)
    return padded_contexts, targets, weights
