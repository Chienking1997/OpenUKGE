from ..dataset import LoadUKGEDataset
from ..word_vec_data import Word2VecData


def load_data(root, num_neg, batch_size, high_threshold=0.70, num_partitions=5, use_pseudo=False):
    dataset = LoadUKGEDataset(root=root,
                              dataset_name='cn15k',
                              num_neg_samples=num_neg,
                              batch_size=batch_size,
                              high_threshold=high_threshold,
                              num_partitions=num_partitions,
                              use_pseudo=use_pseudo)
    dataloader = dataset.get_dataloader()
    return dataloader


def prepare_triplets_for_word2vec(input_data, min_count):
    return Word2VecData(input_data, min_count)


if __name__ == '__main__':
    load_data(root='data', num_neg=2, batch_size=1024)
