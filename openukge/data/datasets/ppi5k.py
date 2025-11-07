from ..dataset import LoadUKGEDataset


def load_data(root, num_neg, batch_size,
              high_threshold=0.80, num_partitions=5,
              use_psl=False, use_pseudo=False):
    dataset = LoadUKGEDataset(root=root,
                              dataset_name='ppi5k',
                              num_neg_samples=num_neg,
                              batch_size=batch_size,
                              high_threshold=high_threshold,
                              num_partitions=num_partitions,
                              use_psl=use_psl,
                              use_pseudo=use_pseudo)
    dataloader = dataset.get_dataloader()
    return dataloader


if __name__ == '__main__':
    load_data(root='data', num_neg=2, batch_size=1024)
