from ..dataset import LoadUKGEDataset, LoadFewShotDataset


def load_data(root, num_neg, batch_size, high_threshold=0.80, num_partitions=5):
    dataset = LoadUKGEDataset(root=root,
                              dataset_name='nl27k',
                              num_neg_samples=num_neg,
                              batch_size=batch_size,
                              high_threshold=high_threshold,
                              num_partitions=num_partitions)
    dataloader = dataset.get_dataloader()
    return dataloader

def load_few_shot_data(root, num_neg, batch_size, max_neighbor, few, type_constrain, has_ont=False, rel_uc=None):
    dataset = LoadFewShotDataset(root=root, dataset_name='nl27k-few-shot',
                                 num_neg_samples=num_neg, batch_size=batch_size,
                                 max_neighbor = max_neighbor, has_ont = has_ont, few=few,
                                 type_constrain=type_constrain, rel_uc=rel_uc)
    dataloader = dataset.get_dataloader()
    return dataloader


if __name__ == '__main__':
    load_data(root='data', num_neg=2, batch_size=1024)
