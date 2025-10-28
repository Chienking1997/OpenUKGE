from .load_data import UKGData, FewShotData
from .ukg_sampler import UKGSampler, FewShotSampler
from .ukg_data_module import UKGDataModule, FewShotModule
from ..utils import download_dataset


class LoadUKGEDataset:
    def __init__(self, root=None,
                 dataset_name=None,
                 num_neg_samples=None,
                 batch_size=None,
                 high_threshold=None,
                 num_partitions=None,
                 use_pseudo=False,
                 config=None):
        if config:
            self.config = config
            self.root = config.root
        self.root = root
        self.dataset_name = dataset_name
        self.num_neg = num_neg_samples
        self.batch_size = batch_size
        self.high_threshold = high_threshold
        self.num_partitions = num_partitions
        self.use_pseudo = use_pseudo

    def get_dataloader(self):
        data_path = download_dataset(dataset_name=self.dataset_name, download_path=self.root)
        dataset = UKGData(dataset_dir=data_path, use_index_file=True, use_pseudo=self.use_pseudo)
        final_data = dataset()
        sampler = UKGSampler(final_data, self.num_neg, self.high_threshold, self.num_partitions)
        dataloader = UKGDataModule(sampler=sampler,
                                   batch_size=self.batch_size,
                                   num_workers=0,
                                   use_pseudo=self.use_pseudo)

        return dataloader



class LoadFewShotDataset:
    def __init__(self, root=None,
                 dataset_name=None,
                 num_neg_samples=None,
                 batch_size=None,
                 max_neighbor = None,
                 has_ont = False,
                 few=None,
                 type_constrain=None,
                 rel_uc=None,
                 config=None):
        if config:
            self.config = config
            self.root = config.root
        self.root = root
        self.dataset_name = dataset_name
        self.num_neg = num_neg_samples
        self.batch_size = batch_size
        self.few = few
        self.type_constrain = type_constrain
        self.rel_uc = rel_uc
        self.max_nbr = max_neighbor
        self.has_ont = has_ont
        # self.base_data = self.dataset_name.replace("-few-shot", "")

    def get_dataloader(self):
        data_path = download_dataset(dataset_name=self.dataset_name, download_path=self.root)
        few_shot_dataset = FewShotData(data_path, self.dataset_name, self.max_nbr, self.has_ont)
        final_data = few_shot_dataset.get_data()
        sampler = FewShotSampler(final_data, self.num_neg, self.batch_size, few=self.few,
                                 type_constrain=self.type_constrain, has_ont=self.has_ont, rel_uc=self.rel_uc)
        dataloader = FewShotModule(sampler=sampler, num_workers=0)

        return dataloader


if __name__ == '__main__':
    dataset = LoadFewShotDataset()