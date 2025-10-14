from .load_data import UKGData
from .ukg_sampler import UKGSampler
from .ukg_data_module import UKGDataModule
from ..utils import download_dataset


class LoadUKGEDataset:
    def __init__(self, root=None,
                 dataset_name=None,
                 num_neg_samples=None,
                 batch_size=None,
                 high_threshold=None,
                 num_partitions=None,
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

    def get_dataloader(self):
        data_path = download_dataset(dataset_name=self.dataset_name, download_path=self.root)
        dataset = UKGData(dataset_dir=data_path, use_index_file=True)
        final_data = dataset.get_data()
        sampler = UKGSampler(final_data, self.num_neg, self.high_threshold, self.num_partitions)
        dataloader = UKGDataModule(sampler=sampler, batch_size=self.batch_size, num_workers=0)

        return dataloader
