import random
import torch


class UKGSampler:
    def __init__(self, data, num_neg, high_threshold, num_partitions):

        self.data = data
        self.num_neg = num_neg
        self.high_threshold = high_threshold
        self.num_partitions = num_partitions
        self.train_triples = data['train']
        self.val_triples = data['val']
        self.test_triples = data['test']
        self.soft_logic_triples = data['soft_logic']
        self.test_neg = data['test_neg']
        self.hr_map = data['hr_map']
        self.all_true_triples = data['all_true']
        self.num_ent = data['num_ent']
        self.num_rel = data['num_rel']
        self.ratio_psl = data['ratio_psl']
        # self.train_batch = None
        self.hr2t = {}
        self.tr2h = {}
        self.hr2t_high_score = {}
        self.tr2h_high_score = {}
        self.hr2t_full = {}
        self.tr2h_full = {}
        self.tr2h_confidence_partitions = []
        self.hr2t_confidence_partitions = []
        self.get_hr2t_rt2h_from_train()
        self.get_high_score_mappings_from_triples(self.high_threshold)
        self.partitioned_mappings_by_confidence(self.num_partitions)
        self.ss_pool = None
        self.ss_pool_score = None
        self.ss_pool_end = 1
        self.ss_pool_base = 0

    def get_hr2t_rt2h_from_train(self):
        """Get the set of hr2t and rt2h from train dataset, the data type is numpy.

        Update:
            self.hr2t_train: The set of hr2t.
            self.rt2h_train: The set of rt2h.
        """

        for head_id, relation_id, tail_id, _ in self.train_triples:
            if (head_id, relation_id) not in self.hr2t:
                self.hr2t[(head_id, relation_id)] = set()
            self.hr2t[(head_id, relation_id)].add(tail_id)

            if (tail_id, relation_id) not in self.tr2h:
                self.tr2h[(tail_id, relation_id)] = set()
            self.tr2h[(tail_id, relation_id)].add(head_id)

    def train_sampling(self, train_batch):
        """Filtering out positive samples and selecting some samples randomly as negative samples.

        Args:
            train_batch: The triples used to be sampled.

        Returns:
            batch_data: The training data.
        """

        batch_data = {}
        neg_ent_sample = []
        pos_triples = []
        probabilities = []
        for data in train_batch:
            head_id, relation_id, tail_id, probability = data
            neg_ent_sample = neg_ent_sample + self.generate_negative_samples(head_id, relation_id, tail_id)
            pos_triples.append((head_id, relation_id, tail_id))
            probabilities.append(probability)

        # self.train_batch = pos_triples
        batch_data["positive_sample"] = torch.LongTensor(pos_triples)
        batch_data['negative_sample'] = torch.LongTensor(neg_ent_sample)
        batch_data["probabilities"] = torch.FloatTensor(probabilities)
        return batch_data

    def train_psl_sampling(self, train_batch):
        """Filtering out positive samples and selecting some samples randomly as negative samples.

        Args:
            train_batch: The triples used to be sampled.

        Returns:
            batch_data: The training data.
        """

        batch_data = {}
        neg_ent_sample = []
        pos_triples = []
        probabilities = []
        psl_sample = []
        psl_pro = []
        for data in train_batch:
            head_id, relation_id, tail_id, probability = data
            neg_ent_sample = neg_ent_sample + self.generate_negative_samples(head_id, relation_id, tail_id)
            pos_triples.append((head_id, relation_id, tail_id))
            probabilities.append(probability)

        psl_ent_num = int(self.ratio_psl * len(train_batch))  # calculate the number of PSL samples in each batch, storage the PSL samples in batch_data['PSL_sample']
        # psl_ent_num = 0
        if len(self.soft_logic_triples) >= psl_ent_num + 1:
            for i in range(psl_ent_num + 1):
                psl_sample.append(self.soft_logic_triples[i][:3])
                psl_pro.append(self.soft_logic_triples[i][3])
            self.soft_logic_triples = self.soft_logic_triples[psl_ent_num + 1:]
        else:
            for i in range(len(self.soft_logic_triples)):
                psl_sample.append(self.soft_logic_triples[i][:3])
                psl_pro.append(self.soft_logic_triples[i][3])
        # psl_batch = random.choice(self.soft_logic_triples)
        # psl_sample = [psl_batch[:3]]
        # psl_pro = [psl_batch[3]]

        batch_data["positive_sample"] = torch.LongTensor(pos_triples)
        batch_data['negative_sample'] = torch.LongTensor(neg_ent_sample)
        batch_data["probabilities"] = torch.FloatTensor(probabilities)
        batch_data["psl_sample"] = torch.LongTensor(psl_sample)
        batch_data["psl_pro"] = torch.FloatTensor(psl_pro)
        return batch_data

    def semi_sampling(self, neg_data, model, n_generated_samples, n_new_samples, n_semi_samples, device, pool_size):

        semi_sample = neg_data.tolist()
        semi_sample_score = [0] * n_generated_samples
        # Initialize the semi-supervised pool as a list if it hasn't been initialized
        if self.ss_pool is None:
            self.ss_pool = [[0, 0, 0] for _ in range(pool_size)]
            self.ss_pool_score = [0] * pool_size

        # Only calculate scores if there are new samples to add

        with torch.no_grad():
            new_semi_scores = model(neg_data.to(device))

        new_semi_samples = semi_sample
        new_semi_scores = new_semi_scores.tolist()

        # Add new samples to the pool
        new_neg_pool_end = self.ss_pool_end + n_new_samples
        remained = max(0, new_neg_pool_end - pool_size)

        self.ss_pool[self.ss_pool_end:min(new_neg_pool_end, pool_size)] = new_semi_samples[
                                                                                  0:n_new_samples - remained]
        self.ss_pool_score[self.ss_pool_end:min(new_neg_pool_end, pool_size)] = new_semi_scores[
                                                                                        0:n_new_samples - remained]
        if remained > 0:
            self.ss_pool[0:remained] = new_semi_samples[n_new_samples - remained:n_new_samples]
            self.ss_pool_score[0:remained] = new_semi_scores[n_new_samples - remained:n_new_samples]

        # Update base and end of pool
        if self.ss_pool_end > self.ss_pool_base:
            self.ss_pool_base += pool_size
        self.ss_pool_end = new_neg_pool_end
        self.ss_pool_base = max(self.ss_pool_base, self.ss_pool_end) % pool_size
        self.ss_pool_end %= pool_size

        # Fetch existing semisupervised samples from the pool if needed
        if n_semi_samples > 0:
            pool_limit = self.ss_pool_end if self.ss_pool_base < self.ss_pool_end else pool_size
            start_index = random.randint(0, pool_limit - n_semi_samples)
            semi_sample[0:n_semi_samples] = self.ss_pool[start_index:start_index + n_semi_samples]
            semi_sample_score[0:n_semi_samples] = self.ss_pool_score[start_index:start_index + n_semi_samples]

        return semi_sample, semi_sample_score

    def generate_negative_samples(self, head_id, relation_id, tail_id):
        neg_samples = []
        for _ in range(self.num_neg):
            corrupted_head = random.randrange(self.num_ent)
            corrupted_tail = random.randrange(self.num_ent)

            while corrupted_head in self.tr2h.get((tail_id, relation_id), set()):
                corrupted_head = random.randrange(self.num_ent)
            neg_samples.append((corrupted_head, relation_id, tail_id))

            while corrupted_tail in self.hr2t.get((head_id, relation_id), set()):
                corrupted_tail = random.randrange(self.num_ent)
            neg_samples.append((head_id, relation_id, corrupted_tail))

        return neg_samples

    @staticmethod
    def generate_hr2t_tr2h_mappings(triples):
        """
        Generate hr2t and tr2h mappings from triples.

        Args:
            triples: List of triples represented as (head_id, relation_id, tail_id, confidence).

        Returns:
            hr2t: Dictionary mapping (head, relation) to a set of tails.
            tr2h: Dictionary mapping (tail, relation) to a set of heads.
        """
        hr2t, tr2h = {}, {}

        # Populate hr2t and tr2h mappings for each triple
        for head_id, relation_id, tail_id, _ in triples:
            hr2t.setdefault((head_id, relation_id), set()).add(tail_id)
            tr2h.setdefault((tail_id, relation_id), set()).add(head_id)

        return hr2t, tr2h

    def partitioned_mappings_by_confidence(self, num_partitions: int) -> None:
        """
        Generate hr2t and tr2h mappings for each confidence partition.

        Args:
            num_partitions: Number of partitions for confidence levels.

        Updates:
            self.hr2t_confidence_partitions: List of hr2t mappings for each confidence partition.
            self.tr2h_confidence_partitions: List of tr2h mappings for each confidence partition.
            self.hr2t_full and self.tr2h_full: Mappings across all triples for compatibility.
        """
        # Partition triples based on confidence levels
        partitioned_triples = self.split_triples_by_confidence(self.all_true_triples, num_partitions)

        # Generate mappings for each partition and store in lists
        for partition in partitioned_triples:
            hr2t, tr2h = self.generate_hr2t_tr2h_mappings(partition)
            self.hr2t_confidence_partitions.append(hr2t)
            self.tr2h_confidence_partitions.append(tr2h)

        # Store the full mappings for use across all triples
        self.hr2t_full, self.tr2h_full = self.generate_hr2t_tr2h_mappings(self.all_true_triples)

    def get_high_score_mappings_from_triples(self, confidence_threshold):
        for head_id, relation_id, tail_id, probability in self.all_true_triples:
            if probability >= confidence_threshold:
                self.hr2t_high_score.setdefault((head_id, relation_id), set()).add(tail_id)
                self.tr2h_high_score.setdefault((tail_id, relation_id), set()).add(head_id)

    @staticmethod
    def split_triples_by_confidence(triples, num_partitions):
        """
        Partition triples into `num_partitions` based on confidence levels.

        Args:
            triples: List of triples represented as (head_id, relation_id, tail_id, confidence).
            num_partitions: Number of partitions to create.

        Returns:
            List of lists, where each sublist contains triples for one confidence partition.
        """
        # Initialize partitions: each partition holds triples within a specific confidence range
        partitions = [[] for _ in range(num_partitions)]

        # Place each triple in the appropriate partition based on its confidence level
        for triple in triples:
            _, _, _, confidence = triple
            partition_index = min(int(confidence * num_partitions), num_partitions - 1)
            partitions[partition_index].append(triple)

        return partitions

    def val_sampling(self, datasets):
        processed_data = {}
        triples = []
        high_triples = []
        probabilities = []
        high_probabilities = []
        for data in datasets:
            head_id, relation_id, tail_id, probability = data
            triples.append((head_id, relation_id, tail_id))
            probabilities.append(probability)
            if probability >= self.high_threshold:
                high_triples.append((head_id, relation_id, tail_id))
                high_probabilities.append(probability)
        processed_data["triples"] = torch.LongTensor(triples)
        processed_data["probabilities"] = torch.FloatTensor(probabilities)
        processed_data["len_triples"] = len(triples)
        processed_data["high_triples"] = torch.LongTensor(high_triples)
        processed_data["high_probabilities"] = torch.FloatTensor(high_probabilities)
        processed_data["len_high_triples"] = len(high_triples)

        processed_data["hr_map"] = self.hr_map
        processed_data["hr2t_high_score"] = self.hr2t_high_score
        processed_data["tr2h_high_score"] = self.tr2h_high_score
        processed_data["num_partitions"] = self.num_partitions
        processed_data["hr2t_partition_mappings"] = self.hr2t_confidence_partitions
        processed_data["tr2h_partition_mappings"] = self.tr2h_confidence_partitions
        processed_data["hr2t_full"] = self.hr2t_full
        processed_data["tr2h_full"] = self.tr2h_full
        return processed_data

    def test_sampling(self, datasets):
        processed_data = {}
        triples = []
        high_triples = []
        probabilities = []
        high_probabilities = []

        for data in datasets:
            head_id, relation_id, tail_id, probability = data
            triples.append((head_id, relation_id, tail_id))
            probabilities.append(probability)
            if probability >= self.high_threshold:
                high_triples.append((head_id, relation_id, tail_id))
                high_probabilities.append(probability)
        processed_data["triples"] = torch.LongTensor(triples)
        processed_data["probabilities"] = torch.FloatTensor(probabilities)
        processed_data["len_triples"] = len(triples)
        processed_data["test_neg"] = torch.LongTensor([row[:3] for row in self.test_neg])
        processed_data["test_neg_pro"] = torch.FloatTensor([row[3] for row in self.test_neg])
        processed_data["high_triples"] = torch.LongTensor(high_triples)
        processed_data["high_probabilities"] = torch.FloatTensor(high_probabilities)
        processed_data["len_high_triples"] = len(high_triples)

        processed_data["hr_map"] = self.hr_map
        processed_data["hr2t_high_score"] = self.hr2t_high_score
        processed_data["tr2h_high_score"] = self.tr2h_high_score
        processed_data["num_partitions"] = self.num_partitions
        processed_data["hr2t_partition_mappings"] = self.hr2t_confidence_partitions
        processed_data["tr2h_partition_mappings"] = self.tr2h_confidence_partitions
        processed_data["hr2t_full"] = self.hr2t_full
        processed_data["tr2h_full"] = self.tr2h_full
        return processed_data

    def get_train_triples(self):
        return self.train_triples

    def get_valid_triples(self):
        return self.val_triples

    def get_test_triples(self):
        return self.test_triples

    def get_soft_logic_data(self):
        return self.soft_logic_triples

    def get_all_true_triples(self):
        return self.all_true_triples

    def get_num_rel(self):
        return self.num_rel

    def get_num_ent(self):
        return self.num_ent


if __name__ == '__main__':
    from load_data import UKGData
    from torch.utils.data import DataLoader

    dataset = UKGData(dataset_dir='../../data/cn15k', use_index_file=True)
    final_data = dataset.get_data()
    Sampler = UKGSampler(final_data, 10, high_threshold=0.85, num_partitions=4)
    dataloader = DataLoader(final_data['train'],
                            batch_size=1024,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=True,
                            shuffle=True,
                            collate_fn=Sampler.train_sampling)
    for batch in dataloader:
        print(batch)
