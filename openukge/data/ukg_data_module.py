from torch.utils.data import DataLoader


class UKGDataModule:
    def __init__(self, sampler=None, batch_size=None, num_workers=None, config=None, use_pseudo = False):
        if config is not None:
            self.config = config
            self.num_workers = self.config.num_workers

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampler = sampler
        self.use_pseudo = use_pseudo
        self.num_ent = self.sampler.get_num_ent()
        self.num_rel = self.sampler.get_num_rel()
        self.train = self.sampler.get_train_triples()
        self.valid = self.sampler.get_valid_triples()
        self.test = self.sampler.get_test_triples()
        self.hr2t = self.sampler.get_hr2t()
        if self.use_pseudo:
            self.pseudo = self.sampler.get_pseudo_data()

    def train_dataloader(self, psl=False):
        if not psl:
            return DataLoader(
                self.train,
                shuffle=True,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                # pin_memory=True,
                drop_last=True,
                collate_fn=self.sampler.train_sampling,
            )
        else:
            return DataLoader(
                self.train,
                shuffle=True,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                # pin_memory=True,
                drop_last=True,
                collate_fn=self.sampler.train_psl_sampling,
            )

    def val_dataloader(self):
        return self.sampler.val_sampling(self.valid)

    def test_dataloader(self):
        return self.sampler.test_sampling(self.test)

    def semi_dataloader(self, neg_data, model, epoch, device, t_new_semi, t_semi_train):
        n_generated_samples = len(neg_data)  # semi sample used for training
        n_new_samples = len(neg_data) // 2 if epoch >= t_new_semi else 0  # new sample for the pool
        # original (M_0.8)
        n_semi_samples = int(min(n_generated_samples * 0.8, max(0, -t_semi_train + epoch) * n_generated_samples * 0.02))
        semi_batch = self.sampler.semi_sampling(neg_data, model, n_generated_samples,
                                                n_new_samples, n_semi_samples, device, 10000000)

        return semi_batch



    def gcn_dataloader(self):
        if self.use_pseudo:
            pseudo_bs = len(self.pseudo) // (len(self.train) // self.batch_size)
            self.sampler.get_adj_matrix(use_pseudo=self.use_pseudo, pseudo_data=self.pseudo)
            return DataLoader(self.pseudo,
                shuffle=True,
                batch_size=pseudo_bs,
                num_workers=self.num_workers,
                # pin_memory=True,
                drop_last=True,
                collate_fn=self.sampler.pseudo_sampling)
        else:
            adj_matrix  = Repeater(self.sampler.get_adj_matrix())
            return adj_matrix


class Repeater:
    def __init__(self, value):
        self.value = value

    def __iter__(self):
        return self

    def __next__(self):
        return self.value


class FewShotModule:
    def __init__(self, sampler, num_workers, config=None):
        self.config = config
        self.sampler = sampler
        self.num_symbols = self.sampler.get_num_symbols()
        self.few = self.sampler.get_few()
        self.ent2id = self.sampler.get_ent2id()
        self.type_constrain = self.sampler.get_type_constrain()
        self.rel2candidates = self.sampler.get_rel2candidates()
        self.rele1_e2 = self.sampler.get_rele1_e2()
        self.symbol2id = self.sampler.get_symbol2id()
        self.e1rel_e2 = self.sampler.get_e1rel_e2()
        self.num_workers = num_workers
        self.train = self.sampler.get_train()
        self.valid = self.sampler.get_valid()
        self.test = self.sampler.get_test()

    def train_dataloader(self):
        return DataLoader(
            self.train,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.sampler.sampling,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.sampler.test_sampling,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.sampler.test_sampling,
        )

    def device_setter(self, device):
        self.sampler.get_device(device)