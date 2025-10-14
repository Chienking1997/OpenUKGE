import os
import random
import pickle


class UKGData:
    def __init__(self, dataset_dir="", use_index_file=False):
        """
        Initialize the UKGData class.

        Args:
            dataset_dir (str): The directory path of the dataset.
            use_index_file (bool): Whether to use the index file.
        """
        self.dataset_dir = dataset_dir
        self.use_index_file = use_index_file
        # Initialize data structures to store the data
        self.train_data = []
        self.val_data = []
        self.test_data = []
        self.test_data_neg = []
        self.soft_logic_data = []
        self.all_data = []
        self.all_true_data = set()
        # Initialize dictionaries to store the mappings
        self.ent_id = {}
        self.rel_id = {}
        self.id_ent = {}
        self.id_rel = {}
        self.RatioOfPSL = 0.0
        # Load and map the dataset
        self.load_and_map_dataset()
        self.hr_map = self.load_hr_map(self.dataset_dir)
        # self.load_hr_map2(self.dataset_dir,'val.tsv', ['train.tsv', 'val.tsv', 'test.tsv'])
        # self.hr_map_sub = None
        # self.get_fixed_hr(n=200)


    def load_and_map_dataset(self):
        """
        Load and map the dataset.

        If the index file exists, load the mappings from the file.
        Otherwise, generate the mappings and save them to the file.
        """
        entity_mapping_file = os.path.join(self.dataset_dir, 'entity_id.csv')
        relation_mapping_file = os.path.join(self.dataset_dir, 'relation_id.csv')

        if self.use_index_file and os.path.exists(entity_mapping_file) and os.path.exists(relation_mapping_file):
            # Load the mappings from the file
            self.ent_id, self.id_ent = self.load_mapping(entity_mapping_file)
            self.rel_id, self.id_rel = self.load_mapping(relation_mapping_file)
            # Load the data
            self.all_data = self.load_all_data(self.dataset_dir)
            self.soft_logic_data = self.load_soft_data(self.dataset_dir)
            self.RatioOfPSL = len(self.soft_logic_data) / len(self.train_data)
            self.test_data_neg = self.load_test_neg_data(self.dataset_dir)
            self.all_true_data = set(self.all_data)
        else:
            # Load the data
            all_data = self.load_all_data(self.dataset_dir)
            # Generate the mappings
            self.ent_id, self.id_ent, self.rel_id, self.id_rel = self.generate_mappings(all_data)
            # Map the data to indices
            self.train_data = self.triple2index(self.train_data, self.ent_id, self.rel_id)
            self.val_data = self.triple2index(self.val_data, self.ent_id, self.rel_id)
            self.test_data = self.triple2index(self.test_data, self.ent_id, self.rel_id)
            self.soft_logic_data = self.load_soft_data(self.dataset_dir)
            self.soft_logic_data = self.triple2index(self.soft_logic_data, self.ent_id, self.rel_id)
            # Combine the data
            self.all_data = self.train_data + self.val_data + self.test_data
            self.all_true_data = set(self.all_data)
            # Save the mappings to the file
            self.save_mapping(self.ent_id, entity_mapping_file)
            self.save_mapping(self.rel_id, relation_mapping_file)
            self.RatioOfPSL = len(self.soft_logic_data) / len(self.train_data)

    @staticmethod
    def load_mapping(file_path):
        """
        Load a mapping from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            tuple: A tuple containing the key-to-id and id-to-key mappings.
        """
        key2id = {}
        id2key = {}
        with open(file_path, mode='r') as infile:
            # Skip the header line
            # next(infile)
            for line in infile:
                key, value = line.strip().split(',')
                key2id[key] = int(value)
                id2key[int(value)] = key
        return key2id, id2key

    @staticmethod
    def load_data(file_path):
        """
        Load data from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            list: A list of tuples containing the data.
        """
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                head, relation, tail, probability = line.strip().split('\t')
                data.append((int(head), int(relation), int(tail), float(probability)))
        return data

    @staticmethod
    def generate_mappings(all_data):
        """
        Generate the mappings from the data.

        Args:
            all_data (list): A list of tuples containing the data.

        Returns:
            tuple: A tuple containing the entity-to-id, id-to-entity, relation-to-id, and id-to-relation mappings.
        """
        # Get all unique entities
        entities = set(item[0] for item in all_data).union(set(item[2] for item in all_data))

        # Get all unique relations
        relations = set(item[1] for item in all_data)

        # Generate the entity-to-id mapping
        ent2id = {entity: idx for idx, entity in enumerate(entities)}

        # Generate the id-to-entity mapping
        id2ent = {idx: entity for entity, idx in ent2id.items()}

        # Generate the relation-to-id mapping
        rel2id = {relation: idx for idx, relation in enumerate(relations)}

        # Generate the id-to-relation mapping
        id2rel = {idx: relation for relation, idx in rel2id.items()}

        return ent2id, id2ent, rel2id, id2rel

    @staticmethod
    def triple2index(triples, ent_id, rel_id):
        """
        Map a list of triples to indices.

        Args:
            triples (list): A list of tuples containing the triples.
            ent_id (dict): The entity-to-id mapping.
            rel_id (dict): The relation-to-id mapping.

        Returns:
            list: A list of tuples containing the mapped triples.
        """
        data = []
        for triple in triples:
            head, relation, tail, probability = triple
            head_id = ent_id[head]
            relation_id = rel_id[relation]
            tail_id = ent_id[tail]
            data.append((head_id, relation_id, tail_id, probability))
        return data

    @staticmethod
    def save_mapping(mapping, file_path):
        """
        Save a mapping to a file.

        Args:
            mapping (dict): The mapping to save.
            file_path (str): The path to the file.
        """
        with open(file_path, mode='w') as outfile:
            for key, value in mapping.items():
                outfile.write(f"{key},{value}\n")
            outfile.close()

    def load_all_data(self, dataset_folder):
        """
        Load all data from a dataset folder.

        Args:
            dataset_folder (str): The path to the dataset folder.

        Returns:
            list: A list of tuples containing the data.
        """
        train_file = os.path.join(dataset_folder, 'train.tsv')
        val_file = os.path.join(dataset_folder, 'val.tsv')
        test_file = os.path.join(dataset_folder, 'test.tsv')

        self.train_data = self.load_data(train_file)
        self.val_data = self.load_data(val_file)
        self.test_data = self.load_data(test_file)

        return self.train_data + self.val_data + self.test_data

    def load_soft_data(self, dataset_folder):
        soft_logic_data = os.path.join(dataset_folder, 'extra', 'softlogic.tsv')
        return self.load_data(soft_logic_data)

    def load_test_neg_data(self, dataset_folder):
        test_neg_file = os.path.join(dataset_folder, 'extra', 'test', 'test_with_neg_beurre.tsv')
        return self.load_data(test_neg_file)

    @staticmethod
    def load_hr_map(data_dir):
        file = os.path.join(data_dir, 'extra', 'test', 'ndcg_test_beurre.pickle')
        with open(file, 'rb') as f:
            hr_map = pickle.load(f)
        return hr_map

    def load_hr_map2(self, data_dir, hr_base_file, supplement_t_files, splitter='\t', line_end='\n'):
        """
        Initialize self.hr_map.
        Load self.hr_map={h:{r:t:w}}}, not restricted to test data
        :param hr_base_file: Get self.hr_map={h:r:{t:w}}} from the file.
        :param supplement_t_files: Add t(only t) to self.hr_map. Don't add h or r.
        :return:
        """
        self.hr_map = {}
        with open(os.path.join(data_dir, hr_base_file)) as f:
            for line in f:
                line = line.rstrip(line_end).split(splitter)
                h = line[0]
                r = line[1]
                t = line[2]
                w = float(line[3])
                # construct hr_map
                if self.hr_map.get(h) == None:
                    self.hr_map[h] = {}
                if self.hr_map[h].get(r) == None:
                    self.hr_map[h][r] = {t: w}
                else:
                    self.hr_map[h][r][t] = w

        count = 0
        for h in self.hr_map:
            count += len(self.hr_map[h])
        print('Loaded ranking test queries. Number of (h,r,?t) queries: %d' % count)

        for file in supplement_t_files:
            with open(os.path.join(data_dir, file)) as f:
                for line in f:
                    line = line.rstrip(line_end).split(splitter)
                    h = line[0]
                    r = line[1]
                    t = line[2]
                    w = float(line[3])

                    # update hr_map
                    if h in self.hr_map and r in self.hr_map[h]:
                        self.hr_map[h][r][t] = w

    def get_fixed_hr(self, outputdir=None, n=500):
        hr_map500 = {}
        dict_keys = []
        for h in self.hr_map.keys():
            for r in self.hr_map[h].keys():
                dict_keys.append([h, r])

        dict_keys = sorted(dict_keys, key=lambda x: len(self.hr_map[x[0]][x[1]]), reverse=True)
        dict_final_keys = []

        for i in range(2525):
            dict_final_keys.append(dict_keys[i])

        count = 0
        for i in range(n):
            temp_key = random.choice(dict_final_keys)
            h = temp_key[0]
            r = temp_key[1]
            for t in self.hr_map[h][r]:
                w = self.hr_map[h][r][t]
                if hr_map500.get(h) is None:
                    hr_map500[h] = {}
                if hr_map500[h].get(r) is None:
                    hr_map500[h][r] = {t: w}
                else:
                    hr_map500[h][r][t] = w

        for h in hr_map500.keys():
            for r in hr_map500[h].keys():
                count = count + 1

        self.hr_map_sub = hr_map500

        if outputdir is not None:
            with open(outputdir, 'wb') as handle:
                pickle.dump(hr_map500, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return hr_map500

    def print_dataset_info(self):
        """
        Print information about the dataset.
        """
        print(f"  Number of triples: {len(self.all_data)}")
        print(f"  Number of non-repetitive triplets: {len(self.all_true_data)}")
        print(f"  Number of entities: {len(self.ent_id)}")
        print(f"  Number of relations: {len(self.rel_id)}")

    def get_data(self):
        """
        Return all processed data and mappings.

        Returns:
            dict: A dictionary containing the processed data and mappings with the following keys:
                - 'train': The training data.
                - 'val': The validation data.
                - 'test': The test data.
                - 'all': All the data.
                - 'all_true': All the non-repetitive data.
                - 'ent_id': The entity-to-id mapping.
                - 'rel_id': The relation-to-id mapping.
                - 'id_ent': The id-to-entity mapping.
                - 'id_rel': The id-to-relation mapping.
                - 'ratio_psl': calculate the ratio of the number of PSL samples to the number of training samples.
        """
        return {
            'train': self.train_data,
            'val': self.val_data,
            'test': self.test_data,
            'test_neg': self.test_data_neg,
            'all': self.all_data,
            'soft_logic': self.soft_logic_data,
            'all_true': self.all_true_data,
            'hr_map': self.hr_map,
            'ent_id': self.ent_id,
            'rel_id': self.rel_id,
            'id_ent': self.id_ent,
            'id_rel': self.id_rel,
            'num_ent': len(self.ent_id),
            'num_rel': len(self.rel_id),
            'ratio_psl': self.RatioOfPSL
        }


if __name__ == '__main__':
    dataset = UKGData(dataset_dir='../../datasets/cn15k', use_index_file=True)
    final_data = dataset.get_data()
    dataset.print_dataset_info()
