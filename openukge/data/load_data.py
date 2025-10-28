import os
import random
import pickle
import json
import math
import copy
from collections import defaultdict as ddict
from statistics import mean, pstdev
from typing import Dict, Tuple, List, Union, Any
from tqdm import tqdm


class UKGData:
    def __init__(self, dataset_dir="", use_index_file=True, use_pseudo=False):
        """
        UKGData: Uncertain Knowledge Graph dataset loader.

        Args:
            dataset_dir (str):
                The directory path of the dataset.
            use_index_file (bool):
                Whether to use predefined index mapping files.
                - True: Entity/relation mapping files exist and will be loaded.
                - False: Entity/relation names in triples will be mapped to IDs dynamically.
            use_pseudo (bool):
                Whether to load pseudo-neighbor triples for UPGAT student model training.
        """

        # Dataset directory path
        self.dataset_dir = dataset_dir

        # Use pre-generated entity/relation index files
        # (If the UKG dataset is already indexed, these mapping files must be provided.
        # Otherwise, ID mapping will be generated automatically based on raw triples.)
        self.use_index_file = use_index_file

        # Core dataset splits
        self.train_data = []       # Training triples
        self.val_data = []         # Validation triples
        self.test_data = []        # Test triples
        self.test_data_neg = []    # Test triples with random negative samples (from BEUrRE-provided datasets)

        # Additional uncertain logical knowledge from soft logic inference
        self.soft_logic_data = []  # Soft logic triples with confidence

        # Combined dataset: train + validation + test
        self.all_data = []

        # Unique true triples stored as a set (used for filtered ranking evaluation)
        self.all_true_data = set()

        # Entity & relation mappings
        self.ent_id = {}  # entity_name -> entity_id
        self.rel_id = {}  # relation_name -> relation_id
        self.id_ent = {}  # entity_id -> entity_name
        self.id_rel = {}  # relation_id -> relation_name

        # Ratio of soft logic data used during training
        self.RatioOfPSL = 0.0

        # Load and map the dataset according to current configuration
        self.load_and_map_dataset()

        # Head-Relation map for NDCG in test stage
        self.hr_map = self.load_hr_map(self.dataset_dir)

        # Whether to load pseudo neighbors from teacher model reasoning
        # (Used only in UPGAT student model training)
        self.use_pseudo = use_pseudo
        if use_pseudo:
            self.pseudo_data = self.load_pseudo_data(self.dataset_dir)

        # Head-Relation map for validation stage
        self.hr_map_val = self.load_hr_map2()

    def __len__(self) -> int:
        """Return total number of triples in the dataset."""
        return len(self.all_data)

    def __str__(self) -> str:
        """
        Human-friendly, readable dataset summary.
        Triggered by print(obj).
        """
        return (
            f"Dataset Statistics:\n"
            f"  Total triples: {len(self.all_data)}\n"
            f"  Unique triples: {len(self.all_true_data)}\n"
            f"  Entities: {len(self.ent_id)}\n"
            f"  Relations: {len(self.rel_id)}"
        )

    def __repr__(self) -> str:
        """
        Developer-focused representation for debugging/logging.
        Triggered in REPL/Jupyter or by simply typing the object.
        """
        return (
            f"{self.__class__.__name__}("
            f"triples={len(self.all_data)}, "
            f"unique={len(self.all_true_data)}, "
            f"entities={len(self.ent_id)}, "
            f"relations={len(self.rel_id)})"
        )

    def __call__(self):
        """
        Callable interface to return dataset contents.

        Returns:
            dict: Packaged dataset structure including:
                - train / val / test: Labeled triple data
                - test_neg: Negative triples for evaluation
                - all / all_true: Full and unique triple sets
                - soft_logic: Weighted PSL triples
                - hr_map: Hierarchical relation structure: hr_map[h][r][t] -> float(weight)
                - ent_id / rel_id: Entity / Relation → ID mappings
                - id_ent / id_rel: Reverse ID → Symbol mappings
                - num_ent / num_rel: Cardinality of entity / relation vocabularies
                - ratio_psl: |PSL| / |Train| ratio
                - pseudo: Optional pseudo-labeled triples (List[Tuple[int,int,int,float]])
        """
        return {
            'train': self.train_data,
            'val': self.val_data,
            'test': self.test_data,
            'test_neg': self.test_data_neg,
            'all': self.all_data,
            'soft_logic': self.soft_logic_data,
            'all_true': self.all_true_data,
            'hr_map': self.hr_map,          # type: Dict[int, Dict[int, Dict[int, float]]]
            'hr_map_val': self.hr_map_val,  # type: Dict[int, Dict[int, Dict[int, float]]]
            'ent_id': self.ent_id,          # type: Dict[str, int]
            'rel_id': self.rel_id,          # type: Dict[str, int]
            'id_ent': self.id_ent,          # type: Dict[int, str]
            'id_rel': self.id_rel,          # type: Dict[int, str]
            'num_ent': len(self.ent_id),
            'num_rel': len(self.rel_id),
            'ratio_psl': self.RatioOfPSL,
            'pseudo': self.pseudo_data if self.use_pseudo else None
        }

    def load_and_map_dataset(self):
        """
        Load triples and construct entity/relation ID mappings.

        If ID mapping files exist and use_index_file=True:
            - Load ent/rel mappings directly from file
            - Load datasets already in mapped ID format

        Otherwise:
            - Load original triples (entity/relation names)
            - Generate new ent/rel mappings
            - Convert triples to ID-based format
            - Save mappings back to disk
        """

        entity_map_file = os.path.join(self.dataset_dir, "entity_id.csv")
        relation_map_file = os.path.join(self.dataset_dir, "relation_id.csv")
        mapping_files_exist = (
                os.path.exists(entity_map_file)
                and os.path.exists(relation_map_file)
        )

        if self.use_index_file and mapping_files_exist:
            # ----- ① Load existing index mappings -----
            self.ent_id, self.id_ent = self.load_mapping(entity_map_file)
            self.rel_id, self.id_rel = self.load_mapping(relation_map_file)

            # Load already indexed data
            self.all_data = self.load_all_data(self.dataset_dir)
            self.soft_logic_data = self.load_soft_data(self.dataset_dir)
            self.test_data_neg = self.load_test_neg_data(self.dataset_dir)

            # Compute ratio for soft logic guidance
            self.RatioOfPSL = len(self.soft_logic_data) / len(self.train_data)
            self.all_true_data = set(self.all_data)

        else:
            # ----- ② Build new mappings from original triples -----
            raw_all = self.load_all_data(self.dataset_dir)

            # Automatically generate ent/rel index dictionaries
            self.ent_id, self.id_ent, self.rel_id, self.id_rel = \
                self.generate_mappings(raw_all)

            # Convert train / val / test triples to index format
            self.train_data = self.triple2index(self.train_data, self.ent_id, self.rel_id)
            self.val_data = self.triple2index(self.val_data, self.ent_id, self.rel_id)
            self.test_data = self.triple2index(self.test_data, self.ent_id, self.rel_id)

            # Save ent/rel mappings for future reuse
            self.save_mapping(self.ent_id, entity_map_file)
            self.save_mapping(self.rel_id, relation_map_file)

            # Convert soft logic triples
            self.soft_logic_data = self.triple2index(
                self.load_soft_data(self.dataset_dir),
                self.ent_id, self.rel_id
            )

            # Combine all true triples
            self.all_data = self.train_data + self.val_data + self.test_data
            self.all_true_data = set(self.all_data)

            # Compute ratio for soft logic guidance
            self.RatioOfPSL = len(self.soft_logic_data) / len(self.train_data)


    @staticmethod
    def load_mapping(file_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Load an entity/relation mapping file (CSV format).

        Each line format should be: <name(string)>,<id(int)>

        Args:
            file_path (str): Path to mapping CSV file.

        Returns:
            Tuple[Dict[str, int], Dict[int, str]]:
                - key2id: mapping from token string -> integer ID
                - id2key: reverse mapping from integer ID -> token string

        Raises:
            FileNotFoundError: If the mapping file does not exist.
            ValueError: If a line is incorrectly formatted.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Mapping file not found: {file_path}")

        key2id: Dict[str, int] = {}
        id2key: Dict[int, str] = {}

        with open(file_path, mode='r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                parts = line.split(',')
                if len(parts) != 2:
                    raise ValueError(f"Invalid mapping line format: '{line}'")

                key, id_str = parts
                try:
                    id_val = int(id_str)
                except ValueError:
                    raise ValueError(
                        f"Mapping ID must be integer, got: '{id_str}' in line '{line}'"
                    )

                key2id[key] = id_val
                id2key[id_val] = key

        return key2id, id2key

    @staticmethod
    def load_data(
            file_path: str
    ) -> List[Tuple[Union[int, str], Union[int, str], Union[int, str], float]]:
        """
        Load triple data from a tab-separated file. Automatically convert
        head/relation/tail to int if they are numeric IDs.

        Expected format per line:
            head<TAB>relation<TAB>tail<TAB>probability
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        def parse_value(x: str) -> Union[int, str]:
            """Convert to int if numeric, otherwise keep as string."""
            return int(x) if x.isdigit() else x

        data: List[Tuple[Union[int, str], Union[int, str], Union[int, str], float]] = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) != 4:
                    raise ValueError(f"Expected 4 fields, got {len(parts)}: '{line}'")

                head, relation, tail, prob_str = parts

                try:
                    prob = float(prob_str)
                except ValueError:
                    raise ValueError(f"Invalid probability: '{prob_str}' in line: '{line}'")

                data.append((
                    parse_value(head),
                    parse_value(relation),
                    parse_value(tail),
                    prob,
                ))

        return data

    @staticmethod
    def generate_mappings(
            all_data: List[Tuple[str, str, str, float]]
    ) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
        """
        Generate bidirectional ID mappings for entities and relations from a dataset.

        Parameters
        ----------
        all_data : List[Tuple[Any, Any, Any, float]]
            A list of (head, relation, tail, probability) tuples.

        Returns
        -------
        Tuple[
            Dict[Union[int, str], int],  # dict mapping entity label -> integer ID
            Dict[int, Union[int, str]],  # dict mapping integer ID -> entity label
            Dict[Union[int, str], int],  # dict mapping relation label -> integer ID
            Dict[int, Union[int, str]]   # dict mapping integer ID -> relation label
        ]
            Bidirectional mappings for entities and relations.
        """
        if not all_data:
            raise ValueError("Input dataset is empty. Cannot generate mappings.")

        # Extract all unique entities from head and tail
        entities = {item[0] for item in all_data}.union({item[2] for item in all_data})

        # Extract all unique relations
        relations = {item[1] for item in all_data}

        # Generate entity mappings
        ent2id: Dict[str, int] = {entity: idx for idx, entity in enumerate(entities)}
        id2ent: Dict[int, str] = {idx: entity for entity, idx in ent2id.items()}

        # Generate relation mappings
        rel2id: Dict[str, int] = {relation: idx for idx, relation in enumerate(relations)}
        id2rel: Dict[int, str] = {idx: relation for relation, idx in rel2id.items()}

        return ent2id, id2ent, rel2id, id2rel

    @staticmethod
    def triple2index(
            triples: List[Tuple[str, str, str, float]],
            ent_id: Dict[str, int],
            rel_id: Dict[str, int]
    ) -> List[Tuple[int, int, int, float]]:
        """
        Convert a list of triples (head, relation, tail, probability) from
        string labels to integer IDs using provided mappings.

        Parameters
        ----------
        triples : List[Tuple[str, str, str, float]]
            List of triples with string labels for head, relation, and tail.
        ent_id : Dict[str, int]
            Mapping from entity label to integer ID.
        rel_id : Dict[str, int]
            Mapping from relation label to integer ID.

        Returns
        -------
        List[Tuple[int, int, int, float]]
            List of triples with head, relation, and tail replaced by their
            corresponding integer IDs. Probability is preserved.

        Raises
        ------
        KeyError
            If a head, relation, or tail in a triple is not found in the mapping.
        ValueError
            If a triple does not have exactly 4 elements.
        """
        if not triples:
            raise ValueError("Input triple list is empty.")
        indexed_data: List[Tuple[int, int, int, float]] = []
        for triple in triples:
            if len(triple) != 4:
                raise ValueError(f"Invalid triple format (expected 4 elements): {triple}")
            head, relation, tail, prob = triple
            try:
                head_id = ent_id[head]
                relation_id = rel_id[relation]
                tail_id = ent_id[tail]
            except KeyError as e:
                raise KeyError(f"Mapping not found for {e.args[0]} in triple {triple}")
            indexed_data.append((head_id, relation_id, tail_id, prob))
        return indexed_data

    @staticmethod
    def save_mapping(mapping, file_path):
        """
        Save a mapping to a file.

        Args:
            mapping (dict): The mapping to save.
            file_path (str): The path to the file.
        """
        with open(file_path, mode='w', encoding='utf-8') as outfile:
            for key, value in mapping.items():
                outfile.write(f"{key},{value}\n")
            outfile.close()

    def load_all_data(self, dataset_folder: str) -> List[Tuple[Any, Any, Any, float]]:
        """
        Load train/validation/test triples from a dataset directory.

        The directory must contain:
            - train.tsv
            - val.tsv
            - test.tsv

        Parameters
        ----------
        dataset_folder : str
            Directory path containing triple data files.

        Returns
        -------
        List[Tuple[Any, Any, Any, float]]
            Combined triple list of train, validation, and test splits.

        Raises
        ------
        FileNotFoundError
            If any required dataset file is missing.
        """
        train_path = os.path.join(dataset_folder, "train.tsv")
        val_path = os.path.join(dataset_folder, "val.tsv")
        test_path = os.path.join(dataset_folder, "test.tsv")

        # check existence before reading -> fail fast
        for p, name in [(train_path, "train"), (val_path, "val"), (test_path, "test")]:
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"Missing dataset file '{name}.tsv' in directory: {dataset_folder}"
                )

        # load data
        self.train_data = self.load_data(train_path)
        self.val_data = self.load_data(val_path)
        self.test_data = self.load_data(test_path)

        # return aggregated list
        return self.train_data + self.val_data + self.test_data

    def load_soft_data(self, dataset_folder):
        soft_logic_data = os.path.join(dataset_folder, 'extra', 'softlogic.tsv')
        return self.load_data(soft_logic_data)

    def load_test_neg_data(self, dataset_folder):
        test_neg_file = os.path.join(dataset_folder, 'extra', 'test', 'test_with_neg_beurre.tsv')
        return self.load_data(test_neg_file)

    def load_pseudo_data(self, dataset_folder):
        pseudo_file = os.path.join(dataset_folder, 'extra', 'pseudo.tsv')
        return self.load_data(pseudo_file)

    @staticmethod
    def load_hr_map(data_dir):
        """
        Load a Head-Relation mapping from test data.
        Args:
            data_dir:

        Returns:

        """
        file = os.path.join(data_dir, 'extra', 'test', 'ndcg_test_beurre.pickle')
        with open(file, 'rb') as f:
            hr_map = pickle.load(f)
        return hr_map

    def load_hr_map2(self):
        """
        Construct a Head-Relation mapping from valid data.

        This mapping is used for NDCG computation.
        Each entry maps:
            hr_map[h][r] -> { t1: confidence1, t2: confidence2, ... }

        Returns:
            dict: Nested dictionary mapping head and relation to tails with confidence.

        """
        # 1. Add val triples
        hr_map = {}

        # 添加训练集 triples
        for h, r, t, w in self.val_data:
            hr_map.setdefault(h, {}).setdefault(r, {})[t] = float(w)

        # count = sum(len(hr_map[h]) for h in hr_map)
        # print(f"Loaded ranking val queries. Number of (h,r,?t) queries: {count}")

        # 2. Add t(only t) to self.hr_map
        for h, r, t, w in self.all_true_data:
            if h in hr_map and r in hr_map[h]:
                hr_map[h][r][t] = float(w)
        return hr_map


class FewShotData:
    """Data processing for few-shot GMUC & GMUC+ data (no numpy, no pandas).

    Attributes are the same as the original version.
    """

    def __init__(self, data_path, dset_name, max_nbr, has_ont):

        self.data_path = data_path
        self.dataset_name = dset_name
        self.max_neighbor = max_nbr
        self.has_ont = has_ont
        self.ent2id = {}
        self.rel2id = {}
        self.symbol2id = {}
        self.rel2candidates = {}
        self.e1rel_e2 = ddict(list)
        self.rele1_e2 = ddict(dict)


        # Load data files
        with open(f"{self.data_path}/train_tasks.json") as f:
            self.train_tasks = json.load(f)
        with open(f"{self.data_path}/dev_tasks.json") as f:
            self.dev_tasks = json.load(f)
        with open(f"{self.data_path}/test_tasks.json") as f:
            self.test_tasks = json.load(f)

        self.task_pool = list(self.train_tasks.keys())
        self.num_tasks = len(self.task_pool)

        with open(f"{self.data_path}/path_graph") as f:
            self.path_graph = f.readlines()

        self.type2ents = ddict(set)
        self.known_rels = ddict(list)

        self.get_tasks()
        self.get_rel2candidates()
        self.get_e1rel_e2()
        self.get_rele1_e2()
        degrees = self.build_graph(max_=self.max_neighbor)

        if self.has_ont:
            self.ent2ic = {}
            self.rel_uc1 = {}
            self.rel_uc2 = {}
            self.get_ontology()

    # ----------------------- #
    #  Entity/Relation setup  #
    # ----------------------- #
    def get_tasks(self):
        """Assign entity/relation IDs and compute symbol2id."""
        eid, rid = -1, -1

        def register_triples(tasks):
            nonlocal eid, rid
            for rel, triples in tasks.items():
                if rel not in self.rel2id:
                    rid += 1
                    self.rel2id[rel] = rid
                for e1, r, e2, s in triples:
                    for e in [e1, e2]:
                        if e not in self.ent2id:
                            eid += 1
                            self.ent2id[e] = eid

        # Register entities/relations
        register_triples(self.train_tasks)
        register_triples(self.dev_tasks)
        register_triples(self.test_tasks)

        # Register from path_graph
        for line in self.path_graph:
            e1, r, e2, s = line.strip().split()
            for rel in [r, r + "_inv"]:
                if rel not in self.rel2id:
                    rid += 1
                    self.rel2id[rel] = rid
            for e in [e1, e2]:
                if e not in self.ent2id:
                    eid += 1
                    self.ent2id[e] = eid

        # Store counts
        self.num_ent = len(self.ent2id)
        self.num_rel = len(self.rel2id)

        # Build symbol2id
        symbol_id = {}
        i = 0
        for key in list(self.rel2id.keys()) + list(self.ent2id.keys()):
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1
        symbol_id['PAD'] = i

        self.symbol2id = symbol_id
        self.num_symbols = len(symbol_id) - 1
        self.pad_id = self.num_symbols

    # ------------------------- #
    #  Candidate construction   #
    # ------------------------- #
    def get_rel2candidates(self):
        """Compute rel2candidates from train/dev/test tasks."""
        # known_rels = path_graph + train_tasks
        for line in self.path_graph:
            e1, rel, e2, s = line.strip().split()
            self.known_rels[rel].append([e1, rel, e2, s])

        for key, triples in self.train_tasks.items():
            self.known_rels[key] = triples

        all_relations = list(self.known_rels.keys()) + list(self.dev_tasks.keys()) + list(self.test_tasks.keys())
        all_triples = list(self.known_rels.values()) + list(self.dev_tasks.values()) + list(self.test_tasks.values())
        assert len(all_relations) == len(all_triples)

        if self.dataset_name == "nl27k-few-shot":
            for ent in self.ent2id.keys():
                if ':' in ent:
                    t = ent.split(':')[1]
                    self.type2ents[t].add(ent)

            for rel, triples in zip(all_relations, all_triples):
                possible_types = {t[2].split(':')[1] for t in triples if ':' in t[2]}
                candidates = []
                for t in possible_types:
                    candidates.extend(self.type2ents[t])
                self.rel2candidates[rel] = list(set(candidates))[:1000]
        else:
            for rel, triples in zip(all_relations, all_triples):
                ents = {e for tri in triples for e in (tri[0], tri[2])}
                self.rel2candidates[rel] = list(ents)[:1000]

    # ------------------------- #
    #   Data structure helpers  #
    # ------------------------- #
    def get_e1rel_e2(self):
        """Map e1+rel to possible e2."""
        all_tasks = list(self.train_tasks.values()) + list(self.dev_tasks.values()) + list(self.test_tasks.values())
        for task in all_tasks:
            for e1, rel, e2, s in task:
                self.e1rel_e2[e1 + rel].append(e2)

    def get_rele1_e2(self):
        """Map rel → head → [(tail, score)]."""
        for task in list(self.dev_tasks.values()) + list(self.test_tasks.values()):
            d = ddict(list)
            for h, r, t, s in task:
                d[h].append([t, s])
            self.rele1_e2[r] = d

    # ------------------------- #
    #     Ontology handling     #
    # ------------------------- #
    def get_ontology(self):
        """Compute IC for entities and UC for relations without pandas/numpy."""
        ont_path = f"{self.data_path}/ontology.csv"

        with open(ont_path, encoding='utf-8') as f:
            lines = [line.strip().split(',') for line in f.readlines() if line.strip()]
        header = lines[0]
        idx_h, idx_rel, idx_t = header.index('h'), header.index('rel'), header.index('t')
        data = lines[1:]

        pairs, concept_set = [], set()
        for h, rel, t in [(row[idx_h], row[idx_rel], row[idx_t]) for row in data]:
            if rel == 'is_A':
                pairs.append((h, t))
                concept_set.update([h, t])

        # Extract domain and type
        relation_set = {row[idx_h] for row in data if row[idx_rel] == 'domain'}
        entity_set = concept_set - relation_set

        # Build hypo_dict
        hypo_dict = {e: [c1 for c1, c2 in pairs if c2 == e] for e in entity_set}
        real_hypo_dict = copy.deepcopy(hypo_dict)
        for e in entity_set:
            for sub in hypo_dict[e]:
                real_hypo_dict[e].extend(real_hypo_dict.get(sub, []))
            real_hypo_dict[e] = list(set(real_hypo_dict[e]))

        # Compute IC
        ent_type_ic = {e: 1 - math.log(len(real_hypo_dict[e]) + 1) / math.log(292) for e in entity_set}

        # Entity → type mapping
        ent2type = {row[idx_h]: row[idx_t] for row in data if row[idx_rel] == 'type'}

        # Compute ent2uc (normalized)
        ent2uc = {}
        for ent in self.ent2id.keys():
            t = ent2type.get(ent)
            if not t or t not in ent_type_ic:
                continue
            ent2uc[ent] = 1 - ent_type_ic[t]
        if ent2uc:
            mu, sigma = mean(ent2uc.values()), pstdev(ent2uc.values()) or 1.0
            for k in ent2uc:
                ent2uc[k] = (ent2uc[k] - mu) / sigma
        self.ent2ic = ent2uc

        # Compute relation UC
        rel2uc1, rel2uc2 = ddict(float), ddict(float)
        for rel, triples in self.known_rels.items():
            domain, range_ = set(), set()
            for h, r, t, s in triples:
                if ':' in h:
                    domain.add(h.split(':')[1])
                if ':' in t:
                    range_.add(t.split(':')[1])
            count = 0
            ic_sum = 0.0
            for d in domain:
                for rg in range_:
                    count += 1
                    ic_sum += (1 - ent_type_ic.get('concept:' + d, 0) +
                               1 - ent_type_ic.get('concept:' + rg, 0)) / 2
            if count > 0:
                rel2uc1[rel] = ic_sum / count
                rel2uc2[rel] = count

        # Normalization (no numpy)
        def normalize_dict(d):
            if not d:
                return {}
            vals = list(d.values())
            mu, sigma = mean(vals), pstdev(vals) or 1.0
            return {k: (v - mu) / sigma for k, v in d.items()}

        self.rel_uc1 = normalize_dict(rel2uc1)
        self.rel_uc2 = normalize_dict(rel2uc2)

    # ------------------------- #
    #     Graph construction    #
    # ------------------------- #
    def build_graph(self, max_=50):
        """Build neighbor info graph without numpy."""
        self.connections = [[[self.pad_id, self.pad_id, self.pad_id] for _ in range(max_)]
                            for _ in range(self.num_ent)]
        self.e1_rele2 = ddict(list)
        self.e1_degrees = ddict(int)

        with open(f"{self.data_path}/path_graph") as f:
            for line in tqdm(f.readlines()):
                e1, rel, e2, s = line.strip().split()
                s = float(s)
                self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2], s))
                self.e1_rele2[e2].append((self.symbol2id[rel + '_inv'], self.symbol2id[e1], s))

        degrees = {}
        for ent, eid in self.ent2id.items():
            neighbors = self.e1_rele2.get(ent, [])[:max_]
            degrees[ent] = len(neighbors)
            self.e1_degrees[eid] = len(neighbors)
            for idx, (r, t, s) in enumerate(neighbors):
                self.connections[eid][idx] = [r, t, s]
        return degrees

    def get_data(self):
        return {
            'rel2candidates': self.rel2candidates,
            'ent2id': self.ent2id,
            'ent2ic': self.ent2ic if self.has_ont else None,
            'symbol2id': self.symbol2id,
            'rel_uc1': self.rel_uc1 if self.has_ont else None,
            'rel_uc2': self.rel_uc2 if self.has_ont else None,
            'e1rel_e2': self.e1rel_e2,
            'rele1_e2': self.rele1_e2,
            'e1_degrees': self.e1_degrees,
            'connections': self.connections,
            'train_tasks' : self.train_tasks,
            'dev_tasks' : self.dev_tasks,
            'test_tasks' : self.test_tasks,
            'num_symbols': self.num_symbols,
        }




if __name__ == '__main__':
    # dataset = UKGData(dataset_dir='../../datasets/cn15k', use_index_file=False)
    # final_data = dataset.get_data()
    # dataset.print_dataset_info()
    dataset = UKGData(dataset_dir='../../data/cn15k', use_index_file=True)
    final_data = dataset()
    print(dataset)
