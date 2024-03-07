import threading
from random import Random
from typing import Dict, Iterator, List, Optional, Union, Tuple, Any

import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
from rdkit import Chem

from chem.features import BatchMolGraph, MolGraph
from chem.features import is_explicit_h, is_reaction, is_adding_hs, is_mol
from chem.rdkit import make_mol
from utils.features_generator import get_features_generator

# Cache of graph featurizations
CACHE_GRAPH = True
SMILES_TO_GRAPH: Dict[str, MolGraph] = {}


def cache_graph() -> bool:

    return CACHE_GRAPH


def set_cache_graph(cache_graph: bool) -> None:

    global CACHE_GRAPH
    CACHE_GRAPH = cache_graph


def empty_cache():

    SMILES_TO_GRAPH.clear()
    SMILES_TO_MOL.clear()


# Cache of RDKit molecules RDKit分子缓存
CACHE_MOL = True
SMILES_TO_MOL: Dict[str, Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]] = {}


def cache_mol() -> bool:

    return CACHE_MOL


def set_cache_mol(cache_mol: bool) -> None:

    global CACHE_MOL
    CACHE_MOL = cache_mol


class MoleculeDatapoint:

    def __init__(self,
                 id: List[Optional[float]] = None,
                 smiles: List[str] = None,
                 targets: List[Optional[float]] = None,
                 envs: List[Optional[float]] = None,
                 R: np.ndarray = None,
                 data_weight: float = None,
                 features_generator: List[str] = None
                 ):

        self.smiles = smiles
        self.id = id
        self.targets = targets
        self.envs = envs
        self.R = R
        self.features_generator = features_generator
        if data_weight is not None:
            self.data_weight = data_weight

        self.is_mol_list = [is_mol(s) for s in smiles]
        self.is_reaction_list = [is_reaction(x) for x in self.is_mol_list]
        self.is_explicit_h_list = [is_explicit_h(x) for x in self.is_mol_list]
        self.is_adding_hs_list = [is_adding_hs(x) for x in self.is_mol_list]

        if self.features_generator is not None:
            self.features = []

            for fg in self.features_generator:
                features_generator = get_features_generator(fg)
                for m, reaction in zip(self.mol, self.is_reaction_list):
                    if not reaction:
                        if m is not None and m.GetNumHeavyAtoms() > 0:
                            self.features.extend(features_generator(m))
                        # for H2
                        elif m is not None and m.GetNumHeavyAtoms() == 0:
                            # not all features are equally long, so use methane as dummy molecule to determine length
                            self.features.extend(np.zeros(len(features_generator(Chem.MolFromSmiles('C')))))
                    else:
                        if m[0] is not None and m[1] is not None and m[0].GetNumHeavyAtoms() > 0:
                            self.features.extend(features_generator(m[0]))
                        elif m[0] is not None and m[1] is not None and m[0].GetNumHeavyAtoms() == 0:
                            self.features.extend(np.zeros(len(features_generator(Chem.MolFromSmiles('C')))))

            self.features = np.array(self.features)

        self.raw_features, self.raw_targets = self.features, self.targets

    @property
    def mol(self) -> List[Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]]:

        mol = make_mols(self.smiles, self.is_reaction_list, self.is_explicit_h_list, self.is_adding_hs_list)
        if cache_mol():
            for s, m in zip(self.smiles, mol):
                SMILES_TO_MOL[s] = m

        return mol

    @property
    def number_of_molecules(self) -> int:

        return len(self.smiles)

    def set_features(self, features: np.ndarray) -> None:

        self.features = features

    def extend_env(self, env: np.ndarray) -> None:

        self.features = np.append(self.features, env) if self.features is not None else env

    def num_tasks(self) -> int:

        return len(self.targets)

    def set_targets(self, targets: List[Optional[float]]):

        self.targets = targets

    def set_envs(self, envs: List[Optional[float]]):

        self.envs = envs

    def reset_features_and_targets(self) -> None:

        self.features, self.targets = self.raw_features, self.raw_targets


class MoleculeDataset(Dataset):

    def __init__(self, data: List[MoleculeDatapoint]):

        self._data = data
        self._batch_graph = None
        self._random = Random()

    def smiles(self, flatten: bool = False) -> Union[List[str], List[List[str]]]:

        if flatten:
            return [smiles for d in self._data for smiles in d.smiles]

        return [d.smiles for d in self._data]

    def mols(self, flatten: bool = False) -> Union[
        List[Chem.Mol], List[List[Chem.Mol]], List[Tuple[Chem.Mol, Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]]]:

        if flatten:
            return [mol for d in self._data for mol in d.mol]

        return [d.mol for d in self._data]

    @property
    def number_of_molecules(self) -> int:

        return self._data[0].number_of_molecules if len(self._data) > 0 else None

    def batch_graph(self) -> List[BatchMolGraph]:

        if self._batch_graph is None:
            self._batch_graph = []

            mol_graphs = []
            for d in self._data:
                mol_graphs_list = []
                for s, m in zip(d.smiles, d.mol):
                    if s in SMILES_TO_GRAPH:
                        mol_graph = SMILES_TO_GRAPH[s]
                    else:
                        mol_graph = MolGraph(m, d.R)
                        if cache_graph():
                            SMILES_TO_GRAPH[s] = mol_graph
                    mol_graphs_list.append(mol_graph)
                mol_graphs.append(mol_graphs_list)

            self._batch_graph = [BatchMolGraph([g[i] for g in mol_graphs]) for i in range(len(mol_graphs[0]))]

        return self._batch_graph

    def features(self) -> List[np.ndarray]:

        if len(self._data) == 0 or self._data[0].features is None:
            return None

        return [d.features for d in self._data]

    def data_weights(self) -> List[float]:

        if not hasattr(self._data[0], 'data_weight'):
            return [1. for d in self._data]

        return [d.data_weight for d in self._data]

    def id(self) -> List[List[Optional[float]]]:
        return [d.id for d in self._data]

    def envs(self):
        return [d.envs for d in self._data]

    def R(self):
        return [d.R for d in self._data]

    def targets(self) -> List[List[Optional[float]]]:

        return [d.targets for d in self._data]

    def num_tasks(self) -> int:

        return self._data[0].num_tasks() if len(self._data) > 0 else None

    def features_size(self) -> int:

        return len(self._data[0].features) if len(self._data) > 0 and self._data[0].features is not None else None

    def normalize_features(self, scaler, replace_nan_token: int = 0):

        if scaler is None:
            features = np.vstack([d.raw_features for d in self._data])
            scaler = StandardScaler(replace_nan_token=replace_nan_token)
            scaler.fit(features)

        for d in self._data:
            d.set_features(scaler.transform(d.raw_features.reshape(1, -1))[0])

        return scaler

    def normalize_targets(self):

        targets = [d.raw_targets for d in self._data]
        scaler = StandardScaler().fit(targets)
        scaled_targets = scaler.transform(targets).tolist()
        self.set_targets(scaled_targets)

        return scaler

    def set_targets(self, targets: List[List[Optional[float]]]) -> None:

        if not len(self._data) == len(targets):
            raise ValueError(
                "number of molecules and targets must be of same length! "
                f"num molecules: {len(self._data)}, num targets: {len(targets)}"
            )
        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i])

    def reset_features_and_targets(self) -> None:

        for d in self._data:
            d.reset_features_and_targets()

    def __len__(self) -> int:

        return len(self._data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:

        return self._data[item]


class MoleculeSampler(Sampler):

    def __init__(self,
                 dataset: MoleculeDataset,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):

        super(Sampler, self).__init__()

        self.dataset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle

        self._random = Random(seed)

        if self.class_balance:
            indices = np.arange(len(dataset))
            has_active = np.array([any(target == 1 for target in datapoint.targets) for datapoint in dataset])

            self.positive_indices = indices[has_active].tolist()
            self.negative_indices = indices[~has_active].tolist()

            self.length = 2 * min(len(self.positive_indices), len(self.negative_indices))
        else:
            self.positive_indices = self.negative_indices = None

            self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:

        if self.class_balance:
            if self.shuffle:
                self._random.shuffle(self.positive_indices)
                self._random.shuffle(self.negative_indices)

            indices = [index for pair in zip(self.positive_indices, self.negative_indices) for index in pair]
        else:
            indices = list(range(len(self.dataset)))

            if self.shuffle:
                self._random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:

        return self.length


def construct_molecule_batch(data: List[MoleculeDatapoint]) -> MoleculeDataset:

    data = MoleculeDataset(data)
    data.batch_graph()  # Forces computation and caching of the BatchMolGraph for the molecules

    return data


class MoleculeDataLoader(DataLoader):

    def __init__(self,
                 dataset: MoleculeDataset,
                 batch_size: int = 50,
                 num_workers: int = 0,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):

        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._class_balance = class_balance
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._context = 'forkserver'  # In order to prevent a hanging
            self._timeout = 3600  # Just for sure that the DataLoader won't hang

        self._sampler = MoleculeSampler(
            dataset=self._dataset,
            class_balance=self._class_balance,
            shuffle=self._shuffle,
            seed=self._seed
        )

        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=construct_molecule_batch,
            multiprocessing_context=self._context,
            timeout=self._timeout
        )

    @property
    def targets(self) -> List[List[Optional[float]]]:

        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')

        return [self._dataset[index].targets for index in self._sampler]

    @property
    def iter_size(self) -> int:

        return len(self._sampler)

    def __iter__(self) -> Iterator[MoleculeDataset]:

        return super(MoleculeDataLoader, self).__iter__()


def make_mols(smiles: List[str], reaction_list: List[bool], keep_h_list: List[bool], add_h_list: List[bool]):

    mol = []
    for s, reaction, keep_h, add_h in zip(smiles, reaction_list, keep_h_list, add_h_list):
        if reaction:
            mol.append(SMILES_TO_MOL[s] if s in SMILES_TO_MOL else (
            make_mol(s.split(">")[0], keep_h, add_h), make_mol(s.split(">")[-1], keep_h, add_h)))
        else:
            mol.append(SMILES_TO_MOL[s] if s in SMILES_TO_MOL else make_mol(s, keep_h, add_h))
    return mol


class StandardScaler:

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None):

        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X: List[List[Optional[float]]]) -> 'StandardScaler':

        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

        return self

    def transform(self, X: List[List[Optional[float]]]) -> np.ndarray:

        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X: List[List[Optional[float]]]) -> np.ndarray:

        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

