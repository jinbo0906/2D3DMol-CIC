from typing import List, Tuple, Union
from itertools import zip_longest
import logging

from rdkit import Chem
import torch
import numpy as np
from torch import nn

from chemprop.rdkit import make_mol


class Featurization_parameters:
    """
    A class holding molecule featurization parameters as attributes.  将分子特征参数作为属性的类
    """

    def __init__(self) -> None:
        # Atom feature sizes
        self.MAX_ATOMIC_NUM = 100
        self.ATOM_FEATURES = {
            'atom type': ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'H', 'P', 'K', 'I', 'Na'],
            'atomic_num': [1, 6, 7, 8, 9, 16, 17, 35, 53, 15, 19, 53, 11],  # list(range(self.MAX_ATOMIC_NUM))
            'degree': [0, 1, 2, 3, 4, 5],  # 原子的度被定义为原子直接邻居的个数
            'formal_charge': [-1, -2, 1, 2, 0],
            'chiral_tag': [0, 1, 2, 3],
            'num_Hs': [0, 1, 2, 3, 4],
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ],

        }
        self.ATOM_FEATURES1 = {
            'electronegativity': {'C': 2.55, 'N': 3.04, 'O': 3.44, 'S': 2.58, 'F': 3.98, 'Cl': 3.16, 'Br': 2.96,
                                  'H': 2.2, 'P': 2.19, 'K': 0.82, 'I': 2.66, 'Na': 0.93},
            'electron_affinities': {'C': 153.9, 'N': 7, 'O': 141, 'S': 200, 'F': 328, 'Cl': 349, 'Br': 324.6,
                                    'H': 72.8, 'P': 72, 'K': 48.4, 'I': 295.2, 'Na': 52.8},
            'ionization_energy': {'C': 1086.5, 'N': 1402.3, 'O': 1313.9, 'S': 999.6, 'F': 1681, 'Cl': 1251.2,
                                  'Br': 1139.9, 'H': 1312, 'P': 1011.8, 'K': 418.8, 'I': 1008.4, 'Na': 495.8},
            'radius': {'C': 67, 'N': 56, 'O': 48, 'S': 88, 'F': 42, 'Cl': 79, 'Br': 94, 'H': 53, 'P': 98, 'K': 243,
                       'I': 115, 'Na': 190}}
        # Distance feature sizes
        self.PATH_DISTANCE_BINS = list(range(10))
        self.THREE_D_DISTANCE_MAX = 20
        self.THREE_D_DISTANCE_STEP = 1
        self.THREE_D_DISTANCE_BINS = list(range(0, self.THREE_D_DISTANCE_MAX + 1, self.THREE_D_DISTANCE_STEP))

        # len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
        self.ATOM_FDIM = sum(len(choices) + 1 for choices in self.ATOM_FEATURES.values()) + 6
        self.EXTRA_ATOM_FDIM = 0
        self.BOND_FDIM = 15
        self.EXTRA_BOND_FDIM = 32
        self.REACTION_MODE = None
        self.EXPLICIT_H = False
        self.REACTION = False
        self.ADDING_H = False


# Create a global parameter object for reference throughout this module  创建一个全局参数对象，以便在整个模块中引用
PARAMS = Featurization_parameters()


def reset_featurization_parameters(logger: logging.Logger = None) -> None:
    """
    Function resets feature parameter values to defaults by replacing the parameters instance.
    """
    if logger is not None:
        debug = logger.debug
    else:
        debug = print
    debug('Setting molecule featurization parameters to default.')
    global PARAMS
    PARAMS = Featurization_parameters()


def get_atom_fdim(overwrite_default_atom: bool = False, is_reaction: bool = False) -> int:
    """
    Gets the dimensionality of the atom feature vector.

    :param overwrite_default_atom: Whether to overwrite the default atom descriptors
    :param is_reaction: Whether to add :code:`EXTRA_ATOM_FDIM` for reaction input when :code:`REACTION_MODE` is not None
    :return: The dimensionality of the atom feature vector.
    """
    if PARAMS.REACTION_MODE:
        return (not overwrite_default_atom) * PARAMS.ATOM_FDIM + is_reaction * PARAMS.EXTRA_ATOM_FDIM
    else:
        return (not overwrite_default_atom) * PARAMS.ATOM_FDIM + PARAMS.EXTRA_ATOM_FDIM


def set_explicit_h(explicit_h: bool) -> None:
    """
    Sets whether RDKit molecules will be constructed with explicit Hs.

    :param explicit_h: Boolean whether to keep explicit Hs from input.
    """
    PARAMS.EXPLICIT_H = explicit_h


def set_adding_hs(adding_hs: bool) -> None:
    """
    Sets whether RDKit molecules will be constructed with adding the Hs to them.
    设置RDKit分子是否通过添加Hs来构造
    :param adding_hs: Boolean whether to add Hs to the molecule.
    """
    PARAMS.ADDING_H = adding_hs


def set_reaction(reaction: bool, mode: str) -> None:
    """
    Sets whether to use a reaction or molecule as input and adapts feature dimensions.
 
    :param reaction: Boolean whether to except reactions as input.
    :param mode: Reaction mode to construct atom and bond feature vectors.

    """
    PARAMS.REACTION = reaction
    if reaction:
        PARAMS.EXTRA_ATOM_FDIM = PARAMS.ATOM_FDIM - PARAMS.MAX_ATOMIC_NUM - 1
        PARAMS.EXTRA_BOND_FDIM = PARAMS.BOND_FDIM
        PARAMS.REACTION_MODE = mode


def is_explicit_h(is_mol: bool = True) -> bool:
    r"""Returns whether to retain explicit Hs (for reactions only)"""
    if not is_mol:
        return PARAMS.EXPLICIT_H
    return False


def is_adding_hs(is_mol: bool = False) -> bool:
    r"""Returns whether to add explicit Hs to the mol (not for reactions)"""
    if is_mol:
        return PARAMS.ADDING_H
    return False


def is_reaction(is_mol: bool = True) -> bool:
    r"""Returns whether to use reactions as input"""
    if is_mol:
        return False
    if PARAMS.REACTION:  # (and not is_mol, checked above)
        return True
    return False


def reaction_mode() -> str:
    r"""Returns the reaction mode"""
    return PARAMS.REACTION_MODE


def set_extra_atom_fdim(extra):
    """Change the dimensionality of the atom feature vector."""
    PARAMS.EXTRA_ATOM_FDIM = extra


def get_bond_fdim(atom_messages: bool = False,
                  overwrite_default_bond: bool = False,
                  overwrite_default_atom: bool = False,
                  is_reaction: bool = False) -> int:
    """
    Gets the dimensionality of the bond feature vector.

    :param atom_messages: Whether atom messages are being used. If atom messages are used,
                          then the bond feature vector only contains bond features.
                          Otherwise it contains both atom and bond features.
    :param overwrite_default_bond: Whether to overwrite the default bond descriptors
    :param overwrite_default_atom: Whether to overwrite the default atom descriptors
    :param is_reaction: Whether to add :code:`EXTRA_BOND_FDIM` for reaction input when :code:`REACTION_MODE:` is not None
    :return: The dimensionality of the bond feature vector.
    """

    if PARAMS.REACTION_MODE:
        return (not overwrite_default_bond) * PARAMS.BOND_FDIM + is_reaction * PARAMS.EXTRA_BOND_FDIM + \
               (not atom_messages) * get_atom_fdim(overwrite_default_atom=overwrite_default_atom,
                                                   is_reaction=is_reaction)
    else:
        return (not overwrite_default_bond) * PARAMS.BOND_FDIM + PARAMS.EXTRA_BOND_FDIM + \
               (not atom_messages) * get_atom_fdim(overwrite_default_atom=overwrite_default_atom,
                                                   is_reaction=is_reaction)


def set_extra_bond_fdim(extra):
    """Change the dimensionality of the bond feature vector."""
    PARAMS.EXTRA_BOND_FDIM = extra


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def get_atomic_features(value, choices):
    atom = value
    if atom in ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'H', 'P', 'K', 'I', 'Na']:
        feature = choices[atom]
        return feature
    else:
        return 0


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.
    为原子构建特征向量
    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features = onek_encoding_unk(atom.GetSymbol(), PARAMS.ATOM_FEATURES['atom type']) + \
                   onek_encoding_unk(atom.GetAtomicNum(), PARAMS.ATOM_FEATURES['atomic_num']) + \
                   onek_encoding_unk(atom.GetTotalDegree(), PARAMS.ATOM_FEATURES['degree']) + \
                   onek_encoding_unk(atom.GetFormalCharge(), PARAMS.ATOM_FEATURES['formal_charge']) + \
                   onek_encoding_unk(int(atom.GetChiralTag()), PARAMS.ATOM_FEATURES['chiral_tag']) + \
                   onek_encoding_unk(int(atom.GetTotalNumHs()), PARAMS.ATOM_FEATURES['num_Hs']) + \
                   onek_encoding_unk(int(atom.GetHybridization()), PARAMS.ATOM_FEATURES['hybridization']) + \
                   [1 if atom.GetIsAromatic() else 0] + \
                   [atom.GetMass() * 0.01] + [
                       get_atomic_features(atom.GetSymbol(), PARAMS.ATOM_FEATURES1['electronegativity']) * 0.1] + [
                       get_atomic_features(atom.GetSymbol(), PARAMS.ATOM_FEATURES1['electron_affinities']) * 0.001] + [
                       get_atomic_features(atom.GetSymbol(), PARAMS.ATOM_FEATURES1['ionization_energy']) * 0.0001] + [
                       get_atomic_features(atom.GetSymbol(), PARAMS.ATOM_FEATURES1[
                           'radius']) * 0.01]  # scaled to about the same range as other features 缩放到与其他功能相同的范围
        if functional_groups is not None:
            features += functional_groups
    return features


def atom_features_zeros(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom containing only the atom number information.

    :param atom: An RDKit atom.
    :return: A list containing the atom features.
    """
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features = onek_encoding_unk(atom.GetAtomicNum() - 1, PARAMS.ATOM_FEATURES['atomic_num']) + \
                   [0] * (PARAMS.ATOM_FDIM - PARAMS.MAX_ATOMIC_NUM - 1)  # set other features to zero
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.
    建立一个键的特征向量
    :param bond: An RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (PARAMS.BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0),
            (bond.GetIsAromatic() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def map_reac_to_prod(mol_reac: Chem.Mol, mol_prod: Chem.Mol):
    """
    Build a dictionary of mapping atom indices in the reactants to the products.

    :param mol_reac: An RDKit molecule of the reactants.
    :param mol_prod: An RDKit molecule of the products.
    :return: A dictionary of corresponding reactant and product atom indices.
    """
    only_prod_ids = []
    prod_map_to_id = {}
    mapnos_reac = set([atom.GetAtomMapNum() for atom in mol_reac.GetAtoms()])
    for atom in mol_prod.GetAtoms():
        mapno = atom.GetAtomMapNum()
        if mapno > 0:
            prod_map_to_id[mapno] = atom.GetIdx()
            if mapno not in mapnos_reac:
                only_prod_ids.append(atom.GetIdx())
        else:
            only_prod_ids.append(atom.GetIdx())
    only_reac_ids = []
    reac_id_to_prod_id = {}
    for atom in mol_reac.GetAtoms():
        mapno = atom.GetAtomMapNum()
        if mapno > 0:
            try:
                reac_id_to_prod_id[atom.GetIdx()] = prod_map_to_id[mapno]
            except KeyError:
                only_reac_ids.append(atom.GetIdx())
        else:
            only_reac_ids.append(atom.GetIdx())
    return reac_id_to_prod_id, only_prod_ids, only_reac_ids


def Envelope(inputs):
    """
    Envelope function that ensures a smooth cutoff
    """
    exponent = 6
    p = exponent + 1
    a = -(p + 1) * (p + 2) / 2
    b = p * (p + 2)
    c = -p * (p + 1) / 2
    env_val = 1 / inputs + a * inputs**(p - 1) + b * inputs**p + c * inputs**(p + 1)
    return np.where(inputs < 3, env_val, np.zeros_like(inputs))


def BesselBasisLayer(inputs, num_radial=64):
    # num_radial = PARAMS.EXTRA_BOND_FDIM  # num_radial=6
    inv_cutoff = np.array(1 / 5, dtype=np.float32)  # cutoff=5   从一个类张量的物体中创建一个常数张量
    # envelope = Envelope(inputs)  # 5
    # Initialize frequencies at canonical positions 在规范位置初始化频率
    freq_init = np.pi * np.arange(1, num_radial + 1, dtype=np.float32)
    # frequencies = self.add_weight(name="frequencies", shape=self.num_radial,
    #                                    dtype=tf.float32, initializer=freq_init, trainable=True)
    d_scaled = inputs * inv_cutoff
    # Necessary for proper broadcasting behaviour
    d_scaled = np.expand_dims(d_scaled, -1)
    d_cutoff = Envelope(d_scaled)
    return np.sin(freq_init * d_scaled)


class MolGraph:
    """
    A :class:`MolGraph` represents the graph structure and featurization of a single molecule.
    表示单个分子的图结构和特征
    A MolGraph computes the following attributes:
    MolGraph计算以下属性
    * :code:`n_atoms`: The number of atoms in the molecule.  # 分子中的原子数
    * :code:`n_bonds`: The number of bonds in the molecule.  # 分子中的键数
    * :code:`f_atoms`: A mapping from an atom index to a list of atom features.  # 从原子索引到原子特征列表的映射
    * :code:`f_bonds`: A mapping from a bond index to a list of bond features.   # 从键索引到键特征列表的映射
    * :code:`a2b`: A mapping from an atom index to a list of incoming bond indices.  # 从原子索引到传入键索引列表的映射
    * :code:`b2a`: A mapping from a bond index to the index of the atom the bond originates from.  # 从键指数到生成键的原子的指数的映射
    * :code:`b2revb`: A mapping from a bond index to the index of the reverse bond.  # 从键指数到反向键指数的映射
    * :code:`overwrite_default_atom_features`: A boolean to overwrite default atom descriptors.
    * :code:`overwrite_default_bond_features`: A boolean to overwrite default bond descriptors.
    * :code:`is_mol`: A boolean whether the input is a molecule.
    * :code:`is_reaction`: A boolean whether the molecule is a reaction.
    * :code:`is_explicit_h`: A boolean whether to retain explicit Hs (for reaction mode)
    * :code:`is_adding_hs`: A boolean whether to add explicit Hs (not for reaction mode)
    * :code:`reaction_mode`:  Reaction mode to construct atom and bond feature vectors
    """

    def __init__(self, mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]],
                 R: np.ndarray,
                 atom_features_extra: np.ndarray = None,
                 bond_features_extra: np.ndarray = None,
                 overwrite_default_atom_features: bool = False,
                 overwrite_default_bond_features: bool = False):
        """
        :param mol: A SMILES or an RDKit molecule.
        :param atom_features_extra: A list of 2D numpy array containing additional atom features to featurize the molecule
        :param bond_features_extra: A list of 2D numpy array containing additional bond features to featurize the molecule
        :param overwrite_default_atom_features: Boolean to overwrite default atom features by atom_features instead of concatenating
        :param overwrite_default_bond_features: Boolean to overwrite default bond features by bond_features instead of concatenating
        """
        self.is_mol = is_mol(mol)
        self.is_reaction = is_reaction(self.is_mol)
        self.is_explicit_h = is_explicit_h(self.is_mol)
        self.is_adding_hs = is_adding_hs(self.is_mol)
        self.reaction_mode = reaction_mode()

        # Convert SMILES to RDKit molecule if necessary 将SMILES转换为RDKit分子
        if type(mol) == str:
            if self.is_reaction:
                mol = (make_mol(mol.split(">")[0], self.is_explicit_h, self.is_adding_hs),
                       make_mol(mol.split(">")[-1], self.is_explicit_h, self.is_adding_hs))
            else:
                mol = make_mol(mol, self.is_explicit_h, self.is_adding_hs)

        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features  从原子索引到原子特征的映射
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features  从键索引映射到concat(in_atom, bond)特性
        self.f_atoms2 = []
        self.a2b = []  # mapping from atom index to incoming bond indices  从原子索引映射到传入键索引
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from  从键索引映射到键所来自原子的索引
        self.b2revb = []  # mapping from bond index to the index of the reverse bond  从键指数映射到反向键的指数
        self.overwrite_default_atom_features = overwrite_default_atom_features
        self.overwrite_default_bond_features = overwrite_default_bond_features

        if not self.is_reaction:
            # Get atom features  得到原子特性
            self.f_atoms = [atom_features(atom) for atom in mol.GetAtoms()]  # 得到原子特征向量

            self.n_atoms = len(self.f_atoms)
            if atom_features_extra is not None and len(atom_features_extra) != self.n_atoms:
                raise ValueError(f'The number of atoms in {Chem.MolToSmiles(mol)} is different from the length of '
                                 f'the extra atom features')

            # Initialize atom to bond mapping for each atom  为每个原子初始化原子到键映射
            for _ in range(self.n_atoms):
                self.a2b.append([])

            # Get bond features  得到键特征
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)  # 返回两个原子之间的键，如果有的话
                    if bond is not None:
                        dij = np.sqrt(np.sum(np.square(R[a1]-R[a2])))

                    if bond is None:
                        continue
                    f_Dij = BesselBasisLayer(dij, 32)
                    f_Dij = f_Dij.tolist()
                    f_bond = bond_features(bond)  # 得到键特征向量
                    f_bond = f_bond+f_Dij
                    # f_bond.append(dij*0.1)

                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                    # Update index mappings  更新索引映射
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)  # b1 = a1 --> a2
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)  # b2 = a2 --> a1
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.n_bonds += 2

            for a1 in range(self.n_atoms):
                dij = []
                for a2 in range(self.n_atoms):
                    d = 100.0 if np.sqrt(np.sum(np.square(R[a1] - R[a2]))) == 0 else 1 / np.sqrt(
                        np.sum(np.square(R[a1] - R[a2])))
                    # d_=BesselBasisLayer(d) * 0.1
                    # d_ = d_.tolist()
                    dij.append(BesselBasisLayer(d, 64))
                dij = np.sum(dij, axis=0)
                # dij.append(d)
                self.f_atoms2.append((dij * 0.01).tolist()+self.f_atoms[a1])


class BatchMolGraph:
    """
    A :class:`BatchMolGraph` represents the graph structure and featurization of a batch of molecules.
    表示一批分子的图形结构和特征
    A BatchMolGraph contains the attributes of a :class:`MolGraph` plus:

    * :code:`atom_fdim`: The dimensionality of the atom feature vector.
    * :code:`bond_fdim`: The dimensionality of the bond feature vector (technically the combined atom/bond features).
    * :code:`a_scope`: A list of tuples indicating the start and end atom indices for each molecule.
    * :code:`b_scope`: A list of tuples indicating the start and end bond indices for each molecule.
    * :code:`max_num_bonds`: The maximum number of bonds neighboring an atom in this batch.
    * :code:`b2b`: (Optional) A mapping from a bond index to incoming bond indices.
    * :code:`a2a`: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph]):
        r"""
        :param mol_graphs: A list of :class:`MolGraph`\ s from which to construct the :class:`BatchMolGraph`.
        """
        self.overwrite_default_atom_features = mol_graphs[0].overwrite_default_atom_features
        self.overwrite_default_bond_features = mol_graphs[0].overwrite_default_bond_features
        self.is_reaction = mol_graphs[0].is_reaction
        self.atom_fdim = get_atom_fdim(overwrite_default_atom=self.overwrite_default_atom_features,
                                       is_reaction=self.is_reaction)
        self.bond_fdim = get_bond_fdim(overwrite_default_bond=self.overwrite_default_bond_features,
                                       overwrite_default_atom=self.overwrite_default_atom_features,
                                       is_reaction=self.is_reaction)

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_atoms2 = [[0] * (self.atom_fdim+64)]
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)
            f_atoms2.extend(mol_graph.f_atoms2)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(1, max(
            len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.f_atoms2 = torch.FloatTensor(f_atoms2)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self, atom_messages: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor,
                                                                   torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                                                   List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the :class:`BatchMolGraph`.

        The returned components are, in order:

        * :code:`f_atoms`
        * :code:`f_bonds`
        * :code:`a2b`
        * :code:`b2a`
        * :code:`b2revb`
        * :code:`a_scope`
        * :code:`b_scope`

        :param atom_messages: Whether to use atom messages instead of bond messages. This changes the bond feature
                              vector to contain only bond features rather than both atom and bond features.
        :return: A tuple containing PyTorch tensors with the atom features, bond features, graph structure,
                 and scope of the atoms and bonds (i.e., the indices of the molecules they belong to).
        """
        if atom_messages:
            f_bonds = self.f_bonds[:, -get_bond_fdim(atom_messages=atom_messages,
                                                     overwrite_default_atom=self.overwrite_default_atom_features,
                                                     overwrite_default_bond=self.overwrite_default_bond_features):]
        else:
            f_bonds = self.f_bonds

        return self.f_atoms, self.f_atoms2, f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """
        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each atom index to all the neighboring atom indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


def mol2graph(mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]],
              atom_features_batch: List[np.array] = (None,),
              bond_features_batch: List[np.array] = (None,),
              overwrite_default_atom_features: bool = False,
              overwrite_default_bond_features: bool = False
              ) -> BatchMolGraph:
    """
    Converts a list of SMILES or RDKit molecules to a :class:`BatchMolGraph` containing the batch of molecular graphs.

    :param mols: A list of SMILES or a list of RDKit molecules. SMILES的列表或RDKit分子的列表
    :param atom_features_batch: A list of 2D numpy array containing additional atom features to featurize the molecule
    :param bond_features_batch: A list of 2D numpy array containing additional bond features to featurize the molecule 二维numpy数组的列表，包含额外的键特征，以表征分  子
    :param overwrite_default_atom_features: Boolean to overwrite default atom descriptors by atom_descriptors instead of concatenating
    :param overwrite_default_bond_features: Boolean to overwrite default bond descriptors by bond_descriptors instead of concatenating
    :return: A :class:`BatchMolGraph` containing the combined molecular graph for the molecules.
    """
    return BatchMolGraph([MolGraph(mol, af, bf,
                                   overwrite_default_atom_features=overwrite_default_atom_features,
                                   overwrite_default_bond_features=overwrite_default_bond_features)
                          for mol, af, bf
                          in zip_longest(mols, atom_features_batch, bond_features_batch)])


def is_mol(mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]) -> bool:
    """Checks whether an input is a molecule or a reaction

    :param mol: str, RDKIT molecule or tuple of molecules
    :return: Whether the supplied input corresponds to a single molecule
    """

    if isinstance(mol, str) and ">" not in mol:
        return True
    elif isinstance(mol, Chem.Mol):
        return True
    return False
