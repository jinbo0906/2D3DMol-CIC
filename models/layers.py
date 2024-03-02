import math
from typing import List, Union, Tuple
from functools import reduce

import numpy as np
from chemprop.nn_utils import get_activation_function, index_select_ND
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import sympy as sym

from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph

from models.utils import bessel_basis, real_sph_harm


class MPN_2D(nn.Module):
    def __init__(self,
                 model_conf,
                 device,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPN_2D, self).__init__()
        self.device = device
        self.model_conf = model_conf
        self.atom_fdim = atom_fdim or get_atom_fdim()
        self.bond_fdim = bond_fdim or get_bond_fdim()
        self.encoder = nn.ModuleList([MPNEncoder_2D(self.model_conf, self.device, self.atom_fdim, self.bond_fdim)
                                      for _ in range(self.model_conf['number_of_molecules'])])

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                features_batch: List[np.ndarray] = None
                ) -> torch.FloatTensor:
        """
        Encodes a batch of molecules.
        编码一批分子
        :param batch: A list of SMILES, a list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if type(batch[0]) != BatchMolGraph:
            # Group first molecules, second molecules, etc for mol2graph
            batch = [[mols[i] for mols in batch] for i in range(len(batch[0]))]

            # TODO: handle atom_descriptors_batch with multiple molecules per input
            batch = [mol2graph(b) for b in batch]

        encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]

        output = reduce(lambda x, y: torch.cat((x, y), dim=1), encodings)

        if self.model_conf['use_input_features']:
            features_batch = torch.from_numpy(np.stack(features_batch)).float().to(self.device)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view(1, -1)
            output = torch.cat([output, features_batch], dim=1)

        return output


class MPN_3D(nn.Module):
    def __init__(self,
                 model_conf,
                 device,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPN_3D, self).__init__()
        self.model_conf = model_conf
        self.atom_fdim = atom_fdim or get_atom_fdim()
        self.bond_fdim = bond_fdim or get_bond_fdim()
        self.device = device
        self.encoder = nn.ModuleList([MPNEncoder_3D(self.model_conf, self.device, self.atom_fdim, self.bond_fdim)
                                      for _ in range(self.model_conf['number_of_molecules'])])

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]]
                ) -> torch.FloatTensor:
        """
        Encodes a batch of molecules.
        编码一批分子
        :param batch: A list of SMILES, a list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if type(batch[0]) != BatchMolGraph:
            # Group first molecules, second molecules, etc for mol2graph
            batch = [[mols[i] for mols in batch] for i in range(len(batch[0]))]

            # TODO: handle atom_descriptors_batch with multiple molecules per input
            batch = [mol2graph(b) for b in batch]

        encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]

        output = reduce(lambda x, y: torch.cat((x, y), dim=1), encodings)

        return output


class MPNEncoder_2D(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, model_conf, device, atom_fdim: int, bond_fdim: int):

        super(MPNEncoder_2D, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = model_conf['layer']['hidden_size']
        self.bias = model_conf['layer']['bias']
        self.depth = model_conf['layer']['depth']
        self.dropout = model_conf['layer']['dropout']
        self.layers_per_message = 1
        # self.undirected = args.undirected
        self.device = device
        self.aggregation = model_conf['layer']['aggregation']
        self.aggregation_norm = model_conf['layer']['aggregation_norm']

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(model_conf['layer']['activate'])

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim
        self.W_i_atom = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        input_dim = self.bond_fdim
        self.W_i_bond = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        w_h_input_size_atom = self.hidden_size + self.bond_fdim
        self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias)

        w_h_input_size_bond = self.hidden_size

        for depth in range(self.depth - 1):
            self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.gru = BatchGRU(self.hidden_size)

        self.lr = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=self.bias)

    def forward(self,
                mol_graph: BatchMolGraph) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.
        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        一个形状为:code: ' (num_molecules, hidden_size) '的PyTorch张量，包含每个分子的编码。
        """
        f_atoms, f_atoms2, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()
        f_atoms, f_atoms2, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_atoms2.to(self.device), f_bonds.to(self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)

        # Input
        input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()

        input_bond = self.W_i_bond(f_bonds)  # num_bonds x hidden_size
        message_bond = self.act_func(input_bond)
        input_bond = self.act_func(input_bond)

        # Message passing
        for depth in range(self.depth - 1):
            agg_message = index_select_ND(message_bond, a2b)
            agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
            message_atom = message_atom + agg_message

            # directed graph
            rev_message = message_bond[b2revb]  # num_bonds x hidden
            message_bond = message_atom[b2a] - rev_message  # num_bonds x hidden

            message_bond = self._modules[f'W_h_{depth}'](message_bond)
            message_bond = self.dropout_layer(self.act_func(input_bond + message_bond))

        agg_message = index_select_ND(message_bond, a2b)
        agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
        agg_message = self.lr(torch.cat([agg_message, message_atom, input_atom], 1))
        agg_message = self.gru(agg_message, a_scope)

        atom_hiddens = self.act_func(self.W_o(agg_message))  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.mean(0))
        mol_vecs = torch.stack(mol_vecs, dim=0)

        return mol_vecs  # num_molecules x hidden


class MPNEncoder_3D(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, model_conf, device, atom_fdim: int, bond_fdim: int):

        super(MPNEncoder_3D, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = model_conf['layer']['hidden_size_3D']
        self.bias = model_conf['layer']['bias']
        self.depth = model_conf['layer']['depth']
        self.dropout = model_conf['layer']['dropout']
        self.layers_per_message = 1
        # self.undirected = args.undirected
        self.device = device
        self.aggregation = model_conf['layer']['aggregation']
        self.aggregation_norm = model_conf['layer']['aggregation_norm']

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(model_conf['layer']['activate'])

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim + 64
        self.W_i_atom = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        self.gru = BatchGRU(self.hidden_size)

        self.W_h = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.W_o = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self,
                mol_graph: BatchMolGraph) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        一个形状为:code: ' (num_molecules, hidden_size) '的PyTorch张量，包含每个分子的编码。
        """
        f_atoms, f_atoms2, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()
        f_atoms, f_atoms2, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_atoms2.to(self.device), f_bonds.to(self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)

        input_atom = self.W_i_atom(f_atoms2)  # num_atoms x hidden_size
        input_atom = self.act_func(input_atom)
        input_atom = self.W_o(input_atom)  # num_atoms x hidden_size
        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()

        agg_message = self.gru(message_atom, a_scope)
        #
        atom_hiddens = self.act_func(self.W_h(agg_message))  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.mean(0))
        mol_vecs = torch.stack(mol_vecs, dim=0)

        return mol_vecs  # num_molecules x hidden


class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True,
                          bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size),
                                1.0 / math.sqrt(self.hidden_size))

    def forward(self, node, a_scope):
        hidden = node
        # message = F.relu(node + self.bias)
        message = F.relu(node)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        # padding
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))

            cur_message = torch.nn.ZeroPad2d((0, 0, 0, MAX_atom_len - cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))

        message_lst = torch.cat(message_lst, 0)
        hidden_lst = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2, 1, 1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)

        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2 * self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)

        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1),
                             cur_message_unpadding], 0)
        return message


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
    return np.where(inputs < 10, env_val, np.zeros_like(inputs))


def BesselBasisLayer(inputs):
    num_radial = 64  # num_radial=6
    inv_cutoff = np.array(1 / 5, dtype=np.float32)  # cutoff=5   从一个类张量的物体中创建一个常数张量
    # envelope = Envelope(inputs)  # 5
    # Initialize frequencies at canonical positions 在规范位置初始化频率
    freq_init = np.pi * np.arange(1, num_radial + 1, dtype=np.float32)
    # frequencies = self.add_weight(name="frequencies", shape=self.num_radial,
    #                                    dtype=tf.float32, initializer=freq_init, trainable=True)
    d_scaled = inputs * inv_cutoff
    # Necessary for proper broadcasting behaviour
    d_scaled = np.expand_dims(d_scaled, -1)
    # d_cutoff = Envelope(d_scaled)
    return np.sin(freq_init * d_scaled)


class SphericalBasisLayer(nn.Module):
    def __init__(self, num_spherical=7, num_radial=6, cutoff=5):
        super(SphericalBasisLayer, self).__init__()

        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical

        self.inv_cutoff = np.array(1 / cutoff, dtype=np.float32)

        # retrieve formulas
        self.bessel_formulas = bessel_basis(num_spherical, num_radial)
        self.sph_harm_formulas = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        # convert to tensorflow functions
        x = sym.symbols('x')
        theta = sym.symbols('theta')
        for i in range(num_spherical):
            self.sph_funcs.append(sym.lambdify([theta], self.sph_harm_formulas[i][0], 'tensorflow'))
            for j in range(num_radial):
                self.bessel_funcs.append(sym.lambdify([x], self.bessel_formulas[i][j], 'tensorflow'))

    def forward(self, d, Angles, id_expand_kj):

        d_scaled = d * self.inv_cutoff
        rbf = [f(d_scaled) for f in self.bessel_funcs]
        rbf = np.stack(rbf, axis=1)

        d_cutoff = Envelope(d_scaled)
        rbf_env = d_cutoff[:, None] * rbf
        rbf_env = tf.gather(rbf_env, id_expand_kj)

        cbf = [f(Angles) for f in self.sph_funcs]
        cbf = np.stack(cbf, axis=1)
        cbf = np.repeat(cbf, self.num_radial, axis=1)

        return rbf_env * cbf

