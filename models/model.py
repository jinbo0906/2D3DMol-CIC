from math import exp
from typing import List, Union, Tuple

import numpy as np
from rdkit import Chem
from torch import nn
import torch

from models import MPN_2D, MPN_3D, BesselBasisLayer
from chemprop.nn_utils import get_activation_function
from chemprop.features import BatchMolGraph


class MoleculeModel(nn.Module):

    def __init__(self, model_conf, run_conf, global_conf):
        super(MoleculeModel, self).__init__()
        self.loss_function = model_conf['loss_function']
        self.device = global_conf['device']

        self.output_size = run_conf['train_conf']['num_tasks']

        self.create_encoder(model_conf, self.device)
        self.create_ffn(model_conf)
        self.w = nn.Parameter(torch.ones(2))

    def create_encoder(self, model_conf, device) -> None:
        """
        Creates the message passing encoder for the model.
        为模型创建消息传递编码器
        """
        self.encoder_2D = MPN_2D(model_conf, device)
        self.encoder_3D = MPN_3D(model_conf, device)

        if model_conf['checkpoint_frzn'] is not None:
            if model_conf['freeze_first_only']:  # Freeze only the first encoder
                for param in list(self.encoder.encoder.children())[0].parameters():
                    param.requires_grad = False
            else:  # Freeze all encoders
                for param in self.encoder.parameters():
                    param.requires_grad = False

    def create_ffn(self, model_conf) -> None:
        """
        Creates the feed-forward layers for the model.
        """
        if model_conf['is_env']:
            first_linear_dim = model_conf['layer']['hidden_size'] + 64 + model_conf['layer']['hidden_size_3D']
        else:
            first_linear_dim = model_conf['layer']['hidden_size'] + model_conf['layer']['hidden_size_3D']
            # first_linear_dim = args.hidden_size * args.number_of_molecules

        dropout = nn.Dropout(model_conf['layer']['dropout'])
        activation = get_activation_function(model_conf['layer']['activate'])

        # Create FFN layers   这里可以修改ffn层
        if model_conf['layer']['ffn_num_layers'] == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, model_conf['layer']['ffn_hidden_size'])
            ]
            for _ in range(model_conf['layer']['ffn_num_layers'] - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(model_conf['layer']['ffn_hidden_size'], model_conf['layer']['ffn_hidden_size']),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(model_conf['layer']['ffn_hidden_size'], self.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def fingerprint(self,
                    batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                    fingerprint_type: str = 'MPN') -> torch.Tensor:
        """
        Encodes the latent representations of the input molecules from intermediate stages of the model.

        :param batch: A list of SMILES, a list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param fingerprint_type: The choice of which type of latent representation to return as the molecular fingerprint. Currently
                                 supported MPN for the output of the MPNN portion of the model or last_FFN for the input to the final readout layer.
        :return: The latent fingerprint vectors.
        """
        if fingerprint_type == 'MPN':
            return self.encoder(batch)
        elif fingerprint_type == 'last_FFN':
            return self.ffn[:-1](self.encoder(batch))
        else:
            raise ValueError(f'Unsupported fingerprint type {fingerprint_type}.')

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                envs_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of SMILES, a list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :return: The output of the :class:`MoleculeModel`, containing a list of property predictions
        """
        # w1 = torch.exp(self.w[0])/torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        output_2D = self.encoder_2D(batch)
        output_3D = self.encoder_3D(batch)

        env_emb = torch.Tensor([BesselBasisLayer(exp(env))*0.1 for env in envs_batch]).float().to(self.device)
        # env_emb = env_emb.to(self.device)

        output1 = torch.cat([output_2D, output_3D], dim=1)
        output = torch.cat([output1, env_emb], dim=1)
        # output = torch.cat([output_2D, env_emb], dim=1)

        output = self.ffn(output)

        return output
