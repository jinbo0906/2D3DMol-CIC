from logging import Logger
import os
import pickle
import re
import csv
from logging import info
from random import Random
from typing import Callable, List, Union, Tuple
import numpy as np
from argparse import Namespace

from chem.nn_utils import NoamLR
from hydra.utils import get_original_cwd
from utils.data import MoleculeDataset, StandardScaler, MoleculeDatapoint, make_mols
from models.model import MoleculeModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from chem.features import is_mol

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer


def save_smiles_splits(data_path: str,
                       save_dir: str,
                       task_names: List[str] = None,
                       train_data: MoleculeDataset = None,
                       val_data: MoleculeDataset = None,
                       test_data: MoleculeDataset = None,
                       smiles_columns: List[str] = None) -> None:
    save_split_indices = True

    data_path = os.path.join(get_original_cwd(), data_path)

    if not isinstance(smiles_columns, list):
        smiles_columns = preprocess_smiles_columns(path=data_path, smiles_columns=smiles_columns)

    with open(data_path) as f:
        reader = csv.DictReader(f)

        indices_by_smiles = {}
        for i, row in enumerate(tqdm(reader)):
            smiles = tuple([row[column] for column in smiles_columns])
            if smiles in indices_by_smiles:
                save_split_indices = False
                info(
                    'Warning: Repeated SMILES found in data, pickle file of split indices cannot distinguish entries and will not be generated.')
                break
            indices_by_smiles[smiles] = i

    if task_names is None:
        task_names = get_task_names(path=data_path, smiles_columns=smiles_columns)

    all_split_indices = []
    for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
        if dataset is None:
            continue

        with open(os.path.join(save_dir, f'{name}_smiles.csv'), 'w') as f:
            writer = csv.writer(f)
            if smiles_columns[0] == '':
                writer.writerow(['smiles'])
            else:
                writer.writerow(smiles_columns)
            for smiles in dataset.smiles():
                writer.writerow(smiles)

        with open(os.path.join(save_dir, f'{name}_full.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(smiles_columns + task_names)
            dataset_targets = dataset.targets()
            for i, smiles in enumerate(dataset.smiles()):
                writer.writerow(smiles + dataset_targets[i])

        if save_split_indices:
            split_indices = []
            for smiles in dataset.smiles():
                index = indices_by_smiles.get(tuple(smiles))
                if index is None:
                    save_split_indices = False
                    info(
                        f'Warning: SMILES string in {name} could not be found in data file, and likely came from a secondary data file. '
                        'The pickle file of split indices can only indicate indices for a single file and will not be generated.')
                    break
                split_indices.append(index)
            else:
                split_indices.sort()
                all_split_indices.append(split_indices)

        if name == 'train':
            data_weights = dataset.data_weights()
            if any([w != 1 for w in data_weights]):
                with open(os.path.join(save_dir, f'{name}_weights.csv'), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['data weights'])
                    for weight in data_weights:
                        writer.writerow([weight])

    if save_split_indices:
        with open(os.path.join(save_dir, 'split_indices.pckl'), 'wb') as f:
            pickle.dump(all_split_indices, f)


def get_task_names(path: str,
                   smiles_columns: Union[str, List[str]] = None,
                   target_columns: List[str] = None,
                   ignore_columns: List[str] = None) -> List[str]:
    if target_columns is not None:
        return target_columns

    columns = get_header(path)

    if not isinstance(smiles_columns, list):
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns)

    ignore_columns = set(smiles_columns + ([] if ignore_columns is None else ignore_columns))

    target_names = [column for column in columns if column not in ignore_columns]

    return target_names


def load_checkpoint(log,
                    model_conf,
                    run_conf,
                    global_conf,
                    path: str,
                    device: torch.device = None) -> MoleculeModel:
    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)

    loaded_state_dict = state['state_dict']

    # Build model
    model = MoleculeModel(model_conf, run_conf, global_conf)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for loaded_param_name in loaded_state_dict.keys():
        # Backward compatibility for parameter names
        if re.match(r'(encoder\.encoder\.)([Wc])', loaded_param_name):
            param_name = loaded_param_name.replace('encoder.encoder', 'encoder.encoder.0')
        else:
            param_name = loaded_param_name

        # Load pretrained parameter, skipping unmatched parameters
        if param_name not in model_state_dict:
            info(f'Warning: Pretrained parameter "{loaded_param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[loaded_param_name].shape:
            info(f'Warning: Pretrained parameter "{loaded_param_name}" '
                 f'of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding '
                 f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            log.info(f'Loading pretrained parameter "{loaded_param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[loaded_param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    model = model.to(device)

    return model


def save_checkpoint(
        path: str,
        model: MoleculeModel,
        scaler: StandardScaler = None,
        args: dict = None,
) -> None:
    # Convert args to namespace for backwards compatibility
    if args is not None:
        args = Namespace(**args)

    data_scaler = {"means": scaler.means, "stds": scaler.stds} if scaler is not None else None

    state = {
        "args": args,
        "state_dict": model.state_dict(),
        "data_scaler": data_scaler,
    }
    torch.save(state, path)


def build_optimizer(model: nn.Module, run_conf: dict) -> Optimizer:
    params = [{"params": model.parameters(), "lr": run_conf['train_conf']['init_lr'], "weight_decay": 0}]

    return Adam(params)


def build_lr_scheduler(
        optimizer: Optimizer, run_conf: dict, total_epochs: List[int] = None
):
    # Learning rate scheduler
    if run_conf['train_conf']['scheduler'] == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(
            run_conf["train_conf"]["epochs"] / run_conf["train_conf"]['sch_step']["stage"]),
                                               gamma=run_conf["train_conf"]['sch_step']["gamma"])
    elif run_conf['train_conf']['scheduler'] == 'NoamLR':
        return NoamLR(
            optimizer=optimizer,
            warmup_epochs=[run_conf['train_conf']['warmup_epochs']],
            total_epochs=total_epochs or [run_conf['train_conf']['epochs']] * 1,
            steps_per_epoch=992 // run_conf['train_conf']['batch_size'],
            init_lr=[run_conf['train_conf']['init_lr']],
            max_lr=[run_conf['train_conf']['max_lr']],
            final_lr=[run_conf['train_conf']['final_lr']],
        )


def get_metric_func(metric: str) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    if metric == 'rmse':
        return rmse

    if metric == 'mse':
        return mean_squared_error

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score


def rmse(targets: List[float], preds: List[float]) -> float:
    return mean_squared_error(targets, preds, squared=False)


def extract_model_pt_paths(root_dir):
    model_pt_paths = []

    # 遍历根目录及其子目录
    for root, dirs, files in os.walk(root_dir):
        # 检查当前目录下是否存在model.pt文件
        if "model.pt" in files:
            # 构造完整路径并添加到列表中
            model_path = os.path.join(root, "model.pt")
            model_pt_paths.append(model_path)

    return model_pt_paths


def extract_pred_csv_paths(root_dir):
    csv_paths = []

    # 遍历根目录及其子目录
    for root, dirs, files in os.walk(root_dir):
        # 检查当前目录下是否存在model.pt文件
        if "preds.csv" in files:
            # 构造完整路径并添加到列表中
            csv_path = os.path.join(root, "preds.csv")
            csv_paths.append(csv_path)

    return csv_paths


def get_data(
        log,
        smi_path: str,
        geom_path: str,
        data_args: dict,
        skip_invalid_smiles: bool = True,
        skip_none_targets: bool = False):
    id_columns, smiles_columns, target_columns, env_columns, features_generator = data_args['id_columns'], \
        data_args['smiles_columns'], data_args['target_columns'], data_args['env_columns'], data_args[
        'features_generator']

    if not isinstance(smiles_columns, list):
        smiles_columns = preprocess_smiles_columns(path=smi_path, smiles_columns=smiles_columns)

    max_data_size = float('inf')

    # Load data  加载smi数据
    with open(smi_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if any([c not in fieldnames for c in smiles_columns]):
            raise ValueError(
                f'Data file did not contain all provided smiles columns: {smiles_columns}. Data file field names are: {fieldnames}')
        if any([c not in fieldnames for c in target_columns]):
            raise ValueError(
                f'Data file did not contain all provided target columns: {target_columns}. Data file field names are: {fieldnames}')

        all_id, all_smiles, all_targets, all_env, all_R = [], [], [], [], []
        for i, row in enumerate(tqdm(reader)):
            smiles = [row[c] for c in smiles_columns]  # 获取SMILES
            targets = []  # 获取目标值
            for column in target_columns:
                value = row[column]
                if value in ['', 'nan']:
                    targets.append(None)
                elif '>' in value or '<' in value:
                    raise ValueError(
                        'Inequality found in target data. To use inequality targets (> or <), the regression loss '
                        'function bounded_mse must be used.')
                else:
                    targets.append(float(value))

            id = []
            for column in id_columns:
                value = row[column]
                id.append(float(value))

            env = []
            for column in env_columns:
                value = row[column]
                env.append(float(value))

            if skip_none_targets and all(x is None for x in targets):
                continue

            all_id.append(id)
            all_smiles.append(smiles)
            all_targets.append(targets)
            all_env.append(env[0])

            if len(all_smiles) >= max_data_size:
                break

    # 加载geom数据
    inhibitor = np.load(geom_path, allow_pickle=True)
    molecula_R = inhibitor['R']
    molecula_N = inhibitor['N']
    N_cumsum = [0]
    N = 0
    for n in molecula_N:
        N = n + N
        N_cumsum.append(N)
    for i in range(len(N_cumsum) - 1):
        R = molecula_R[N_cumsum[i]:N_cumsum[i + 1]]
        all_R.append(R)

    data = MoleculeDataset([
        MoleculeDatapoint(
            id=id,
            smiles=smiles,
            targets=targets,
            envs=envs,
            R=r,
            features_generator=features_generator
        ) for i, (id, smiles, targets, envs, r) in tqdm(enumerate(zip(all_id, all_smiles, all_targets, all_env, all_R)),
                                                        total=len(all_smiles))
    ])

    # Filter out invalid SMILES  过滤无效的SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            log.info(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    return MoleculeDataset([datapoint for datapoint in tqdm(data)
                            if all(s != '' for s in datapoint.smiles) and all(m is not None for m in datapoint.mol)
                            and all(m.GetNumHeavyAtoms() > 0 for m in datapoint.mol if not isinstance(m, tuple))
                            and all(
            m[0].GetNumHeavyAtoms() + m[1].GetNumHeavyAtoms() > 0 for m in datapoint.mol if isinstance(m, tuple))])


def get_data_from_smiles(smiles: List[List[str]],
                         skip_invalid_smiles: bool = True,
                         logger: Logger = None,
                         features_generator: List[str] = None) -> MoleculeDataset:
    debug = logger.debug if logger is not None else print

    data = MoleculeDataset([
        MoleculeDatapoint(
            smiles=smile,
            features_generator=features_generator
        ) for smile in smiles
    ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def get_header(path: str) -> List[str]:
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def preprocess_smiles_columns(path: str,
                              smiles_columns: Union[str, List[str]] = None,
                              number_of_molecules: int = 1) -> List[str]:
    if smiles_columns is None:
        if os.path.isfile(path):
            columns = get_header(path)
            smiles_columns = columns[:number_of_molecules]
        else:
            smiles_columns = [None] * number_of_molecules
    else:
        if not isinstance(smiles_columns, list):
            smiles_columns = [smiles_columns]
        if os.path.isfile(path):
            columns = get_header(path)
            if len(smiles_columns) != number_of_molecules:
                raise ValueError('Length of smiles_columns must match number_of_molecules.')
            if any([smiles not in columns for smiles in smiles_columns]):
                raise ValueError('Provided smiles_columns do not match the header of data file.')

    return smiles_columns


def get_smiles(path: str,
               smiles_columns: Union[str, List[str]] = None,
               number_of_molecules: int = 1,
               header: bool = True,
               flatten: bool = False
               ) -> Union[List[str], List[List[str]]]:
    if smiles_columns is not None and not header:
        raise ValueError('If smiles_column is provided, the CSV file must have a header.')

    if not isinstance(smiles_columns, list) and header:
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns,
                                                   number_of_molecules=number_of_molecules)

    with open(path) as f:
        if header:
            reader = csv.DictReader(f)
        else:
            reader = csv.reader(f)
            smiles_columns = list(range(number_of_molecules))

        smiles = [[row[c] for c in smiles_columns] for row in reader]

    if flatten:
        smiles = [smile for smiles_list in smiles for smile in smiles_list]

    return smiles


def get_invalid_smiles_from_file(path: str = None,
                                 smiles_columns: Union[str, List[str]] = None,
                                 header: bool = True,
                                 reaction: bool = False,
                                 ) -> Union[List[str], List[List[str]]]:
    smiles = get_smiles(path=path, smiles_columns=smiles_columns, header=header)

    invalid_smiles = get_invalid_smiles_from_list(smiles=smiles, reaction=reaction)

    return invalid_smiles


def get_invalid_smiles_from_list(smiles: List[List[str]], reaction: bool = False) -> List[List[str]]:
    invalid_smiles = []

    # If the first SMILES in the column is a molecule, the remaining SMILES in the same column should all be a molecule.
    # Similarly, if the first SMILES in the column is a reaction, the remaining SMILES in the same column should all
    # correspond to reaction. Therefore, get `is_mol_list` only using the first element in smiles.
    is_mol_list = [is_mol(s) for s in smiles[0]]
    is_reaction_list = [True if not x and reaction else False for x in is_mol_list]
    is_explicit_h_list = [False for x in is_mol_list]  # set this to False as it is not needed for invalid SMILES check
    is_adding_hs_list = [False for x in is_mol_list]  # set this to False as it is not needed for invalid SMILES check

    for mol_smiles in smiles:
        mols = make_mols(smiles=mol_smiles, reaction_list=is_reaction_list, keep_h_list=is_explicit_h_list,
                         add_h_list=is_adding_hs_list)
        if any(s == '' for s in mol_smiles) or \
                any(m is None for m in mols) or \
                any(m.GetNumHeavyAtoms() == 0 for m in mols if not isinstance(m, tuple)) or \
                any(m[0].GetNumHeavyAtoms() + m[1].GetNumHeavyAtoms() == 0 for m in mols if isinstance(m, tuple)):
            invalid_smiles.append(mol_smiles)

    return invalid_smiles


def split_data(data: MoleculeDataset,
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0,
               num_folds: int = 1) -> Tuple[MoleculeDataset,
                                               MoleculeDataset,
                                               MoleculeDataset]:

    random = Random(seed)

    if split_type == 'random':
        indices = list(range(len(data)))
        random.shuffle(indices)

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = [data[i] for i in indices[:train_size]]
        val = [data[i] for i in indices[train_size:train_val_size]]
        test = [data[i] for i in indices[train_val_size:]]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'non-random':
        indices = list(range(len(data)))

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = [data[i] for i in indices[:train_size]]
        val = [data[i] for i in indices[train_size:train_val_size]]
        test = [data[i] for i in indices[train_val_size:]]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    else:
        raise ValueError(f'split_type "{split_type}" not supported.')
