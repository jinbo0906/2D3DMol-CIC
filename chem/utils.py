from argparse import Namespace
import csv
from datetime import timedelta
from functools import wraps
import logging
import os
import pickle
from time import time
from typing import Any, Callable, List, Tuple
import collections

import torch

from tqdm import tqdm

from chemprop.data import StandardScaler, MoleculeDataset, preprocess_smiles_columns, get_task_names
from models import MoleculeModel


def makedirs(path: str, isfile: bool = False) -> None:
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. :code:`isfile == True`),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != "":
        os.makedirs(path, exist_ok=True)


def overwrite_state_dict(
    loaded_param_name: str,
    model_param_name: str,
    loaded_state_dict: collections.OrderedDict,
    model_state_dict: collections.OrderedDict,
    logger: logging.Logger = None,
) -> collections.OrderedDict:
    """
    Overwrites a given parameter in the current model with the loaded model.
    :param loaded_param_name: name of parameter in checkpoint model.
    :param model_param_name: name of parameter in current model.
    :param loaded_state_dict: state_dict for checkpoint model.
    :param model_state_dict: state_dict for current model.
    :param logger: A logger.
    :return: The updated state_dict for the current model.
    """
    debug = logger.debug if logger is not None else print

    if model_param_name not in model_state_dict:
        debug(f'Pretrained parameter "{model_param_name}" cannot be found in model parameters.')

    elif model_state_dict[model_param_name].shape != loaded_state_dict[loaded_param_name].shape:
        debug(
            f'Pretrained parameter "{loaded_param_name}" '
            f"of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding "
            f"model parameter of shape {model_state_dict[model_param_name].shape}."
        )

    else:
        debug(f'Loading pretrained parameter "{model_param_name}".')
        model_state_dict[model_param_name] = loaded_state_dict[loaded_param_name]

    return model_state_dict


def load_frzn_model(
    model: torch.nn,
    path: str,
    current_args: Namespace = None,
    cuda: bool = None,
    logger: logging.Logger = None,
) -> MoleculeModel:
    """
    Loads a model checkpoint.
    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :param cuda: Whether to move model to cuda.
    :param logger: A logger.
    :return: The loaded MoleculeModel.
    """
    debug = logger.debug if logger is not None else print

    loaded_mpnn_model = torch.load(path, map_location=lambda storage, loc: storage)
    loaded_state_dict = loaded_mpnn_model["state_dict"]
    loaded_args = loaded_mpnn_model["args"]

    model_state_dict = model.state_dict()

    if loaded_args.number_of_molecules == 1 and current_args.number_of_molecules == 1:
        encoder_param_names = [
            "encoder.encoder.0.W_i.weight",
            "encoder.encoder.0.W_h.weight",
            "encoder.encoder.0.W_o.weight",
            "encoder.encoder.0.W_o.bias",
        ]
        if current_args.checkpoint_frzn is not None:
            # Freeze the MPNN
            for param_name in encoder_param_names:
                model_state_dict = overwrite_state_dict(
                    param_name, param_name, loaded_state_dict, model_state_dict
                )

        if current_args.frzn_ffn_layers > 0:
            ffn_param_names = [
                [f"ffn.{i*3+1}.weight", f"ffn.{i*3+1}.bias"]
                for i in range(current_args.frzn_ffn_layers)
            ]
            ffn_param_names = [item for sublist in ffn_param_names for item in sublist]

            # Freeze MPNN and FFN layers
            for param_name in encoder_param_names + ffn_param_names:
                model_state_dict = overwrite_state_dict(
                    param_name, param_name, loaded_state_dict, model_state_dict
                )

        if current_args.freeze_first_only:
            debug(
                "WARNING: --freeze_first_only flag cannot be used with number_of_molecules=1 (flag is ignored)"
            )

    elif loaded_args.number_of_molecules == 1 and current_args.number_of_molecules > 1:
        # TODO(degraff): these two `if`-blocks can be condensed into one
        if (
            current_args.checkpoint_frzn is not None
            and current_args.freeze_first_only
            and current_args.frzn_ffn_layers <= 0
        ):  # Only freeze first MPNN
            encoder_param_names = [
                "encoder.encoder.0.W_i.weight",
                "encoder.encoder.0.W_h.weight",
                "encoder.encoder.0.W_o.weight",
                "encoder.encoder.0.W_o.bias",
            ]
            for param_name in encoder_param_names:
                model_state_dict = overwrite_state_dict(
                    param_name, param_name, loaded_state_dict, model_state_dict
                )
        if (
            current_args.checkpoint_frzn is not None
            and not current_args.freeze_first_only
            and current_args.frzn_ffn_layers <= 0
        ):  # Duplicate encoder from frozen checkpoint and overwrite all encoders
            loaded_encoder_param_names = [
                "encoder.encoder.0.W_i.weight",
                "encoder.encoder.0.W_h.weight",
                "encoder.encoder.0.W_o.weight",
                "encoder.encoder.0.W_o.bias",
            ] * current_args.number_of_molecules

            model_encoder_param_names = [
                [
                    (
                        f"encoder.encoder.{mol_num}.W_i.weight",
                        f"encoder.encoder.{mol_num}.W_h.weight",
                        f"encoder.encoder.{mol_num}.W_o.weight",
                        f"encoder.encoder.{mol_num}.W_o.bias",
                    )
                ]
                for mol_num in range(current_args.number_of_molecules)
            ]
            model_encoder_param_names = [
                item for sublist in model_encoder_param_names for item in sublist
            ]

            for loaded_param_name, model_param_name in zip(
                loaded_encoder_param_names, model_encoder_param_names
            ):
                model_state_dict = overwrite_state_dict(
                    loaded_param_name, model_param_name, loaded_state_dict, model_state_dict
                )

        if current_args.frzn_ffn_layers > 0:
            raise ValueError(
                f"Number of molecules from checkpoint_frzn ({loaded_args.number_of_molecules}) "
                f"must equal current number of molecules ({current_args.number_of_molecules})!"
            )

    elif loaded_args.number_of_molecules > 1 and current_args.number_of_molecules > 1:
        if loaded_args.number_of_molecules != current_args.number_of_molecules:
            raise ValueError(
                f"Number of molecules in checkpoint_frzn ({loaded_args.number_of_molecules}) "
                f"must either match current model ({current_args.number_of_molecules}) or equal 1."
            )

        if current_args.freeze_first_only:
            raise ValueError(
                f"Number of molecules in checkpoint_frzn ({loaded_args.number_of_molecules}) "
                "must be equal to 1 for freeze_first_only to be used!"
            )

        if (current_args.checkpoint_frzn is not None) & (not (current_args.frzn_ffn_layers > 0)):
            encoder_param_names = [
                [
                    (
                        f"encoder.encoder.{mol_num}.W_i.weight",
                        f"encoder.encoder.{mol_num}.W_h.weight",
                        f"encoder.encoder.{mol_num}.W_o.weight",
                        f"encoder.encoder.{mol_num}.W_o.bias",
                    )
                ]
                for mol_num in range(current_args.number_of_molecules)
            ]
            encoder_param_names = [item for sublist in encoder_param_names for item in sublist]

            for param_name in encoder_param_names:
                model_state_dict = overwrite_state_dict(
                    param_name, param_name, loaded_state_dict, model_state_dict
                )

        if current_args.frzn_ffn_layers > 0:
            encoder_param_names = [
                [
                    (
                        f"encoder.encoder.{mol_num}.W_i.weight",
                        f"encoder.encoder.{mol_num}.W_h.weight",
                        f"encoder.encoder.{mol_num}.W_o.weight",
                        f"encoder.encoder.{mol_num}.W_o.bias",
                    )
                ]
                for mol_num in range(current_args.number_of_molecules)
            ]
            encoder_param_names = [item for sublist in encoder_param_names for item in sublist]
            ffn_param_names = [
                [f"ffn.{i+3+1}.weight", f"ffn.{i+3+1}.bias"]
                for i in range(current_args.frzn_ffn_layers)
            ]
            ffn_param_names = [item for sublist in ffn_param_names for item in sublist]

            for param_name in encoder_param_names + ffn_param_names:
                model_state_dict = overwrite_state_dict(
                    param_name, param_name, loaded_state_dict, model_state_dict
                )

        if current_args.frzn_ffn_layers >= current_args.ffn_num_layers:
            raise ValueError(
                f"Number of frozen FFN layers ({current_args.frzn_ffn_layers}) "
                f"must be less than the number of FFN layers ({current_args.ffn_num_layers})!"
            )

    # Load pretrained weights
    model.load_state_dict(model_state_dict)

    return model


def load_scalers(
    path: str,
) -> Tuple[StandardScaler, StandardScaler, StandardScaler, StandardScaler]:
    """
    Loads the scalers a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: A tuple with the data :class:`~chemprop.data.scaler.StandardScaler`
             and features :class:`~chemprop.data.scaler.StandardScaler`.
    """
    state = torch.load(path, map_location=lambda storage, loc: storage)

    if state["data_scaler"] is not None:
        scaler = StandardScaler(state["data_scaler"]["means"], state["data_scaler"]["stds"])
    else:
        scaler = None

    if state["features_scaler"] is not None:
        features_scaler = StandardScaler(
            state["features_scaler"]["means"], state["features_scaler"]["stds"], replace_nan_token=0
        )
    else:
        features_scaler = None

    if "atom_descriptor_scaler" in state.keys() and state["atom_descriptor_scaler"] is not None:
        atom_descriptor_scaler = StandardScaler(
            state["atom_descriptor_scaler"]["means"],
            state["atom_descriptor_scaler"]["stds"],
            replace_nan_token=0,
        )
    else:
        atom_descriptor_scaler = None

    if "bond_feature_scaler" in state.keys() and state["bond_feature_scaler"] is not None:
        bond_feature_scaler = StandardScaler(
            state["bond_feature_scaler"]["means"],
            state["bond_feature_scaler"]["stds"],
            replace_nan_token=0,
        )
    else:
        bond_feature_scaler = None

    return scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler


def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.
    创建一个带有流处理程序和两个文件处理程序的记录器
    If a logger with that name already exists, simply returns that logger.
    Otherwise, creates a new logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of :code:`quiet`.
    One file handler (:code:`verbose.log`) saves all logs, the other (:code:`quiet.log`) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e., print only important info).
    :return: The logger.
    """

    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)

        fh_v = logging.FileHandler(os.path.join(save_dir, "verbose.log"))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, "quiet.log"))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger


def timeit(logger_name: str = None) -> Callable[[Callable], Callable]:
    """
    Creates a decorator which wraps a function with a timer that prints the elapsed time.

    :param logger_name: The name of the logger used to record output. If None, uses :code:`print` instead.
    :return: A decorator which wraps a function with a timer that prints the elapsed time.
    """

    def timeit_decorator(func: Callable) -> Callable:
        """
        A decorator which wraps a function with a timer that prints the elapsed time.

        :param func: The function to wrap with the timer.
        :return: The function wrapped with the timer.
        """

        @wraps(func)
        def wrap(*args, **kwargs) -> Any:
            start_time = time()
            result = func(*args, **kwargs)
            delta = timedelta(seconds=round(time() - start_time))
            info = logging.getLogger(logger_name).info if logger_name is not None else print
            info(f"Elapsed time = {delta}")

            return result

        return wrap

    return timeit_decorator


def save_smiles_splits(
    data_path: str,
    save_dir: str,
    task_names: List[str] = None,
    features_path: List[str] = None,
    train_data: MoleculeDataset = None,
    val_data: MoleculeDataset = None,
    test_data: MoleculeDataset = None,
    logger: logging.Logger = None,
    smiles_columns: List[str] = None,
) -> None:
    """
    Saves a csv file with train/val/test splits of target data and additional features.
    Also saves indices of train/val/test split as a pickle file. Pickle file does not support repeated entries
    with the same SMILES or entries entered from a path other than the main data path, such as a separate test path.

    :param data_path: Path to data CSV file.
    :param save_dir: Path where pickle files will be saved.
    :param task_names: List of target names for the model as from the function get_task_names().
        If not provided, will use datafile header entries.
    :param features_path: List of path(s) to files with additional molecule features.
    :param train_data: Train :class:`~chemprop.data.data.MoleculeDataset`.
    :param val_data: Validation :class:`~chemprop.data.data.MoleculeDataset`.
    :param test_data: Test :class:`~chemprop.data.data.MoleculeDataset`.
    :param smiles_columns: The name of the column containing SMILES. By default, uses the first column.
    :param logger: A logger for recording output.
    """
    makedirs(save_dir)

    info = logger.info if logger is not None else print
    save_split_indices = True

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
                    "Warning: Repeated SMILES found in data, pickle file of split indices cannot distinguish entries and will not be generated."
                )
                break
            indices_by_smiles[smiles] = i

    if task_names is None:
        task_names = get_task_names(path=data_path, smiles_columns=smiles_columns)

    features_header = []
    if features_path is not None:
        for feat_path in features_path:
            with open(feat_path, "r") as f:
                reader = csv.reader(f)
                feat_header = next(reader)
                features_header.extend(feat_header)

    all_split_indices = []
    for dataset, name in [(train_data, "train"), (val_data, "val"), (test_data, "test")]:
        if dataset is None:
            continue

        with open(os.path.join(save_dir, f"{name}_smiles.csv"), "w") as f:
            writer = csv.writer(f)
            if smiles_columns[0] == "":
                writer.writerow(["smiles"])
            else:
                writer.writerow(smiles_columns)
            for smiles in dataset.smiles():
                writer.writerow(smiles)

        with open(os.path.join(save_dir, f"{name}_full.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(smiles_columns + task_names)
            dataset_targets = dataset.targets()
            for i, smiles in enumerate(dataset.smiles()):
                writer.writerow(smiles + dataset_targets[i])

        if features_path is not None:
            dataset_features = dataset.features()
            with open(os.path.join(save_dir, f"{name}_features.csv"), "w") as f:
                writer = csv.writer(f)
                writer.writerow(features_header)
                writer.writerows(dataset_features)

        if save_split_indices:
            split_indices = []
            for smiles in dataset.smiles():
                index = indices_by_smiles.get(tuple(smiles))
                if index is None:
                    save_split_indices = False
                    info(
                        f"Warning: SMILES string in {name} could not be found in data file, and "
                        "likely came from a secondary data file. The pickle file of split indices "
                        "can only indicate indices for a single file and will not be generated."
                    )
                    break
                split_indices.append(index)
            else:
                split_indices.sort()
                all_split_indices.append(split_indices)

        if name == "train":
            data_weights = dataset.data_weights()
            if any([w != 1 for w in data_weights]):
                with open(os.path.join(save_dir, f"{name}_weights.csv"), "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["data weights"])
                    for weight in data_weights:
                        writer.writerow([weight])

    if save_split_indices:
        with open(os.path.join(save_dir, "split_indices.pckl"), "wb") as f:
            pickle.dump(all_split_indices, f)
