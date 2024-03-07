import math
from collections import defaultdict
from typing import Dict, List, Callable

from chem.nn_utils import compute_pnorm, compute_gnorm
from tensorboardX import SummaryWriter
import torch
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

from models.model import MoleculeModel
from models.utils import activate_dropout
from utils.data import MoleculeDataset, MoleculeDataLoader, StandardScaler
from utils.util import get_metric_func


def train(logger,
          run_conf,
          model: MoleculeModel,
          data_loader: MoleculeDataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          n_iter: int = 0,
          writer: SummaryWriter = None
          ) -> int:

    model.train()
    loss_sum = iter_count = 0

    for batch in tqdm(data_loader, total=len(data_loader), leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, target_batch, data_weights_batch, envs_batch = batch.batch_graph(), \
            batch.features(), batch.targets(), batch.data_weights(), batch.envs()
        mask = torch.tensor([[x is not None for x in tb] for tb in target_batch],
                            dtype=torch.bool)  # shape(batch, tasks)
        targets = torch.tensor([[0 if x is None else x for x in tb] for tb in target_batch])  # shape(batch, tasks)

        target_weights = torch.ones(targets.shape[1]).unsqueeze(0)
        data_weights = torch.tensor(data_weights_batch).unsqueeze(1)  # shape(batch,1)

        # Run model
        model.zero_grad()
        preds = model(mol_batch, features_batch, envs_batch)

        # if math.isnan(preds):
        #     print(batch.id(), batch.smiles())
        #     continue

        # Move tensors to correct device
        torch_device = preds.device
        mask = mask.to(torch_device)
        targets = targets.to(torch_device)
        target_weights = target_weights.to(torch_device)
        data_weights = data_weights.to(torch_device)

        # 计算损失
        loss = loss_func(preds, targets) * target_weights * data_weights * mask
        loss = loss.sum() / mask.sum()
        loss_sum += loss.item()
        iter_count += 1
        loss.backward()

        if run_conf["train_conf"]["grad_clip"] != 0:
            nn.utils.clip_grad_norm_(model.parameters(), run_conf["train_conf"]["grad_clip"])

        optimizer.step()

        scheduler.step()

        n_iter += len(batch)

        # Log and/or add to tensorboard
        if (n_iter // run_conf['train_conf']['batch_size']) % run_conf['train_conf']['log_frequency'] == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum = iter_count = 0

            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            logger.info(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    return n_iter


def predict(
        model: MoleculeModel,
        data_loader: MoleculeDataLoader,
        disable_progress_bar: bool = False,
        scaler: StandardScaler = None,
        return_unc_parameters: bool = False,
        dropout_prob: float = 0.0,
) -> List[List[float]]:
    model.eval()

    # Activate dropout layers to work during inference for uncertainty estimation
    if dropout_prob > 0.0:
        def activate_dropout_(model):
            return activate_dropout(model, dropout_prob)

        model.apply(activate_dropout_)

    preds = []

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch = batch.batch_graph()
        features_batch = batch.features()
        envs_batch = batch.envs()

        # Make predictions
        with torch.no_grad():
            batch_preds = model(
                mol_batch,
                features_batch,
                envs_batch
            )

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds


def evaluate(
        model: MoleculeModel,
        data_loader: MoleculeDataLoader,
        num_tasks: int,
        metrics: List[str],
        dataset_type: str,
        scaler: StandardScaler = None) -> Dict[str, List[float]]:

    preds = predict(
        model=model,
        data_loader=data_loader,
        scaler=scaler
    )

    results = evaluate_predictions(
        preds=preds,
        targets=data_loader.targets,
        num_tasks=num_tasks,
        metrics=metrics
    )

    return results


def evaluate_predictions(
        preds: List[List[float]],
        targets: List[List[float]],
        num_tasks: int,
        metrics: List[str]
) -> Dict[str, List[float]]:
    metric_to_func = {metric: get_metric_func(metric) for metric in metrics}

    if len(preds) == 0:
        return {metric: [float('nan')] * num_tasks for metric in metrics}

    # valid_preds and valid_targets have shape (num_tasks, data_size)
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(len(preds)):
            if targets[j][i] is not None:  # Skip those without targets
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])

    # Compute metric. Spectra loss calculated for all tasks together, others calculated for tasks individually.
    results = defaultdict(list)

    for i in range(num_tasks):
        # # Skip if all targets or preds are identical, otherwise we'll crash during classification

        if len(valid_targets[i]) == 0:
            continue

        for metric, metric_func in metric_to_func.items():
            results[metric].append(metric_func(valid_targets[i], valid_preds[i]))

    results = dict(results)

    return results