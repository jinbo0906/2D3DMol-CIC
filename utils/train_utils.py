import json
import os
from collections import defaultdict
from typing import Dict, List, Callable

import numpy as np
import pandas as pd
from chemprop.nn_utils import NoamLR, compute_pnorm, compute_gnorm
from tensorboardX import SummaryWriter
import torch
from torch import nn
from tqdm import trange, tqdm
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler
from torch.optim import Optimizer

from models.model import MoleculeModel
from models.loss_func import get_loss_func
from chemprop.constants import MODEL_FILE_NAME
from utils.data_utils import MoleculeDataset, set_cache_graph, MoleculeDataLoader, StandardScaler
from utils.util_utils import split_data, get_metric_func, save_smiles_splits, save_checkpoint, load_checkpoint, \
    build_optimizer, build_lr_scheduler

from chemprop.utils import makedirs


def run_training(run_conf: dict,
                 model_conf: dict,
                 data_conf: dict,
                 global_conf: dict,
                 data: MoleculeDataset,
                 seed: int,
                 log,
                 save_dir) -> Dict[str, List[float]]:
    # Split data 分割数据集
    log.info(f'Splitting data with seed {seed}')
    train_data, val_data, test_data = split_data(data=data,
                                                 split_type=run_conf['data_conf']['split_type'],
                                                 seed=seed,
                                                 num_folds=run_conf['train_conf']['num_folds']
                                                 )

    if run_conf['train_conf']['save_smiles_splits']:
        save_smiles_dir = os.path.join(save_dir, run_conf['train_conf']['splits_save_path'])
        save_smiles_splits(
            data_path=data_conf['observe_data']['smi_train_csv_file'],
            save_dir=save_smiles_dir,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            smiles_columns=data_conf['observe_data']['smiles_columns']
        )

    # train_data_size = len(train_data)

    log.info(f'Total size = {len(data):,} | '
             f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')
    log.info('Fitting scaler')
    scaler = train_data.normalize_targets()  # 归一化目标值

    # Get loss function
    loss_func = get_loss_func(run_conf)

    # Set up test set evaluation  设置测试集评估
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), run_conf['train_conf']['num_tasks']))

    # Automatically determine whether to cache
    if len(data) <= run_conf['train_conf']['cache_cutoff']:
        set_cache_graph(True)
        num_workers = 0
    else:
        set_cache_graph(False)
        num_workers = run_conf['train_conf']['num_workers']

    # Create data loaders 创建数据加载器
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=run_conf['train_conf']['batch_size'],
        num_workers=num_workers,
        class_balance=run_conf['train_conf']['class_balance'],
        shuffle=True,
        seed=seed
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=run_conf['train_conf']['batch_size'],
        num_workers=num_workers
    )
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=run_conf['train_conf']['batch_size'],
        num_workers=num_workers
    )

    # Train ensemble of models
    for model_idx in range(model_conf['ensemble_size']):
        # Tensorboard writer
        save_model_dir = os.path.join(save_dir, f'model_{model_idx}')
        makedirs(save_model_dir)
        try:
            writer = SummaryWriter(log_dir=save_model_dir)
        except:
            writer = SummaryWriter(logdir=save_model_dir)

        # Load/build model 加载/构建模型
        # checkpoint_paths = run_conf['train_conf']['checkpoint_paths']
        # if checkpoint_paths is not None:
        #     log.info(f'Loading model {model_idx} from {checkpoint_paths[model_idx]}')
        #     model = load_checkpoint(log, model_conf, run_conf, global_conf, checkpoint_paths[model_idx],
        #                             device=global_conf['device'])
        # else:
        log.info(f'Building model {model_idx}')
        model = MoleculeModel(model_conf, run_conf, global_conf)

        log.info(model)
        if global_conf['device'] == 'cuda':
            log.info('Moving model to cuda')
        model = model.to(global_conf['device'])

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_model_dir, MODEL_FILE_NAME), model, scaler, run_conf)

        # Optimizers
        optimizer = build_optimizer(model, run_conf)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, run_conf)

        # Run training
        best_score = float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in trange(run_conf['train_conf']['epochs']):
            log.info(f'Epoch {epoch}')
            n_iter = train(
                logger=log,
                run_conf=run_conf,
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                n_iter=n_iter,
                writer=writer,
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            log.info(f'lr:{optimizer.param_groups[0]["lr"]}')
            val_scores = evaluate(
                model=model,
                data_loader=val_data_loader,
                num_tasks=run_conf['train_conf']['num_tasks'],
                metrics=run_conf['train_conf']['metrics'],
                dataset_type=run_conf['train_conf']['dataset_type'],
                scaler=scaler
            )
            for metric, scores in val_scores.items():
                # Average validation score
                avg_val_score = np.nanmean(scores)
                log.info(f'Validation {metric} = {avg_val_score:.6f}')
                writer.add_scalar(f'validation_{metric}', avg_val_score, n_iter)
            # Save model checkpoint if improved validation score 如果改进验证分数，则保存模型检查点
            avg_val_score = np.nanmean(val_scores['rmse'])
            if (avg_val_score < best_score) or (epoch > 200 and avg_val_score > best_score):
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_model_dir, MODEL_FILE_NAME), model, scaler, run_conf)

        # Evaluate on test set using model with the best validation score 使用验证得分最高的模型对测试集进行评估
        log.info(f'Model {model_idx} best validation rmse = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(log, model_conf, run_conf, global_conf, os.path.join(save_model_dir, MODEL_FILE_NAME),
                                device=global_conf['device'])

        test_preds = predict(
            model=model,
            data_loader=test_data_loader,
            scaler=scaler
        )

        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=run_conf['train_conf']['num_tasks'],
            metrics=run_conf['train_conf']['metrics']
        )

        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds)

        # Average test score
        for metric, scores in test_scores.items():
            avg_test_score = np.nanmean(scores)
            log.info(f'Model {model_idx} test {metric} = {avg_test_score:.6f}')
            writer.add_scalar(f'test_{metric}', avg_test_score, 0)

        writer.close()

    # Evaluate ensemble on test set
    avg_test_preds = (sum_test_preds / model_conf['ensemble_size']).tolist()
    ensemble_scores = evaluate_predictions(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=run_conf['train_conf']['num_tasks'],
        metrics=run_conf['train_conf']['metrics']
    )

    for metric, scores in ensemble_scores.items():
        # Average ensemble score
        avg_ensemble_test_score = np.nanmean(scores)
        log.info(f'Ensemble test {metric} = {avg_ensemble_test_score:.6f}')

    # Save scores
    with open(os.path.join(save_dir, 'test_scores.json'), 'w') as f:
        json.dump(ensemble_scores, f, indent=4, sort_keys=True)

    # Optionally save test preds
    if run_conf['train_conf']['save_preds']:
        test_preds_dataframe = pd.DataFrame(data={'smiles': test_data.smiles()})
        # test_real_dataframe = pd.DataFrame(data={'smiles': test_data.smiles()})
        dataframe_name = ['pre', 'real']
        for i, task_name in enumerate(dataframe_name):
            if task_name == 'pre':
                test_preds_dataframe[task_name] = [pred[i] for pred in avg_test_preds]
            if task_name == 'real':
                test_preds_dataframe[task_name] = [real[i] for real in test_targets]
        test_preds_dataframe.to_csv(os.path.join(save_dir, 'test_preds.csv'), index=False)
        # test_real_dataframe.to_csv(os.path.join(save_dir, 'test_real.csv'), index=False)

    return ensemble_scores


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

        optimizer.step()

        if isinstance(scheduler, NoamLR):
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


def activate_dropout(module: nn.Module, dropout_prob: float):
    if isinstance(module, nn.Dropout):
        module.p = dropout_prob
        module.train()


def run_testing(log, model_path, test_data, run_conf, model_conf, global_conf, save_pre_dir):
    test_data, _, _ = split_data(data=test_data, split_type='non-random',
                                 sizes=(1, 0.0, 0.0),
                                 num_folds=run_conf['train_conf']['num_folds']
                                 )
    scaler = test_data.normalize_targets()  # 归一化目标值
    # Set up test set evaluation  设置测试集评估
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), run_conf['train_conf']['num_tasks']))
    num_workers = 0
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=run_conf['train_conf']['batch_size'],
        num_workers=num_workers
    )
    model = load_checkpoint(log, model_conf, run_conf, global_conf, model_path,
                            device=global_conf['device'])

    test_preds = predict(
        model=model,
        data_loader=test_data_loader,
        scaler=scaler
    )
    if len(test_preds) != 0:
        sum_test_preds += np.array(test_preds)
    avg_test_preds = (sum_test_preds / model_conf['ensemble_size']).tolist()
    test_preds_dataframe = pd.DataFrame(data={'smiles': test_data.smiles()})
    # test_real_dataframe = pd.DataFrame(data={'smiles': test_data.smiles()})
    dataframe_name = ['pre']
    for i, task_name in enumerate(dataframe_name):
        if task_name == 'pre':
            test_preds_dataframe[task_name] = [pred[i] for pred in avg_test_preds]

    preds_save_path = os.path.join(save_pre_dir, 'preds.csv')
    test_preds_dataframe.to_csv(preds_save_path, index=False)

    return preds_save_path
