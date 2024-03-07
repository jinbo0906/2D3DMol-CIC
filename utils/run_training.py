import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from chem.nn_utils import param_count_all
from tensorboardX import SummaryWriter
from tqdm import trange

from models.model import MoleculeModel
from models.loss_func import get_loss_func
from chem.constants import MODEL_FILE_NAME
from utils.data import MoleculeDataset, set_cache_graph, MoleculeDataLoader
from utils.util import split_data, save_smiles_splits, save_checkpoint, load_checkpoint, build_optimizer, \
    build_lr_scheduler
from utils.train import train, predict, evaluate, evaluate_predictions

from chem.utils import makedirs


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

        log.info(f'Building model {model_idx}')
        # 初始化模型
        model = MoleculeModel(model_conf, run_conf, global_conf)

        log.info(model)
        log.info(f'Number of parameters = {param_count_all(model):,}')

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
        best_score = -float('inf')
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

            scheduler.step()

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
