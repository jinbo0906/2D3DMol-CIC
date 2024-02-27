import os
import numpy as np
import pandas as pd

from utils import MoleculeDataLoader, load_checkpoint, split_data, predict


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
