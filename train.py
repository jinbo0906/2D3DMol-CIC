from collections import defaultdict
import numpy as np
import csv

from chemprop.utils import makedirs
import os
import hydra
from hydra.utils import get_original_cwd
import queue
import logging
import random
import pandas as pd
from omegaconf import OmegaConf

import torch
from torch.utils.tensorboard import SummaryWriter
from utils import get_data, run_training, run_testing, extract_model_pt_paths

from chemprop.constants import TEST_SCORES_FILE_NAME


def global_seed(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class TrainingSystem:

    def __init__(self, conf):

        self.log = logging.getLogger("Train")

        self.global_conf = conf["global_conf"]
        self.model_conf = conf["model_conf"]
        self.run_conf = conf["run_conf"]
        self.data_conf = conf["data_conf"]
        self.test_conf = conf["test_conf"]

        self.tensorboard_writer = SummaryWriter("./tensorboard_log")
        self.log.info(OmegaConf.to_yaml(conf))

        # --------------
        # global
        # --------------
        self.log.info("init global conf...")

        global_seed(self.global_conf["seed"])  # seed
        self.device = torch.device(self.global_conf["device"])  # device
        self.log.info(f"Device: {self.device}")
        self.project_root = get_original_cwd()  # root
        self.model_save_queue = queue.Queue()

        # --------------
        # data
        # --------------
        self._data_init()

    def _data_init(self):
        self.log.info("init data...")

        # observe data
        observe_data_conf = self.data_conf["observe_data"]
        smi_train_csv_file_path = os.path.join(self.project_root, observe_data_conf['smi_train_csv_file'])
        geom_train_csv_file_path = os.path.join(self.project_root, observe_data_conf['geom_train_npz_file'])
        self.data = get_data(
            smi_path=smi_train_csv_file_path,
            geom_path=geom_train_csv_file_path,
            data_args=observe_data_conf
        )

    def train_loop(self):
        self.log.info("Run training on different random seeds for each fold")
        all_scores = defaultdict(list)
        for fold_num in range(self.run_conf["train_conf"]["num_folds"]):
            self.log.info(f'Fold {fold_num}')
            curr_seed = self.global_conf["seed"] + fold_num
            save_dir = os.path.join('./', f'fold_{fold_num}')
            makedirs(save_dir)
            self.data.reset_features_and_targets()
            model_scores = run_training(self.run_conf, self.model_conf, self.data_conf, self.global_conf,
                                        self.data, curr_seed, self.log, save_dir)

            for metric, scores in model_scores.items():
                all_scores[metric].append(scores)
        all_scores = dict(all_scores)
        # Convert scores to numpy arrays
        for metric, scores in all_scores.items():
            all_scores[metric] = np.array(scores)

        # Report results
        self.log.info(f'{self.run_conf["train_conf"]["num_folds"]}-fold cross validation')
        # Report scores for each fold
        for fold_num in range(self.run_conf["train_conf"]["num_folds"]):
            for metric, scores in all_scores.items():
                self.log.info(f'\tSeed {self.global_conf["seed"] + fold_num} ==> test {metric} = {np.nanmean(scores[fold_num]):.6f}')

        # Report scores across folds
        for metric, scores in all_scores.items():
            avg_scores = np.nanmean(scores, axis=1)  # average score for each model across tasks
            mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
            self.log.info(f'Overall test {metric} = {mean_score:.6f} +/- {std_score:.6f}')

        # Save scores
        with open(os.path.join(self.run_conf['save_folder'], TEST_SCORES_FILE_NAME), 'w') as f:
            writer = csv.writer(f)

            header = ['Task']
            for metric in self.run_conf['train_conf']['metrics']:
                header += [f'Mean {metric}', f'Standard deviation {metric}'] + \
                          [f'Fold {i} {metric}' for i in range(self.run_conf["train_conf"]["num_folds"])]
            writer.writerow(header)

            for task_num, task_name in enumerate(self.data_conf['observe_data']['target_columns']):
                row = [task_name]
                for metric, scores in all_scores.items():
                    task_scores = scores[:, task_num]
                    mean, std = np.nanmean(task_scores), np.nanstd(task_scores)
                    row += [mean, std] + task_scores.tolist()
                writer.writerow(row)

        # Determine mean and std score of main metric
        avg_scores = np.nanmean(all_scores['rmse'], axis=1)
        mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)

        # Optionally merge and save test preds
        all_preds = pd.concat([pd.read_csv(os.path.join(self.run_conf['save_folder'], f'fold_{fold_num}', 'test_preds.csv'))
                               for fold_num in range(self.run_conf["train_conf"]["num_folds"])])
        all_preds.to_csv(os.path.join(self.run_conf['save_folder'], 'test_preds.csv'), index=False)

        return mean_score, std_score

    def test_loop(self):

        self.log.info("test start...")

        # 加载数据
        smi_test_csv_file_path = os.path.join(self.project_root, self.test_conf['test_path'])
        geom_test_csv_file_path = os.path.join(self.project_root, self.test_conf['geom_prepath'])
        test_data = get_data(
            smi_path=smi_test_csv_file_path,
            geom_path=geom_test_csv_file_path,
            data_args=self.data_conf
        )

        # 加载model path
        model_paths = extract_model_pt_paths(self.test_conf['checkpoint_dir'])
        preds_csv_paths = []
        for i, model_path in enumerate(model_paths):
            save_pre_dir = os.path.join(self.test_conf['preds_path'], f'model_{i}')
            makedirs(save_pre_dir)
            preds_save_path = run_testing(self.log, model_path, test_data, self.run_conf, self.model_conf, self.global_conf, save_pre_dir)
            preds_csv_paths.append(preds_save_path)
        # 计算预测平均值
        # 初始化一个空的DataFrame来存储分子名和平均预测结果
        averaged_preds = pd.DataFrame(columns=["smiles", "average_preds"])
        # 遍历所有preds目录
        for i, csv_path in enumerate(preds_csv_paths):
            # 读取csv文件
            df = pd.read_csv(csv_path)

            # 如果是第一次读取，直接将第一列（假设为"Molecule"）和第二列添加到averaged_preds中
            if i == 0:
                averaged_preds["smiles"] = df["smiles"]
                averaged_preds["average_preds"] = df.iloc[:, 1]  # 假设第二列就是预测结果
            else:
                # 否则，只累加预测结果并除以已处理的模型数量
                averaged_preds["average_preds"] += df.iloc[:, 1]

        # 计算平均值并填充到Average_Prediction列
        averaged_preds["average_preds"] /= len(preds_csv_paths)

        averaged_preds.to_csv(os.path.join(self.test_conf['preds_path'], 'preds_mean.csv'), index=False)


@hydra.main(version_base=None, config_path="conf", config_name="Basic")
def train_setup(cfg):
    train_system = TrainingSystem(cfg)

    if cfg["run_conf"]["main_conf"]["run_mode"] == "train":
        train_system.train_loop()
    elif cfg["run_conf"]["main_conf"]["run_mode"] == "test":
        train_system.test_loop()
    else:
        raise ValueError("run_mode: {} is not supported".format(cfg["run_conf"]["main_conf"]["run_mode"]))


if __name__ == "__main__":
    train_setup()
