import numpy as np
import time
import gc
import json
from tqdm import tqdm
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
from models import get_model, model_saver, observe_loss_func


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

        # --------------
        # data
        # --------------
        self._data_init()

        # --------------
        # model
        # --------------
        self._model_init()

        # -------------
        # optimizer
        # -------------
        self._optim_init()

        # ------------
        # loss func
        # ------------
        self._loss_init()

        # ------------
        # others
        # -------------
        self.best_model_path = None
        self.model_save_queue = queue.Queue(maxsize=5)

    def _model_init(self):
        self.log.info("init model...")
        self.model = get_model(self.model_conf).to(self.device)
        self.log.info(self.model)
        if self.model_conf["load_checkpoint"]:
            self.model.load_state_dict(torch.load(os.path.join(self.project_root, self.model_conf["checkpoint_path"])))

    def _optim_init(self):
        self.log.info("init optimizer...")
        if self.run_conf["optim"] == "adam":
            self.optim = torch.optim.Adam(self.model.parameters(), **self.run_conf["optim_conf"])

        if self.run_conf["sch"] == "Identify":
            self.sch = None
        elif self.run_conf["sch"] == "step":
            step_sch_conf = self.run_conf["sch_step"]
            self.sch = torch.optim.lr_scheduler.StepLR(self.optim, step_size=int(
                self.run_conf["train_conf"]["epoch"] / step_sch_conf["stage"]),
                                                       gamma=step_sch_conf["gamma"])
        elif self.run_conf["sch"] == "cos":
            self.sch = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=int(
                self.run_conf["train_conf"]["epoch"]), eta_min=self.run_conf["optim_conf"]["lr"]/30)
        else:
            raise ValueError("sch: {} is not supported".format(self.run_conf["sch"]))

    def _data_init(self):
        self.log.info("init data...")
        pass

    def _loss_init(self):
        self.log.info("init loss...")
        pass

    def train_loop(self):
        pass

    def eval_loop(self, step):
        pass

    def test_loop(self):
        pass


@hydra.main(version_base=None, config_path="conf", config_name="Basic")
def train_setup(cfg):
    train_system = TrainingSystem(cfg)

    if cfg["run_conf"]["main_conf"]["run_mode"] == "train":
        train_system.train_loop()
        train_system.test_loop()
    elif cfg["run_conf"]["main_conf"]["run_mode"] == "test":
        train_system.test_loop()
    else:
        raise ValueError("run_mode: {} is not supported".format(cfg["run_conf"]["main_conf"]["run_mode"]))


if __name__ == "__main__":
    train_setup()
