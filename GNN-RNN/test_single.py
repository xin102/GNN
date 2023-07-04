import json

import torch
import numpy as np
from torch.utils.data import DataLoader

from models.CNN import GNNRNN
from configs.project_config import ProjectConfig
from utils.tools import SeedMethods
from configs.pretrain_config import DataShapeConfig
import os
from data.dataset import DatasetFactory
from test_full import test_full
from configs.fine_tune_config import FineTuneConfig
from configs.pretrain_config import PretrainConfig

device = ProjectConfig.device
saving_root = PretrainConfig.saving_root
model = GNNRNN().to(device)
seed = PretrainConfig.seed
past_len = DataShapeConfig.past_len
pred_len = DataShapeConfig.pred_len
batch_size = PretrainConfig.batch_size
num_works = ProjectConfig.num_workers
exps_config = FineTuneConfig.exps_config
specific_cfg = PretrainConfig.pre_test_config

use_future_fea = DataShapeConfig.use_feature
use_static = DataShapeConfig.use_static

if __name__ == '__main__':
    print("pid:", os.getpid())
    SeedMethods.seed_torch(seed=seed)
    Model = GNNRNN()

    DS = DatasetFactory.get_dataset_type(use_future_fea, use_static)
    # load mean and standard deviation
    train_means = np.loadtxt(saving_root / "train_means.csv", dtype="float32")
    train_stds = np.loadtxt(saving_root / "train_stds.csv", dtype="float32")
    train_x_mean = train_means[:-1]
    train_y_mean = train_means[-1]
    train_x_std = train_stds[:-1]
    train_y_std = train_stds[-1]
    with open(saving_root / "y_stds_dict.json", "rt") as f:
        y_stds_dict = json.load(f)

    SeedMethods.seed_torch(seed=seed)
    root_now = saving_root / "pretrain_test_single" / "basin1"
    # load test dataset
    ds_test = DS.get_instance(past_len, pred_len, "test", specific_cfg, x_mean=train_x_mean, y_mean=train_y_mean,
                              x_std=train_x_std, y_std=train_y_std, y_stds_dic=y_stds_dict)
    test_loader = DataLoader(ds_test,batch_size,shuffle=False,num_workers=num_works,drop_last=True)
    net_type = "CNN-GNN-LSTM"
    test_type_list = ["new","max_nse","min_mse"]
    area_no = "3"
    for test_type in test_type_list:
        best_path = list(saving_root.glob(f"{test_type}*.pkl"))

        assert (len(best_path) == 1)
        best_path = best_path[0]
        best_model = Model.to(device)
        # load model
        best_model.load_state_dict(torch.load(best_path, map_location=device))
        token = [net_type,area_no,test_type]
        test_full(best_model,test_loader,device,root_now,only_metrics=False,token=token)

