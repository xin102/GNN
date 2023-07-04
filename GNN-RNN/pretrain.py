import torch
import os
import numpy as np
import json
from shutil import copytree
from torch.utils.data import DataLoader

from models.CNN import GNNRNN, GraphSAGE, OneConv

from configs.project_config import ProjectConfig
from configs.dataset_config import DataShapeConfig
from configs.pretrain_config import PretrainConfig
from lr_strategies import WarmUp
from data.dataset import DatasetFactory
from train_full import train_full
from utils.tools import SeedMethods

project_root = ProjectConfig.project_root
device = ProjectConfig.device
num_workers = ProjectConfig.num_workers

past_len = DataShapeConfig.past_len
pred_len = DataShapeConfig.pred_len
src_len = DataShapeConfig.src_len
tgt_len = DataShapeConfig.tgt_len
src_size = DataShapeConfig.src_size
tgt_size = DataShapeConfig.tgt_size
use_future_fea = DataShapeConfig.use_feature
use_static = DataShapeConfig.use_static

pre_train_config = PretrainConfig.pre_train_config
pre_val_config = PretrainConfig.pre_val_config
pre_test_config = PretrainConfig.pre_test_config
loss_func = PretrainConfig.loss_func
n_epochs = PretrainConfig.n_epochs
batch_size = PretrainConfig.batch_size
learning_rate = PretrainConfig.learning_rate
scheduler_paras = PretrainConfig.scheduler_paras

seed = PretrainConfig.seed
saving_message = PretrainConfig.saving_message
saving_root = PretrainConfig.saving_root

if __name__ == '__main__':
    print("pid:", os.getpid())
    SeedMethods.seed_torch(seed=seed)

    # save training results
    saving_root.mkdir(parents=True, exist_ok=True)
    configs_path = project_root / "configs"
    configs_saving = saving_root / "configs"
    if configs_saving.exists():
        raise RuntimeError("config files already exists!")
    copytree(configs_path, configs_saving)


    dataset = DatasetFactory.get_dataset_type(use_future_fea, use_static)
    ds_train = dataset.get_instance(past_len, pred_len, "train", specific_cfg=pre_train_config)
    train_loader = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              drop_last=True)

    train_x_mean, train_y_mean = ds_train.get_means()
    train_x_std, train_y_std = ds_train.get_stds()
    y_stds_dict = ds_train.y_stds_dict

    train_mean = np.concatenate((train_x_mean, train_y_mean), axis=0)
    train_std = np.concatenate((train_x_std, train_y_std), axis=0)
    np.savetxt(saving_root / "train_means.csv", train_mean)
    np.savetxt(saving_root / "train_stds.csv", train_std)
    with open(saving_root / "y_stds_dict.json", "wt") as f:
        json.dump(y_stds_dict, f)

    ds_val = dataset.get_instance(past_len, pred_len, "val", specific_cfg=pre_val_config, x_mean=train_x_mean,
                                  y_mean=train_y_mean, x_std=train_x_std, y_std=train_y_std, y_stds_dic=y_stds_dict)
    val_loader = DataLoader(dataset=ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            drop_last=True)

    ds_test = dataset.get_instance(past_len, pred_len, "test", specific_cfg=pre_test_config, x_mean=train_x_mean,
                                   y_mean=train_y_mean, x_std=train_x_std, y_std=train_y_std, y_stds_dic=y_stds_dict)
    test_loader = DataLoader(dataset=ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             drop_last=True)

    model = GNNRNN().to(device)
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = WarmUp(**scheduler_paras).get_scheduler(optimizer)
    loss_func = loss_func().to(device)

    # test data
    # cnt = 1
    # for x_seq,y_seq_past,y_seq_future in test_loader:
    #     print(cnt)
    #     cnt += 1



    # training
    train_full(model, train_loader, val_loader, optimizer, scheduler, loss_func, n_epochs, device, saving_root)
