import torch
from configs.dataset_config import DataSetConfig
from configs.dataset_config import DataShapeConfig
from configs.project_config import ProjectConfig
from torch import nn


# loss function
class LogCosh(nn.Module):
    def __init__(self):
        super(LogCosh, self).__init__()

    def forward(self, pred, true):
        loss = torch.log(torch.cosh(pred - true))
        return torch.mean(loss)


class PretrainLearningConfig:
    loss_type = 'log_cosh'
    # set loss function
    loss_func = LogCosh
    scale_factor = 1
    n_epochs = 200
    batch_size = 64
    learning_rate = 0.001
    # warm-up
    scheduler_paras = {"warm_up_epochs": n_epochs * 0.25, "decay_rate": 0.99}
    learning_config_info = f"{loss_type}_n{n_epochs}_bs{batch_size}_lr{learning_rate}"


class PretrainConfig(PretrainLearningConfig):
    seed = None
    model_info = 'GNN-RNN'

    pre_train_id = f"{DataSetConfig.forcing_type}{DataSetConfig.basin_mark}@{DataShapeConfig.data_shape_info}" \
                   f"@{DataSetConfig.train_start.date()}~{DataSetConfig.train_end.date()}"
    pre_val_id = f"{DataSetConfig.forcing_type}{DataSetConfig.basin_mark}@{DataShapeConfig.data_shape_info}" \
                 f"@{DataSetConfig.valid_start.date()}~{DataSetConfig.valid_end.date()}"
    pre_test_id = f"{DataSetConfig.forcing_type}{DataSetConfig.basin_mark}@{DataShapeConfig.data_shape_info}" \
                  f"@{DataSetConfig.test_start.date()}~{DataSetConfig.test_end.date()}"

    # save path
    final_train_data_path = ProjectConfig.final_data_root / f"{pre_train_id}_serialized_train.pkl"
    final_val_data_path = ProjectConfig.final_data_root / f"{pre_val_id}_serialized_val.pkl"
    final_test_data_path = ProjectConfig.final_data_root / f"{pre_test_id}_serialized_test.pkl"

    # training configuration information
    pre_train_config = {
        "camels_root": DataSetConfig.camels_root,
        "basin_list": DataSetConfig.global_basin_list,
        "forcing_type": DataSetConfig.forcing_type,
        "start_date": DataSetConfig.train_start,
        "end_date": DataSetConfig.train_end,
        "final_data_path": final_train_data_path
    }
    pre_val_config = {
        "camels_root": DataSetConfig.camels_root,
        "basin_list": DataSetConfig.global_basin_list,
        "forcing_type": DataSetConfig.forcing_type,
        "start_date": DataSetConfig.valid_start,
        "end_date": DataSetConfig.valid_end,
        "final_data_path": final_val_data_path
    }
    pre_test_config = {
        "camels_root": DataSetConfig.camels_root,
        "basin_list": DataSetConfig.global_basin_list,
        "forcing_type": DataSetConfig.forcing_type,
        "start_date": DataSetConfig.test_start,
        "end_date": DataSetConfig.test_end,
        "final_data_path": final_test_data_path
    }

    saving_message = f"{model_info}@{DataShapeConfig.data_shape_info}@{DataSetConfig.dataset_info}@{PretrainLearningConfig.learning_config_info}@seed{seed}"
    saving_root = ProjectConfig.run_root / saving_message
