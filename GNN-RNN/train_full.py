import torch
import time
import os
from model_train import train_epoch
import matplotlib.pyplot as plt
from utils.tools import count_parameters, BoardWriter
from model_eval import eval_model


class BestModelLog:
    def __init__(self, init_model, saving_root, metric_name, high_better: bool,threshold):
        self.high_better = high_better
        self.saving_root = saving_root
        self.metric_name = metric_name
        worst = float("-inf") if high_better else float("inf")
        self.best_epoch = -1
        self.best_value = worst
        self.best_model_path = self.saving_root / f"{threshold}_{self.metric_name}_{self.best_epoch}_{self.best_value}.pkl"
        torch.save(init_model.state_dict(), self.best_model_path)

    # update file
    def update(self, model, new_value, epoch,threshold):
        if ((self.high_better is True) and (new_value > self.best_value)) or (
                (self.high_better is False) and (new_value < self.best_value)):
            # delete old file
            os.remove(self.best_model_path)
            # update model
            self.best_value = new_value
            self.best_epoch = epoch
            self.best_model_path = self.saving_root / f"{threshold}_{self.metric_name}_{self.best_epoch}_{self.best_value}.pkl"
            torch.save(model.state_dict(), self.best_model_path)


def train_full(model, train_loader, val_loader, optimizer, scheduler, loss_func, n_epochs, device, saving_root,using_board=False):
    # record evaluation indicators
    log_file = saving_root / f"log_train.csv"
    with open(log_file, "wt") as file:
        file.write(f"Parameters count:{count_parameters(model)}\n")
        file.write("epoch,train_loss,val_mse,val_nse\n")
    if using_board:
        tb_root = saving_root / "tb_log"
        writer = BoardWriter(tb_root)
    else:
        writer = None

    # update model
    mse_log = BestModelLog(model, saving_root, "min_mse", high_better=False)
    nse_log = BestModelLog(model, saving_root, "max_nse", high_better=True)
    new_log = BestModelLog(model, saving_root, "new", high_better=True)
    x_num = []
    y_num = []
    y_num_val = []
    loss_num = []
    for i in range(n_epochs):
        # print(f"Training process:{i}/{n_epochs}")
        train_loss_iterated = train_epoch(model, train_loader, optimizer, scheduler, loss_func, device,i)
        x_num.append(i)
        y_num.append(train_loss_iterated)
        # loss_num.append(loss_list)
        print(f"Training process:{i}/{n_epochs} : loss_mean:{train_loss_iterated}")

        mse_val, nse_val, val_loss_iteration = eval_model(model, val_loader, device, loss_func)
        y_num_val.append(val_loss_iteration)

        # 使用tensorboard
        if writer is not None:
            writer.writer_board("train_loss", metric_value=train_loss_iterated, epoch=i)
            writer.writer_board("val_mse", metric_value=mse_val, epoch=i)
            writer.writer_board("val_nse", metric_value=nse_val, epoch=i)
        # write data to log_file
        with open(log_file, "at") as file:
            file.write(f"{i},{train_loss_iterated},{mse_val},{nse_val}\n")

        mse_log.update(model, mse_val, i)
        nse_log.update(model, nse_val, i)
        new_log.update(model, i, i)
