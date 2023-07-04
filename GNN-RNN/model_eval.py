import torch
from torch import nn
from utils.metric import calc_nse_torch, calc_mse
import matplotlib.pyplot as plt
from build_graph import data_graph


def eval_model(model, data_loader, device,loss_func,threshold):
    '''
    :param model: training model
    :param data_loader: val data
    :param device: cuda0
    :return: mse,nse
    '''
    model.eval()
    mse = nn.MSELoss()
    cnt = 0
    mse_mean, nse_mean,loss_mean = 0, 0, 0
    g = data_graph(threshold)
    # g = g.to(device)
    with torch.no_grad():
        for x_seq, y_seq_past, y_seq_future in data_loader:
            x_seq, y_seq_past, y_seq_future = x_seq.to(device), y_seq_past.to(device), y_seq_future.to(device)
            # output->(10,64,1)
            output = model(x_seq, y_seq_past,g)
            # length = output.shape[1]
            y_data = y_seq_future.squeeze(dim=3)
            # y_data->(10,64,1)
            y_data = y_data.permute((1, 0, 2))
            loss = loss_func(output,y_data)
            dim_0 = output.shape[0] * output.shape[1]
            output = output.reshape(dim_0,-1)
            y_data = y_data.reshape(dim_0,-1)
            mse_value = mse(y_data, output).item()
            nse_value = calc_nse_torch(y_data,output)
            cnt += 1
            mse_mean = mse_mean + (mse_value - mse_mean) / cnt  # Welford’s method
            nse_mean = nse_mean + (nse_value - nse_mean) / cnt  # Welford’s method
            loss_mean = loss_mean + (loss.item() - loss_mean) / cnt
        return mse_mean, nse_mean,loss_mean


def eval_obs_preds(model, data_loader, device):
    model.eval()
    threshold = None
    g = data_graph(threshold)
    # g = g.to(device)
    obs = torch.tensor([]).to(device)
    preds = torch.tensor([]).to(device)
    id = 0
    with torch.no_grad():
        for x_seq, y_seq_past, y_seq_future in data_loader:
            x_seq,y_seq_past,y_seq_future = x_seq.to(device),y_seq_past.to(device),y_seq_future.to(device)
            # print(f"---------{id}----------")
            id += 1
            output = model(x_seq, y_seq_past,g)
            # output = output.to(device)
            output = torch.squeeze(output)
            preds = torch.cat((preds, output), dim=1)
            # #print(preds.shape)
            # length = output.shape[1]
            y_data = y_seq_future.squeeze(dim=3).to(device)
            # y_data->(10,64,1)
            y_data = y_data.permute((1, 0, 2))

            y_data = torch.squeeze(y_data)
            obs = torch.cat((obs, y_data), dim=1)
            # print(obs.shape)
            # obs.append(y_data)
        obs = obs.to("cpu")
        preds = preds.to("cpu")

        return obs,preds


