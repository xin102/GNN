import numpy as np
import torch

epsilon = 1e-6


# shape:(batch_size,pred_len,tgt_size)
# obs:observations
# sim:simulation
def calc_nse(obs: np.ndarray, sim: np.ndarray) -> np.array:
    denominator = np.sum((obs - np.expand_dims(np.mean(obs, axis=1), axis=1)) ** 2, axis=1)
    numerator = np.sum((obs - sim) ** 2, axis=1)
    nse = 1 - numerator / denominator
    nse_mean = np.mean(nse)
    return nse_mean, nse[:, ]


def calc_rmse(obs: np.ndarray, sim: np.ndarray) -> np.array:
    mse = np.mean((obs - sim) ** 2, axis=1)
    rmse = np.sqrt(mse)
    rmse_mean = np.mean(rmse)
    return rmse_mean, rmse[:, ]


def calc_bias(obs: np.array, sim: np.array):
    numerator = np.sum(sim - obs, axis=1)
    denominator = np.sum(obs, axis=1)
    bias = numerator / denominator

    bias_mean = np.mean(bias)
    # Return mean bias, and bias of all locations, respectively
    return bias_mean, bias[:, ]


def calc_mse(obs: np.array, sim: np.array):
    mse = np.mean((obs - sim) ** 2, axis=1)
    mse_mean = np.mean(mse)
    return mse_mean, mse[:, ]


def calc_nse_torch(obs, sim):
    with torch.no_grad():
        denominator = torch.sum((obs - torch.mean(obs, dim=0)) ** 2, dim=0)
        numerator = torch.sum((sim - obs) ** 2, dim=0)
        nse = torch.tensor(1).to(sim.device) - numerator / denominator

        nse_mean = torch.mean(nse)
        # Return mean NSE, and NSE of all locations, respectively
        return nse_mean


def calc_atpe_2(obs, sim):
    numerator = np.sum(np.abs(sim - obs), axis=1)
    denominator = np.sum(obs, axis=1)
    atpe = numerator / denominator
    atpe_mean = np.mean(atpe)
    return atpe_mean, atpe[:, ]


def calc_log_nse(obs, sim):
    obs = np.maximum(epsilon,obs)
    sim = np.maximum(epsilon,sim)
    numerator = np.sum((np.log(obs) - np.log(sim)) ** 2, axis=1)
    denominator = np.sum((np.log(obs) - np.log(np.mean(obs, axis=1)[:, np.newaxis])) ** 2, axis=1)
    log_nse = 1 - numerator / denominator
    log_nse_mean = np.mean(log_nse)
    return log_nse_mean, log_nse[:, ]


def calc_kge(obs: np.array, sim: np.array):
    mean_obs = np.mean(obs, axis=1)
    mean_sim = np.mean(sim, axis=1)

    std_obs = np.std(obs, axis=1)
    std_sim = np.std(sim, axis=1)

    beta = mean_sim / mean_obs
    alpha = std_sim / std_obs
    numerator = np.mean(((obs - mean_obs[:,np.newaxis]) * (sim - mean_sim[:,np.newaxis])), axis=1)
    denominator = std_obs * std_sim
    gamma = numerator / denominator
    kge = 1 - np.sqrt((beta - 1) ** 2 + (alpha - 1) ** 2 + (gamma - 1) ** 2)

    kge_mean = np.mean(kge)
    # Return mean KEG, and KGE of all locations, respectively
    return kge_mean, kge[:,]