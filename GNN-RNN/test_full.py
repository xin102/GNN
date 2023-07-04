import numpy as np
from matplotlib import pyplot as plt

from utils.metric import calc_nse, calc_mse, calc_log_nse, calc_atpe_2, calc_rmse, calc_bias, calc_kge
from utils.tools import count_parameters, saving_obs_pred
from model_eval import eval_obs_preds


def test_full(test_model, test_loader, device, saving_root, only_metrics: bool,  token, date_index=None):
    data_file = 'file_path'
    print(f"Parameters count:{count_parameters(test_model)}")
    log_file = saving_root / f"log_test.csv"
    obs, pred = eval_obs_preds(test_model, test_loader, device)
    dim_0, dim_1 = obs.shape[0], obs.shape[1]
    obs = obs.numpy()
    pred = pred.numpy()

    _, mses_test = calc_mse(obs, pred)
    test_mse_mean = mses_test.mean()
    obs_rescaled = test_loader.dataset.local_rescale(obs, variable='output')
    pred_rescaled = test_loader.dataset.local_rescale(pred, variable='output')
    obs_rescaled = np.maximum(0, obs_rescaled)
    pred_rescaled = np.maximum(0, pred_rescaled)
    _, nses_test = calc_nse(obs_rescaled, pred_rescaled)
    # set 0
    nses_test = np.maximum(nses_test, 0)
    # print("-------------------nse_test----------------------")
    # print(nses_test)
    # print(f"median:{np.median(nses_test)}")
    test_nse_mean = nses_test.mean()
    # print(f"Testing mean mse: {test_mse_mean}, mean nse:{test_nse_mean}")
    # print("-------------------log_nse----------------------")
    log_nse_mean, log_nse = calc_log_nse(obs_rescaled, pred_rescaled)
    # print(log_nse)
    # print("-------------------atpe_2%----------------------")
    atpe_mean, atpe = calc_atpe_2(obs_rescaled, pred_rescaled)
    # print(atpe)
    # print("-------------------rmse----------------------")
    rmse_mean, rmse = calc_rmse(obs_rescaled, pred_rescaled)
    # print(rmse)
    # print("-------------------bias----------------------")
    bias_mean, bias = calc_bias(obs_rescaled, pred_rescaled)
    # print(bias)
    # print("-------------------kge----------------------")
    kge_mean, kge = calc_kge(obs_rescaled, pred_rescaled)
    # print(kge)

    print("--------------------median--------------------")
    print(f"median mse:{np.median(mses_test)}")
    print(f"median nse:{np.median(nses_test)}")
    print(f"median log_nse:{np.median(log_nse)}")
    print(f"median atpe_2%:{np.median(atpe)}")
    print(f"median rmse:{np.median(rmse)}")
    print(f"median bias:{np.median(bias)}")
    print(f"median kge:{np.median(kge)}")

    print("--------------------mean--------------------")
    print(f"mean mse: {test_mse_mean}")
    print(f"mean nse:{test_nse_mean}")
    print(f"mean log_nse:{log_nse_mean}")
    print(f"mean atpe_2%:{atpe_mean}")
    print(f"mean rmse:{rmse_mean}")
    print(f"mean bias:{bias_mean}")
    print(f"mean kge:{kge_mean}")

    # with open(data_file, "a+") as file:
    #     file.write(
    #         f"{token[0]},{token[1]},{token[2]},{np.median(mses_test)},{np.median(nses_test)},{np.median(log_nse)},{np.median(atpe)},{np.median(rmse)},{np.median(bias)},{np.median(kge)},"
    #         f"{test_mse_mean},{test_nse_mean},{log_nse_mean},{atpe_mean},{rmse_mean},{bias_mean},{kge_mean}\n")

