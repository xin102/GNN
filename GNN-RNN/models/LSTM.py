import torch
import torch.nn as nn
from configs.project_config import ProjectConfig
from configs.pretrain_config import PretrainLearningConfig
from Layers import LSTM_layer

batch_size = PretrainLearningConfig.batch_size
device = ProjectConfig.device


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.input_size = 8
        self.hidden_size = 64
        self.num_layers = 2
        self.out_size = 1

        self.dropout_rate = 0.1
        self.lstm = LSTM_layer(self.input_size, self.hidden_size, self.num_layers, self.out_size).to(device)

    def forward(self, x_seq, y_seq_past, g):
        y_seq_past_mean = torch.mean(y_seq_past, dim=2)
        # y_seq_past_mean = torch.mean(y, dim=2)
        y_seq_past = torch.cat((y_seq_past, y_seq_past_mean.unsqueeze(dim=2)), dim=2)
        x_seq_no_runoff = x_seq.permute((1, 0, 3, 2))
        x_seq = torch.cat((x_seq, y_seq_past), dim=3)
        batch_len = x_seq.shape[0]
        basin_len = x_seq.shape[1]
        seq_len = x_seq.shape[2]
        feature_len = x_seq.shape[3]
        # reshape-->(10,64,6,16)
        x_seq = x_seq.permute((1, 0, 3, 2))



        # LSTM forward
        x_data_lstm = x_seq.permute((0, 1, 3, 2))
        res_data = torch.tensor([]).to(device)
        for i in range(basin_len):
            # x_data_single->(64,16,6)
            x_data_single = x_data_lstm[i].squeeze(dim=0)
            # res_lstm->(64,16,1)
            res_lstm = self.lstm(x_data_single)
            res_data = torch.cat((res_data, res_lstm.unsqueeze(dim=0)), dim=0)
        return res_data
