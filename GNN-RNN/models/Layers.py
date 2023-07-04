import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv
from configs.project_config import ProjectConfig
from configs.pretrain_config import PretrainLearningConfig

batch_size = PretrainLearningConfig.batch_size
device = ProjectConfig.device


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# class ConvNet(nn.Module):
#     def __init__(self, input_channels, hidden_1, hidden_2, output_channels, kernel_size, padding):
#         super(ConvNet, self).__init__()
#         self.input_channels = input_channels
#         self.hidden1 = hidden_1
#         self.hidden2 = hidden_2
#         self.output_channels = output_channels
#         self.kernel_size = kernel_size
#         self.padding = padding
#         self.net = nn.Sequential(
#             nn.Conv1d(in_channels=self.input_channels, out_channels=self.hidden1, kernel_size=self.kernel_size),
#             nn.Conv1d(in_channels=self.hidden1, out_channels=self.hidden2, kernel_size=self.kernel_size),
#             nn.Conv1d(in_channels=self.hidden2, out_channels=self.output_channels, kernel_size=self.kernel_size),
#             nn.MaxPool1d(kernel_size=self.kernel_size, stride=1),
#             nn.Flatten()
#         )
#
#     def forward(self, x):
#         ans = self.net(x).unsqueeze(dim=2)
#         return ans


class OneConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding):
        super(OneConv, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv = nn.Sequential(
            nn.ReplicationPad1d((self.padding, 0)),
            nn.Conv1d(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.ReplicationPad1d((self.padding, 0)),
            nn.AvgPool1d(kernel_size=self.kernel_size, stride=1),
        )
        # self.linear = nn.Linear(12,16)

    def forward(self, x):
        conv_ans = self.conv(x)
        return conv_ans



class GraphSAGE(nn.Module):
    # def __init__(self, input_fea, hidden_fea, output_fea, activation=None):
    def __init__(self, input_fea, hidden1_fea, output_fea, activation=None):
        super(GraphSAGE, self).__init__()
        self.layer1 = SAGEConv(in_feats=input_fea, out_feats=hidden1_fea, aggregator_type='pool',activation=nn.functional.relu)
        self.layer2 = SAGEConv(in_feats=hidden1_fea, out_feats=output_fea, aggregator_type='pool',activation=nn.functional.relu)
        # self.layer = nn.ModuleList()
        # self.layer.append(SAGEConv(in_feats=input_fea, out_feats=hidden1_fea, aggregator_type='pool',
        #                            activation=nn.functional.relu))
        # self.layer.append(SAGEConv(in_feats=hidden1_fea, out_feats=output_fea, aggregator_type='pool',
        #                            activation=nn.functional.relu))

    def forward(self, g, h):
        # node feature
        # for i, (layer, block) in enumerate(zip(self.layer, blocks)):
        #     h = layer(block, h)
        h = self.layer1(g,h)
        h = self.layer2(g,h)
        return h


class LSTM_layer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_layer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bias=True, batch_first=True)
        self.linear1 = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        # h_0,c_0 initialization
        h_0 = torch.rand(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.rand(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # output = (batch_size,seq_len,num_direction*hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        output = self.linear1(output)
        # output = self.linear2(output)
        output = output[:, -1, :]
        return output