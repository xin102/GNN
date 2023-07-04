import torch
import torch.nn as nn
from configs.project_config import ProjectConfig
from configs.pretrain_config import PretrainLearningConfig
from Layers import OneConv,GraphSAGE,LSTM_layer

batch_size = PretrainLearningConfig.batch_size
device = ProjectConfig.device


class GNNRNN(nn.Module):
    def __init__(self):
        super(GNNRNN, self).__init__()
        self.input_channels = 1
        self.out_channels = 2
        self.kernel_size = 3
        self.padding = 2

        self.input_fea = 8
        self.hidden1_fea = 10
        # self.hidden2_fea = 10
        self.output_fea = 8

        self.input_size = 8
        self.hidden_size = 64
        self.num_layers = 2
        self.out_size = 1

        self.dropout_rate = 0.1

        self.cnn = OneConv(self.input_channels, self.out_channels, self.kernel_size, self.padding).to(device)
        self.gnn = GraphSAGE(self.input_fea,self.hidden1_fea,self.output_fea).to(device)
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
        x_seq = x_seq.permute((1, 0, 3, 2))

        # CNN forward（runoff）
        y_data = torch.tensor([]).to(device)
        y_seq_runoff = y_seq_past.permute((1, 0, 3, 2))  # (basin_len,64,1,16)
        for i in range(basin_len):
            y_seq_single = y_seq_runoff[i].squeeze()  # (64,1,16)
            y_seq_single = y_seq_single.unsqueeze(dim=1)
            res_cnn = self.cnn(y_seq_single)
            y_data = torch.cat((y_data, res_cnn.unsqueeze(dim=0)), dim=0)
        x_data = torch.cat((x_seq, y_data), dim=2)

        g = g.to(device)
        x_gnn = x_data.permute((0,1,3,2))
        x_gnn = x_gnn.reshape(basin_len,-1,self.output_fea)
        res_gnn = self.gnn(g,x_gnn)
        res_gnn = res_gnn.reshape(basin_len, batch_len, seq_len, self.output_fea)
        x_data_lstm = res_gnn



        '''
        # gnn forward
        # x_data->(basin_len,batch_size,fea_len,seq_len)->(basin_len,64,7,16)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        x_data = torch.transpose(x_data, 2, 3)  # (basin_len,128,16,7)
        x_data = x_data.reshape(basin_len, -1, 8)
        g.ndata['feas'] = x_data.to('cpu')
        gnn_data = torch.tensor([]).to(device)
        dataloader = dgl.dataloading.NodeDataLoader(g=g, nids=g.nodes(), block_sampler=sampler)
        # sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 5])
        # collator = dgl.dataloading.NodeCollator(g, g.nodes(), sampler)
        # dataloader = torch.utils.data.DataLoader(collator.dataset, collate_fn=collator.collate,
        #                                          batch_size=1, shuffle=False, drop_last=False)
        for input_nodes, out_nodes, blocks in dataloader:
            input_feas = blocks[0].srcdata['feas'].to(device)  # (num_neigh,batch_size,7*16)
            # input_feas = torch.transpose(input_feas,0,1)    # (batch_size,num_neigh,7*16)
            # ans = torch.tensor([]).to(device)
            blocks = [block_.int().to(device) for block_ in blocks]
            res = self.gnn(blocks, input_feas)
            # res = self.gnn(blocks,res)
            gnn_data = torch.cat((gnn_data, res), dim=0)

        # gnn_data->(basin_len,64,7,16)
        gnn_res = gnn_data.reshape(basin_len, batch_len, seq_len, 8)
        x_data_lstm = gnn_res
        '''



        res_data = torch.tensor([]).to(device)
        for i in range(basin_len):
            # x_data_single->(64,16,6)
            x_data_single = x_data_lstm[i].squeeze(dim=0)
            # res_lstm->(64,16,1)
            res_lstm = self.lstm(x_data_single)
            res_data = torch.cat((res_data, res_lstm.unsqueeze(dim=0)), dim=0)
        # res_data->(10,64,16,1)
        return res_data
