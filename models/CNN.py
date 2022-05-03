import torch
from torch import nn
class cnn(nn.Module):
    def __init__(self, input_length, vocab_size, output_size, hidden_size, kernel_sizes, num_filters, 
                 dropout_rate=0.1, pool_size=2, num_mlp_layers=2, num_cnn_layers=1, stride=1, sigmoid=False):
        super(cnn, self).__init__()

        self.convs = nn.ModuleList([])
        
        self.cnn_output_size = 0
        for i in range(len(kernel_sizes)):
            o_size = input_length
            conv = nn.Sequential(
                nn.Conv1d(in_channels=vocab_size,
                          out_channels=num_filters,
                          kernel_size=kernel_sizes[i],
                          stride=stride),
                nn.ReLU()
            )
            o_size = int((o_size - kernel_sizes[i])/stride) + 1
            for j in range(num_cnn_layers-1):
                conv.add_module(f"{i}-pool{j}", nn.MaxPool1d(kernel_size=pool_size))
                conv.add_module(f"{i}-conv{j}", nn.Conv1d(in_channels=num_filters,
                                                      out_channels=num_filters,
                                                      kernel_size=kernel_sizes[i], stride=stride))
                conv.add_module(f"{i}-relu{j}",nn.ReLU())
                o_size = int((o_size - pool_size)/pool_size) + 1
                o_size = int((o_size - kernel_sizes[i])/stride) + 1
            self.cnn_output_size += o_size * num_filters
            self.convs.append(conv)
        self.mlps = nn.Sequential()
        if num_mlp_layers == 1:
            self.mlps.add_module("mlp", nn.Linear(self.cnn_output_size, output_size))
        else:
            for i in range(num_mlp_layers-1):
                if i == 0:
                    self.mlps.add_module(f"mlp{i}", nn.Linear(self.cnn_output_size, hidden_size))
                else:
                    self.mlps.add_module(f"mlp{i}", nn.Linear(hidden_size, hidden_size))
            self.mlps.add_module(f"mlp{num_mlp_layers}", nn.Linear(hidden_size, output_size))
        if sigmoid:
            self.mlps.add_module("sigmoid", nn.Sigmoid())
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        x = x.permute([0,2,1])
        # CNN Layers
        cnn_results = []
        # Convolutional Layers
        for i in range(len(self.convs)):
            # Convolution
            cnn = self.convs[i](x)
            cnn = cnn.view(cnn.size(0), -1)
            cnn_results.append(cnn)
        # Concatenate
        x = torch.cat(cnn_results, 1)
        # Dropout
        x = self.dropout(x)
        # MLP Layers
        y = self.mlps(x)
        return y
