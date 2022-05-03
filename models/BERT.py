import torch
from torch import nn
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, BertModel
device = torch.device("cuda")

class bert(nn.Module):
    def __init__(self, config):
        super(bert, self).__init__()
        self.encoder = BertForSequenceClassification.from_pretrained(**config).to(device)

    def forward(self, text, label=None):
        output = self.encoder(text, labels=label)[:2]
        return output