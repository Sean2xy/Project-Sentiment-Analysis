import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig

class BertBaseUncased(torch.nn.Module):
    def __init__(self):
        super(BertBaseUncased, self).__init__()
        config = BertConfig.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            hidden_dropout_prob=0.1,
            output_attentions = False,
            output_hidden_states = False
        )
        self.bert =BertForSequenceClassification.from_pretrained('bert-base-uncased',config=config)

    def forward(self,ids,mask):
        output = self.bert(
            ids,
            attention_mask = mask
        )
        return output


