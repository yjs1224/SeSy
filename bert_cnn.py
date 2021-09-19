import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel

from transformers import BertPreTrainedModel, BertModel


class TCModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.dropout_prob = config.hidden_dropout_prob
        self.num_labels = 2  # ["neural", "happy", "angry", "sad", "fear", "surprise"]
        self.bidirectional = True
        # self.size = 128
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.filter_num = 128
        self.filter_size = [3,4,5]
        self.cnns = nn.ModuleList([nn.Conv1d(self.hidden_size, self.filter_num, ker) for ker in self.filter_size])
        # self.cnns = nn.Conv1d(self.hidden_size, self.filter_num, kernel_size=self.filter_size)
        self.relu = nn.ReLU()
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(self.filter_num * len(self.filter_size), self.num_labels)
        self.init_weights()

    def forward(self, input_ids,attention_mask,token_type_ids,
                dependency_matrix=None, labels=None, position_ids=None, head_mask=None):
        # input_ids     : [batch_size, sequence_length]
        # attention_mask: [batch_size, sequence_length]
        # token_type_ids: [batch_size, sequence_length]
        input_ids_len = torch.sum(input_ids != 0, dim=-1).float()

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        x = outputs[0]
        x = x.permute(0,2,1)
        clf_input = []
        for cnn in self.cnns:
            x_tmp = cnn(x)
            x_tmp = self.max_pool(x_tmp)
            x_tmp = self.relu(x_tmp)
            clf_input.append(x_tmp.squeeze(2))

        clf_input = torch.cat(clf_input, 1)
        clf_input = self.dropout(clf_input)
        logits = self.classifier(clf_input)
        # logits = F.softmax(logits)

        return logits
