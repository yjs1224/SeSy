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
        self.size = 128
        self.seq_len = 128
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.dropout_prob)
        # self.classifier = nn.Sequential(nn.Linear(self.seq_len*self.hidden_size, self.size), nn.Linear(self.size,self.num_labels))
        # self.classifier = nn.Sequential(nn.Linear(self.hidden_size, self.size),
        #                                 nn.Linear(self.size, self.num_labels))
        self.classifier = nn.Linear(self.hidden_size,self.num_labels)
        self.init_weights()

    def forward(self,  input_ids, attention_mask,token_type_ids,
                dependency_matrix=None, labels=None, position_ids=None, head_mask=None):
        # input_ids     : [batch_size, sequence_length]
        # attention_mask: [batch_size, sequence_length]
        # token_type_ids: [batch_size, sequence_length]
        input_ids_len = torch.sum(input_ids != 0, dim=-1).float()

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        clf_input = outputs[1]
        clf_input = self.dropout(clf_input)
        logits = self.classifier(clf_input)
        # logits = F.softmax(logits)
        return logits