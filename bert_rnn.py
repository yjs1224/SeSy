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

        self.size = 128
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.dropout_prob)


        self.rnn = nn.LSTM(self.hidden_size, self.size, num_layers=2, dropout=self.dropout_prob, bidirectional=True)
        self.classifier = nn.Linear(self.size*2*2, self.num_labels)


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

        clf_input = x.permute(1,0,2)
        rnn_out_last_layer, rnn_out_last_time = self.rnn(clf_input)
        rnn_out_last_time_hidden = rnn_out_last_time[0]
        # clf_input = rnn_out.view(x.shape[1],x.shape[0],2,self.size).permute(1,0,2,3)
        clf_input = rnn_out_last_time_hidden.view(2, 2, x.shape[0], self.size).permute(1,0,2,3).permute(2,1,0,3)
        clf_input = clf_input.reshape((x.shape[0], 2*2*self.size))  # batch, seqlen*direction_num, feature_num

        logits = self.classifier(clf_input)
        # logits = F.softmax(logits)

        return logits
