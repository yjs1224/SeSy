import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel


class GATLayer(nn.Module):
    def __init__(self,in_features,out_features,dropout,alpha,concat=True,get_att=False):
        super(GATLayer,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.get_att = get_att
        self.dropout = dropout
        self.W = nn.Parameter(torch.zeros(size=(in_features,out_features)).cuda())
        nn.init.xavier_uniform_(self.W.data,gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features,1)).cuda())
        nn.init.xavier_uniform_(self.a.data,gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self,input,adj):
        h = torch.matmul(input,self.W)
        a_input = self._prepare_attentional_mechanism_input(h)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        B, M, E = Wh.shape  # (batch_zize, number_nodes, out_features)
        Wh_repeated_in_chunks = Wh.repeat_interleave(M, dim=1)  # (B, M*M, E)
        Wh_repeated_alternating = Wh.repeat(1, M, 1)  # (B, M*M, E)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)  # (B, M*M,2E)
        return all_combinations_matrix.view(B, M, M, 2 * E)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_features) + '->' + str(self.out_features) + ')'


class GAT2(nn.Module):
    def __init__(self,n_feat,n_hid,out_features,dropout,alpha,n_heads):
        super(GAT2,self).__init__()
        self.hidden = n_hid
        self.max_length = 128
        self.dropout = 0.1
        self.attentions = [GATLayer(n_feat, n_hid, dropout=self.dropout,alpha=alpha, concat=True,get_att=False) for _ in range(n_heads)]
        # self.attentions_adj = [GATLayer(n_feat, self.max_length,  alpha=alpha, concat=True,get_att=True) for _ in range(n_heads)]
        for i,attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.out_att = GATLayer(n_hid * n_heads, out_features, dropout=self.dropout, alpha=alpha, concat=False)
    
    def forward(self,x_input,adj):
        x = F.dropout(x_input, self.dropout, training=self.training)
        x = torch.cat([att(x,adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.leakyrelu(self.out_att(x, adj))
        return x


class GAT(nn.Module):
    def __init__(self,n_feat,n_hid,out_features,dropout,alpha,n_heads):
        super(GAT,self).__init__()
        self.hidden = n_hid
        self.max_length = 128
        self.dropout = 0.1
        self.attentions = [GATLayer(n_feat, n_hid, dropout=self.dropout,alpha=alpha, concat=True,get_att=False) for _ in range(n_heads)]
        # self.attentions_adj = [GATLayer(n_feat, self.max_length,  alpha=alpha, concat=True,get_att=True) for _ in range(n_heads)]
        for i,attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        
        self.out_att = GATLayer(n_hid * n_heads, out_features, dropout=self.dropout, alpha=alpha, concat=False)
    
    def forward(self,x_input,adj):
        x = F.dropout(x_input, self.dropout, training=self.training)
        x = torch.cat([att(x,adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class TCModel(BertPreTrainedModel):
    def __init__(self, config, node_dim=128, **kwargs):
        super().__init__(config)
        self.hidden_size  = config.hidden_size
        self.dropout_prob = config.hidden_dropout_prob
        self.num_labels   = 2 #
        self.bidirectional = True

        self.size = node_dim
        
        self.bert = BertModel(config)
        self.num_nodes = 128
        self.dropout = nn.Dropout(self.dropout_prob)
        self.gat = GAT(
                n_feat=self.hidden_size,
                n_hid=self.size,
                out_features=self.size,
                alpha = 0.2,
                n_heads = 8,
                dropout=self.dropout_prob
            )

        self.pool = nn.AdaptiveMaxPool1d(1)
        # self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.size, self.num_labels)
        self.init_weights()

    def sentence_graph_attention(self, sentence, graph, mask):
        # graph and sentence has same dimention
        sentence = sentence.unsqueeze(1)
        d_k = sentence.size(-1)
        scores = torch.matmul(sentence, graph.transpose(1, 2)) / math.sqrt(d_k) # temperature = d_k ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1)==0, -1e9)
        att = F.softmax(scores, dim=-1)
        context = torch.matmul(att, graph).squeeze(1)
        return context


    def forward(self,input_ids,attention_mask,token_type_ids,
                dependency_matrix,labels=None,position_ids=None,head_mask=None):
        # input_ids     : [batch_size, sequence_length]
        # attention_mask: [batch_size, sequence_length]
        # token_type_ids: [batch_size, sequence_length]
        input_ids_len = torch.sum(input_ids != 0,dim=-1).float()

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        pooled_output = outputs[0] # [batch_size, node,hidden_size]
        pooled_output = self.dropout(pooled_output)
        gat_out = self.gat(pooled_output, dependency_matrix)
        clf_input = self.pool(gat_out.permute(0, 2, 1)).squeeze(-1)
        logits = self.classifier(clf_input)
        # logits = F.softmax(logits)
        return logits
