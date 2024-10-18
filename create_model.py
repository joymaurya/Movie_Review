import torch
from torch import nn


class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.Embedding=nn.Embedding(num_embeddings=10000,embedding_dim=128,padding_idx=0,max_norm=True)
    self.RNN=nn.RNN(input_size=128,hidden_size=128,nonlinearity="relu",batch_first=True,num_layers=1)
    self.final_layer=nn.Linear(in_features=128,out_features=1)
  def forward(self,x):
    x=self.Embedding(x)
    rnn_output,h_n=self.RNN(x)
    x=self.final_layer(h_n[-1])
    return x