"""
Source: Attention is all you need.
In this file I am going to implement the transformers architecture from scratch. Block by block I am going to 
design and implement all the modules as per given in the research paper.
"""

import torch
import torch.nn as nn

class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__(self, MultiHeadedAttention)
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size//self.heads

        assert (self.embed_size == self.heads*self.head_dim), "Embed size needs be heads*heads_dim"
        self.fc1 = nn.Linear(in_features=embed_size,out_features=embed_size)

    def forward(self,query,key,values):
        query = self.fc1(query)
        key = self.fc1(key)
        values = self.fc1(values)

        B = query.shape[0]
        que_len,key_len,val_len = query.shape[1],key.shape[1],values.shape[1]

        assert (key_len==val_len), "Length of keys and values differ, which will give shape mismatch errors."

        query = query.reshape(B,que_len,self.heads,self.head_dim).permute(0,2,1,3)
        key_transpose = key.reshape(B,key_len,self.heads,self.head_dim).permute(0,2,3,1)
        values = values.reshape(B,val_len,self.heads,self.head_dim).permute(0,2,1,3)

        energy = torch.matmul(query,key_transpose)/self.embed_size**(1/2)

        attention = torch.softmax(energy)
        attention_score = torch.matmul(attention,values).transpose(1,2)

        return self.fc1(attention_score)

