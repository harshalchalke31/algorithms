"""
Source: Attention is all you need.
In this file I am going to implement the transformers architecture from scratch. Block by block I am going to 
design and implement all the modules as per given in the research paper.
"""

import torch
import torch.nn as nn
import math

class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadedAttention,self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size//self.heads

        assert (self.embed_size == self.heads*self.head_dim), "Embed size needs be heads*heads_dim"
        self.query = nn.Linear(in_features=embed_size,out_features=embed_size)
        self.key = nn.Linear(in_features=embed_size,out_features=embed_size)
        self.values = nn.Linear(in_features=embed_size,out_features=embed_size)
        self.out = nn.Linear(in_features=embed_size,out_features=embed_size)

    def forward(self,query,key,values,mask):
        query = self.query(query)
        key = self.key(key)
        values = self.values(values)

        B = query.shape[0]
        que_len,key_len,val_len = query.shape[1],key.shape[1],values.shape[1]

        assert key_len==val_len, "Length of keys and values differ, which will give shape mismatch errors."

        query = query.reshape(B,que_len,self.heads,self.head_dim).permute(0,2,1,3)
        key_transpose = key.reshape(B,key_len,self.heads,self.head_dim).permute(0,2,3,1)
        values = values.reshape(B,val_len,self.heads,self.head_dim).permute(0,2,1,3)

        energy = torch.matmul(query,key_transpose)/math.sqrt(self.head_dim)

        if mask is not None:
            energy = energy.masked_fill(mask==0,value=float("-1e10"))

        attention = torch.softmax(energy,dim=-1)
        attention_score = torch.matmul(attention,values).transpose(1,2)
        

        return self.out(attention_score.reshape(B,que_len,self.embed_size))
    
class TransformerBlock(nn.Module):
    def __init__(self,embed_size, heads, dropout, expansion_factor):
        super(TransformerBlock,self).__init__()
        self.attention = MultiHeadedAttention(embed_size=embed_size,
                                              heads=heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=embed_size,out_features=expansion_factor*embed_size),
            nn.GELU(),
            nn.Linear(in_features=expansion_factor*embed_size,out_features=embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self,query,key,values,mask):
        normed_query = self.norm1(query)
        attention_scores = self.attention(normed_query,key,values,mask)

        add1 = query+ self.dropout(attention_scores)
        normed_add1 = self.norm2(add1)
        feedforward = self.feed_forward(normed_add1)
        add2 = add1 + self.dropout(feedforward)
        return add2

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, expansion_factor, dropout, max_length):
        super(Encoder,self).__init__()
        self.embed_size = embed_size
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size=embed_size,
                                 heads=heads,
                                 dropout=dropout,
                                 expansion_factor=expansion_factor)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size,self.embed_size)
        self.positional_embedding = nn.Embedding(max_length,self.embed_size)

    def forward(self,x,mask=None):
        B,seq_length = x.shape
        positions = torch.arange(0,seq_length).expand(B,seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x)+self.positional_embedding(positions))

        for layer in self.layers:
            out = layer(out,out,out,mask)

        return out

class Decoderblock(nn.Module):
    def __init__(self, embed_size, heads,dropout ,expansion_factor):
        super(Decoderblock,self).__init__()
        self.attention = MultiHeadedAttention(embed_size=embed_size,heads=heads)
        self.transformer_block = TransformerBlock(embed_size=embed_size,
                                                  heads=heads,
                                                  dropout=dropout,
                                                  expansion_factor=expansion_factor)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_size)
    
    def forward(self,x,key,values,src_mask,trg_mask):
        norm_x = self.norm(x)
        attention = self.attention(norm_x,norm_x,norm_x,trg_mask)
        query = x + self.dropout(attention)
        out = self.transformer_block(query,key,values,src_mask)
        return out
    
