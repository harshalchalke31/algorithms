import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt


class PatchEmbedding(nn.Module):
    def __init__(self,
                 IMG_SIZE,
                 EMBED_DIM,
                 PATCH_SIZE,
                 IN_CHANNELS
                ):
        super().__init__()
        self.PATCH_SIZE = PATCH_SIZE
        self.proj = nn.Conv2d(in_channels=IN_CHANNELS,
                              out_channels=EMBED_DIM,
                              kernel_size=PATCH_SIZE,
                              stride=PATCH_SIZE)
        num_patches = (IMG_SIZE // PATCH_SIZE)**2
        self.cls_token = nn.Parameter(torch.zeros(1,1,EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1,1+num_patches,EMBED_DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # Projection -> (B, D, H, W)  -> (B, D, H/P, W/P)
        x = self.proj(x)

        # Flatten and Transpose -> (B, D, H/P, W/P) -> (B, D, N) -> (B,N,D)
        # Where N = (H/P) * (W/P), 
        # Thus, we have N patches each with embedding dimension D
        x = x.flatten(2).transpose(1,2)

        # Prepend CLS token -> (1, 1, D) -> (B, 1 , D) -> (B, N+1, D)
        cls_token = self.cls_token.expand(B,-1,-1)
        x = torch.concat((cls_token,x),dim=1)
        x = x+self.pos_embed
        return x

class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 DROPOUT_RATE
                ):
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=in_features,
            out_features=out_features
        )
        self.fc2 = nn.Linear(
            in_features=out_features,
            out_features=in_features
        )
        self.dropout = nn.Dropout(p=DROPOUT_RATE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x

class EncoderBlock(nn.Module):
    def __init__(self,
                 EMBED_DIM,
                 MLP_RATIO,
                 NUM_HEADS,
                 DROPOUT_RATE
                ):
        super().__init__()
        self.MLP_DIM = int(EMBED_DIM * MLP_RATIO)
        self.drop1 = nn.Dropout(DROPOUT_RATE)
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.attn = nn.MultiheadAttention(embed_dim=EMBED_DIM,
                                          num_heads=NUM_HEADS,
                                          dropout=DROPOUT_RATE,
                                          batch_first=True)
        
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = MLP(in_features=EMBED_DIM,
                       out_features=self.MLP_DIM,
                       DROPOUT_RATE=DROPOUT_RATE)
        self.drop2 = nn.Dropout(DROPOUT_RATE)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.drop1(self.attn(normed, normed, normed)[0])
        x = x + self.drop2(self.mlp(self.norm2(x)))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self,
                 EMBED_DIM,
                 NUM_HEADS,
                 NUM_CLASSES,
                 DEPTH,
                 DROPOUT_RATE,
                 IMG_SIZE,
                 PATCH_SIZE,
                 IN_CHANNELS,
                 MLP_RATIO,
                 INIT_WEIGHTS = False
                ):
        super().__init__()
        self.patch_embed = PatchEmbedding(IMG_SIZE=IMG_SIZE,
                                          EMBED_DIM=EMBED_DIM,
                                          PATCH_SIZE=PATCH_SIZE,
                                          IN_CHANNELS=IN_CHANNELS)

        self.encoder = nn.Sequential(*[
            EncoderBlock(EMBED_DIM=EMBED_DIM,
                               MLP_RATIO=MLP_RATIO,
                               NUM_HEADS=NUM_HEADS,
                               DROPOUT_RATE=DROPOUT_RATE)
            for _ in range(DEPTH)
            ])
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(in_features=EMBED_DIM,
                              out_features=NUM_CLASSES)
        if INIT_WEIGHTS:
            self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.patch_embed.cls_token, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight,std=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.LayerNorm):
                nn.init.constant_(m.bias,0)
                nn.init.constant_(m.weight,1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)  # Use trunc_normal for patch projection
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)
