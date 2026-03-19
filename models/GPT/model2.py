from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F
import math
# from models.GPT.kan import KAN
from mamba_ssm import Mamba
from mamba_ssm import Mamba2
import random


class CE_Loss(nn.Module):
    def __init__(self, levels):
        super(CE_Loss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.levels = levels
        self.starts = [0]
        for level in levels:
            self.starts.append(self.starts[-1] + level)

    def forward(self, pose_up_pred, pose_down_pred, pose_up_tgt, pose_down_tgt):
        loss = 0
        pose_down_pred, pose_up_pred = rearrange(pose_down_pred, 'b t c -> (b t) c').float(), rearrange(pose_up_pred, 'b t c -> (b t) c').float()
        pose_down_tgt, pose_up_tgt = rearrange(pose_down_tgt, 'b t n-> (b t) n').long(), rearrange(pose_up_tgt, 'b t n -> (b t) n').long()
        for i in range(len(self.levels)):
            pose_down_tgt1, pose_up_tgt1 = pose_down_tgt[:, i], pose_up_tgt[:, i]
            pose_down_pred1, pose_up_pred1 = pose_down_pred[:, self.starts[i]:self.starts[i]+self.levels[i]], pose_up_pred[:, self.starts[i]:self.starts[i]+self.levels[i]]
            loss += (self.cls_loss(pose_up_pred1, pose_up_tgt1) + self.cls_loss(pose_down_pred1, pose_down_tgt1))
        return loss


class GPT(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device
        self.levels = [7, 5, 5, 5, 5]
        self.starts = [0]
        for level in self.levels:
            self.starts.append(self.starts[-1] + level)
        self.dance_decoder = Dance_Decoder(levels=self.levels)
        self.cls_loss = CE_Loss(levels=self.levels)
        self.window_size, self.train_size = 22, 29

    def forward(self, music_librosa, pose_up, pose_down, label):
        music_src = music_librosa[:, :-1, :]
        pose_up_src, pose_down_src = pose_up[:, :-1], pose_down[:, :-1]
        pose_up_tgt, pose_down_tgt = pose_up[:, 1:], pose_down[:, 1:]
        pose_up_pred, pose_down_pred = self.dance_decoder(music_src, pose_up_src, pose_down_src, label)  
        loss = self.cls_loss(pose_up_pred, pose_down_pred, pose_up_tgt, pose_down_tgt)
        return loss

    def inference(self, music_librosas, pose_up_index1, pose_down_index1, label):
        b, t, _ = music_librosas.shape
        music_librosa = music_librosas[:, :1, :]
        pose_up_indexs, pose_down_indexs = pose_up_index1[:, :1], pose_down_index1[:, :1]
        pose_up_index, pose_down_index = pose_up_index1[:, :1], pose_down_index1[:, :1]
        for i in range(1, t):
            pose_up_preds, pose_down_preds = self.dance_decoder(music_librosa, pose_up_index, pose_down_index, label)
            pose_up_pred, pose_down_pred = self.infer_result(pose_up_preds, pose_down_preds)
            pose_up_indexs = torch.cat([pose_up_indexs, pose_up_pred], dim=1)
            pose_down_indexs = torch.cat([pose_down_indexs, pose_down_pred], dim=1)
            if i < self.window_size:
                music_librosa = music_librosas[:, :i+1, :]
                pose_up_index = pose_up_indexs
                pose_down_index = pose_down_indexs         
            else:
                music_librosa = music_librosas[:, i-self.window_size+1:i+1, :]
                pose_up_index = pose_up_indexs[:, i-self.window_size+1:i+1]
                pose_down_index = pose_down_indexs[:, i-self.window_size+1:i+1]       
        return pose_up_indexs, pose_down_indexs

    def infer_result(self, pose_up_preds, pose_down_preds):
        pose_up_pred1, pose_down_pred1 = [], []
        for i in range(len(self.levels)):
            _, pose_up_pred = torch.max(pose_up_preds[:, -1:, self.starts[i]:self.starts[i]+self.levels[i]], dim=-1)
            _, pose_down_pred = torch.max(pose_down_preds[:, -1:, self.starts[i]:self.starts[i]+self.levels[i]], dim=-1) 
            pose_up_pred1.append(pose_up_pred)
            pose_down_pred1.append(pose_down_pred)
        pose_up_pred1, pose_down_pred1 = torch.stack(pose_up_pred1, dim=-1), torch.stack(pose_down_pred1, dim=-1)
        return pose_up_pred1, pose_down_pred1


class Dance_Decoder(nn.Module):
    def __init__(self, input_dim=512, output_dim=54, hidden_dim=512, num_layers=8, nhead=8, levels=None):
        super().__init__()
        self.music_MLP = nn.Sequential(
            nn.Linear(35, 128),
            nn.Linear(128, hidden_dim)
        )  
        self.decoder_layers = nn.ModuleList([DecoderLayer(hidden_dim, nhead) for _ in range(num_layers)])
        self.output_dim = output_dim
        self.layer = num_layers
        self.levels = levels
        output_dim = sum(levels)
        self.MLP_Output = nn.Sequential(
            nn.Linear(hidden_dim, 768),
            nn.Linear(768, output_dim)
        )
        self.pose_up_emb = nn.ModuleList([nn.Embedding(level, hidden_dim) for level in levels])
        self.pose_down_emb = nn.ModuleList([nn.Embedding(level, hidden_dim) for level in levels])
        # self.pose_up_emb = nn.Embedding(output_dim, hidden_dim)
        # self.pose_down_emb = nn.Embedding(output_dim, hidden_dim)

    def forward(self, music_src, pose_up_src, pose_down_src, label_src):
        b, t, _ = pose_up_src.shape
        pose_up_feat = torch.zeros((b, t, 512)).to(music_src.device)
        pose_down_feat = torch.zeros((b, t, 512)).to(music_src.device)
        for i in range(len(self.levels)):
            pose_up_feat += self.pose_up_emb[i](pose_up_src[:, :, i])
            pose_down_feat += self.pose_down_emb[i](pose_down_src[:, :, i])
        music_feat = self.music_MLP(music_src)
        feat_src = torch.cat([pose_up_feat, pose_down_feat, music_feat], dim=1)
        for i in range(self.layer):
            feat_src = self.decoder_layers[i](feat_src, label_src)
        pose_pred = self.MLP_Output(feat_src)
        pose_up_pred, pose_down_pred = pose_pred[:, :t, :], pose_pred[:, t:t*2, :]
        return pose_up_pred, pose_down_pred



class DecoderLayer(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8):
        super().__init__()
        self.expert1_num, self.expert2_num = 4, 16
        self.genre_swab = nn.ModuleList([SWABlock() for _ in range(self.expert2_num)])
        self.global_swab = SWABlock()
        self.global_mamba = MambaBlock()

    def forward(self, feat_src, label):

        feat_src = self.global_mamba(feat_src)
        feat_src = self.global_swab(feat_src)
        B, T, C = feat_src.shape
        output = torch.zeros_like(feat_src)

        for expert_idx in range(self.expert2_num):
            mask = (label == expert_idx)
            expert_input = feat_src[mask]  # [num_samples, T, C]
            expert_output = self.genre_swab[expert_idx](expert_input)
            output[mask] = expert_output
        return output

class MambaBlock(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.mamba1 = Mamba(d_model=512, d_state=16, d_conv=4, expand=2)
        self.mamba2 = Mamba(d_model=512, d_state=16, d_conv=4, expand=2)
        self.mamba3 = Mamba(d_model=512, d_state=16, d_conv=4, expand=2)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(512)

    def forward(self, x):
        b, T, c = x.shape
        t = T // 3
        mamba_feat1 = self.mamba1(x[:, :t, :])
        x[:, :t, :] = self.norm1(x[:, :t, :] + mamba_feat1)
        mamba_feat2 = self.mamba2(x[:, t:2*t, :])
        x[:, t:2*t, :] = self.norm2(x[:, t:2*t, :] + mamba_feat2)
        mamba_feat3 = self.mamba3(x[:, 2*t:, :])
        x[:, 2*t:, :] = self.norm3(x[:, 2*t:, :] + mamba_feat3)
        return x

class SWABlock(nn.Module):
    def __init__(self, n_embd=512, pdrop=0.4):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)
        self.attn = SWACrossConditionalSelfAttention()
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x    

class LayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x):
        b, T, c = x.shape
        t = T // 3
        x_feat = torch.zeros_like(x).to(x.device)  
        x_feat[:, :t, :] += self.ln1(x[:, :t, :])
        x_feat[:, t:2*t, :] += self.ln2(x[:, t:2*t, :])
        x_feat[:, 2*t:, :] += self.ln3(x[:, 2*t:, :])
        return x_feat   

class SWACrossConditionalSelfAttention(nn.Module):
    def __init__(self, n_embd=512, pdrop=0.4, block_size=240, n_head=8):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(pdrop)
        self.resid_drop = nn.Dropout(pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.register_buffer("mask", self.get_mask())

    def get_mask(self, ):
        window_size, train_size = 22, 29
        mask = torch.triu(torch.ones((train_size, train_size), dtype=torch.bool), 1)
        for i in range(window_size, train_size):
            mask[i, :i-window_size+1] = True
        mask = mask.view(1, 1, 29, 29)
        return mask

    def forward(self, x):
        B, T, C = x.size()  # T = 2*t (music up down)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        t = T // 3
        mask = self.mask[:, :, :t, :t].repeat(1, 1, 3, 3)
        att = att.masked_fill(mask==1, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_drop(self.proj(y))
        return y
