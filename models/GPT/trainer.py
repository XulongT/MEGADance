import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from utils.features.geometric import geometric_features
from utils.features.kinetic import kinetic_features
import yaml
from types import SimpleNamespace
from einops import rearrange
from utils.utils import denormalize, normalize, root_postprocess, keypoint_from_smpl, rotation_6d_to_angle_axis
from utils.load_model import load_model
from models.GPT.model2 import GPT
from models.FSQ.trainer import SepFSQ as FSQ
from models.CLS.classification import CLS
from smplx import SMPL
import pickle
import json
import time

class Trainer(nn.Module):
    def __init__(self, device=None):
        super().__init__()

        self.cls_model = CLS(device)
        self.cls_model = load_model(self.cls_model, 'CLS', 30)
        self.cls_model.eval()

        self.fsq = FSQ(device)
        self.fsq = load_model(self.fsq, 'FSQ', 120)
        self.fsq.eval()
        
        self.gpt = GPT(device)

    def forward(self, data, device, epoch):
        self.gpt.train()
        music_librosa, smpl_trans_gt, smpl_poses_gt, smpl_root_vel_gt, smpl_root_init, _, label = self.fsq.preprocess(data, device)
        pose_up_index, pose_down_index = self.fsq.pose_encode(smpl_poses_gt, smpl_root_vel_gt)
        pose_up_index, pose_down_index = pose_up_index[:, :, 0], pose_down_index[:, :, 0]
        loss = self.gpt(music_librosa, pose_up_index, pose_down_index, label)
        loss = {'total': loss}
        return loss

    
    def inference(self, data, device):
        self.gpt.eval()
        music_librosa, smpl_trans_gt, smpl_poses_gt, smpl_root_vel_gt, smpl_root_init_gt, _, label = self.fsq.preprocess(data, device)
        music_librosa_d = music_librosa[:, ::8, :]
        music_beat = data['music_librosa'][:, :, 33].clone()
  
        pose_up_index, pose_down_index = self.fsq.pose_encode(smpl_poses_gt, smpl_root_vel_gt)
        pose_up_index, pose_down_index = pose_up_index[:, :, 0], pose_down_index[:, :, 0]
        pose_up_index_pred, pose_down_index_pred = self.gpt.inference(music_librosa_d, pose_up_index, pose_down_index, label)
        pose_up_index_pred, pose_down_index_pred = rearrange(pose_up_index_pred, 'b t -> b t 1'), rearrange(pose_down_index_pred, 'b t -> b t 1')
        root_vel_decoded, pose_decoded = self.fsq.pose_decode(pose_up_index_pred, pose_down_index_pred) 

        smpl_trans_pred, smpl_root_vel_pred, smpl_poses_pred, keypoints_pred, _, smpl_poses_axis_pred = self.fsq.postprocess(pose_decoded, root_vel_decoded, smpl_root_init_gt)
        smpl_trans_gt, smpl_root_vel_gt, smpl_poses_gt, keypoints_gt, _, smpl_poses_axis_gt = self.fsq.postprocess(smpl_poses_gt, smpl_root_vel_gt, smpl_root_init_gt)
        
        dance_pred_genre_feature = self.cls_model.dance_encode(smpl_trans_pred, smpl_poses_pred)
        dance_gt_genre_feature = self.cls_model.dance_encode(smpl_trans_gt, smpl_poses_gt)
        dance_pred_kinetic_feature = kinetic_features(keypoints_pred)
        dance_gt_kinetic_feature = kinetic_features(keypoints_gt)
        dance_pred_geometric_feature = geometric_features(keypoints_pred)
        dance_gt_geometric_feature = geometric_features(keypoints_gt)

        result = {
            'smpl_poses_pred': smpl_poses_axis_pred, 'smpl_trans_pred': smpl_trans_pred, \
            'keypoints_pred': keypoints_pred, 'dance_pred_genre_feature': dance_pred_genre_feature, \
            'dance_pred_kinetic_feature': dance_pred_kinetic_feature, 'dance_pred_geometric_feature': dance_pred_geometric_feature, \
            'smpl_poses_gt': smpl_poses_axis_gt, 'smpl_trans_gt': smpl_trans_gt, \
            'keypoints_gt': keypoints_gt, 'dance_gt_genre_feature': dance_gt_genre_feature, \
            'dance_gt_kinetic_feature': dance_gt_kinetic_feature, 'dance_gt_geometric_feature': dance_gt_geometric_feature, \
            'music_beat': music_beat, 'label': label, 'file_name': data['file_name']
        }

        loss = {
            'total': 0
        }
        return result, loss


    def demo(self, data, device):
        self.gpt.eval()
        music_librosa, smpl_trans_gt, smpl_poses_gt, smpl_root_vel_gt, smpl_root_init_gt, _, label = self.fsq.preprocess(data, device)
        music_librosa_d = music_librosa[:, ::8, :]
        start_time = time.time()
        print(f'music Length : {music_librosa_d.shape[1]/3.75:.4f} second')
        pose_up_index, pose_down_index = self.fsq.pose_encode(smpl_poses_gt, smpl_root_vel_gt)
        pose_up_index, pose_down_index = pose_up_index[:, :, 0], pose_down_index[:, :, 0]
        pose_up_index_pred, pose_down_index_pred = self.gpt.inference(music_librosa_d, pose_up_index, pose_down_index, label)
        pose_up_index_pred, pose_down_index_pred = rearrange(pose_up_index_pred, 'b t -> b t 1'), rearrange(pose_down_index_pred, 'b t -> b t 1')
        root_vel_decoded, pose_decoded = self.fsq.pose_decode(pose_up_index_pred, pose_down_index_pred) 
        smpl_trans_pred, smpl_root_vel_pred, smpl_poses_pred, keypoints_pred, _, smpl_poses_axis_pred = self.fsq.postprocess(pose_decoded, root_vel_decoded, smpl_root_init_gt)
        end_time = time.time()
        print(f"generation time: {end_time - start_time:.4f} 秒")
        for i in range(music_librosa.shape[0]):
            with open(os.path.join(data['root_dir'][i], 'keypoint', data['file_name'][i].replace('.wav', '.json')), 'w') as f:
                json.dump({'keypoints': keypoints_pred[i].cpu().detach().numpy().tolist()}, f, indent=4)     
            with open(os.path.join(data['root_dir'][i], 'smpl', data['file_name'][i].replace('.wav', '.pkl')), 'wb') as f:
                pickle.dump({'smpl_trans': smpl_trans_pred[i].cpu().detach().numpy(), 'smpl_poses': smpl_poses_axis_pred[i].cpu().detach().numpy()}, f)  

