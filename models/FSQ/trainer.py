import numpy as np
import torch.nn as nn
from .fsq import Conv1DEncoder, Conv1DQuantizer, Conv1DDecoder
import yaml
from types import SimpleNamespace
import torch
from einops import rearrange
from utils.utils import normalize, denormalize, root_postprocess, keypoint_from_smpl, keypoint_from_smpl1, rotation_6d_to_angle_axis
from smplx import SMPL
import torch as t

smpl_up = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
smpl_down = [0, 1, 2, 4, 5, 7, 8, 10, 11]

def _loss_fn(x_target, x_pred):
    return t.mean(t.abs(x_pred - x_target)) 
    
class Motion_Loss(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.acc = 0.25
        self.vel = 0.5

    def forward(self, x_out, x_target):
        recons_loss = t.zeros(()).to(x_target.device)
        regularization = t.zeros(()).to(x_target.device)
        velocity_loss = t.zeros(()).to(x_target.device)
        acceleration_loss = t.zeros(()).to(x_target.device)

        recons_loss += _loss_fn(x_target, x_out)
        regularization += t.mean((x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1])**2)
        velocity_loss +=  _loss_fn( x_out[:, 1:] - x_out[:, :-1], x_target[:, 1:] - x_target[:, :-1])
        acceleration_loss +=  _loss_fn(x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1], x_target[:, 2:] + x_target[:, :-2] - 2 * x_target[:, 1:-1])

        loss = recons_loss + self.vel * velocity_loss + self.acc * acceleration_loss

        return loss

class SepFSQ(nn.Module):
    def __init__(self, device):
        super().__init__()
        with open('./config/fsq.yaml', 'r') as f:
            hps = yaml.safe_load(f)
            hps = SimpleNamespace(**hps)
        print(hps)
        self.hps = SimpleNamespace(**hps.solo)
        self.smpl = SMPL('./Pretrained/SMPL/SMPL_FEMALE.pkl', batch_size=1).to(device)
        self.smpl.eval()

        self.Pose_UP_Encoder = Conv1DEncoder(self.hps, len(smpl_up)*6)
        self.Pose_UP_Quantizer = Conv1DQuantizer(self.hps)
        self.Pose_UP_Decoder = Conv1DDecoder(self.hps, len(smpl_up)*6)

        self.Pose_DOWN_Encoder = Conv1DEncoder(self.hps, len(smpl_down)*6+3)
        self.Pose_DOWN_Quantizer = Conv1DQuantizer(self.hps)
        self.Pose_DOWN_Decoder = Conv1DDecoder(self.hps, len(smpl_down)*6+3)

        self.L1_Loss = nn.L1Loss()
        self.Pose_Loss = Motion_Loss(self.hps)

        mean, std = torch.load('./Pretrained/mean.pt'), torch.load('./Pretrained/std.pt')
        self.smpl_trans_mean, self.smpl_poses_mean, self.smpl_root_vel_mean  = \
            mean['smpl_trans_mean'].to(device).float(), mean['smpl_poses_mean'].to(device).float(), \
            mean['smpl_root_vel_mean'].to(device).float()
        self.smpl_trans_std, self.smpl_poses_std, self.smpl_root_vel_std = \
            std['smpl_trans_std'].to(device).float(), std['smpl_poses_std'].to(device).float(), \
            std['smpl_root_vel_std'].to(device).float()

        # mean, std = torch.load('./Pretrained/retrieval/mean.pt'), torch.load('./Pretrained/retrieval/std.pt')
        self.music_librosa_mean = mean['music_librosa_mean'].to(device).float()
        self.music_librosa_std = std['music_librosa_std'].to(device).float()

    def forward(self, data, device):

        
        music_librosa, _, smpl_poses_gt, smpl_root_vel_gt, _, keypoint_gt, _ = self.preprocess(data, device)

        pose = smpl_poses_gt
        b, t, c = pose.size()
        pose = pose.view(b, t, 24, 6)

        pose_up = pose[:, :, smpl_up, :].view(b, t, -1)        
        pose_up_gt = pose_up.detach().clone()
        pose_up_encoded = self.Pose_UP_Encoder(pose_up)
        _, pose_up_quantised = self.Pose_UP_Quantizer(pose_up_encoded)
        pose_up_decoded = self.Pose_UP_Decoder(pose_up_quantised)

        root_vel = smpl_root_vel_gt
        pose_down = pose[:, :, smpl_down, :].view(b, t, -1)
        pose_down = torch.cat([root_vel, pose_down], dim=2)
        pose_down_gt = pose_down.detach().clone()
        pose_down_encoded = self.Pose_DOWN_Encoder(pose_down)
        _, pose_down_quantised = self.Pose_DOWN_Quantizer(pose_down_encoded)
        pose_down_decoded = self.Pose_DOWN_Decoder(pose_down_quantised)

        smpl_loss = self.Pose_Loss(pose_up_decoded, pose_up.float()) + self.Pose_Loss(pose_down_decoded, pose_down.float())
        keypoint_pred = self.get_keypoint(pose_up_decoded, pose_down_decoded)
        # keypoint_gt = self.get_keypoint(pose_up_gt, pose_down_gt)
        keypoint_loss = self.Pose_Loss(keypoint_pred, keypoint_gt)
        total_loss = keypoint_loss + smpl_loss

        loss = {'smpl_loss': smpl_loss, 'keypoint_loss': keypoint_loss, 'total': total_loss}
        return None, loss

    def get_keypoint(self, pose_up_decoded, pose_down_decoded):

        b, t, _ = pose_up_decoded.shape
        root_vel_decoded = pose_down_decoded[:, :, :3]
        pose_down_decoded = pose_down_decoded[:, :, 3:]
        pose_decoded = torch.zeros(b, t, 24, 6).cuda().float()
        pose_decoded[:, :, smpl_up] = pose_up_decoded.view(b, t, len(smpl_up), 6)
        pose_decoded[:, :, smpl_down] = pose_down_decoded.view(b, t, len(smpl_down), 6)
        pose_decoded = rearrange(pose_decoded, 'b t c1 c2 -> b t (c1 c2)')

        root_vel_decoded = denormalize(root_vel_decoded, self.smpl_root_vel_mean, self.smpl_root_vel_std)
        root_vel_decoded = rearrange(root_vel_decoded, 'b t c -> (b t) c')
        pose_decoded = denormalize(pose_decoded, self.smpl_poses_mean, self.smpl_poses_std)
        pose_decoded = rearrange(pose_decoded, 'b t (c1 c2) -> (b t c1) c2', c2=6)

        from utils.utils import rotation_6d_to_matrix, rotation_matrix_to_angle_axis
        pose_decoded = rotation_matrix_to_angle_axis(rotation_6d_to_matrix(pose_decoded))
        pose_decoded = rearrange(pose_decoded, '(b t c1) c2 -> (b t) (c1 c2)', t=t, c1=24, c2=3)
        keypoint_pred = self.smpl.forward(
            global_orient=pose_decoded[:, :3].reshape(-1, 3).float(),
            body_pose=pose_decoded[:, 3:].reshape(-1, 69).float(),
            transl=root_vel_decoded.float(),
        ).joints[:, :24, :]

        keypoint_pred = rearrange(keypoint_pred, '(b t) c1 c2 -> b t (c1 c2)', t=t)
        return keypoint_pred

    def inference(self, data, device):
        music_librosa, smpl_trans_gt, smpl_poses_gt, smpl_root_vel_gt, smpl_root_init_gt, _, _ = self.preprocess(data, device)
   
        pose_up_index, pose_down_index = self.pose_encode(smpl_poses_gt, smpl_root_vel_gt)
        # self.add_by_index_bincount(self.up_codebook, pose_up_index)
        # self.add_by_index_bincount(self.down_codebook, pose_down_index)
        root_vel_decoded, smpl_pose_decoded = self.pose_decode(pose_up_index, pose_down_index)
    
        smpl_trans_gt, smpl_root_vel_gt, smpl_poses_gt, keypoints_gt, keypoints_gt1, smpl_poses_axis_gt = self.postprocess(smpl_poses_gt, smpl_root_vel_gt, smpl_root_init_gt)
        smpl_trans_pred, smpl_root_vel_pred, smpl_poses_pred, keypoints_pred, keypoints_pred1,  smpl_poses_axis_pred = self.postprocess(smpl_pose_decoded, root_vel_decoded, smpl_root_init_gt)

        result = {
            'smpl_poses_pred': smpl_poses_axis_pred, 'smpl_poses_6d_pred': smpl_poses_pred, 'smpl_trans_pred': smpl_trans_pred, \
            'keypoints_pred': keypoints_pred, 'keypoints_static_pred': keypoints_pred1, 'smpl_root_vel_pred': smpl_root_vel_pred, \
            'smpl_poses_gt': smpl_poses_axis_gt, 'smpl_poses_6d_gt': smpl_poses_gt, 'smpl_trans_gt': smpl_trans_gt, \
            'keypoints_gt': keypoints_gt,  'keypoints_static_gt': keypoints_gt1,  'smpl_root_vel_gt': smpl_root_vel_gt, \
            'file_name': data['file_name'], \
        }

        return result, {'total': 0}

    def add_by_index_bincount(self, a, b):

        indices = b.view(-1).long()  

        counts = torch.bincount(indices, minlength=a.size(0))
        

        a.add_(counts)
        return a

    def pose_encode(self, smpl_poses_gt, smpl_root_vel_gt):

        pose = smpl_poses_gt
        b, t, c = pose.size()
        pose = pose.view(b, t, 24, 6)

        pose_up = pose[:, :, smpl_up, :].view(b, t, -1)        
        pose_up_encoded = self.Pose_UP_Encoder(pose_up)
        pose_up_index, pose_up_quantised = self.Pose_UP_Quantizer(pose_up_encoded)

        root_vel = smpl_root_vel_gt
        pose_down = pose[:, :, smpl_down, :].view(b, t, -1)
        pose_down = torch.cat([root_vel, pose_down], dim=2)
        pose_down_encoded = self.Pose_DOWN_Encoder(pose_down)
        pose_down_index, pose_down_quantised = self.Pose_DOWN_Quantizer(pose_down_encoded)

        pose_up_index = pose_up_index[0]
        pose_down_index = pose_down_index[0]

        return pose_up_index, pose_down_index

    def pose_decode(self, pose_up_idx, pose_down_idx):
        b, t, _ = pose_up_idx.shape
        t = 8 * t

        pose_up_quantised = self.Pose_UP_Quantizer.get_feature_from_index(pose_up_idx)
        pose_down_quantised = self.Pose_DOWN_Quantizer.get_feature_from_index(pose_down_idx)

        pose_up_decoded = self.Pose_UP_Decoder(pose_up_quantised)
        pose_down_decoded = self.Pose_DOWN_Decoder(pose_down_quantised)
        root_vel_decoded = pose_down_decoded[:, :, :3]
        pose_down_decoded = pose_down_decoded[:, :, 3:]

        pose_decoded = torch.zeros(b, t, 24, 6).cuda().float()
        pose_decoded[:, :, smpl_up] = pose_up_decoded.view(b, t, len(smpl_up), 6)
        pose_decoded[:, :, smpl_down] = pose_down_decoded.view(b, t, len(smpl_down), 6)
        pose_decoded = rearrange(pose_decoded, 'b t c1 c2 -> b t (c1 c2)')

        return root_vel_decoded, pose_decoded
        
    def preprocess(self, data, device):
        music_librosa = data['music_librosa'].to(device).float()
        music_librosa = normalize(music_librosa, self.music_librosa_mean, self.music_librosa_std)

        if 'label' in data.keys():
            label = data['label'].to(device).long()
        else:
            label = None
        
        keypoint = data['keypoint'].to(device).float()
        smpl_trans_gt = data['smpl_trans'].to(device).float()
        smpl_poses_gt = data['smpl_poses'].to(device).float()
        smpl_root_vel_gt = data['smpl_root_vel'].to(device).float()
        smpl_root_init = data['smpl_root_init'].to(device).float()

        smpl_trans_gt = normalize(smpl_trans_gt, self.smpl_trans_mean, self.smpl_trans_std)
        smpl_poses_gt = normalize(smpl_poses_gt, self.smpl_poses_mean, self.smpl_poses_std)
        smpl_root_vel_gt = normalize(smpl_root_vel_gt, self.smpl_root_vel_mean, self.smpl_root_vel_std)
    
        return music_librosa, smpl_trans_gt, smpl_poses_gt, smpl_root_vel_gt, smpl_root_init, keypoint, label

    def postprocess(self, smpl_poses, smpl_root_vel, smpl_root_init):
        smpl_root_vel = denormalize(smpl_root_vel, self.smpl_root_vel_mean, self.smpl_root_vel_std)
        smpl_trans = root_postprocess(smpl_root_init, smpl_root_vel)
        smpl_poses = denormalize(smpl_poses, self.smpl_poses_mean, self.smpl_poses_std)
        smpl_poses_axis = rotation_6d_to_angle_axis(smpl_poses)
        keypoints, keypoints_static = keypoint_from_smpl1(smpl_trans, smpl_poses_axis, smpl_root_vel, self.smpl)
        # keypoints = keypoint_from_smpl(smpl_trans, smpl_poses_axis, self.smpl)
        # keypoints_static = None
        return smpl_trans, smpl_root_vel, smpl_poses, keypoints, keypoints_static, smpl_poses_axis