from smplx import SMPL
import torch
import torch as t
from torch.nn import functional as F
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from smplx import SMPL
import kornia as tgm
# from pytorch3d.transforms import (matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix)

def keypoint_from_smpl1(smpl_trans, smpl_pred, smpl_root_vel, smpl):
    b, t, _ = smpl_trans.shape
    keypoints_r = torch.zeros((b, t, 45, 3)).cuda()
    J, stride = 23, 1
    for i in range(0, b, stride):
        upper_bound = min(i+stride, b)
        smpl_trans_slice = smpl_trans[i:upper_bound].reshape(-1, 3).cuda()
        global_orient = smpl_pred[i:upper_bound, :, :3].reshape(-1, 3).cuda()
        body_pose = smpl_pred[i:upper_bound, :, 3:].reshape(-1, 3*J).cuda()
        keypoints = smpl.forward(
            global_orient=global_orient.float(),
            body_pose=body_pose.float(),
            transl=smpl_trans_slice.float(),
        ).joints.reshape(upper_bound - i, t, -1, 3)
        keypoints_r[i:upper_bound] = keypoints
    keypoints_r = keypoints_r[:, :, :24, :].detach().cpu()

    b, t, _ = smpl_trans.shape
    smpl_trans = smpl_root_vel
    keypoints_r1 = torch.zeros((b, t, 45, 3)).cuda()
    J, stride = 23, 1
    for i in range(0, b, stride):
        upper_bound = min(i+stride, b)
        smpl_trans_slice = smpl_trans[i:upper_bound].reshape(-1, 3).cuda()
        global_orient = smpl_pred[i:upper_bound, :, :3].reshape(-1, 3).cuda()
        body_pose = smpl_pred[i:upper_bound, :, 3:].reshape(-1, 3*J).cuda()
        keypoints = smpl.forward(
            global_orient=global_orient.float(),
            body_pose=body_pose.float(),
            transl=smpl_trans_slice.float(),
        ).joints.reshape(upper_bound - i, t, -1, 3)
        keypoints_r1[i:upper_bound] = keypoints
    keypoints_r1 = keypoints_r1[:, :, :24, :].detach().cpu()

    return keypoints_r, keypoints_r1


def keypoint_from_smpl(smpl_trans, smpl_pred, smpl):
    b, t, _ = smpl_trans.shape
    keypoints_r = torch.zeros((b, t, 45, 3))
    J, stride = 23, 8
    for i in range(0, b, stride):
        upper_bound = min(i+stride, b)
        smpl_trans_slice = smpl_trans[i:upper_bound].reshape(-1, 3).cuda()
        global_orient = smpl_pred[i:upper_bound, :, :3].reshape(-1, 3).cuda()
        body_pose = smpl_pred[i:upper_bound, :, 3:].reshape(-1, 3*J).cuda()
        keypoints = smpl.forward(
            global_orient=global_orient.float(),
            body_pose=body_pose.float(),
            transl=smpl_trans_slice.float(),
        ).joints.reshape(upper_bound - i, t, -1, 3)
        keypoints_r[i:upper_bound] = keypoints.detach().cpu()
    return keypoints_r[:, :, :24, :]


def root_preprocess(x):
    b, t, _ = x.shape
    root_init = x[:, 0, :3].clone()
    root_vel = torch.zeros((b, t, 3)).to(x.device)
    root_vel[:, :-1, :3] = x[:, 1:, :3] - x[:, :-1, :3] # root position
    return root_init, root_vel

def root_postprocess(root_init, x):
    b, t, _ = x.shape
    global_pos = torch.zeros((b, t, 3)).to(x.device)
    global_pos[:, 0, :3] = root_init.clone()
    for i in range(1, t):
        global_pos[:, i, :3] = global_pos[:, i-1, :3] + x[:, i-1, :3].clone()
    return global_pos

def normalize(data, mean=0, std=1):
    # std[std < 1e-5] = 1e-5
    return (data - mean) / std

def denormalize(data, mean=0, std=1):
    # std[std < 1e-5] = 1e-5
    return data * std + mean 

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:

    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:


    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def rotation_matrix_to_angle_axis(rotation_matrices):

    traces = torch.diagonal(rotation_matrices, dim1=-2, dim2=-1).sum(-1)
    cos_theta = (traces - 1) / 2

    buffer = 1e-6
    cos_theta = torch.clamp(cos_theta, -1 + buffer, 1 - buffer)
    angles = torch.acos(cos_theta)
    small_angles = angles < 1e-6

 
    omegas = torch.empty(rotation_matrices.shape[:-2] + (3,), device=rotation_matrices.device, dtype=rotation_matrices.dtype)
    omegas[..., 0] = rotation_matrices[..., 2, 1] - rotation_matrices[..., 1, 2]
    omegas[..., 1] = rotation_matrices[..., 0, 2] - rotation_matrices[..., 2, 0]
    omegas[..., 2] = rotation_matrices[..., 1, 0] - rotation_matrices[..., 0, 1]

    sin_angles = torch.sin(angles)

    safe_sin_angles = torch.where(small_angles, torch.tensor(1e-8, device=angles.device), sin_angles)
    omegas /= (2 * safe_sin_angles.unsqueeze(-1))

    angle_axis = angles.unsqueeze(-1) * omegas

    angle_axis = torch.where(small_angles.unsqueeze(-1), omegas * (1 - (angles.unsqueeze(-1) ** 2) / 6), angle_axis)

    return angle_axis

def batch_rotation_matrix_to_angle_axis(rotation_matrices):

    traces = torch.diagonal(rotation_matrices, dim1=-2, dim2=-1).sum(-1)
    cos_theta = (traces - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1, 1)
    angles = torch.acos(cos_theta)
    small_angles = angles < 1e-6
    

    omegas = torch.empty(rotation_matrices.shape[:-2] + (3,), device=rotation_matrices.device, dtype=rotation_matrices.dtype)
    omegas[..., 0] = rotation_matrices[..., 2, 1] - rotation_matrices[..., 1, 2]
    omegas[..., 1] = rotation_matrices[..., 0, 2] - rotation_matrices[..., 2, 0]
    omegas[..., 2] = rotation_matrices[..., 1, 0] - rotation_matrices[..., 0, 1]
    
    sin_angles = torch.sin(angles)
    with torch.no_grad(): 
        omegas /= (2 * sin_angles.unsqueeze(-1))
        omegas[small_angles] = 0.0 
    
    angle_axis = angles.unsqueeze(-1) * omegas
    angle_axis[small_angles] = 0.0  
    
    return angle_axis
    traces = np.trace(rotation_matrices, axis1=3, axis2=4)
    cos_theta = (traces - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    angles = np.arccos(cos_theta)
    small_angles = angles < 1e-6
    
    omegas = np.empty(rotation_matrices.shape[:-1])
    omegas[..., 0] = rotation_matrices[..., 2, 1] - rotation_matrices[..., 1, 2]
    omegas[..., 1] = rotation_matrices[..., 0, 2] - rotation_matrices[..., 2, 0]
    omegas[..., 2] = rotation_matrices[..., 1, 0] - rotation_matrices[..., 0, 1]
    
    sin_angles = np.sin(angles)
    with np.errstate(divide='ignore', invalid='ignore'):
        omegas /= (2 * sin_angles[..., np.newaxis])
        omegas[small_angles] = 0  
    
    angle_axis = angles[..., np.newaxis] * omegas
    angle_axis[small_angles] = 0  
    
    return angle_axis

def euler_to_rotation_matrix_torch(angles):

    b = angles.shape[0]
    R = torch.zeros((b, 3, 3), dtype=angles.dtype, device=angles.device)
    
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    

    sin_roll, sin_pitch, sin_yaw = sin[:, 0], sin[:, 1], sin[:, 2]
    cos_roll, cos_pitch, cos_yaw = cos[:, 0], cos[:, 1], cos[:, 2]
    

    R[:, 0, 0] = cos_yaw * cos_pitch
    R[:, 0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
    R[:, 0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
    
    R[:, 1, 0] = sin_yaw * cos_pitch
    R[:, 1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
    R[:, 1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
    
    R[:, 2, 0] = -sin_pitch
    R[:, 2, 1] = cos_pitch * sin_roll
    R[:, 2, 2] = cos_pitch * cos_roll
    
    return R


def rotation_matrix_to_rotation_6d(smpl_poses):
    J = 24
    b, t, _ = smpl_poses.shape
    smpl_poses = euler_to_rotation_matrix_torch(smpl_poses.reshape(-1, 3))
    smpl_poses = matrix_to_rotation_6d(smpl_poses).reshape(b, t, -1)
    return smpl_poses


def rotation_6d_to_angle_axis(smpl_poses):
    J = 24
    b, t, _ = smpl_poses.shape
    smpl_poses = smpl_poses.reshape(-1, 6)
    smpl_poses = rotation_6d_to_matrix(smpl_poses).reshape(-1, 3, 3)
    smpl_poses = batch_rotation_matrix_to_angle_axis(smpl_poses)
    smpl_poses = smpl_poses.reshape(b, t, -1)
    return smpl_poses


# def similarity_matrix(music_features, dance_features):
#     music_norm = F.normalize(music_features, p=2, dim=1)
#     dance_norm = F.normalize(dance_features, p=2, dim=1)
#     cos_sim = F.cosine_similarity(music_norm.unsqueeze(1), dance_norm.unsqueeze(0), dim=2)
#     return cos_sim


def similarity_matrix(music_features, dance_features):


    music_norm = music_features / music_features.norm(dim=2, keepdim=True)
    dance_norm = dance_features / dance_features.norm(dim=2, keepdim=True)
    

    music_norm = music_norm.permute(1, 0, 2)  
    dance_norm = dance_norm.permute(1, 0, 2)  
    

    similarity_matrix = torch.bmm(music_norm, dance_norm.transpose(1, 2))
    similarity_matrix = torch.mean(similarity_matrix, dim=0)
    
    return similarity_matrix

def compute_rank(sims):

    sims_sort = torch.argsort(sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)
    lst = torch.diag(sims_sort_2)

    metrics = {}
    # metrics["R1"] = (100 * float(torch.sum(lst == 0)) / len(lst))
    metrics["R5"] = (100 * float(torch.sum(lst < 5)) / len(lst))
    metrics["R10"] = (100 * float(torch.sum(lst < 10)) / len(lst))
    metrics["MedR"] = float(torch.median(lst.float()) + 1)
    metrics["MeanR"] = float(torch.mean(lst.float()) + 1)

    return metrics



import torch
import numpy as np
import random

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False