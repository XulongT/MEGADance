import torch
from torch.utils.data import Dataset
import os
import pickle as pkl
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle
from utils.utils import root_preprocess, root_postprocess, normalize, denormalize, \
                        rotation_matrix_to_rotation_6d, rotation_6d_to_angle_axis
from utils.utils import keypoint_from_smpl
import torch.nn.functional as F
from einops import rearrange


def createEvalDataset(root_dir='./data', batch_size=32, stride=10000, sample_len=1024, start=0, end=1):
    with open(os.path.join(root_dir, 'test.txt'), 'r') as file:
        file_names = [line.strip() for line in file]
    smpl_poses, smpl_trans, musics = [], [], []

    file_names_plus = []
    for file_name in tqdm(file_names):
        motion_file_path = os.path.join(root_dir, 'motion', file_name+'.pkl')
        motion_file = open(motion_file_path, 'rb')
        motion_data = pickle.load(motion_file)
        smpl_pose, smpl_tran = motion_data['smpl_poses'], motion_data['smpl_trans']

        librosa_file_path = os.path.join(root_dir, 'librosa', file_name+'.pkl')
        librosa_file = open(librosa_file_path, 'rb')
        librosa_music = pickle.load(librosa_file)['music']

        total_length = min(smpl_pose.shape[0], smpl_tran.shape[0], librosa_music.shape[0]//8*8)
        for i in range(start, total_length+end-sample_len, stride):
            smpl_poses.append(smpl_pose[i:i+sample_len, :])
            smpl_trans.append(smpl_tran[i:i+sample_len, :])
            musics.append(librosa_music[i:i+sample_len, :])
            file_names_plus.append(file_name)

    smpl_poses, smpl_trans, musics = np.array(smpl_poses), np.array(smpl_trans), np.array(musics)
    print(smpl_poses.shape, smpl_trans.shape, musics.shape)
    print('Eval Dataset len: ', smpl_poses.shape[0])
    eval_dataloader = DataLoader(EvalDataset(smpl_poses, smpl_trans, musics, file_names_plus), batch_size=batch_size, shuffle=False)
    return eval_dataloader



class EvalDataset(Dataset):
    def __init__(self, smpl_poses, smpl_trans, music_librosa, file_names):
        
        music_librosa = torch.from_numpy(music_librosa)
        smpl_root_init, smpl_root_vel = root_preprocess(torch.from_numpy(smpl_trans))
        smpl_poses = torch.from_numpy(smpl_poses)
        keypoints = get_keypoint(smpl_poses.clone(), smpl_root_vel.clone())
        smpl_poses = rotation_matrix_to_rotation_6d(smpl_poses)

        self.keypoints = keypoints
        self.music_librosa = music_librosa
        self.smpl_trans = smpl_trans
        self.smpl_poses = smpl_poses
        self.smpl_root_vel = smpl_root_vel
        self.smpl_root_init = smpl_root_init

        self.file_names = file_names
        

    def __len__(self):
        return len(self.smpl_poses)

    def __getitem__(self, idx):
        return {'smpl_trans': self.smpl_trans[idx], 'smpl_poses': self.smpl_poses[idx],  \
                'smpl_root_vel': self.smpl_root_vel[idx], 'smpl_root_init': self.smpl_root_init[idx], \
                'keypoint': self.keypoints[idx], 'music_librosa': self.music_librosa[idx], 'file_name': self.file_names[idx]}


def get_keypoint(smpl_poses, smpl_trans):
    from smplx import SMPL
    from tqdm import tqdm
    b, t, _  = smpl_poses.shape
    keypoints = torch.zeros((b, t, 72)).float()
    print('obtain keypoints from SMPL...')
    smpl = SMPL('./Pretrained/SMPL/SMPL_FEMALE.pkl', batch_size=1).cuda()
    stride = 32
    for i in tqdm(range(0, b, stride)):
        ub = min(i+stride, b)
        keypoint_pred = smpl.forward(
            global_orient=smpl_poses[i:ub, :, :3].reshape(-1, 3).float().cuda(),
            body_pose=smpl_poses[i:ub, :, 3:].reshape(-1, 69).float().cuda(),
            transl=smpl_trans[i:ub, :, :].reshape(-1, 3).float().cuda(),
        ).joints[:, :24, :].reshape(ub-i, t, -1)
        keypoints[i:ub] = keypoint_pred.detach().cpu()
    torch.cuda.empty_cache()
    return keypoints