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
import librosa

style1s = [
    'Folk', 'Classic', 'Mix', 'Street'
]
style2s = [
    'Dai', 'ShenYun', 'Wei', 'Korean', 'Urban', 'Hiphop', 'Popping', 'Miao', 'HanTang', \
    'Breaking', 'Kun', 'Locking', 'Jazz', 'Choreography', 'Chinese', 'DunHuang'
]



def createEvalDataset(root_dir='./data', batch_size=32, stride=512, sample_len=1024, start=0, end=1):
    with open(os.path.join(root_dir, 'test.txt'), 'r') as file:
        file_names = [line.strip() for line in file]
    smpl_poses, smpl_trans, music_librosas, music_muqs, labels = [], [], [], [], []

    for file_name in tqdm(file_names):
        motion_file_path = os.path.join(root_dir, 'motion', file_name+'.pkl')
        motion_file = open(motion_file_path, 'rb')
        motion_data = pickle.load(motion_file)
        smpl_pose, smpl_tran = motion_data['smpl_poses'], motion_data['smpl_trans']

        librosa_file_path = os.path.join(root_dir, 'librosa', file_name+'.pkl')
        librosa_file = open(librosa_file_path, 'rb')
        music_librosa = pickle.load(librosa_file)['music']

        muq_file_path = os.path.join(root_dir, 'muq', file_name+'.pkl')
        muq_file = open(muq_file_path, 'rb')
        music_muq = pickle.load(muq_file)['music']

        label_file_path = os.path.join(root_dir, 'label', file_name+'.json')
        label_file = open(label_file_path, 'r')
        label = style2s.index(json.load(label_file).get('style2'))
        # label = style1s.index(json.load(label_file).get('style1'))

        total_len = min(smpl_pose.shape[0], smpl_pose.shape[0], music_librosa.shape[0], music_muq.shape[0])//8*8
        for i in range(start, total_len+end-sample_len, stride):
            smpl_poses.append(smpl_pose[i:i+sample_len, :])
            smpl_trans.append(smpl_tran[i:i+sample_len, :])
            music_librosas.append(music_librosa[i:i+sample_len, :])
            music_muqs.append(music_muq[i:i+sample_len, :])
            labels.append(label)

    smpl_poses, smpl_trans, music_librosas, music_muqs, labels = \
        np.array(smpl_poses), np.array(smpl_trans), np.array(music_librosas), np.array(music_muqs), np.array(labels)
    print(smpl_poses.shape, smpl_trans.shape, music_librosas.shape, music_muqs.shape, labels.shape)
    print('Test Dataset len: ', smpl_poses.shape[0])
    eval_dataloader = DataLoader(EvalDataset(smpl_poses, smpl_trans, music_librosas, music_muqs, labels, file_names), batch_size=batch_size, shuffle=False)
    return eval_dataloader



class EvalDataset(Dataset):
    def __init__(self, smpl_poses, smpl_trans, music_librosas, music_muqs, labels, file_names):
        
        music_librosas = torch.from_numpy(music_librosas)
        music_muqs = torch.from_numpy(music_muqs)
        smpl_root_init, smpl_root_vel = root_preprocess(torch.from_numpy(smpl_trans))
        smpl_poses = torch.from_numpy(smpl_poses)
        smpl_poses = rotation_matrix_to_rotation_6d(smpl_poses)

        self.music_muq = music_muqs
        self.music_librosa = music_librosas
        self.smpl_trans = smpl_trans
        self.smpl_poses = smpl_poses
        self.smpl_root_vel = smpl_root_vel
        self.smpl_root_init = smpl_root_init

        self.file_name = file_names
        self.label = labels
        

    def __len__(self):
        return len(self.smpl_poses)

    def __getitem__(self, idx):
        return {'smpl_trans': self.smpl_trans[idx], 'smpl_poses': self.smpl_poses[idx],  \
                'smpl_root_vel': self.smpl_root_vel[idx], 'smpl_root_init': self.smpl_root_init[idx], \
                'music_librosa': self.music_librosa[idx], 'music_muq': self.music_muq[idx], \
                'label': self.label[idx]}
