import torch
import torch.nn as nn
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
import json

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
    smpl_poses, smpl_trans, musics, muqs, labels = [], [], [], [], []
    file_name_plus = []

    for file_name in tqdm(file_names):
        motion_file_path = os.path.join(root_dir, 'motion', file_name+'.pkl')
        motion_file = open(motion_file_path, 'rb')
        motion_data = pickle.load(motion_file)
        smpl_pose, smpl_tran = motion_data['smpl_poses'], motion_data['smpl_trans']
        librosa_file_path = os.path.join(root_dir, 'librosa', file_name+'.pkl')
        librosa_file = open(librosa_file_path, 'rb')
        librosa_music = pickle.load(librosa_file)['music']
        muq_file_path = os.path.join(root_dir, 'muq', file_name+'.pkl')
        muq_file = open(muq_file_path, 'rb')
        muq_music = pickle.load(muq_file)['music']
        label_file_path = os.path.join(root_dir, 'label', file_name+'.json')
        label_file = open(label_file_path, 'r')
        label = style2s.index(json.load(label_file).get('style2'))

        total_length = min(smpl_pose.shape[0], librosa_music.shape[0], muq_music.shape[0])//8*8
        for i in range(start, total_length+end-sample_len, stride):
            smpl_poses.append(smpl_pose[i:i+sample_len, :])
            smpl_trans.append(smpl_tran[i:i+sample_len, :])
            musics.append(librosa_music[i:i+sample_len, :])
            muqs.append(muq_music[i:i+sample_len, :])
            labels.append(label)
            file_name_plus.append(file_name)

    smpl_poses, smpl_trans, musics, muqs, labels = np.array(smpl_poses), np.array(smpl_trans), np.array(musics), np.array(muqs), np.array(labels)
    print(smpl_poses.shape, smpl_trans.shape, musics.shape, muqs.shape, labels.shape)
    print('Eval Dataset len: ', smpl_poses.shape[0])
    eval_dataloader = DataLoader(EvalDataset(smpl_poses, smpl_trans, musics, muqs, labels, file_name_plus), batch_size=batch_size, shuffle=False)
    return eval_dataloader


def createDemoDataset(root_dir='./data', batch_size=1):
    music_path = os.path.join(root_dir, 'music')
    librosa_path = os.path.join(root_dir, 'librosa')
    keypoint_path = os.path.join(root_dir, 'keypoint')
    smpl_path = os.path.join(root_dir, 'smpl')
    os.makedirs(librosa_path, exist_ok=True)
    os.makedirs(keypoint_path, exist_ok=True)
    os.makedirs(smpl_path, exist_ok=True)
    from utils.librosa_extraction.get_librosa import extract_librosa
    extract_librosa(music_path, librosa_path)
    file_names = os.listdir(music_path)
    smpl_poses, smpl_trans, musics, labels = [], [], [], []
    file_names_plus = []
    for file_name in tqdm(file_names):
        motion_file_path = os.path.join('./Pretrained/Init1.pkl')
        with open(motion_file_path, 'rb') as file:
            data = pickle.load(file)
            smpl_pose, smpl_tran = data['smpl_poses'], data['smpl_trans']
            
        librosa_file = open(os.path.join(librosa_path, file_name.replace('.wav', '.pkl')), 'rb')
        librosa_music = pickle.load(librosa_file)['music']
        min_len = librosa_music.shape[0]//8*8
        for label in style2s:
            smpl_poses.append(smpl_pose)
            smpl_trans.append(smpl_tran)
            # label_file_path = os.path.join(root_dir, 'label', file_name.replace('.wav', '.json'))
            # label_file = open(label_file_path, 'r')
            # label = style2s.index(json.load(label_file).get('style2'))
            # label = style1s.index(json.load(label_file).get('style1'))
            musics.append(librosa_music[:min_len])
            labels.append(style2s.index(label))
            file_names_plus.append(file_name.replace('.wav', f'_{label}.wav'))

    smpl_poses, smpl_trans, musics, labels = np.array(smpl_poses), np.array(smpl_trans), np.array(musics), np.array(labels)
    print(smpl_poses.shape, smpl_trans.shape, musics.shape, labels.shape)
    print('Demo Dataset len: ', smpl_poses.shape[0])

    demo_dataloader = DataLoader(DemoDataset(smpl_poses, smpl_trans, musics, labels, file_names_plus, root_dir), batch_size=batch_size, shuffle=False)
    return demo_dataloader



class EvalDataset(Dataset):
    def __init__(self, smpl_poses, smpl_trans, music_librosa, muq, labels, file_names):
        
        music_muq = torch.from_numpy(muq)
        music_librosa = torch.from_numpy(music_librosa)
        smpl_root_init, smpl_root_vel = root_preprocess(torch.from_numpy(smpl_trans))
        smpl_poses = torch.from_numpy(smpl_poses)
        smpl_poses = rotation_matrix_to_rotation_6d(smpl_poses)

        self.music_muq = muq
        self.music_librosa = music_librosa
        self.smpl_trans = smpl_trans
        self.smpl_poses = smpl_poses
        self.smpl_root_vel = smpl_root_vel
        self.smpl_root_init = smpl_root_init

        self.file_names = file_names
        self.labels = labels
        

    def __len__(self):
        return len(self.smpl_poses)

    def __getitem__(self, idx):

        return {'smpl_trans': self.smpl_trans[idx], 'smpl_poses': self.smpl_poses[idx],  \
                'smpl_root_vel': self.smpl_root_vel[idx], 'smpl_root_init': self.smpl_root_init[idx], \
                'music_librosa': self.music_librosa[idx], 'music_muq': self.music_muq[idx], \
                'file_name': self.file_names[idx], 'label': self.labels[idx], 'keypoint': torch.zeros(1)}


class DemoDataset(Dataset):
    def __init__(self, smpl_poses, smpl_trans, music_librosa, labels, file_names, root_dir):
        
        smpl_root_init, smpl_root_vel = root_preprocess(torch.from_numpy(smpl_trans))
        smpl_poses = torch.from_numpy(smpl_poses)
        smpl_poses = rotation_matrix_to_rotation_6d(smpl_poses)

        self.music_librosa = music_librosa
        self.smpl_trans = smpl_trans
        self.smpl_poses = smpl_poses
        self.smpl_root_vel = smpl_root_vel
        self.smpl_root_init = smpl_root_init

        self.file_names = file_names
        self.labels = labels
        self.root_dir = root_dir
        

    def __len__(self):
        return len(self.smpl_poses)

    def __getitem__(self, idx):
        return {'smpl_trans': self.smpl_trans[idx], 'smpl_poses': self.smpl_poses[idx],  \
                'smpl_root_vel': self.smpl_root_vel[idx], 'smpl_root_init': self.smpl_root_init[idx], \
                'music_librosa': self.music_librosa[idx], 'file_name': self.file_names[idx], \
                'label': self.labels[idx], 'root_dir': self.root_dir, 'keypoint': torch.zeros(1)}

