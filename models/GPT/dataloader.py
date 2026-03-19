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

def read_txt_to_list(file_path):
    return [int(line.strip()) for line in open(file_path).readlines() if line.strip()]

def contains_anomaly(anomalies, start, end):
    return any(start <= p < end for p in anomalies)

    
def createTrainDataset(root_dir='./data', batch_size=32, stride=30, sample_len=240, start=0, end=1):

    with open(os.path.join(root_dir, 'train.txt'), 'r') as file:
        file_names = [line.strip() for line in file]
    smpl_poses, smpl_trans, musics, labels = [], [], [], []

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
        # label = style1s.index(json.load(label_file).get('style1'))
        anomaly_file_path = os.path.join(root_dir, 'anomaly', file_name+'.txt')
        anomaly_list = read_txt_to_list(anomaly_file_path)
        total_length = min(smpl_pose.shape[0], librosa_music.shape[0]//8*8)
        for i in range(start, total_length+end-sample_len, stride):
            if contains_anomaly(anomaly_list, i, i+sample_len):
                continue
            smpl_poses.append(smpl_pose[i:i+sample_len, :])
            smpl_trans.append(smpl_tran[i:i+sample_len, :])
            musics.append(librosa_music[i:i+sample_len:8, :])
            labels.append(label)

    smpl_poses, smpl_trans, musics, labels = np.array(smpl_poses), np.array(smpl_trans), np.array(musics), np.array(labels)
    print(smpl_poses.shape, smpl_trans.shape, musics.shape, labels.shape)
    print('Train Dataset len: ', smpl_poses.shape[0])
    train_dataloader = DataLoader(TrainDataset(smpl_poses, smpl_trans, musics, labels, file_names), batch_size=batch_size, shuffle=True, num_workers=16)
    return train_dataloader


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
        # label = style1s.index(json.load(label_file).get('style1'))
        anomaly_file_path = os.path.join(root_dir, 'anomaly', file_name+'.txt')
        anomaly_list = read_txt_to_list(anomaly_file_path)

        total_length = min(smpl_pose.shape[0], librosa_music.shape[0], muq_music.shape[0])//8*8
        for i in range(start, total_length+end-sample_len, stride):
            if contains_anomaly(anomaly_list, i, i+sample_len):
                continue
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


def createComparisonDataset(root_dir='./data', baseline='Lodge', batch_size=32):
    baseline_dir = os.path.join(root_dir, baseline)
    gt_dir = os.path.join(root_dir, 'GT')
    music_dir = os.path.join(root_dir, '../music', 'librosa')
    label_dir = os.path.join(root_dir, '../label')
    file_names = os.listdir(baseline_dir)
    smpl_poses_preds, smpl_trans_preds, smpl_poses_gts, smpl_trans_gts, musics_librosa, labels = [], [], [], [], [], []
    for file_name in tqdm(file_names):
        motion_file_path_pred = os.path.join(baseline_dir, file_name)
        motion_file_pred = open(motion_file_path_pred, 'rb')
        motion_data_pred = pickle.load(motion_file_pred)
        smpl_pose_pred, smpl_tran_pred = motion_data_pred['smpl_poses'], motion_data_pred['smpl_trans']
        smpl_poses_preds.append(smpl_pose_pred)
        smpl_trans_preds.append(smpl_tran_pred)

        motion_file_path_gt = os.path.join(gt_dir, file_name)
        motion_file_gt = open(motion_file_path_gt, 'rb')
        motion_data_gt = pickle.load(motion_file_gt)
        smpl_pose_gt, smpl_tran_gt = motion_data_gt['smpl_poses'], motion_data_gt['smpl_trans']
        smpl_poses_gts.append(smpl_pose_gt)
        smpl_trans_gts.append(smpl_tran_gt)

        librosa_file_path = os.path.join(music_dir, file_name)
        librosa_file = open(librosa_file_path, 'rb')
        music_librosa = pickle.load(librosa_file)['music']
        musics_librosa.append(music_librosa)

        label_file_path = os.path.join(label_dir, file_name.split('_')[0]+'.json')
        label_file = open(label_file_path, 'r')
        label = style2s.index(json.load(label_file).get('style2'))
        labels.append(label)

    smpl_poses_preds, smpl_trans_preds, smpl_poses_gts, smpl_trans_gts, musics_librosa, labels = \
        np.array(smpl_poses_preds), np.array(smpl_trans_preds), np.array(smpl_poses_gts), np.array(smpl_trans_gts), np.array(musics_librosa), np.array(labels)
    print(smpl_poses_preds.shape, smpl_trans_preds.shape, smpl_poses_gts.shape, smpl_trans_gts.shape, musics_librosa.shape)
    print('Comparison Dataset len: ', smpl_poses_preds.shape[0])
    eval_dataloader = DataLoader(ComparisonDataset(smpl_poses_preds, smpl_trans_preds, smpl_poses_gts, smpl_trans_gts, musics_librosa, labels), batch_size=batch_size, shuffle=False)
    return eval_dataloader


class TrainDataset(Dataset):
    def __init__(self, smpl_poses, smpl_trans, music_librosa, labels, file_names):

        music_librosa = torch.from_numpy(music_librosa)
        smpl_trans = torch.from_numpy(smpl_trans)
        smpl_root_init, smpl_root_vel = root_preprocess(smpl_trans)
        smpl_poses = rotation_matrix_to_rotation_6d(torch.from_numpy(smpl_poses))
        
        music_librosa_mean, music_librosa_std = torch.mean(music_librosa, dim=(0, 1, 2)), torch.std(music_librosa, dim=(0, 1, 2))

        smpl_trans_mean, smpl_trans_std = torch.mean(smpl_trans, dim=(0, 1, 2)), torch.std(smpl_trans, dim=(0, 1, 2))
        smpl_poses_mean, smpl_poses_std = torch.mean(smpl_poses, dim=(0, 1, 2)), torch.std(smpl_poses, dim=(0, 1, 2))
        smpl_root_vel_mean, smpl_root_vel_std = torch.mean(smpl_root_vel, dim=(0, 1, 2)), torch.std(smpl_root_vel, dim=(0, 1, 2))

        # torch.save({
        #     'music_librosa_mean': music_librosa_mean,
        #     'smpl_trans_mean': smpl_trans_mean,
        #     'smpl_poses_mean': smpl_poses_mean,
        #     'smpl_root_vel_mean': smpl_root_vel_mean,
        # }, './Pretrained/mean.pt')
        # torch.save({
        #     'music_librosa_std': music_librosa_std,
        #     'smpl_trans_std': smpl_trans_std,
        #     'smpl_poses_std': smpl_poses_std,
        #     'smpl_root_vel_std': smpl_root_vel_std,
        # }, './Pretrained/std.pt')

        self.music_librosa = music_librosa
        self.smpl_trans = smpl_trans
        self.smpl_poses = smpl_poses
        self.smpl_root_vel = smpl_root_vel
        self.smpl_root_init = smpl_root_init
        self.filenames = file_names
        self.labels = labels


    def __len__(self):
        return len(self.smpl_poses)
    
    def __getitem__(self, idx):
        smpl_trans = self.smpl_trans[idx]
        smpl_poses = self.smpl_poses[idx]
        smpl_root_vel = self.smpl_root_vel[idx]
        music_librosa = self.music_librosa[idx]
        label = self.labels[idx]

        return {'smpl_trans': smpl_trans, 'smpl_poses': smpl_poses, 'smpl_root_init': self.smpl_root_init[idx], \
                'smpl_root_vel': smpl_root_vel, 'music_librosa': music_librosa, 'label': label, 'keypoint': torch.zeros(1)}

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

class ComparisonDataset(Dataset):
    def __init__(self, smpl_poses_preds, smpl_trans_preds, smpl_poses_gts, smpl_trans_gts, musics_librosa, label):
        
        
        self.smpl_poses_preds = rotation_matrix_to_rotation_6d(torch.from_numpy(smpl_poses_preds))
        self.smpl_trans_preds = torch.from_numpy(smpl_trans_preds)
        self.smpl_poses_gts = rotation_matrix_to_rotation_6d(torch.from_numpy(smpl_poses_gts))
        self.smpl_trans_gts = torch.from_numpy(smpl_trans_gts)
        self.musics_librosa = torch.from_numpy(musics_librosa)
        self.labels = label

    def __len__(self):
        return len(self.smpl_poses_preds)

    def __getitem__(self, idx):
        return {'smpl_poses_pred': self.smpl_poses_preds[idx], 'smpl_trans_pred': self.smpl_trans_preds[idx], \
                'smpl_poses_gt': self.smpl_poses_gts[idx], 'smpl_trans_gt': self.smpl_trans_gts[idx], \
                'music_librosa': self.musics_librosa[idx], 'label': self.labels[idx]}