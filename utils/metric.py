import torch
import torch.nn as nn
import os
from smplx import SMPL
import torch
import numpy as np
import pickle as pkl
from utils.features.kinetic import extract_kinetic_features
from utils.features.geometric import extract_geometric_features
from utils.utils import similarity_matrix, compute_rank
from scipy import linalg
from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
from einops import rearrange
from tqdm import tqdm

class Metric():
    def __init__(self, root_dir='./data'):
        
        self.MAE_Loss = torch.nn.L1Loss()
        self.MSE_Loss = nn.MSELoss()

        self.smpl_poses_gt = []
        self.smpl_root_vel_gt = []
        self.smpl_poses_pred = []
        self.smpl_root_vel_pred = []

        self.keypoints_pred = []
        self.keypoints_gt = []

        self.dance_gt_genre_feature = []
        self.dance_pred_genre_feature = []

        self.dance_gt_kinetic_feature = []
        self.dance_pred_kinetic_feature = []

        self.dance_gt_geometric_feature = []
        self.dance_pred_geometric_feature = []

        self.music_beat = []
        self.label = []


    def update(self, result):
        self.smpl_poses_pred.append(result['smpl_poses_pred'].detach().cpu())
        self.smpl_poses_gt.append(result['smpl_poses_gt'].detach().cpu())
        self.keypoints_pred.append(result['keypoints_pred'])
        self.keypoints_gt.append(result['keypoints_gt'])
        self.dance_pred_genre_feature.append(result['dance_pred_genre_feature'].detach().cpu())
        self.dance_gt_genre_feature.append(result['dance_gt_genre_feature'].detach().cpu())
        self.dance_pred_kinetic_feature.append(result['dance_pred_kinetic_feature'])
        self.dance_gt_kinetic_feature.append(result['dance_gt_kinetic_feature'])
        self.dance_pred_geometric_feature.append(result['dance_pred_geometric_feature'])
        self.dance_gt_geometric_feature.append(result['dance_gt_geometric_feature'])
        self.music_beat.append(result['music_beat'])
        self.label.append(result['label'].detach().cpu())
        
    def result(self):
        smpl_poses_gt, smpl_poses_pred = torch.cat(self.smpl_poses_gt, dim=0), torch.cat(self.smpl_poses_pred, dim=0)
        keypoints_gt, keypoints_pred = torch.cat(self.keypoints_gt, dim=0), torch.cat(self.keypoints_pred, dim=0)
        music_beat, label = torch.cat(self.music_beat, dim=0), torch.cat(self.label, dim=0)
        dance_gt_genre_feature, dance_pred_genre_feature = torch.cat(self.dance_gt_genre_feature, dim=0), torch.cat(self.dance_pred_genre_feature, dim=0)
        dance_gt_kinetic_feature, dance_pred_kinetic_feature = torch.cat(self.dance_gt_kinetic_feature, dim=0), torch.cat(self.dance_pred_kinetic_feature, dim=0)
        dance_gt_geometric_feature, dance_pred_geometric_feature = torch.cat(self.dance_gt_geometric_feature, dim=0), torch.cat(self.dance_pred_geometric_feature, dim=0)
        
        print('Beat Similarity of Pred:', calculate_beat_similarity(music_beat, keypoints_pred))
        print('Beat Similarity of GT:', calculate_beat_similarity(music_beat, keypoints_gt))

        print('Geometric Feature:')
        dance_gt_geometric_feature, dance_pred_geometric_feature = self.normalize(dance_gt_geometric_feature, dance_pred_geometric_feature)
        print('Dance Diversity of GT:', calculate_avg_distance(dance_gt_geometric_feature))
        print('Dance Diversity of Pred:', calculate_avg_distance(dance_pred_geometric_feature))
        print('Dance FID of Pred and GT:', calc_fid(dance_pred_geometric_feature, dance_gt_geometric_feature))

        print('Kinetic Feature:')
        dance_gt_kinetic_feature, dance_pred_kinetic_feature = self.normalize(dance_gt_kinetic_feature, dance_pred_kinetic_feature)
        print('Dance Diversity of GT:', calculate_avg_distance(dance_gt_kinetic_feature))
        print('Dance Diversity of Pred:', calculate_avg_distance(dance_pred_kinetic_feature))
        print('Dance FID of Pred and GT:', calc_fid(dance_pred_kinetic_feature, dance_gt_kinetic_feature))

        print('Genre Feature:')
        print('Dance Genre Accurary of GT:', compute_acc(dance_gt_genre_feature, label))
        print('Dance Genre Accurary of Pred:', compute_acc(dance_pred_genre_feature, label))
        dance_gt_genre_feature, dance_pred_genre_feature = self.normalize(dance_gt_genre_feature, dance_pred_genre_feature)
        print('Dance Diversity of GT:', calculate_avg_distance(dance_gt_genre_feature))
        print('Dance Diversity of Pred:', calculate_avg_distance(dance_pred_genre_feature))
        print('Dance FID of Pred and GT:', calc_fid(dance_pred_genre_feature, dance_gt_genre_feature))

        
    def normalize(self, gt, pred):
        mean = gt.mean(axis=0)
        std = gt.std(axis=0)
        return (gt - mean) / (std + 1e-10), (pred - mean) / (std + 1e-10)
 

def compute_acc(pred_logits, gt_labels):

    pred_classes = torch.argmax(pred_logits, dim=1)
    

    pred_np = pred_classes.cpu().numpy()
    gt_np = gt_labels.cpu().numpy()

    correct = (pred_classes == gt_labels).sum().item()
    accuracy = correct / gt_labels.size(0)
    return accuracy

def calculate_beat_similarity(music_beat, keypoints):
    # print(music_beat, keypoint)
    music_beat, keypoints = np.array(music_beat), np.array(keypoints)
    b, t, _, _ = keypoints.shape
    ba_score = []
    for i in tqdm(range(b)):
        mb = get_mb(music_beat[i])
        db = get_db(keypoints[i])
        ba = BA(mb, db)
        ba_score.append(ba)
    return np.mean(ba_score)

def BA(music_beats, motion_beats):
    if len(music_beats) == 0:
        return 0
    ba = 0
    for bb in music_beats:
        ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
    return (ba / len(music_beats))

def get_mb(music_beats):
    t = music_beats.shape[0]
    beats = music_beats.astype(bool)
    beat_axis = np.arange(t)
    beat_axis = beat_axis[beats]
    return beat_axis

def get_db(keypoints):
    t, _, _ = keypoints.shape
    keypoints = keypoints.reshape(t, 24, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats



def calc_fid(kps_gen, kps_gt):

    kps_gt, kps_gen = np.array(kps_gt), np.array(kps_gen)

    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)

    mu1, mu2, sigma1, sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



def calculate_avg_distance(feat):
    feat = np.array(feat)
    n, c = feat.shape
    diff = feat[:, np.newaxis, :] - feat[np.newaxis, :, :]
    sq_diff = np.sum(diff**2, axis=2)
    distances = np.sqrt(sq_diff)
    total_distance = np.sum(np.triu(distances, 1))
    avg_distance = total_distance / ((n * (n - 1)) / 2)
    return avg_distance


from sklearn.metrics import precision_score, recall_score
def compute_metrics(pred_logits, gt_labels):

    # pred class
    pred_classes = torch.argmax(pred_logits, dim=1)
    

    pred_np = pred_classes.cpu().numpy()
    gt_np = gt_labels.cpu().numpy()
    
    # acc
    correct = (pred_classes == gt_labels).sum().item()
    accuracy = correct / gt_labels.size(0)
    
    # precision and recall
    precision = precision_score(gt_np, pred_np, average='macro', zero_division=0)
    recall = recall_score(gt_np, pred_np, average='macro', zero_division=0)
    
    return accuracy, precision, recall

class CLSMetric():
    def __init__(self, root_dir='./data'):
        self.dance_pred = []
        self.dance_gt = []

    def update(self, result):  
        self.dance_pred.append(result['dance_pred'].detach().cpu())
        self.dance_gt.append(result['dance_gt'].detach().cpu())    

    def result(self):
        dance_gt, dance_pred = torch.cat(self.dance_gt, dim=0), torch.cat(self.dance_pred, dim=0)
        acc, precision, recall = compute_metrics(dance_pred, dance_gt)
        print(f'Dance Genre Classification:')
        print(f'Accuracy: {acc:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {2*(precision*recall)/(precision+recall+1e-8):.4f}')  



if __name__ =='__main__':
    metric = Metric()

    with open('./data1/gt/kinetic_features.pkl', 'rb') as file:
        gt_features_k = pkl.load(file)['kinetic_features']
    
    gt_features_k = np.repeat(gt_features_k, 10, 0)

    pred_features_k = np.random.permutation(gt_features_k)
    
    gt_features_k, pred_features_k = normalize(gt_features_k, pred_features_k)
    fid_k = calc_fid(pred_features_k, gt_features_k)
    print(fid_k)