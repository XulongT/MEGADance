import numpy as np
import pickle as pkl
import pickle
import sys
from features.kinetic import extract_kinetic_features
from features.manual import extract_manual_features
import os
from smplx import SMPL
import torch
from tqdm import tqdm 
import json
from einops import rearrange
from utils.utils import keypoint_preprocess, keypoint_postprocess 
# smpl = SMPL('./Pretrained/SMPL/SMPL_MALE.pkl', batch_size=1).cuda()

def gt_process(root_dir):
    gt_dir = os.path.join(root_dir, 'gt')
    os.makedirs(gt_dir, exist_ok=True)
    kinetic_feature_path = os.path.join(gt_dir, 'kinetic_features.pkl')
    manual_feature_path = os.path.join(gt_dir, 'manual_features.pkl')

    keypoint_input_path = os.path.join(root_dir, 'test')
    keypoint_output_path = os.path.join(gt_dir, 'Keypoints')
    os.makedirs(keypoint_output_path, exist_ok=True)

    total_keypoints = []
    kinetic_features, manual_features = [], []
    for file_name in tqdm(sorted(os.listdir(keypoint_input_path))):
        file_path = os.path.join(keypoint_input_path, file_name)
        file =  open(file_path, 'r')
        keypoints = np.array(json.load(file)['keypoints'])[:, :1200, :]
        for i in range(keypoints.shape[0]):
            _, t, _ = keypoints.shape
            keypoint = keypoints[i, :t//8*8, :]
            with open(os.path.join(keypoint_output_path, file_name.replace('.json', '_{}.pkl').format(i)), 'wb') as f:
                pickle.dump({'keypoints': keypoint}, f)    
            total_keypoints.append(keypoint)

    kinetic_features, manual_features = calculate_pred_features_k(total_keypoints)
    print(kinetic_features.shape, manual_features.shape)

    with open(kinetic_feature_path, 'wb') as file:
        pickle.dump({'kinetic_features': kinetic_features}, file)
    with open(manual_feature_path, 'wb') as file:
        pickle.dump({'manual_features': manual_features}, file)


from concurrent.futures import ProcessPoolExecutor
def calculate_features_for_keypoint(keypoint):
    root = keypoint[:1, :3].copy()
    keypoint = keypoint - np.tile(root, (1, 24))
    keypoint = keypoint.reshape(-1, 24, 3)
    kinetic_feature = np.array(extract_kinetic_features(keypoint))
    manual_feature = np.array(extract_manual_features(keypoint))
    return kinetic_feature, manual_feature


def calculate_pred_features_k(keypoints):
    kinetic_features, manual_features = [], []
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(calculate_features_for_keypoint, keypoints), total=len(keypoints)))
    kinetic_features, manual_features = zip(*results)
    return np.array(kinetic_features), np.array(manual_features)

if __name__ =='__main__':
    gt_dir = './data'
    gt_process(gt_dir)