import os
import numpy as np
import pickle as pkl
import matplotlib
import json
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from mpl_toolkits.mplot3d import Axes3D
from einops import rearrange

def adjust_pose(joint):
    np_dance_trans = np.zeros([3, 25]).copy()
    joint = np.transpose(joint)

    # head
    np_dance_trans[:, 0] = joint[:, 15]
    
    #neck
    np_dance_trans[:, 1] = joint[:, 12]
    
    # left up
    np_dance_trans[:, 2] = joint[:, 16]
    np_dance_trans[:, 3] = joint[:, 18]
    np_dance_trans[:, 4] = joint[:, 20]

    # right up
    np_dance_trans[:, 5] = joint[:, 17]
    np_dance_trans[:, 6] = joint[:, 19]
    np_dance_trans[:, 7] = joint[:, 21]

    
    np_dance_trans[:, 8] = joint[:, 0]
    
    np_dance_trans[:, 9] = joint[:, 1]
    np_dance_trans[:, 10] = joint[:, 4]
    np_dance_trans[:, 11] = joint[:, 7]

    np_dance_trans[:, 12] = joint[:, 2]
    np_dance_trans[:, 13] = joint[:, 5]
    np_dance_trans[:, 14] = joint[:, 8]

    np_dance_trans[:, 15] = joint[:, 15]
    np_dance_trans[:, 16] = joint[:, 15]
    np_dance_trans[:, 17] = joint[:, 15]
    np_dance_trans[:, 18] = joint[:, 15]

    np_dance_trans[:, 19] = joint[:, 11]
    np_dance_trans[:, 20] = joint[:, 11]
    np_dance_trans[:, 21] = joint[:, 8]

    np_dance_trans[:, 22] = joint[:, 10]
    np_dance_trans[:, 23] = joint[:, 10]
    np_dance_trans[:, 24] = joint[:, 7]

    np_dance_trans = np.transpose(np_dance_trans)

    return np_dance_trans

pose_edge_list = [        
    [ 0,  1], [ 1,  8],                                         # body
    [ 1,  2], [ 2,  3], [ 3,  4],                               # right arm
    [ 1,  5], [ 5,  6], [ 6,  7],                               # left arm
    [ 8,  9], [ 9, 10], [10, 11], [11, 24], [11, 22], [22, 23], # right leg
    [ 8, 12], [12, 13], [13, 14], [14, 21], [14, 19], [19, 20]  # left leg
]
pose_color_list = [
    [153,  0, 51], [153,  0,  0],
    [153, 51,  0], [153,102,  0], [153,153,  0],
    [102,153,  0], [ 51,153,  0], [  0,153,  0],
    [  0,153, 51], [  0,153,102], [  0,153,153], [  0,153,153], [  0,153,153], [  0,153,153],
    [  0,102,153], [  0, 51,153], [  0,  0,153], [  0,  0,153], [  0,  0,153], [  0,  0,153]
]
def plot_line(joint, ax):
    for i, e in enumerate(pose_edge_list):
        ax.plot([joint[e[0]][0], joint[e[1]][0]], [joint[e[0]][1], joint[e[1]][1]], [joint[e[0]][2], joint[e[1]][2]], \
                    color=(pose_color_list[i][0]/255, pose_color_list[i][1]/255, pose_color_list[i][2]/255))

def swap(joint):
    tmp = np.zeros_like(joint)
    tmp[:, :, :, 0] = joint[:, :, :, 0]
    tmp[:, :, :, 1] = joint[:, :, :, 2]
    tmp[:, :, :, 2] = joint[:, :, :, 1]
    return tmp

def calculate_coordinate_range(joints):
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    for joint in joints:
        x = joint[:, 0]
        y = joint[:, 1]
        min_x = min(min_x, np.min(x))
        max_x = max(max_x, np.max(x))
        min_y = min(min_y, np.min(y))
        max_y = max(max_y, np.max(y))

    print('min_x, max_x, min_y, max_y', min_x, max_x, min_y, max_y)
    return min_x, max_x, min_y, max_y

def save_img(k, all_joints3d, image_path):
    elev = 20  # 仰角，0度是从X-Y平面看，90度是从上往下看
    azim = 60  # 方位角，0度是从Y轴的负方向看，90度是从X轴看
    min_lin, max_lin = np.min(all_joints3d[:, :, :, :].reshape(-1, 3), axis=0), np.max(all_joints3d[:, :, :, :].reshape(-1, 3), axis=0)
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    joints3d = all_joints3d[:, k]
    for joint in joints3d:
        joint = adjust_pose(joint[:, :3])
        ax.scatter(joint[:, 0], joint[:, 1], joint[:, 2], color='black', s=10)
        plot_line(joint, ax)
    ax.set_xlim(min_lin[0], max_lin[0])
    ax.set_ylim(min_lin[1], max_lin[1])
    ax.set_zlim(min_lin[2], max_lin[2])
    ax.view_init(elev=elev, azim=azim)
    plt.savefig(os.path.join(image_path, '{}.png'.format(k)))
    plt.cla()
    plt.close()


def vis(keypoints_path, music_path, video_path, image_path):
    
    for keypoints_file in tqdm(sorted(os.listdir(keypoints_path))):

        os.makedirs(image_path, exist_ok=True)
        print(os.path.join(keypoints_path, keypoints_file))

        video_file = video_path + '/{}'.format(keypoints_file.replace('.json', '.mp4'))
        music_file = music_path + '/{}'.format(keypoints_file.replace('.json', '.wav'))
        video_file_new = video_file.replace('.mp4', '_audio.mp4')
        if os.path.exists(video_file_new):
            continue

        # all_joints3d -> n, t, j, 3
        all_joints3d = np.array(json.load(open(os.path.join(keypoints_path, keypoints_file), "rb"))['keypoints'])
        all_joints3d = rearrange(all_joints3d, 't c1 c2 -> 1 t c1 c2')
        all_joints3d = swap(all_joints3d)

        timestep = list(range(all_joints3d.shape[1]))
        with ProcessPoolExecutor(max_workers=16) as executor:
            executor.map(save_img, timestep, [all_joints3d]*len(timestep), [image_path]*len(timestep))
        
        cmd = f"ffmpeg -r 30 -i {image_path}/%d.png -vb 20M -vcodec mpeg4 -y {video_file} -loglevel quiet"
        os.system(cmd)

        cmd_audio = f"ffmpeg -i {video_file} -i {music_file} -map 0:v -map 1:a -c:v copy -shortest -y {video_file_new} -loglevel quiet"
        os.system(cmd_audio)
        if os.path.exists(image_path):
            shutil.rmtree(image_path)
        if os.path.exists(video_file):
            os.remove(video_file)

def visual(demo_name):
    root_dir = './demo'
    mus_path = os.path.join(root_dir, '{}/music'.format(demo_name))
    keypoints_path = os.path.join(root_dir, '{}/keypoint'.format(demo_name))
    image_path = os.path.join(root_dir, '{}/image'.format(demo_name))
    video_path = os.path.join(root_dir, '{}/video'.format(demo_name))
    os.makedirs(video_path, exist_ok=True)
    vis(keypoints_path, mus_path, video_path, image_path)

if __name__ =='__main__':
    visual(demo_name='1')



