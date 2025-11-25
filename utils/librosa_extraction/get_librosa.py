# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


import os
import sys
import json
import random
import argparse
import essentia
import essentia.streaming
from essentia.standard import *
import librosa
import numpy as np
from utils.librosa_extraction.extractor import FeatureExtractor
import pickle
from tqdm import tqdm
import torch

extractor = FeatureExtractor()

def extract_librosa(input_dir, output_dir):

    
    sampling_rate = 15360
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print('---------- Extract features from raw audio ----------')
    musics = []
    dances = []
    fnames = []
    train = []
    test = []

    audio_fnames = sorted(os.listdir(input_dir))

    ii = 0
    for audio_fname in tqdm(audio_fnames):
        video_file = audio_fname
        if os.path.exists(os.path.join(output_dir, video_file.replace('.wav', '.pkl'))):
            continue
        sr = sampling_rate
        loader = essentia.standard.MonoLoader(filename=os.path.join(input_dir, video_file), sampleRate=sr)
        audio = loader()
        audio = np.array(audio).T
        feature = extract_acoustic_feature(audio, sr)
        filename = os.path.join(output_dir, audio_fname.replace('.wav', '.pkl'))
        with open(filename, 'wb') as file:
            pickle.dump({'music': feature}, file)


def extract_acoustic_feature(audio, sr):

    melspe_db = extractor.get_melspectrogram(audio, sr) 
    mfcc = extractor.get_mfcc(melspe_db)
    mfcc_delta = extractor.get_mfcc_delta(mfcc)
    audio_harmonic, audio_percussive = extractor.get_hpss(audio)
    chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, sr,  octave=4)
    if chroma_cqt.shape[1] > mfcc.shape[1]:
        chroma_cqt = chroma_cqt[:, :-1]

    rms = extractor.get_rms(audio)
    onset_env = extractor.get_onset_strength(audio_percussive, sr)
    # tempogram = extractor.get_tempogram(onset_env, sr)
    onset_beat, onset_peak = extractor.get_onset_beat(onset_env, sr)
    onset_env = onset_env.reshape(1, -1)

    feature = np.concatenate([
        mfcc, # 20
        # mfcc_delta, # 20
        chroma_cqt, # 12
        onset_env, # 1
        onset_beat, # 1
        onset_peak, # 1
        # rms, # 1
        # tempogram
    ], axis=0)

    feature = feature.transpose(1, 0)
    print(f'acoustic feature -> {feature.shape}')

    return feature


if __name__ == '__main__':
    extract_librosa('./data/music/input_music_seg', './data/music/librosa') 
