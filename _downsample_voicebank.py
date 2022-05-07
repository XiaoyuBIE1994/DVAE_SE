#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inria.fr)
License agreement in LICENSE.txt
"""

import os
import argparse
import librosa
import soundfile as sf
from tqdm import tqdm

wavdir = '/mnt/beegfs/perception/xbie/VoiceBankDemand/clean_trainset_28spk_wav'
traindir = '/mnt/beegfs/perception/xbie/VoiceBankDemand/clean_trainset_26spk_wav_16k'
valdir = '/mnt/beegfs/perception/xbie/VoiceBankDemand/clean_valset_2spk_wav_16k/'

file_list = librosa.util.find_files(wavdir, ext='wav')

for wavfile in tqdm(file_list):
    path, filename = os.path.split(wavfile)
    speaker = filename[1:4]
    x, fs = sf.read(wavfile)
    y = librosa.resample(x, fs, 16000)
    new_filename = filename[:-4] + '_16k.wav'
    
    if speaker == '226' or speaker == '287':
        filepath = os.path.join(valdir, new_filename)
    else:
        filepath = os.path.join(traindir, new_filename)

    sf.write(filepath, y, 16000)


wavdir = '/mnt/beegfs/perception/xbie/VoiceBankDemand/clean_testset_wav'
testdir = '/mnt/beegfs/perception/xbie/VoiceBankDemand/clean_testset_wav_16k'

file_list = librosa.util.find_files(wavdir, ext='wav')

for wavfile in tqdm(file_list):
    path, filename = os.path.split(wavfile)
    speaker = filename[1:4]
    x, fs = sf.read(wavfile)
    y = librosa.resample(x, fs, 16000)
    new_filename = filename[:-4] + '_16k.wav'
    filepath = os.path.join(testdir, new_filename)
    sf.write(filepath, y, 16000)
