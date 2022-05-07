#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inria.fr)
License agreement in LICENSE.txt
"""

import os
import json
import soundfile as sf

data_dir = './data/VoiceBankDemand/noisy_testset_wav_16k/'
log_file = './data/VoiceBankDemand/log_testset.txt'
log_json_257 = './data/VoiceBankDemand/VoiceBankDemand_testset_p257.json'
log_json_232 = './data/VoiceBankDemand/VoiceBankDemand_testset_p232.json'
json_dir = './data/VoiceBankDemand/json_acc/'

if not os.path.isdir(json_dir):
    os.makedirs(json_dir, exist_ok=True)

"""
write p232 and p257 in global
"""
p_id = ['257', '232']
json_p257 = {}
json_p232 = {}

with open(log_file, 'r') as f:
    log_testset = f.readlines()

for log_info in log_testset:
    filename, noise_type, noise_level = log_info.strip().split(' ')
    p_id = filename[1:4]
    filename16k = filename + '_16k.wav'

    mix_info = {}
    mix_info['p_id'] = p_id
    mix_info['utt_name'] = filename
    mix_info['noisy_wav'] = f'{{noisy_root}}/{filename16k}'
    mix_info['clean_wav'] = f'{{clean_root}}/{filename16k}'
    mix_info['noise_type'] = noise_type
    mix_info['snr'] = float(noise_level)

    x, fx = sf.read(mix_info['noisy_wav'].format(noisy_root=data_dir))
    len_audio = len(x) / 16000
    mix_info['length'] = len_audio

    if p_id == '257':
        json_p257[filename] = mix_info
    elif p_id == '232':
        json_p232[filename] = mix_info

with open(log_json_257, 'w') as f:
    print('p257, total audio: ', len(json_p257))
    json.dump(json_p257, f, indent=1)

with open(log_json_232, 'w') as f:
    print('p232, total audio: ', len(json_p232))
    json.dump(json_p232, f, indent=1)

"""
split each speaker into 40 sub-set
- p232 has 393 audio files
- p257 has 431 audio files
"""
with open(log_json_232, 'r') as f:
    dataset = json.load(f)

idx = 0
subdata = {}
print('Total length',  len(dataset))
for i, (key, value) in enumerate(dataset.items(), start=1):
    
    subdata[key] = value
    if i % 10 == 0 or i == len(dataset):
        filename = 'p232_sub{}.json'.format(idx)
        print('Saving ', idx, filename)
        sub_json = json_dir + filename
        with open(sub_json, 'w') as f:
            json.dump(subdata, f, indent=1)
        subdata = {}
        idx += 1

with open(log_json_257, 'r') as f:
    dataset = json.load(f)

idx = 0
subdata = {}
print('Total length',  len(dataset))
for i, (key, value) in enumerate(dataset.items(), start=1):

    subdata[key] = value
    if i % 11 == 0 or i == len(dataset):
        filename = 'p257_sub{}.json'.format(idx)
        print('Saving ', idx, filename)
        sub_json = json_dir + filename
        with open(sub_json, 'w') as f:
            json.dump(subdata, f, indent=1)
        subdata = {}
        idx += 1