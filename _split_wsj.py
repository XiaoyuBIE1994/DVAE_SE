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

data_dir = './data/QUT_WSJ0/test/'
json_dir = './data/QUT_WSJ0/json_acc/'

if not os.path.isdir(json_dir):
    os.makedirs(json_dir, exist_ok=True)

"""
split each speaker into 10 sub-set
"""
log_test = './data/QUT_WSJ0/QUT_WSJ0_test_dataset.json'
with open(log_test, 'r') as f:
    dataset = json.load(f)

idx = 0
subdata = {}
print('Total length',  len(dataset))
for i, val in enumerate(dataset, start=1):    
    filename = val['utt_name']
    p_id = filename[0:3]
    noise_type = val['noise_env']
    noise_level = val['snr']
    noisy_name = f'{filename}_{noise_type}_{noise_level}'
    clean_name = f'{p_id}/{filename}'

    mix_info = {}
    mix_info['p_id'] = p_id
    mix_info['utt_name'] = filename
    mix_info['noisy_wav'] = f'{{noisy_root}}/{noisy_name}.wav'
    mix_info['clean_wav'] = f'{{clean_root}}/{clean_name}.wav'
    mix_info['noise_type'] = noise_type
    mix_info['snr'] = float(noise_level)


    subdata[filename] = mix_info

    if i % 8 == 0 or i == len(dataset):
        filename = 'QUT_test_sub{}.json'.format(idx)
        print('Saving ', idx, filename)
        sub_json = json_dir + filename
        with open(sub_json, 'w') as f:
            json.dump(subdata, f, indent=1)
        subdata = {}
        idx += 1