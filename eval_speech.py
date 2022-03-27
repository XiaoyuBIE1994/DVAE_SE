#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

import os
import sys
from tqdm import tqdm
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from src.utils.eval_metric import compute_median, EvalMetrics
torch.manual_seed(0)
np.random.seed(0)

# from src.learning_algo import LearningAlgorithm
from src.learning_algo_ss import LearningAlgorithm_ss as LearningAlgorithm

test_dir = './data/VoiceBankDemand/clean_testset_wav_16k/'
# test_dir = './data/clean_speech/wsj0_si_et_05'

trained_model = 'SRNN_final_epoch293.pt'

saved_model_dir = './trained_models/vb_srnn'
state_file = os.path.join(saved_model_dir, trained_model)
cfg_file = os.path.join(saved_model_dir, 'config.ini')
params = {'cfg': cfg_file}
learning_algo = LearningAlgorithm(params=params)
learning_algo.build_model()
dvae = learning_algo.model
dvae.load_state_dict(torch.load(state_file, map_location='cuda'))
eval_metrics = EvalMetrics(metric='all')
dvae.eval()
print('Total params: %.2fM' % (sum(p.numel() for p in dvae.parameters()) / 1000000.0))

list_rmse = []
list_sisdr = []
list_pesq = []
list_pesq_wb = []
list_pesq_nb = []
list_estoi = []

file_list = librosa.util.find_files(test_dir, ext='wav')
print('test audio: ', len(file_list))

tot_len = 0
for audio_file in tqdm(file_list):
    
    # print(audio_file)

    root, file = os.path.split(audio_file)
    filename, _ = os.path.splitext(file)
    recon_audio = os.path.join(ret_dir, 'recon_{}.wav'.format(filename))
    orig_audio = os.path.join(ret_dir, 'orig_{}.wav'.format(filename))

    wlen_sec = 64e-3
    hop_percent = 0.25
    fs = 16000
    zp_percent = 0
    wlen = wlen_sec * fs
    wlen = int(np.power(2, np.ceil(np.log2(wlen)))) # pwoer of 2
    hop = int(hop_percent * wlen)
    nfft = wlen + zp_percent * wlen
    win = np.sin(np.arange(0.5, wlen+0.5) / wlen * np.pi)
    trim = False

    x, fs_x = sf.read(audio_file)

    tot_len += len(x) / fs_x
    
    if trim:
        x, _ = librosa.effects.trim(x, top_db=30)
    
    #####################
    # Scaling on waveform
    #####################
    # scale = 1
    scale = np.max(np.abs(x)) # normalized by Max(|x|)
    x = x / scale

    # STFT
    X = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=wlen, window=win)

    # Prepare data input        
    data_orig = np.abs(X) ** 2 # (x_dim, seq_len)
    data_orig = torch.from_numpy(data_orig.astype(np.float32)).to(dvae.device) 
    data_orig = data_orig.permute(1,0).unsqueeze(1) #  (x_dim, seq_len) => (seq_len, 1, x_dim)

    # Reconstruction
    with torch.no_grad():
        data_recon = torch.exp(dvae(data_orig))
        # data_recon = torch.exp(dvae(data_orig, use_pred = 1))

    data_recon = data_recon.to('cpu').detach().squeeze().permute(1,0).numpy()

    # Re-synthesis
    X_recon = np.sqrt(data_recon) * np.exp(1j * np.angle(X))
    x_recon = librosa.istft(X_recon, hop_length=hop, win_length=wlen, window=win)
    
    # Wrtie audio file
    scale_norm = 1 / (np.maximum(np.max(np.abs(x_recon)), np.max(np.abs(x)))) * 0.9
    sf.write(recon_audio, scale_norm*x_recon, fs_x)
    sf.write(orig_audio, scale_norm*x, fs_x)

    rmse, sisdr, pesq, pesq_wb, pesq_nb, estoi = eval_metrics.eval(audio_est=recon_audio, audio_ref=orig_audio)
    
    # print('File: {}, rmse: {:.4f}, pesq: {:.4f}, estoi: {:.4f}'.format(filename, rmse, pesq, estoi))

    list_rmse.append(rmse)
    list_sisdr.append(sisdr)
    list_pesq.append(pesq)
    list_pesq_wb.append(pesq_wb)
    list_pesq_nb.append(pesq_nb)
    list_estoi.append(estoi)

np_rmse = np.array(list_rmse)
np_sisdr = np.array(list_sisdr)
np_pesq = np.array(list_pesq)
np_pesq_wb = np.array(list_pesq_wb)
np_pesq_nb = np.array(list_pesq_nb)
np_estoi = np.array(list_estoi)

print('Re-synthesis finished')

print("Mean evaluation")
print('mean rmse score: {:.4f}'.format(np.mean(np_rmse)))
print('mean sisdr score: {:.1f}'.format(np.mean(np_sisdr)))
print('mean pypesq wb score: {:.2f}'.format(np.mean(np_pesq)))
print('mean pesq wb score: {:.2f}'.format(np.mean(np_pesq_wb)))
print('mean pesq nb score: {:.2f}'.format(np.mean(np_pesq_nb)))
print('mean estoi score: {:.2f}'.format(np.mean(np_estoi)))

rmse_median, rmse_ci = compute_median(np_rmse)
sisdr_median, sisdr_ci = compute_median(np_sisdr)
pesq_median, pesq_ci = compute_median(np_pesq)
pesq_wb_median, pesq_wb_ci = compute_median(np_pesq_wb)
pesq_nb_median, pesq_nb_ci = compute_median(np_pesq_nb)
estoi_median, estoi_ci = compute_median(np_estoi)

print("Median evaluation")
print('median rmse score: {:.4f} +/- {:.4f}'.format(rmse_median, rmse_ci))
print('median sisdr score: {:.1f} +/- {:.1f}'.format(sisdr_median, sisdr_ci))
print('median pypesq score: {:.2f} +/- {:.2f}'.format(pesq_median, pesq_ci))
print('median pesq wb score: {:.2f} +/- {:.2f}'.format(pesq_wb_median, pesq_wb_ci))
print('median pesq nb score: {:.2f} +/- {:.2f}'.format(pesq_nb_median, pesq_nb_ci))
print('median estoi score: {:.2f} +/- {:.2f}'.format(estoi_median, estoi_ci))

print('Total length: {:.2f}'.format(tot_len/3600))