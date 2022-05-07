#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inria.fr)
License agreement in LICENSE.txt
"""

import os
import sys
import argparse
from tqdm import tqdm
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from src.utils import myconf
from src.utils.eval_metric import compute_median, EvalMetrics
from src.model import build_VAE, build_DKF, build_SRNN, build_RVAE
from src.model_ss import build_SRNN_ss

torch.manual_seed(0)
np.random.seed(0)


def run(dvae, file_list, eval_metrics, STFT_dict):
    list_rmse = []
    list_sisdr = []
    list_pesq = []
    list_pesq_wb = []
    list_pesq_nb = []
    list_estoi = []

    for audio_file in tqdm(file_list):
        
        # print(audio_file)

        root, file = os.path.split(audio_file)
        filename, _ = os.path.splitext(file)

        nfft = STFT_dict['nfft']
        hop = STFT_dict['hop']
        wlen = STFT_dict['wlen']
        win - STFT_dict['win']
        trim = STFT_dict['trim']

        x, fs_x = sf.read(audio_file)
        
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
        # scale_norm = 1 / (np.maximum(np.max(np.abs(x_recon)), np.max(np.abs(x)))) * 0.9
        # sf.write(recon_audio, scale_norm*x_recon, fs_x)
        # sf.write(orig_audio, scale_norm*x, fs_x)

        rmse, sisdr, pesq, pesq_wb, pesq_nb, estoi = eval_metrics.eval(x_est=x_recon, x_ref=x, fs=fs_x)
        
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_dict', type=str, default=None, help='pretrained model state')
    parser.add_argument('--testset', type=str, default='wsj', choices=['wsj', 'vb'], help='test on wsj or voicebank')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='use cuda or cpu')
    args = parser.parse_args()

    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    eval_metrics = EvalMetrics(metric='all')


    # File path config
    if args.testset == 'wsj':
        file_list = librosa.util.find_files('./data/clean_speech/wsj0_si_et_05', ext='wav')
    elif args.testset == 'vb':
        file_list = librosa.util.find_files('./data/VoiceBankDemand/clean_testset_wav_16k', ext='wav')
    print(f'Test on {args.testset}, totl audio files {len(file_list)}')

    # load DVAE model
    state_file = args.state_dict
    cfg_file = os.path.join(os.path.dirname(state_file), 'config.ini')
    cfg = myconf()
    cfg.read(cfg_file)
    model_name = cfg.get('Network', 'name')

    if model_name == 'VAE':
        dvae = build_VAE(cfg=cfg, device=device)
    elif model_name == 'DKF':
        dvae = build_DKF(cfg=cfg, device=device)
    elif model_name == 'RVAE':
        dvae = build_RVAE(cfg=cfg, device=device)
    elif model_name == 'SRNN':
        dvae = build_SRNN_ss(cfg=cfg, device=device)

    dvae.load_state_dict(torch.load(state_file, map_location=device))
    dvae.eval()
    print(f'Evaluate model: {model_name}')
    print('Total params: %.2fM' % (sum(p.numel() for p in dvae.parameters()) / 1e6))

    # load STFT params

    wlen_sec = cfg.getfloat('STFT', 'wlen_sec')
    hop_percent = cfg.getfloat('STFT', 'hop_percent')
    fs = cfg.getint('STFT', 'fs')
    zp_percent = cfg.getint('STFT', 'zp_percent')
    wlen = wlen_sec * fs
    wlen = int(np.power(2, np.ceil(np.log2(wlen)))) # pwoer of 2
    hop = int(hop_percent * wlen)
    nfft = wlen + zp_percent * wlen
    win = np.sin(np.arange(0.5, wlen+0.5) / wlen * np.pi)
    trim = cfg.getboolean('STFT', 'trim')

    STFT_dict = {}
    STFT_dict['nfft'] = nfft
    STFT_dict['hop'] = hop
    STFT_dict['wlen'] = wlen
    STFT_dict['win'] = win
    STFT_dict['trim'] = trim

    print('='*80)
    print('STFT params')
    print(f'fs: {fs}')
    print(f'wlen: {wlen}')
    print(f'hop: {hop}')
    print(f'nfft: {nfft}')
    print('='*80)

    run(dvae, file_list, eval_metrics, STFT_dict)
