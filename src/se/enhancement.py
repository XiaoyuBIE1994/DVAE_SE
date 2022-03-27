#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt

"""
import os
import numpy as np
import torch
import pickle
import librosa
import soundfile as sf
import matplotlib.pyplot as plt 
from .em_algorithm import VEM
from src.utils import myconf
from src.model import build_VAE, build_DKF, build_RVAE
from src.model_ss import build_SRNN_ss

def set_require_grad(layer_to_optim):
    for layer in layer_to_optim:
        for para in layer.parameters():
            para.requires_grad = True


def enhance(mix_file, output_file, clean_file, saved_model='./', state_dict_file= 'model.pt', nmf_rank=25, niter=200,
            nepochs_E_step=1, nsamples_E_step=1, nsamples_WF=1, lr=1e-2, device='cpu', monitor_perf=False):
    
    # Read config
    cfg_file = os.path.join(saved_model, 'config.ini')
    cfg = myconf()
    cfg.read(cfg_file)
    beta = cfg.getfloat('Training', 'beta')

    # Build DVAE model, set only encoder to be optimzed
    dvae_type = cfg.get('Network', 'name')
    if dvae_type == 'VAE':
        dvae = build_VAE(cfg=cfg, device=device)
        layer_to_optim = [dvae.mlp_x_z, dvae.inf_mean, dvae.inf_logvar]
    elif dvae_type == 'DKF':
        dvae = build_DKF(cfg=cfg, device=device)
        layer_to_optim = [dvae.mlp_x_gx, dvae.rnn_gx, dvae.mlp_ztm1_g, dvae.mlp_g_z, dvae.inf_mean, dvae.inf_logvar]
    elif dvae_type == 'RVAE':
        dvae = build_RVAE(cfg=cfg, device=device)
        layer_to_optim = [dvae.mlp_x_gx, dvae.rnn_g_x, dvae.mlp_z_gz, dvae.rnn_g_z, dvae.mlp_g_z, dvae.inf_mean, dvae.inf_logvar]
    elif dvae_type == 'SRNN':
        dvae = build_SRNN_ss(cfg=cfg, device=device)
        layer_to_optim = [dvae.mlp_hx_g, dvae.rnn_g, dvae.mlp_gz_z, dvae.inf_mean, dvae.inf_logvar] # SRNN-GM v1
        # layer_to_optim = [dvae.mlp_x_h, dvae.rnn_h, dvae.mlp_hx_g, dvae.rnn_g, dvae.mlp_gz_z, dvae.inf_mean, dvae.inf_logvar] # SRNN-GM v2
    else:
        raise ValueError('Wrong DVAE type')

    # Load model weights and initialize the optimizer
    ckpt_file = os.path.join(saved_model, state_dict_file)
    dvae.load_state_dict(torch.load(ckpt_file, map_location=device))
    for para in dvae.parameters():
        para.requires_grad = False
    set_require_grad(layer_to_optim)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dvae.parameters()), lr=lr)

    # Read STFT params
    wlen_sec = cfg.getfloat('STFT', 'wlen_sec')
    hop_percent = cfg.getfloat('STFT', 'hop_percent')
    fs = cfg.getint('STFT', 'fs')
    zp_percent = cfg.getint('STFT', 'zp_percent')
    wlen = wlen_sec * fs
    wlen = int(np.power(2, np.ceil(np.log2(wlen)))) # pwoer of 2
    hop = int(hop_percent * wlen)
    nfft = wlen + zp_percent * wlen
    win = np.sin(np.arange(0.5, wlen+0.5) / wlen * np.pi)

    # Read mix audio
    x, fs_x = sf.read(mix_file)
    x = x / np.max(np.abs(x))
    X = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=wlen, window=win)

    # Initialize noise matrix
    F, N = X.shape
    eps = np.finfo(float).eps
    W_init = np.maximum(np.random.rand(F, nmf_rank), eps)
    H_init = np.maximum(np.random.rand(nmf_rank, N), eps)
    g_init = np.ones(N)
    
    # Load VEM algo
    vem_algo = VEM(X=X, W=W_init, H=H_init, g=g_init, dvae=dvae, optimizer=optimizer, beta=beta,
                  niter=niter, lr=lr, nepochs_E_step=nepochs_E_step,
                  nsamples_WF=nsamples_WF, device=device)

    # Run enhancement
    if monitor_perf:
        # loss, rmse, pesq, estoi = vem_algo.run_monitor_EM(fs=fs, hop=hop, wlen=wlen, win=win, audio_ref=clean_file)
        # save_pre, _ = os.path.splitext(output_file)
        # eval_pckl_file = save_pre + '_eval.pckl'
        # with open(eval_pckl_file, 'wb') as f:
        #     pickle.dump([loss, rmse, pesq, estoi], f)
        # plot_monitor_performance(loss, rmse, pesq, estoi, save_pre)
        # time_consume = 0  # temporal
        pass # TODO
    else:
        loss, time_consume = vem_algo.run_EM()
    
    S_hat = vem_algo.S_hat
    N_hat = vem_algo.N_hat

    s_hat = librosa.istft(S_hat, hop_length=hop, win_length=wlen, window=win)
    n_hat = librosa.istft(N_hat, hop_length=hop, win_length=wlen, window=win)

    sf.write(output_file, s_hat, fs_x)
    
    return loss, time_consume