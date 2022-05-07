#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inria.fr)
License agreement in LICENSE.txt
"""


import os
import json
import torch
import random
import numpy as np
import argparse
from datetime import datetime
from src.utils import get_logger, EvalMetrics
from src.se.enhancement import enhance

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def speech_enhance(params):

    # Load file json
    with open(params['json_file'], 'r') as f:
        dataset = json.load(f)

    # Init logger
    log_filename = os.path.join(params['ckpt_dir'], 'log_{}.txt'.format(params['exp_name']))
    logger = get_logger(log_filename, params['log_type'])
    logger.info('============================================')
    logger.info('========= Speech Enhancement Start =========')
    logger.info('============================================')
    
    # Init evaluation
    eval_metrics = EvalMetrics(metric='all')

    for ind_mix, mix_info in dataset.items():

        utt_name = mix_info['utt_name']
        mix_file = mix_info['noisy_wav'].format(noisy_root=params['noisy_root'])
        clean_file = mix_info['clean_wav'].format(clean_root=params['clean_root'])
        recon_file = os.path.join(params['enhance_dir'], utt_name + '.wav')
        
        start_time = datetime.now()

        # Enhance algo, clean_file only used if we run monitor performance
        loss, time_consume = enhance(mix_file=mix_file, output_file=recon_file, clean_file='',
                                     saved_model=params['saved_model'], state_dict_file=params['state_dict_file'], 
                                     nmf_rank=params['nmf_rank'], niter=params['niter'], nepochs_E_step=params['nepochs_E_step'], 
                                     nsamples_E_step=params['nsamples_E_step'], nsamples_WF=params['nsamples_WF'], lr=params['lr'],
                                     device=params['device'])

        x_recon, fs_x = sf.read(recon_file)
        x_ref, _ = sf.read(clean_file)
        rmse, sisdr, pesq, pesq_wb, pesq_nb, estoi = eval_metrics.eval(x_est=x_recon, x_ref=x, fs=fs_x)
        end_time = datetime.now()
        elapsed = (end_time - start_time).seconds

        logger.info("File: {}\t rmse: {:.4f}\t sisdr: {:.2f}\t pypesq: {:.2f}\t pesq wb: {:.2f}\t pesq nb: {:.2f}\t estoi: {:.2f}\t time: {:.1f}s".format(utt_name, rmse, sisdr, pesq, pesq_wb, pesq_nb, estoi, elapsed))

        dataset[ind_mix]['time_cost'] = elapsed
        dataset[ind_mix]['rmse'] = rmse
        dataset[ind_mix]['sisdr'] = sisdr
        dataset[ind_mix]['pesq'] = pesq
        dataset[ind_mix]['pesq_wb'] = pesq_wb
        dataset[ind_mix]['pesq_nb'] = pesq_nb
        dataset[ind_mix]['estoi'] = estoi

    # Write eval. resuls of audios
    json_filename = os.path.join(params['ckpt_dir'], 'log_{}.json'.format(params['exp_name']))
    with open(json_filename, 'w') as f:
        json.dump(dataset, f, indent=1)


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        self.parser.add_argument('--exp_name', type=str, default='p232', help='experiment name')
        self.parser.add_argument('--dataset', type=str, default='VB', choices=['VB', 'WSJ'], help='dataset')
        self.parser.add_argument('--saved_model', type=str, default = './', help='name for saved model')
        self.parser.add_argument('--state_dict_file', type=str, default = 'dvae_final_epochXXX.pt', help='dvae trained state')
        self.parser.add_argument('--json_file', type=str, default='VoiceBankDemand_testset.json', help='json file for audios to be enhanced')
        self.parser.add_argument('--ckpt_dir', type=str, default='/tmp', help='path for checkpoint')
        self.parser.add_argument('--enhance_dir', type=str, default='/tmp', help='path to denoised data')
        self.parser.add_argument('--log_type', type=int, default = 1, choices=[1, 2], help='1 file, 2 stream')
        self.parser.add_argument('--niter', type=int, default = 1, help='iterations for VEM')
        self.parser.add_argument('--nmf_rank', type=int, default = 1, help='NMF rank')
        self.parser.add_argument('--nepochs_E_step', type=int, default = 1, help='training iters for E-step')
        self.parser.add_argument('--nsamples_E_step', type=int, default = 1, help='sampling number for E-step')
        self.parser.add_argument('--nsamples_WF', type=int, default = 1, help='sampling in Wienner filter')
        self.parser.add_argument('--lr', type=float, default = 1e-3, help='learning rate in E-step')
        self.parser.add_argument('--device', type=str, default = 'cpu', help='cpu or cuda')

    def get_params(self):
        self._initial()
        self.opt = self.parser.parse_args()
        params = vars(self.opt)
        return params


if __name__ == '__main__':

    params = Options().get_params()
    setup_seed(123456789)

    if params['dataset'] == 'VB':
        params['noisy_root'] = './data/VoiceBankDemand/noisy_testset_wav_16k'
        params['clean_root'] = './data/VoiceBankDemand/clean_testset_wav_16k'
    elif params['dataset'] == 'WSJ':
        params['noisy_root'] = './data/QUT_WSJ0/test'
        params['clean_root'] = './data/wsj_clean/wsj0_si_et_05'

    if not os.path.isdir(params['ckpt_dir']):
        os.makedirs(params['ckpt_dir'], exist_ok=True)
    if not os.path.isdir(params['enhance_dir']):
        os.makedirs(params['enhance_dir'], exist_ok=True)
    
    speech_enhance(params)
