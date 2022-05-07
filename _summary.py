#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inria.fr)
License agreement in LICENSE.txt
"""

import os
import argparse
import json
import numpy as np
from src.utils import compute_median

def run(json_files):
    list_rmse = []
    list_sisdr = []
    list_pesq = []
    list_pesq_wb = []
    list_pesq_nb = []
    list_estoi = []

    for filename in json_files:
        with open(filename, 'r') as f:
            dataset = json.load(f)
        for audio_file, se_log in dataset.items():
            list_rmse.append(se_log['rmse'])
            list_sisdr.append(se_log['sisdr'])
            list_pesq.append(se_log['pesq'])
            list_pesq_wb.append(se_log['pesq_wb'])
            list_pesq_nb.append(se_log['pesq_nb'])
            list_estoi.append(se_log['estoi'])
    
    np_rmse = np.array(list_rmse)
    np_sisdr = np.array(list_sisdr)
    np_pesq = np.array(list_pesq)
    np_pesq_wb = np.array(list_pesq_wb)
    np_pesq_nb = np.array(list_pesq_nb)
    np_estoi = np.array(list_estoi)

    print("Mean evaluation")
    print('mean rmse score: {:.4f}'.format(np.mean(np_rmse)))
    print('mean sisdr score: {:.1f}'.format(np.mean(np_sisdr)))
    print('mean pypesq score: {:.2f}'.format(np.mean(np_pesq)))
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
    parser.add_argument('--ret_dir', type=str, default=None, help='json log path')
    args = parser.parse_args()

    json_files = []

    for root, dirs, files in os.walk(args.ret_dir):
        for filename in files:
            if filename.endswith('.json'):
                json_files.append(os.path.join(root, filename))

    run(json_files)
    
