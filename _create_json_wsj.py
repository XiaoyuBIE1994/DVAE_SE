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

"""
write launch script for QUT-WSJ
"""
saved_model = 'ALL_2022-01-02-17h48_SRNN-ss_z_dim=16'
state_dict_file = 'SRNN_final_epoch225.pt'
cpu_type = 'cpu_med'
# cpu_type = 'cpu_prod'
lr = '1e-3'
iter_num = 300

for i in range(82):
    lines = ['#!/bin/bash\n',
            '#SBATCH --job-name=sub_{}\n'.format(i),
            '#SBATCH --output=/workdir/biex/saved_models_se/log_se_wsj/se_wsj_{}.stdout\n'.format(i),
            '#SBATCH --error=/workdir/biex/saved_models_se/log_se_wsj/se_wsj_{}.stderr\n'.format(i),
            '#SBATCH --mail-user=xiaoyu.bie@inria.fr\n',
            '#SBATCH --mail-type=ALL\n',
            '#SBATCH --nodes=1\n',
            '#SBATCH --ntasks=1\n',
            '#SBATCH --cpus-per-task=4\n',
            '#SBATCH --partition={}\n'.format(cpu_type),
            '#SBATCH --mem=16G\n',
            '#SBATCH --time=4:00:00\n',
            '\n\n',
            '# load module\n',
            'module purge\n',
            'module load anaconda3/2021.05/gcc-9.2.0\n',
            'source activate dvae\n',
            '\n\n',
            '# submit job, w/0 pre-train\n',
            'python /gpfs/users/biex/GitPublish/dvae-se/test_wsj.py \\\n',
            '        --exp_name sub_{} \\\n'.format(i),
            '        --saved_model /workdir/biex/saved_models_se/{} \\\n'.format(saved_model),
            '        --state_dict_file {} \\\n'.format(state_dict_file),
            '        --json_file /workdir/biex/data/QUT_WSJ0/json_acc/QUT_test_sub{}.json \\\n'.format(i),
            '        --mix_dir /workdir/biex/data/QUT_WSJ0/test \\\n',
            '        --clean_dir /workdir/biex/data/clean_speech/wsj0_si_et_05 \\\n',
            '        --ckpt_dir /workdir/biex/saved_models_se/{}/ckpt_wsj_iter{}_lr{} \\\n'.format(saved_model, iter_num, lr),
            '        --enhance_dir /workdir/biex/saved_models_se/{}/ckpt_wsj_iter{}_lr{}/enhance_wav \\\n'.format(saved_model, iter_num, lr),
            '        --log_type 1 \\\n',
            '        --niter {} \\\n'.format(iter_num),
            '        --nmf_rank 8 \\\n',
            '        --nepochs_E_step 1 \\\n',
            '        --nsamples_E_step 1 \\\n',
            '        --nsamples_WF 1 \\\n',
            '        --lr {} \\\n'.format(lr),
            '        --device cpu']

    filename = './script_wsj/ruche_test_wsj_sub{}.sh'.format(i)
    with open(filename, 'w') as f:
        f.writelines(lines)
    os.chmod(filename, 0o777)

launch_list = []
for i in range(82):
    launch_list.append(f'sbatch ./script_wsj/ruche_test_wsj_sub{i}.sh\n')
launch_file = './ruche_test_wsj.sh'
with open(launch_file, 'w') as f:
    f.writelines(launch_list)
os.chmod(launch_file, 0o777)

enhance_dir = f'/workdir/biex/saved_models_se/{saved_model}/ckpt_wsj_iter{iter_num}_lr{lr}/enhance_wav'
if not os.path.isdir(enhance_dir):
    os.makedirs(enhance_dir)

# launch_list = []
# for i in range(82):
#     id_job = 1084226 + i
#     launch_list.append(f'scancel {id_job}\n')
# launch_file = './ruche_cancel.sh'
# with open(launch_file, 'w') as f:
#     f.writelines(launch_list)
# os.chmod(launch_file, 0o777)