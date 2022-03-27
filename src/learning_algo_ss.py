#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt
"""


import os
import shutil
import socket
import datetime
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from .utils import myconf, get_logger, loss_ISD, loss_KLD
from .dataset import speech_dataset
from .model_ss import build_SRNN_ss


class LearningAlgorithm_ss():

    """
    Basical class for model building, including:
    - read common paramters for different models
    - define data loader
    - define loss function as a class member
    """

    def __init__(self, params):
        # Load config parser
        self.params = params
        self.config_file = self.params['cfg']
        if not os.path.isfile(self.config_file):
            raise ValueError('Invalid config file path')    
        self.cfg = myconf()
        self.cfg.read(self.config_file)
        self.model_name = self.cfg.get('Network', 'name')
        self.dataset_name = self.cfg.get('DataFrame', 'dataset_name')

        # Get host name and date
        self.hostname = socket.gethostname()
        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%M")
        
        # Load model parameters
        self.use_cuda = self.cfg.getboolean('Training', 'use_cuda')
        self.device = 'cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu'


    def build_model(self):
        if self.model_name == 'SRNN':
            self.model = build_SRNN_ss(cfg=self.cfg, device=self.device)
        else:
            print('Error: wrong model type')
        

    def init_optimizer(self):
        # Init optimizer (Adam by default)
        optimization  = self.cfg.get('Training', 'optimization')
        lr = self.cfg.getfloat('Training', 'lr')
        if optimization == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        return optimizer



    def get_basic_info(self):
        basic_info = []
        basic_info.append('HOSTNAME: ' + self.hostname)
        basic_info.append('Time: ' + self.date)
        basic_info.append('Device for training: ' + self.device)
        if self.device == 'cuda':
            basic_info.append('Cuda verion: {}'.format(torch.version.cuda))
        basic_info.append('Model name: {}'.format(self.model_name))
        basic_info.append('Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters()) / 1000000.0))
        
        return basic_info


    def debug(self):
        # Build model
        self.build_model()
        # Create data loader
        train_dataloader, val_dataloader, train_num, val_num = speech_dataset.build_dataloader(self.cfg)
        print('Train on {}'.format(self.dataset_name))
        print('Training samples: {}'.format(train_num))
        print('Validation samples: {}'.format(val_num))


    def train(self):
        ############
        ### Init ###
        ############

        # Build model
        self.build_model()

        # Set module.training = True
        self.model.train()
        torch.autograd.set_detect_anomaly(True)

        # Create directory for results
        if not self.params['reload']:
            saved_root = self.cfg.get('User', 'saved_root')
            z_dim = self.cfg.getint('Network','z_dim')
            tag = self.cfg.get('Network', 'tag')
            filename = "{}_{}_{}_z_dim={}".format(self.dataset_name, self.date, tag, z_dim)
            save_dir = os.path.join(saved_root, filename)
            if not(os.path.isdir(save_dir)):
                os.makedirs(save_dir)
        else:
            tag = self.cfg.get('Network', 'tag')
            save_dir = self.params['model_dir']

        # Save the model configuration
        save_cfg = os.path.join(save_dir, 'config.ini')
        shutil.copy(self.config_file, save_cfg)

        # Create logger
        log_file = os.path.join(save_dir, 'log.txt')
        logger_type = self.cfg.getint('User', 'logger_type')
        logger = get_logger(log_file, logger_type)

        # Print basical infomation
        for log in self.get_basic_info():
            logger.info(log)
        logger.info('In this experiment, result will be saved in: ' + save_dir)

        # Print model infomation (optional)
        if self.cfg.getboolean('User', 'print_model'):
            for log in self.model.get_info():
                logger.info(log)

        # Init optimizer
        optimizer = self.init_optimizer()

        # Create data loader
        train_dataloader, val_dataloader, train_num, val_num = speech_dataset.build_dataloader(self.cfg)
        logger.info('Train on {}'.format(self.dataset_name))
        logger.info('Training samples: {}'.format(train_num))
        logger.info('Validation samples: {}'.format(val_num))
        
        ######################
        ### Batch Training ###
        ######################

        # Load training parameters
        epochs = self.cfg.getint('Training', 'epochs')
        early_stop_patience = self.cfg.getint('Training', 'early_stop_patience')
        save_frequency = self.cfg.getint('Training', 'save_frequency')
        beta = self.cfg.getfloat('Training', 'beta')

        # Create python list for loss
        if not self.params['reload']:
            train_loss = np.zeros((epochs,))
            val_loss = np.zeros((epochs,))
            train_recon = np.zeros((epochs,))
            train_kl = np.zeros((epochs,))
            val_recon = np.zeros((epochs,))
            val_kl = np.zeros((epochs,))
            best_val_loss = np.inf
            cpt_patience = 0
            cur_best_epoch = epochs
            best_state_dict = self.model.state_dict()
            best_optim_dict = optimizer.state_dict()
            start_epoch = -1
            if self.params['use_pretrain']:
                state_dict_file = self.params['pretrain_dict']
                self.model.load_state_dict(torch.load(state_dict_file, map_location=self.device))
                best_state_dict = self.model.state_dict()
                logger.info('Loading pre-trained model: {}'.format(state_dict_file))
        else:
            cp_file = os.path.join(save_dir, '{}_checkpoint.pt'.format(self.model_name))
            checkpoint = torch.load(cp_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            start_epoch = checkpoint['epoch']
            loss_log = checkpoint['loss_log']
            logger.info('Resuming trainning: epoch: {}'.format(start_epoch))
            train_loss = np.pad(loss_log['train_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            val_loss = np.pad(loss_log['val_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            train_recon = np.pad(loss_log['train_recon'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            train_kl = np.pad(loss_log['train_kl'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            val_recon = np.pad(loss_log['val_recon'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            val_kl = np.pad(loss_log['val_kl'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            best_val_loss = checkpoint['best_val_loss']
            cpt_patience = 0
            cur_best_epoch = start_epoch
            best_state_dict = self.model.state_dict()
            best_optim_dict = optimizer.state_dict()

        # Schedule sampling
        ss_start = self.cfg.getint('Training', 'ss_start')
        ss_end = self.cfg.getint('Training', 'ss_end')

        # Train with mini-batch SGD
        for epoch in range(start_epoch+1, epochs):

            start_time = datetime.datetime.now()
            
            # Schedule Sampling
            if epoch < ss_start:
                ## monotocinal annealing
                kl_warm = epoch / ss_start
                ## cyclical annealing
                # cyc_len = ss_start // 2
                # tau = (epoch % cyc_len) / cyc_len
                # kl_warm = 2 * tau if tau < 0.5 else 1
                use_pred = 0
            elif epoch >= ss_start and epoch < ss_end:
                kl_warm = 1
                ss_step = epoch - ss_start
                use_pred = ss_step / (ss_end - ss_start)
            else:
                kl_warm = 1
                use_pred = 1

            if epoch == 0:
                logger.info('=====> KL warm up start')
            elif epoch == ss_start:
                logger.info('=====> KL warm up end, schedule sampling start')
            elif epoch == ss_end:
                logger.info('=====> schedule sampling end')

            # Batch training
            for _, batch_data in enumerate(train_dataloader):
                
                batch_data = batch_data.to(self.device)

                # (batch_size, x_dim, seq_len) -> (seq_len, batch_size, x_dim)
                batch_data = batch_data.permute(2, 0, 1)
                seq_len, bs, _ = batch_data.shape
                recon_batch_data = torch.exp(self.model(batch_data, use_pred)) # output log-variance

                loss_recon = loss_ISD(batch_data, recon_batch_data)
                loss_recon = loss_recon / (seq_len * bs)

                loss_kl = loss_KLD(self.model.z_mean, self.model.z_logvar, self.model.z_mean_p, self.model.z_logvar_p)
                loss_kl = loss_kl / (seq_len * bs)

                loss_tot = loss_recon + beta * loss_kl
                optimizer.zero_grad()
                loss_tot.backward()
                optimizer.step()

                train_loss[epoch] += loss_tot.item() * bs
                train_recon[epoch] += loss_recon.item() * bs
                train_kl[epoch] += loss_kl.item() * bs
                
            # Validation
            for _, batch_data in enumerate(val_dataloader):

                batch_data = batch_data.to(self.device)
                
                # (batch_size, x_dim, seq_len) -> (seq_len, batch_size, x_dim)
                batch_data = batch_data.permute(2, 0, 1)
                seq_len, bs, _ = batch_data.shape
                recon_batch_data = torch.exp(self.model(batch_data, use_pred)) # output log-variance
                
                loss_recon = loss_ISD(batch_data, recon_batch_data)
                loss_recon = loss_recon / (seq_len * bs)

                loss_kl = loss_KLD(self.model.z_mean, self.model.z_logvar, self.model.z_mean_p, self.model.z_logvar_p)
                loss_kl = loss_kl / (seq_len * bs)

                loss_tot = loss_recon + beta * loss_kl

                val_loss[epoch] += loss_tot.item() * bs
                val_recon[epoch] += loss_recon.item() * bs
                val_kl[epoch] += loss_kl.item() * bs

            # Loss normalization
            train_loss[epoch] = train_loss[epoch]/ train_num
            val_loss[epoch] = val_loss[epoch] / val_num
            train_recon[epoch] = train_recon[epoch] / train_num 
            train_kl[epoch] = train_kl[epoch]/ train_num
            val_recon[epoch] = val_recon[epoch] / val_num 
            val_kl[epoch] = val_kl[epoch] / val_num
            
            # Early stop patiance (valid only after prob=1)
            if epoch < ss_end or val_loss[epoch] < best_val_loss:
                best_val_loss = val_loss[epoch]
                cpt_patience = 0
                best_state_dict = self.model.state_dict()
                best_optim_dict = optimizer.state_dict()
                cur_best_epoch = epoch
            else:
                cpt_patience += 1

            # Training time
            end_time = datetime.datetime.now()
            interval = (end_time - start_time).seconds / 60
            logger.info('Epoch: {} training time {:.2f}m'.format(epoch, interval))
            logger.info('Train => tot: {:.2f} recon {:.2f} KL {:.2f} Val => tot: {:.2f} recon {:.2f} KL {:.2f}'.format(train_loss[epoch], train_recon[epoch], train_kl[epoch], val_loss[epoch], val_recon[epoch], val_kl[epoch]))

            # Stop traning if early-stop triggers, only active after fully using prediction
            if cpt_patience == early_stop_patience and use_pred == 1:
                logger.info('Early stop patience achieved')
                break

            # Save model parameters regularly
            if epoch % save_frequency == 0:
                loss_log = {'train_loss': train_loss[:cur_best_epoch+1],
                            'val_loss': val_loss[:cur_best_epoch+1],
                            'train_recon': train_recon[:cur_best_epoch+1],
                            'train_kl': train_kl[:cur_best_epoch+1], 
                            'val_recon': val_recon[:cur_best_epoch+1], 
                            'val_kl': val_kl[:cur_best_epoch+1]}
                save_file = os.path.join(save_dir, self.model_name + '_checkpoint.pt')
                torch.save({'epoch': cur_best_epoch,
                            'best_val_loss': best_val_loss,
                            'cpt_patience': cpt_patience,
                            'model_state_dict': best_state_dict,
                            'optim_state_dict': best_optim_dict,
                            'loss_log': loss_log
                        }, save_file)
                logger.info('Epoch: {} ===> checkpoint stored with current best epoch: {}'.format(epoch, cur_best_epoch))
        
        # Save the final weights of network with the best validation loss
        save_file = os.path.join(save_dir, self.model_name + '_final_epoch' + str(cur_best_epoch) + '.pt')
        torch.save(best_state_dict, save_file)
        
        # Save the training loss and validation loss
        train_loss = train_loss[:epoch+1]
        val_loss = val_loss[:epoch+1]
        train_recon = train_recon[:epoch+1]
        train_kl = train_kl[:epoch+1]
        val_recon = val_recon[:epoch+1]
        val_kl = val_kl[:epoch+1]
        loss_file = os.path.join(save_dir, 'loss_model.pckl')
        with open(loss_file, 'wb') as f:
            pickle.dump([train_loss, val_loss, train_recon, train_kl, val_recon, val_kl], f)


        # Save the loss figure
        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_loss, label='training loss')
        plt.plot(val_loss, label='validation loss')
        plt.legend(fontsize=16, title=self.model_name, title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        fig_file = os.path.join(save_dir, 'loss_{}.png'.format(tag))
        plt.savefig(fig_file)

        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_recon, label='Training')
        plt.plot(val_recon, label='Validation')
        plt.legend(fontsize=16, title='{}: Recon. Loss'.format(self.model_name), title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        fig_file = os.path.join(save_dir, 'loss_recon_{}.png'.format(tag))
        plt.savefig(fig_file) 

        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_kl, label='Training')
        plt.plot(val_kl, label='Validation')
        plt.legend(fontsize=16, title='{}: KL Divergence'.format(self.model_name), title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        fig_file = os.path.join(save_dir, 'loss_KLD_{}.png'.format(tag))
        plt.savefig(fig_file)