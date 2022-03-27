#!/usr/bin/env python
# -*- coding: utf-8 -*-
## copy from data_utils.py.
## for norm_frame
## in fact nothong changed

import numpy as np
import torch

## func utils for norm
def normExPI(img, P0,P1,P2):
    # P0: orig
    # P0-P1: axis x
    # P0-P1-P2: plane xoy

    X0 = P0 #np.concatenate((P0,np.array([1])),axis=0)
    X1 = (P1-P0) / np.linalg.norm((P1-P0)) + P0
    D = np.cross(P2-P0, P1-P0)
    X2 = D / np.linalg.norm(D) + P0

    X3 = np.cross(X2-P0, X1-P0) + P0
    X = np.concatenate((np.array([X0,X1,X2,X3]).transpose(), np.array([[1, 1, 1,1]])), axis = 0)
    Q = np.array([[0,0,0],[1,0,0],[0,0,1], [0,1,0]]).transpose()

    M  = Q.dot(np.linalg.pinv(X))

    img_norm = img.copy()
    for i in range(len(img)):
        tmp = img[i]
        tmp = np.concatenate((tmp,np.array([1])),axis=0)
        img_norm[i] =  M.dot(tmp)
    return img_norm

def normExPI_xoz(img, P0,P1,P2):
    # P0: orig
    # P0-P1: axis x
    # P0-P2: axis z

    X0 = P0
    X1 = (P1-P0) / np.linalg.norm((P1-P0)) + P0
    X2 = (P2-P0) / np.linalg.norm((P2-P0)) + P0
    X3 = np.cross(X2-P0, X1-P0) + P0
    ### x2 determine z -> x2 determine plane xoz
    X2 = np.cross(X1-P0, X3-P0) + P0

    X = np.concatenate((np.array([X0,X1,X2,X3]).transpose(), np.array([[1, 1, 1,1]])), axis = 0)
    Q = np.array([[0,0,0],[1,0,0],[0,0,1], [0,1,0]]).transpose()
    M  = Q.dot(np.linalg.pinv(X))

    img_norm = img.copy()
    for i in range(len(img)):
        tmp = img[i]
        tmp = np.concatenate((tmp,np.array([1])),axis=0)
        img_norm[i] =  M.dot(tmp)
    return img_norm

def get_relation_coord(P0_m,P1_m,P2_m, P0_f,P1_f,P2_f):
    # P0: orig
    # P0-P1: axis x
    # P0-P2: axis z
    # return coord:  9 dim (vector_x, vector_z, vector_o)

    X0_m = P0_m
    X1_m = (P1_m-P0_m) / np.linalg.norm((P1_m-P0_m)) + P0_m
    X2_m = (P2_m-P0_m) / np.linalg.norm((P2_m-P0_m)) + P0_m
    X0_f = P0_f
    X1_f = (P1_f-P0_f) / np.linalg.norm((P1_f-P0_f)) + P0_f
    X2_f = (P2_f-P0_f) / np.linalg.norm((P2_f-P0_f)) + P0_f

    D1 = X1_f - X1_m + np.array((1,0,0))
    D2 = X2_f - X2_m + np.array((0,0,1))
    D0 = X0_f - X0_m + np.array((0,0,0))

    return np.concatenate((D1,D2,D0))


#############################################################
## norm_frame
## used in pi3d.py

def normExPI_2p_by_frame(seq):
    nb, dim = seq.shape # nb_frames, dim=108
    seq_norm = seq.copy()
    for i in range(nb):
        img = seq[i].reshape((-1,3))
        #P0 = (img[10] + img[11])/2
        #P1 = img[29]
        #P2 = img[11]
        #img_norm = normExPI(img, P0,P1,P2)
        P0 = (img[10] + img[11])/2
        P1 = img[11]
        P2 = img[3]
        img_norm = normExPI_xoz(img, P0,P1,P2)
        seq_norm[i] = img_norm.reshape(dim)
    return seq_norm

def normExPI_2p_by_frame_deltaf(seq):
    # (m, delta_f) = (m, f - m)
    nb, dim = seq.shape # nb_frames, dim=108
    seq_norm = seq.copy()
    for i in range(nb):
        img = seq[i].reshape((-1,3))
        P0 = (img[10] + img[11])/2
        P1 = img[11]
        P2 = img[3]
        img_norm = normExPI_xoz(img, P0,P1,P2)
        img_norm = np.concatenate((img_norm[:int(dim/6)],\
                img_norm[int(dim/6):]-img_norm[:int(dim/6)]), axis=0)
        seq_norm[i] = img_norm.reshape(dim)
    return seq_norm
def normExPI_2p_by_frame_deltam(seq):
    # (delta_m, f) = (m - f, f)
    nb, dim = seq.shape # nb_frames, dim=108
    seq_norm = seq.copy()
    for i in range(nb):
        img = seq[i].reshape((-1,3))
        P0 = (img[10+18] + img[11+18])/2
        P1 = img[11+18]
        P2 = img[3+18]
        img_norm = normExPI_xoz(img, P0,P1,P2)
        ##TODO
        #img_norm = np.concatenate((img_norm[:int(dim/6)] - img_norm[int(dim/6):],\
        #        img_norm[int(dim/6):]), axis=0)
        seq_norm[i] = img_norm.reshape(dim)
    return seq_norm


###################################################
### unnorm
### used in **_eval.py

def norm_frame_torch(seq):
    # used in mian_pi_3d_eval_2p.py, change f based to m based, for normExPI_2p_by_frame_deltam
    # in torch.Size([32, 10, 26, 3])
    a,b,c,d = seq.shape
    seq = seq.detach().cpu().numpy().reshape((a,b,c*d))
    seq_norm = seq
    for i in range(a):
        seq_norm[i] = normExPI_2p_by_frame(seq[i])
    seq_norm = torch.from_numpy(seq_norm.reshape((a,b,c,d))).cuda()
    return seq_norm


def unnorm_abs2Indep(seq):
    # used for output of main_pi_3d_eval_2p.py & main_pi_3d_eval_13kpts.py
    # in:  torch.size(32, 75, 13or26, 3)
    # out: torch.size(32, 75, 13or26, 3)
    seq = seq.detach().cpu().numpy()
    bz, frame, nb, dim = seq.shape
    seq_norm = seq
    for j in range(bz):
        for i in range(frame):
            img = seq[j][i]

            P0_m = (img[10] + img[11])/2
            P1_m = img[11]
            P2_m = img[3]
            if nb == 36:
                img_norm_m = normExPI_xoz(img[:int(nb/2)], P0_m,P1_m,P2_m)
                P0_f = (img[18+10] + img[18+11])/2
                P1_f = img[18+11]
                P2_f = img[18+3]
                img_norm_f = normExPI_xoz(img[int(nb/2):], P0_f,P1_f,P2_f)
                img_norm = np.concatenate((img_norm_m, img_norm_f))
            elif nb == 18:
                img_norm = normExPI_xoz(img, P0_m,P1_m,P2_m)
            seq_norm[j][i] = img_norm.reshape((nb,dim))
    seq = torch.from_numpy(seq_norm).cuda()
    return seq # nb_frames, dim=32*6+6or9



##TODO:18
def unnorm_Indep2abs(seq): ##TODO:18
    # inverse of normExPI_indep_by_frame
    # used for output of main_pi_3d_eval_2p_indepnorm.py
    # in:  torch.size(batch_size=32, nb_frames=75, 29, 3)
    # out:  torch.size(32,75,26,3)
    seq = seq.detach().cpu().numpy()
    p1 = seq[:,:,:13]

    #p2 = seq[:,:,13:26]
    #D1,D2,D0 = seq[:,:,26], seq[:,:,27],seq[:,:,28]
    p2_norm = seq[:,:,13:26]
    for i in range(len(seq)):
        for t in range(len(seq[0])):
            j = seq[i][t]
            p2 = j[13:26]
            D1,D2,D0 = j[26],j[27],j[28] #x,z,0
            # calcul M p2_abs = M.dot(p2) #Q=MX
            X = np.concatenate((np.array([[0,0,0],[1,0,0],[0,1,0], [0,0,1]]).transpose(), np.array([[1, 1, 1,1]])), axis = 0)
            #embed()
            #exit()
            Qx = D1#+np.array([1,0,0])
            Qz = D2#+np.array([0,0,1])
            Qy = np.cross(Qz-D0,Qx-D0)+D0
            Q = np.array([D0,Qx,Qy,Qz]).transpose()
            M  = Q.dot(np.linalg.pinv(X))
            for k in range(13):
                tmp = p2[k]
                tmp = np.concatenate((tmp,np.array([1])),axis=0)
                p2_norm[i][t][k] =  M.dot(tmp)

    seq = torch.from_numpy(np.concatenate((p1,p2_norm),axis=2)).cuda()
    return seq

#########################################################
### Indep Norm by frame

def normExPI_indep_by_frame(seq):
    nb, dim = seq.shape # nb_frames, dim=108=18*2*3
    seq_norm = []
    for i in range(nb):
        img = seq[i].reshape((-1,3))

        P0_m = (img[10] + img[11])/2
        P1_m = img[11]
        P2_m = img[3]
        img_norm_m = normExPI_xoz(img[:int(dim/6)], P0_m,P1_m,P2_m)
        P0_f = (img[28] + img[29])/2
        P1_f = img[29]
        P2_f = img[21]
        img_norm_f = normExPI_xoz(img[int(dim/6):], P0_f,P1_f,P2_f)
        relation_coord = get_relation_coord(P0_m,P1_m,P2_m, P0_f,P1_f,P2_f).reshape((-1,3))

        #img_norm = np.concatenate((img_norm_m, img_norm_f))
        #embed()
        img_norm = np.concatenate((img_norm_m, img_norm_f, relation_coord))
        seq_norm.append(img_norm.reshape(img_norm.shape[0]*3))
    return np.array(seq_norm) # nb_frames, dim=108+6or9


#####################################################
### h36m

def h36m_normExPI_indep_by_frame(seq):
    nb, dim = seq.shape # nb_frames, dim=32*2*3
    seq_norm = []
    for i in range(nb):
        img = seq[i].reshape((-1,3))

        P0_m = (img[6] + img[1])/2
        P1_m = img[1]
        P2_m = img[13]
        img_norm_m = normExPI_xoz(img[:int(dim/6)], P0_m,P1_m,P2_m)
        P0_f = (img[32+6] + img[32+1])/2
        P1_f = img[32+1]
        P2_f = img[32+13]
        img_norm_f = normExPI_xoz(img[int(dim/6):], P0_f,P1_f,P2_f)
        relation_coord = get_relation_coord(P0_m,P1_m,P2_m, P0_f,P1_f,P2_f).reshape((-1,3))
        #img_norm = np.concatenate((img_norm_m, img_norm_f))
        img_norm = np.concatenate((img_norm_m, img_norm_f, relation_coord))
        seq_norm.append(img_norm.reshape(img_norm.shape[0]*3))
    return np.array(seq_norm) # nb_frames, dim=32*6+6or9

'''
if __name__ == '__main__':
    img = np.random.rand(5,3) *1000
    p0,p1,p2 = img[0],img[1],img[2]
    img_norm = normExPI(img,p0,p1,p2)
    print(img, img_norm)
'''


