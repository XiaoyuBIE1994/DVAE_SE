#!/usr/bin/env python
# encoding: utf-8

import numpy as np
# from IPython import embed
from .data_utils import rotmat2expmap

PARENT_H36M = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
    17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1

PARENT_EXPI = [3,0,0,-1,3,3,4,5,6,7,3,3,10,11,12,13,14,15] #18

def _toposort_visit(parents, visited, toposorted, joint):
    parent = parents[joint]
    visited[joint] = True
    if parent != joint and not visited[parent]:
        _toposort_visit(parents, visited, toposorted, parent)
    toposorted.append(joint)


def check_toposorted(parents, toposorted):
    # check that array contains all/only joint indices
    assert sorted(toposorted) == list(range(len(parents)))

    # make sure that order is correct
    to_topo_order = {
        joint: topo_order
        for topo_order, joint in enumerate(toposorted)
    }
    for joint in toposorted:
        assert to_topo_order[joint] >= to_topo_order[parents[joint]]

    # verify that we have only one root
    joints = range(len(parents))
    assert sum(parents[joint] == joint for joint in joints) == 1


def toposort(parents):
    """Return toposorted array of joint indices (sorted root-first)."""
    toposorted = []
    visited = np.zeros_like(parents, dtype=bool)
    for joint in range(len(parents)):
        if not visited[joint]:
            _toposort_visit(parents, visited, toposorted, joint)

    #check_toposorted(parents, toposorted)

    return np.asarray(toposorted)

def _norm_bvecs(bvecs):
     """Norm bone vectors, handling small magnitudes by zeroing bones."""
     bnorms = np.linalg.norm(bvecs, axis=-1)
     mask_out = bnorms <= 1e-5
     _, broad_mask = np.broadcast_arrays(bvecs, mask_out[..., None])
     bvecs[broad_mask] = 0
     bnorms[mask_out] = 1
     return bvecs / bnorms[..., None]

def skew(x):
    return np.array([[0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]])
def norm_vec(u):
     return u / np.sqrt(np.dot(u, u))

def xyz_to_expmap(xyz_seq, parents):
    """Converts a tree of (x, y, z) positions into the parameterisation used in
    the SRNN paper, "modelling human motion with binary latent variables"
    paper, etc. Stores inter-frame offset in root joint position."""
    assert xyz_seq.ndim == 3 and xyz_seq.shape[2] == 3, \
        "Wanted TxJx3 array containing T skeletons, each with J (x, y, z)s"

    exp_seq = np.zeros_like(xyz_seq)
    toposorted = toposort(parents)
    # [1:] ignores the root; apart from that, processing order doesn't actually
    # matter
    for child in toposorted[1:]:
        parent = parents[child]
        bones = xyz_seq[:, parent] - xyz_seq[:, child]
        grandparent = parents[parent]
        if grandparent == parent:
            # we're the root; parent bones will be constant (x,y,z)=(0,-1,0)
            parent_bones = np.zeros_like(bones)
            parent_bones[:, 1] = -1
        else:
            # we actually have a parent bone :)
            parent_bones = xyz_seq[:, grandparent] - xyz_seq[:, parent]

        # normalise parent and child bones
        norm_bones = _norm_bvecs(bones)
        norm_parent_bones = _norm_bvecs(parent_bones)
        '''
        # cross product will only be used to get axis around which to rotate
        cross_vecs = np.cross(norm_parent_bones, norm_bones)
        norm_cross_vecs = _norm_bvecs(cross_vecs)
        # dot products give us rotation angle
        angles = np.arccos(np.sum(norm_bones * norm_parent_bones, axis=-1))
        log_map = norm_cross_vecs * angles[..., None]
        '''
        log_map=[]
        for i in range(len(xyz_seq)):
            U = norm_bones[i]
            V = norm_parent_bones[i]
            axis = norm_vec(np.cross(U, V))
            cosine_theta = np.dot(U, V) / (np.sqrt(np.dot(U, U)) * np.sqrt(np.dot(V, V)))
            sine_theta = np.sqrt(1 - cosine_theta ** 2)
            R = cosine_theta * np.identity(3) + sine_theta * skew(axis) + (1 - cosine_theta) * np.outer(axis, axis)
            # print(">>>R2:", R)
            log_map.append(rotmat2expmap(R))

        exp_seq[:, child] = np.array(log_map)

    # root will store distance from previous frame
    root = toposorted[0]
    exp_seq[1:, root] = xyz_seq[1:, root] - xyz_seq[:-1, root]

    return exp_seq

## not used
def _xyz_to_expmap(xyz_seq, parent):
    # input xyz: n,32,3
    # output
    nb_f, nb_j, _ = xzy_seq.shape
    exp_seq = np.zeros_like(xyz_seq)

    for i in range(nb_j):
        p = parent[i] #parent
        bone = xyz_seq[:, p] - xyz_seq[:, i]
        if p <0: # root, parent_bone (x,y,z)=(0,-1,0)
            root=p
            p_bone = np.zeros_like(bone)
            p_bone[:,1] = -1
        else:
            gp = parents[p] #grandparent
            p_bone = xyz_seq[:, gp] - xyz_seq[:, p]

        norm_bone = _norm_bvecs(bone)
        norm_p_bone = _norm_bvecs(p_bone)

        # cross product will only be used to get axis around which to rotate
        cross_vecs = np.cross(norm_p_bone, norm_bone)
        norm_cross_vecs = _norm_bvecs(cross_vecs)
        # dot products give us rotation angle
        angles = np.arccos(np.sum(norm_bone * norm_p_bone, axis=-1))
        log_map = norm_cross_vecs * angles[..., None]
        exp_seq[:, i] = log_map

    exp_seq[1:, root] = xyz_seq[1:, root] - xyz_seq[:-1, root]
    return exp_seq




