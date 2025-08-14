# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:31:41 2023

@author: arad
"""
import numpy as np
import scipy.sparse as sp

####################################################
# This tool is to generate meta-path based adjacency
# matrix given original links.
#این ابزار برای تولید ماتریس مجاورت مبتنی بر متا مسیر با یال های اصلی است.
####################################################

pa = np.genfromtxt("./acm/pa.txt")
ps = np.genfromtxt("./acm/ps.txt")


A = 7167
P = 4019
S=60


pa_ = sp.coo_matrix((np.ones(pa.shape[0]),(pa[:,0], pa[:, 1])),shape=(P,A)).toarray()
ps_ = sp.coo_matrix((np.ones(ps.shape[0]),(ps[:,0], ps[:, 1])),shape=(P,S)).toarray()

pap = np.matmul(pa_.T, pa_) > 0
pap = sp.coo_matrix(pap)
sp.save_npz("./acm/mypap.npy", pap)
"""
apc = np.matmul(pa_.T, pc_) > 0
apcpa = np.matmul(apc, apc.T) > 0
apcpa = sp.coo_matrix(apcpa)
sp.save_npz("./dblp/apcpa.npz", apcpa)

apt = np.matmul(pa_.T, pt_) > 0
aptpa = np.matmul(apt, apt.T) > 0
aptpa = sp.coo_matrix(aptpa)
sp.save_npz("./dblp/aptpa.npz", aptpa)
"""