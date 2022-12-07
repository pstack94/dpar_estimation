from dipy.core.gradients import gradient_table
from dipy.sims.phantom import add_noise
import sys
from dipy.sims.phantom import single_tensor
import matplotlib.pyplot as plt
from sklearn.decomposition import non_negative_factorization
from itertools import permutations
import nibabel as nib
import scipy.io as sio
from os.path import dirname, join as pjoin
import numpy as np
from scipy.optimize import nnls
import os
import random
from scipy.stats import dirichlet
from scipy.special import erf
from scipy.stats import truncnorm
from matplotlib import pyplot
import shutil
from tensorflow import keras
from itertools import permutations
from scipy.special import erf


def open_vol(name):
    img= nib.load(name)
    vol= img.get_fdata()
    return vol, img

def open_bvals(in_name):
    with open(in_name, 'r') as f:
        array = []
        for line in f: # read rest of lines
            array.append([float(x) for x in line.split()])
    array=np.array(array)
    return array.T

def proc_bvals(bvals, v):
    bvals_proc = np.zeros(bvals.shape)

    for i in range(len(v)):
        c = bvals>=v[i]-60
        c2 = bvals<v[i]+60
        bvals_proc[c&c2] = v[i]
    return bvals_proc.reshape(bvals_proc.shape[0],)

def calc_SMT_matrix(bvals, vol_all):
    #calcular SMT
    shells= np.unique(bvals).shape[0]
    nx,_ = vol_all.shape

    V = np.ones((nx,shells))

    b0 = vol_all[:,bvals==0]
    volumen_n = vol_all/np.mean(b0,axis=1).reshape(-1,1)

    u_bv = np.unique(bvals)

    for i in range(shells):
        V[:,i] = np.mean(volumen_n[:,bvals==u_bv[i]], axis=1)
    return V

def open_data_matlab():
    mat_fname = pjoin('./Library/Training_3D_SF_features.mat')
    mat_contents = sio.loadmat(mat_fname)

    Ms=mat_contents['Ms']
    Fs=mat_contents['Fs']
    R1=mat_contents['R1']
    R2=mat_contents['R2']
    R3=mat_contents['R3']
    L1=mat_contents['lambda1']
    L2=mat_contents['lambda2']
    L3=mat_contents['lambda3']

    R = np.zeros((3,3,16,16,5,3))
    R[:,:,:,:,:,0] = R1
    R[:,:,:,:,:,1] = R2
    R[:,:,:,:,:,2] = R3

    lmbda = np.zeros((3,3,16,16,5))
    lmbda[:,0,:,:,:] = L1
    lmbda[:,1,:,:,:] = L2
    lmbda[:,2,:,:,:] = L3

    return Ms, Fs, R, lmbda


def genera_fantasma(dp_min, dp_max, dfree, n_bvals, n_bvecs, v):
    samples = 1280
    Ms, Fs, R, lmbda = open_data_matlab()
    nx, ny, nz = Ms.shape

    # definir difuciones
    dp1 = dp_max - dp_min
    dpars_wm = np.random.rand(nx, ny, nz)*dp1 + dp_min
    coef_agua = dfree

    #abrir bvecs and bvals
    bvecs= open_bvals(n_bvecs)
    bvals= open_bvals(n_bvals)

    bvals= bvals.reshape(bvals.shape[0],)
    bvals = proc_bvals(bvals, v)

    gtab = gradient_table(bvals, bvecs)

    #definir tensores
    evals_bola = np.array([coef_agua,coef_agua,coef_agua])
    evecs=np.array([[1,0,0],[0,1,0],[0,0,1]])


    n1,n2,n3 = Ms.shape

    #icsfs = genera_icsf_norm_trunc(mu, sigma, low_l, up_l,tipo, samples)
    icsfs = np.random.rand(n1, n2, nz)*0.6 + 0.3
    icsfs_f = np.zeros((n1, n2, nz))
    dpars = np.zeros((n1, n2, nz))
    E = np.zeros((n1, n2, nz,bvals.shape[0]))

    CSF = single_tensor(gtab, S0=1, evals=evals_bola, evecs=evecs, snr=None)

    H_wm = np.random.rand(n1, n2, nz)
    #H_dot = np.random.rand(n1, n2, nz)*0.02
    #H_dot = np.random.rand(n1, n2, nz)*0.10 # para hacer como Coronado 6 Abril 2022
    H_dot = np.random.normal(0.1, 0.01, samples).reshape(n1, n2, nz)
    H_csf = 1 - H_wm

    cont = 0
    for i in range(n1):
        for j in range(n2):
            for k in range(nz):
                n_manojos = Ms[i,j,k]
                fs = Fs[:,i,j,k]
                r = R[:,:,i,j,k,:]
                WM= 0
                icsf_wm = icsfs[i,j,k]
                dpar_wm = dpars_wm[i,j,k]
                dpars[i,j,k] = dpar_wm
                evals_intra=np.array([0,0,dpar_wm])
                evals_extra=np.array([dpar_wm*(1-icsf_wm), dpar_wm*(1-icsf_wm), dpar_wm])
                for l in range(n_manojos):
                    #simular materia blanca
                    intra_wm = single_tensor(gtab, S0=1, evals=evals_intra, evecs=r[:,:,l], snr=None)
                    extra_wm = single_tensor(gtab, S0=1, evals=evals_extra, evecs=r[:,:,l], snr=None)
                    WM += fs[l] * (icsf_wm *intra_wm  + (1-icsf_wm)*extra_wm)
                icsfs_f[i,j,k]  = icsf_wm
                E[i,j,k,:] = (1-H_dot[i,j,k])*(H_wm[i,j,k]*WM + H_csf[i,j,k]*CSF) + H_dot[i,j,k]
                cont+=1

    return E, icsfs_f, dpars, bvals, H_wm, H_csf, H_dot

def data_phantom(dp_min, dp_max, dfree, n_bvals, n_bvecs, v, snr, f_out):
    snr_str = str(snr)
    dir_SMT = f_out+'/SMT_dpar_noise'
    dir_LBL = f_out+'/Labels_dpar_noise'

    if not os.path.exists(dir_SMT):
        os.makedirs(dir_SMT)

    if not os.path.exists(dir_LBL):
        os.makedirs(dir_LBL)


    for i in range(0,100):
        E, icsfs_f, dpars, bvals, H_wm, H_csf, H_dot= genera_fantasma(dp_min, dp_max, dfree, n_bvals, n_bvecs, v)
        nx, ny, nz, _ = E.shape
        mask_all = np.ones((nx,ny,nz), dtype=bool)


        E_noisy = add_noise(E, snr=snr, S0=1, noise_type='rician')

        V1 = calc_SMT_matrix(bvals, E_noisy[mask_all])

        l_wm1 = H_wm[mask_all].reshape(-1,1)
        l_csf1 = H_csf[mask_all].reshape(-1,1)
        l_dot1 = H_dot[mask_all].reshape(-1,1)
        icsf1 = icsfs_f[mask_all].reshape(-1,1)
        l_dpars_wm1 = dpars[mask_all].reshape(-1,1)

        idx = str(i)

        np.savetxt(dir_SMT + '/SMT_noise'+idx+'.txt', V1, fmt='%f')
        np.savetxt(dir_LBL + '/L_WM_'+idx+'.txt', l_wm1, newline=" ", fmt='%f')
        np.savetxt(dir_LBL + '/L_CSF_'+idx+'.txt', l_csf1, newline=" ", fmt='%f')
        np.savetxt(dir_LBL + '/L_DOT_'+idx+'.txt', l_dot1, newline=" ", fmt='%f')
        np.savetxt(dir_LBL + '/ICSF_'+idx+'.txt', icsf1, newline=" ", fmt='%f')
        np.savetxt(dir_LBL + '/L_dpar_WM'+idx+'.txt', l_dpars_wm1, newline=" ", fmt='%f')
    shutil.make_archive(f_out+'/data', 'zip', f_out)
    shutil.rmtree(dir_SMT)
    shutil.rmtree(dir_LBL)

def generate_signal_ball(coef_agua, bvals, bvecs):
    gtab = gradient_table(bvals, bvecs)
    evals_bola = np.array([coef_agua,coef_agua,coef_agua])
    evecs=np.array([[1,0,0],[0,1,0],[0,0,1]])
    CSF = single_tensor(gtab, S0=1, evals=evals_bola, evecs=evecs, snr=None)
    return CSF.reshape(1,-1)

def base_SMT_csf_noisy(snr,bvals, bvecs,N, diff_free):
    smt_csf_noisy = 0
    for i in range(N):
        s_csf = generate_signal_ball(diff_free, bvals, bvecs)
        s_noisy = add_noise(s_csf, snr, S0=1, noise_type='rician')
        smt_csf_noisy +=  calc_SMT_matrix(bvals, s_noisy)

    smt_csf_noisy/=N

    return smt_csf_noisy



def SMT_teorica(icsf_wm, dpar_wm, b):
    smt_wm = np.zeros(b.shape[0] + 1)
    dext_wm = dpar_wm*(1-icsf_wm)

    eb_int_wm =np.sqrt(np.pi)*erf(np.sqrt(b*dpar_wm))/(2*np.sqrt(b*dpar_wm))
    eb_ext_wm = np.exp(-b*dext_wm) * np.sqrt(np.pi)*erf(np.sqrt(b*(dpar_wm - dext_wm)))/(2*np.sqrt(b*(dpar_wm - dext_wm)))
    eb_wm = icsf_wm*eb_int_wm + (1-icsf_wm)*eb_ext_wm


    smt_wm[0] = 1
    smt_wm[1:] = eb_wm

    return smt_wm
#function that estimates dpar with fixed search

def SMT_to_WMparams_fix(V, icsf_wm,model_csf,bvals,bvecs, snr, dpar_min, dpar_max, diff_free):
    b = np.unique(bvals)
    dpars = np.arange(dpar_min, dpar_max,0.005e-3)
    dpars = np.random.permutation(dpars)
    error_min = 1e9
    h_dot = 0.1
    N = 100
    smt_csf_noisy = base_SMT_csf_noisy(snr,bvals, bvecs,N,diff_free)


    b_wm = np.mean(V, axis=0)
    h_csf = np.mean(model_csf.predict(V))
    h_wm = 1 - h_csf - h_dot

    for dpar in dpars:

        smt_wm = SMT_teorica(icsf_wm, dpar, b[1:])

        base_prop = h_wm*smt_wm + h_csf*smt_csf_noisy + h_dot

        error = np.linalg.norm(b_wm - base_prop)

        if error< error_min:
            error_min = error
            base_f = base_prop
            h_wmff = h_wm
            h_csff = h_csf
            h_dotf = h_dot
            icsff = icsf_wm
            dparf = dpar
    return icsff, dparf, h_wmff, h_csff

#function that estimates dpar with fixed search
def SMT_to_WMparams3(V, model_csf,bvals,bvecs, snr,  dpar_min, dpar_max, diff_free):

    b = np.unique(bvals)

    dpars = np.arange(dpar_min, dpar_max, 0.005e-3)
    dpars = np.random.permutation(dpars)

    icsfs = np.arange(0.3,1.0,0.01)
    icsfs = np.random.permutation(icsfs)

    error_min = 1e9

    h_dot = 0.1

    N = 100
    smt_csf_noisy = base_SMT_csf_noisy(snr,bvals, bvecs,N,diff_free)


    b_wm = np.mean(V, axis=0)
    h_csf = np.mean(model_csf.predict(V))

    for dpar in dpars:
        for icsf_wm in icsfs:
            h_wm = 1 - h_csf - h_dot
            if h_csf < 0 or h_csf > 1:
                continue
            smt_wm = SMT_teorica(icsf_wm, dpar, b[1:])

            base_prop = h_wm*smt_wm + h_csf*smt_csf_noisy + h_dot
            error = np.linalg.norm(b_wm - base_prop)
            if error< error_min:
                    error_min = error
                    base_f = base_prop
                    h_wmff = h_wm
                    h_csff = h_csf
                    h_dotf = h_dot
                    icsff = icsf_wm
                    dpar_f = dpar
    return icsff, dpar_f, h_wmff, h_csff

def SMT_to_icsf_dpar_fixed_noisy_csf(b_wm, snr, H_csf, bvals, bvecs, dpar, dcsf):
    icsfs = np.arange(0.01,1.05,0.01) #es buen idea aumentar la resolucion
    min_error = 1e9
    b = np.unique(bvals)
    smt_csf = base_SMT_csf_noisy(snr,bvals, bvecs,100,dcsf)

    for icsf_wm in icsfs:
        smt_wm = SMT_teorica(icsf_wm, dpar, b[1:])

        h_wm = 1 - H_csf -0.1
        base_prop = h_wm*smt_wm + H_csf*smt_csf +0.1

        error = np.mean(np.abs(b_wm.reshape(-1,1) -base_prop.reshape(-1,1)))

        if error < min_error:
            min_error = error
            base_final = smt_wm
            icsf_final = icsf_wm
    return icsf_final
