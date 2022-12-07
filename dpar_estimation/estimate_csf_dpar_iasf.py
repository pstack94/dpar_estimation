import sys
sys.path.append('./Library')
import lib as lib
import argparse
import numpy as np
from tensorflow import keras
import nibabel as nib
parser = argparse.ArgumentParser(description='This function estimates the free diffusion compartment, mean Dpar and voxel to voxel IASF using Artificial Neuronal Networks and SphericalMean')
parser.add_argument('dwifilename', nargs='?', help = 'path and name of the DWI')
parser.add_argument('maskfilename', nargs='?', help = 'path and name of the Mask')
parser.add_argument('model', nargs='?', help = 'path and name of the ANN model')
parser.add_argument('dp_min', type = float, help = 'minimum parallel difusivity i.e 0.5e-3')
parser.add_argument('dp_max', type = float, help = 'maximum parallel difusivity i.e 1e-3')
parser.add_argument('dfree', type = float, help = 'free water difusivity i.e 1.44e-3')
parser.add_argument('bvalfilename', nargs='?', help = 'path and name of the bvals')
parser.add_argument('bvecfilename', nargs='?', help = 'path and name of the bvecs')
parser.add_argument('f_out', nargs='?', help = 'path where the output is going to be stored')
parser.add_argument('shells',nargs="+", type=int, help = 'type the protocol shells ie. -shells 0 500 2000 4500 6000 8000')
parser.add_argument('snr', type=int, default = 22, help = 'type the snr i.e: 22')
parser.add_argument('-type_search',nargs="?", default = 'free', help = 'type of search ie. free or fixed')
parser.add_argument('-iasf', type= float, default = 0.675, help = 'Select a fixed IASF ')

args = parser.parse_args()


#python estimate_csf.py /home/alram/NAS/NAS/datos/rat/taill_rat_verona_exvivo/CTR_07/DIFF_eddy_ms.nii /home/alram/NAS/NAS/datos/rat/taill_rat_verona_exvivo/CTR_07/mask_test.nii /home/alram/NAS/NAS/datos/rat/taill_rat_verona_exvivo/modelo_csf_verona_snr26.h5 /home/alram/NAS/NAS/datos/rat/taill_rat_verona_exvivo/CTR_07/DIFF_eddy_ms.bval /home/alram/NAS/NAS/datos/rat/taill_rat_verona_exvivo/CTR_07/prueba_csf.nii -shells 0 500 2000 4500 6000 8000

#open DWI and mask
vol, img = lib.open_vol(args.dwifilename)
mask,_ = lib.open_vol(args.maskfilename)
mask = mask!=0

#open and process bvals
bvals = lib.open_bvals(args.bvalfilename)
bvals = lib.proc_bvals(bvals, args.shells)
bvecs = lib.open_bvals(args.bvecfilename)

#compute spherical mean
V = lib.calc_SMT_matrix(bvals, vol[mask])
b_wm = np.mean(V, axis=0)
b = np.unique(bvals)
print(V.shape)

#open ANN
model_csf = keras.models.load_model(args.model)

#evaluate ANN
print('Evaluating Model')
h_csf = model_csf.predict(V)
print('Mean water compartment in ROI: ', np.mean(model_csf.predict(V)))

#save CSF map
vol_csf = np.zeros(mask.shape)
vol_csf[mask] = h_csf[:,0]
csf_output= nib.Nifti1Image(vol_csf, img.affine, header= img.header)
nib.save(csf_output, args.f_out+'/H_CSF.nii')

#perform free search or fixed search to find Dpar

if (args.type_search == 'free'): 
    print('Finding mean Dpar in ROI, free search: ')
    icsf_wm, dpar_wm, h_wm, h_csf2= lib.SMT_to_WMparams3(V, model_csf,bvals,bvecs,args.snr, args.dp_min, args.dp_max, args.dfree)
    np.savetxt(args.f_out+'/dpar.txt', np.array(dpar_wm).reshape(-1,1),  delimiter=',')
elif(args.type_search == 'fixed'): 
    print('Finding mean Dpar in ROI, fixed search: ')
    icsf_wm, dpar_wm, h_wm, h_csf2 = lib.SMT_to_WMparams_fix(V,args.iasf,model_csf,bvals,bvecs,args.snr, args.dp_min, args.dp_max, args.dfree)
    np.savetxt(args.f_out+'/dpar.txt', np.array(dpar_wm).reshape(-1,1),  delimiter=',')
else: 
    print('Not valid search type')
print('Mean Dpar in ROI: ', dpar_wm) 
#save dpar in txt file: 

#compute voxel to voxel ICSF
print('Computing voxel to voxel IASF') 
icsf_csf_noisy = []
for j in range(V.shape[0]):
    icsf_csf_noisy.append(lib.SMT_to_icsf_dpar_fixed_noisy_csf(V[j,:], args.snr, h_csf[j,0], bvals, bvecs, dpar = dpar_wm, dcsf=args.dfree))

#saveICSF
vol_icsf = np.zeros(mask.shape)
vol_icsf[mask] = icsf_csf_noisy
icsf_output= nib.Nifti1Image(vol_icsf, img.affine, header= img.header)
nib.save(icsf_output, args.f_out+'/H_IASF.nii')
print('Finished with no errors')

