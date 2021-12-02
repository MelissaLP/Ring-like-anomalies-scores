from utils import getting_data, cuda_circle, generate_cls
import time
import healpy as hp
import numpy as np
import pandas as pd
import camb


print('Load data')
path = '/data/gravwav/lopezm/MasterThesis/RealData/'
path_store = '/data/gravwav/lopezm/MasterThesis/Results/'

sevem = getting_data(path+'COM_CMB_IQU-sevem_1024_R2.02_full.fits',
                     "I_STOKES")
smica = getting_data(path+'COM_CMB_IQU-smica_1024_R2.02_full.fits',
                     "I_STOKES")
nilc = getting_data(path+'COM_CMB_IQU-nilc_1024_R2.02_full.fits',
                    "I_STOKES")
commander = getting_data(path+'COM_CMB_IQU-commander_1024_R2.02_full.fits',
                         "I_STOKES")
full = getting_data(path+'LFI_SkyMap_070_1024_R2.01_full.fits',
                    "I_STOKES")


all_maps = [commander, full]
all_names = [ 'commander', 'full']

for maps, names in zip(all_maps, all_names):
    print(names)
    appended_data = []
    for radius in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
        start = time.time()
        data = cuda_circle(maps, radius, 1024, names)
        # store DataFrame in list
        appended_data.append(data[1])
        end = time.time()
        print(radius, end - start)
    # see pd.concat documentation for more info
    appended_data = pd.concat(appended_data)
    # ONLY use pickle for correct formating
    appended_data.to_pickle(path_store+'RealMaps/'+names+'.csv')

    
cl_tt = generate_cls()
names_sim = 'sevem'

for i in range(0, 25):  # ID of the simulation
    print(i)
    # From the CL we construct the sky map with synfast.
    # NSIDE=1024, \ell <=1500 and FWHM=10'
    # (in radians it is 0.00290888) according to Planck 2018
    map_cl = hp.sphtfunc.synfast(cl_tt,
                                 nside=1024,
                                 lmax=1500,
                                 fwhm=0.00290888,
                                 alm=False)
    sim_map = map_cl*10**(-6)
    appended_data = []
    start = time.time()
    for radius in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:

        data = cuda_circle(sim_map, radius, 1024, names_sim)
        # store DataFrame in list
        appended_data.append(data[1])

    end = time.time()
    print(radius, end - start)

    # see pd.concat documentation for more info
    appended_data = pd.concat(appended_data)
    # ONLY use pickle for correct formating
    appended_data.to_pickle(path_store+'SimulatedMaps/'+names+'/sim_map_'+str(i)+'.csv')
