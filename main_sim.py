from utils import getting_data, cuda_circle, generate_cls
import time
import healpy as hp
import numpy as np
import pandas as pd
import camb
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--id', metavar='1', type=int, nargs=1,
                    help='id of map', default=1)
parser.add_argument('--mask', metavar='1', type=str, nargs=1,
                    help='id of jobs', default=1)

args = parser.parse_args()
ids = args.id[0]
names_sim = args.mask[0]

print('Load data')

path_store = '/data/gravwav/lopezm/MasterThesis/Results/'
    
cl_tt = generate_cls()

# ID of the simulation
print(ids)
# From the CL we construct the sky map with synfast.
# NSIDE=1024, \ell <=1500 and FWHM=10'
# (in radians it is 0.00290888) according to Planck 2018

for i in range(ids*1, ids*1+1):
	map_cl = hp.sphtfunc.synfast(cl_tt,nside=1024,lmax=1500,fwhm=0.00290888,alm=False)
	sim_map = map_cl*10**(-6)
	appended_data = []
	#start = time.time()
	for radius in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
    		start = time.time()
    		data = cuda_circle(sim_map, radius, 1024, names_sim)
    		# store DataFrame in list
    		appended_data.append(data[1])
    		end = time.time()
    		print(radius, end - start)
	#end = time.time()
	#print(radius, end - start)

	# see pd.concat documentation for more info
	appended_data = pd.concat(appended_data)
	# ONLY use pickle for correct formating
	appended_data.to_pickle(path_store+'SimulatedMaps/'+names_sim+'/sim_map_'+str(i)+'_'+str(names_sim)+'.csv')

