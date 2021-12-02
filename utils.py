import healpy as hp
import numpy as np
from accelerate import minus, temperature, A_positive
from accelerate import A_negative, angular_distance, linearity
import pandas as pd
import itertools
from scipy.stats import norm
import camb


def getting_data(input_filename, istokes):
    """
    This function calls the data set,
    which is smoothed and the monopoles and dipoles are removed.
    We also remove \ell >1500, in accordance with Penrose et al.
    
    Input
    ------
    input_filename: path to map
    istokes: to get temperature field
    
    Output
    ------
    v: smooth temperature map according to Penrose et al.
        
    """
    
    read_map = hp.read_map(input_filename,
                           (istokes,),
                           nest=False,
                           dtype=None)
    remove_dipole = hp.pixelfunc.remove_dipole(read_map,
                                               nest=False,
                                               fitval=False)
    v = hp.sphtfunc.smoothing(remove_dipole, lmax=1500)

    return v


def getting_mask(input_filename, tmask):
    read_map = hp.read_map(input_filename,
                           (tmask,),
                           nest=False,
                           dtype=None).astype(np.bool_)
    return read_map


def define_grid(types, width):
    # Define the grid of the map
    step = 80
    epsilon = 0.0025

    P = np.arange(-np.pi, np.pi, 2*np.pi/step)
    
    if types == 'full':
        T=np.concatenate((np.arange(np.pi/9,np.pi/2,7*np.pi/(18*step)),
                          np.arange(-np.pi/2,-np.pi/9,7*np.pi/(18*step))))
        
    else:
        T = np.concatenate((np.arange(0, np.pi/2, 7*np.pi/(18*step)),
                            np.arange(-np.pi/2, 0, 7*np.pi/(18*step))))

    # And the size of the rings

    radius_in = np.arange(0.000,
                          0.0425,
                          epsilon)
    radius = (np.arange(0.000,
                        0.0425,
                        epsilon)+width)
    # 0.02 is for width
    return T, P, radius_in, radius


def define_mask(cleaning, N):

    if cleaning == 'full':
        mask = np.zeros(hp.nside2npix(N), dtype=bool)
        pixel_theta, pixel_phi = hp.pix2ang(N, np.arange(hp.nside2npix(N)))
        mask[(pixel_theta > np.pi/9) & (pixel_theta < np.pi-np.pi/9)] = 1
    else:
        path_to_mask = '/data/gravwav/lopezm/MasterThesis/RealData/'
        mask = getting_mask(path_to_mask+'COM_CMB_IQU-'+str(cleaning)+'_1024_R2.02_full.fits',
                            "TMASK")
    return mask


def compute_circle(N, t, p, r, r_in):
    # Compute query (outer circle) and query_in (inner circle)
    query = hp.query_disc(nside=N,
                          vec=hp.ang2vec(np.pi/2-t, p),
                          radius=r, inclusive=False, nest=False)
    query_in = hp.query_disc(nside=N,
                             vec=hp.ang2vec(np.pi/2-t, p),
                             radius=r_in, inclusive=False, nest=False)

    # Inverse intersection of query (outer circle) and query_in (inner circle)
    inner = minus(query, query_in)
    return inner


def drop_masked_pixels(v, inner):

    circle_temperature = v[inner]
    inner_unmasked = np.where(circle_temperature > -10**30)[0]
    return inner_unmasked


def store_data(radius_, coords, gradient, r_coeff):

    d = pd.DataFrame()
    d['radius'] = radius_
    d['coordinates'] = coords
    d['gradient'] = gradient
    d['r_coeff'] = r_coeff
    # d.drop(d[ d['gradient'] == -10 ].index, inplace=True)

    # Calculate by families of rings the normalized gradient
    d['std_gradient'] = d['gradient'].groupby(d['radius']).transform('std')
    d['gradient_norm'] = d['gradient']/d['std_gradient']

    return d


def store_final_data(radii, widths, coordinates, A_pos, A_neg, normalized_gradient, r_values):

    df = pd.DataFrame()
    df['radius'] = radii
    df['width'] = widths
    df['coordinates'] = coordinates
    df['A+'] = A_pos
    df['A-'] = A_neg
    df['Norm'] = normalized_gradient
    df['Pearson coefficient'] = r_values

    return df


def cuda_circle(v, width, N, types):
    # We need the data, name of the file, the width and NSIDE

    T, P, radius_in, radius = define_grid(types, width)

    radius_ = []
    gradient = []
    r_coeff = []
    coords = []

    mask = define_mask(types, N)
    v[mask] = hp.UNSEEN
    minimum = 1000000
    for t, p in itertools.product(T, P):
        for r, r_in in zip(radius, radius_in):

            inner = compute_circle(N, t, p, r, r_in)
            inner_new = drop_masked_pixels(v, inner)

            if len(inner_new) != 0:

                # pixels refers to the ID of pixels inside the annulus
                pixels = np.array(hp.pix2vec(nside=N,
                                             ipix=inner)).T
                # angular distance from center to pixels
                distance = angular_distance(hp.ang2vec(np.pi/2-t, p),
                                            pixels)
                if len(distance) < minimum: minimum = len(distance)
                # Compute slopes and Pearson coefficients
                grad, r_value = linearity(distance, temperature(inner, v))

                # Store values

                radius_.append(r)
                gradient.append(grad)
                r_coeff.append(r_value)
                coords.append([t, p])
    print('min size ', minimum)
    # Store in a Data Frame and drop masked values
    d = store_data(radius_, coords, gradient, r_coeff)
    # Obtain CDFs from each family of rings to calculate A+, A-
    grouped = d.groupby(['radius'])
    radii = []
    A_pos = []
    A_neg = []
    normalized_gradient = []
    r_values = []
    widths = []
    coordinates = []

    for x in grouped.groups:  # We group according to outer radius

        normalized_gradient.append(np.array(grouped.get_group(x)['gradient_norm']))
        widths.append(width)

        # Get Pearson coefficients and coordinates
        r_values.append(np.array(grouped.get_group(x)['r_coeff']))
        coordinates.append(np.array(grouped.get_group(x)['coordinates']))

        data = np.sort(np.array(grouped.get_group(x)['gradient_norm']))

        # Compute A functions
        F = norm.cdf(data)
        # Get normalized CDFs and radii
        radii.append(np.round(np.unique(grouped.get_group(x)['radius'])[0]-width, decimals=3))
        A_pos.append(A_positive(F))
        A_neg.append(A_negative(F))

    # Store in a Data Frame all parameters ordered according to outer radius
    df = store_final_data(radii, widths, coordinates,
                          A_pos, A_neg,
                          normalized_gradient, r_values)

    return (d, df)


def generate_cls():
    lmin, lmax=2, 2000 # \ell range where the monopole and dipole are zero
    #Set Planck 2018 cosmological parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.66, ombh2=0.02242,
                       omch2=0.11933, mnu=0.120,nnu=2.99,
                       omk=0.0007, tau=0.0561, YHe=0.242)
    pars.InitPower.set_params(As=np.exp(3.047)/10**10, ns=0.9665)

    #Set \ell max again
    pars.set_for_lmax(lmax, lens_potential_accuracy=1) 
    pars.Want_CMB = True #Because we want the CMB

    powers = camb.get_results(pars).get_cmb_power_spectra(pars, CMB_unit="muK")
    l = np.arange(lmin, lmax)
    #cl = powers['total'][lmin:lmax,0]/(l*(l+1))*2*np.pi

    # Ordering TT, EE, BB, TE angular power expectra (CL). 
    # We add 2 zeros because the monopoles and the dipoles are zero 
    # *WARNING* this is very important, if not we get other simulation complitely different

    cl_tt = np.zeros(2).tolist()
    cl_tt.extend((powers['total'][lmin:lmax,0]/(l*(l+1))*2*np.pi).tolist())
    cl_tt = np.array(cl_tt) #Simulation of the CMB temperature CL
    cl_ee = np.zeros(2).tolist()
    cl_ee.extend((powers['total'][lmin:lmax,1]/(l*(l+1))*2*np.pi).tolist())
    cl_ee = np.array(cl_ee) #Simulation of the CMB E polarization CL
    cl_bb = np.zeros(2).tolist()
    cl_bb.extend((powers['total'][lmin:lmax,2]/(l*(l+1))*2*np.pi).tolist())
    cl_bb = np.array(cl_bb) #Simulation of the CMB B polarization CL
    cl_te = np.zeros(2).tolist()
    cl_te.extend((powers['total'][lmin:lmax,3]/(l*(l+1))*2*np.pi).tolist())
    cl_te = np.array(cl_te)

    return cl_tt
