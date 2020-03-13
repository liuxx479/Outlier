'''generates GRF for outlier detection training set'''

from scipy import *
import numpy as np
import os
from astropy.io import fits
from classy import Class
from lenstools import ConvergenceMap
from lenstools import GaussianNoiseGenerator
from astropy.units import deg
from emcee.utils import MPIPool 

########## fiducial parameters ##########

num_pixel_side = 256
side_angle = 4.27 * deg

# LCDM parameters
A_s = 2.1e-9
h=0.7
OmegaB = 0.046
OmegaM = 0.3
n_s = 0.97
tau = 0.054 ## only for primary CMB, not used for now


### derived parameters
ombh2 = OmegaB*h**2
omch2 = (OmegaM-OmegaB)*h**2
H0=h*100
### accuracy parameters
lmax=5000

##### Generate Clkk from class

def clkk_gen (Omega_m, A_se9, zs=1.0):
    A_s = A_se9*1e-9
    omch2 = (OmegaM-OmegaB)*h**2
    LambdaCDM = Class()
    LambdaCDM.set({'omega_b':ombh2,'omega_cdm':omch2,'h':h,'A_s':A_s,'n_s':n_s})
    LambdaCDM.set({'output':'mPk,sCl',
                   'P_k_max_1/Mpc':10.0,
                   'l_switch_limber':100,
                   'selection':'dirac',
                   'selection_mean':zs,
                   'l_max_lss':lmax,
                   'non linear':'halofit'
                  })
    # run class
    LambdaCDM.compute()

    si8=LambdaCDM.sigma8()

    cls=LambdaCDM.density_cl(lmax)
    ell=cls['ell'][2:]
    clphiphi=cls['ll'][0][2:]
    clkk=1.0/4 * (ell+2.0)*(ell+1.0)*(ell)*(ell-1.0)*clphiphi

    return si8, ell, clkk

######### GRF initiate

gen = GaussianNoiseGenerator(shape=(num_pixel_side,num_pixel_side),side_angle=side_angle)#,label="convergence")

A_se9_find = lambda om, S8: -2.1+7.915*S8/sqrt(om/0.3) 

def GRF_from_PS (omS8seed, zs=1.0):
    om, S8, iseed = omS8seed
    print iseed, om, S8
    #fn_GRF='GRFs/GRF_si%.4f_om%.4f.fits'%(si8, om)
    #fn_cl='cls/clkk_si%.4f_om%.4f.npy'%(si8, om)
    fn_GRF='GRFs/GRF_%06d.fits'%(iseed)
    fn_cl='cls/clkk_%06d.npy'%(iseed)
    A_se9 = A_se9_find(om, S8)
    si8, ell, clkk = clkk_gen (A_se9, om, zs=1.0)
    gaussian_map = gen.fromConvPower(np.array([ell,clkk]),seed=int(iseed),kind="linear",bounds_error=False,fill_value=0.0)
    gaussian_map.save(fn_GRF)
    save(fn_cl,[ell, clkk])
    return iseed, om, si8, S8, A_se9

####### compute
###seed(94720)
#S8_arr = (rand(20000)-0.5)*0.4+0.77
#om_arr = (rand(20000) - 0.5)*0.4 + 0.27
##A_se9_find = lambda om, S8: -2.1+7.915*S8/sqrt(om/0.3) 
#A_se9_arr = A_se9_find(om_arr, S8_arr)
#si8B_arr = S8_arr/sqrt(om_arr/0.3)
### savetxt('GRF_params.txt',array([om_arr, S8_arr, A_se9_arr]).T, header='om\tS8\tAs*1e9')
om_arr, S8_arr, A_se9_arr=loadtxt('GRF_params.txt').T

inputs=array([om_arr,S8_arr, arange(len(om_arr))]).T##omS8seed
#### test on laptop - pass
# out=map(GRF_from_PS, inputs[:3])

##### MPIPool
pool=MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

out=pool.map(GRF_from_PS, inputs)
pool.close()

savetxt('GRF_params_output.txt',out,header='seed\tom\tsi8\tS8\tAs*1e9')

print 'done-done-done'
