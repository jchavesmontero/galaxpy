Jonas Chaves-Montero

Argonne National Laboratory

October 2019

CONTENT OF THE FOLDER
---------------------

This folder contains all the star formation histories (SFHs) used to generate galaxy colors in Chaves-Montero et al. 2019. 

SFHs drawn from UniverseMachine and IllustrisTNG are located in "umachine_sfh.npy" and "illustris_sfh.npy", respectively. The times at which UniverseMachine and IllustrisTNG SFHs are tabulated can be found in "umachine_tz.npy" and "illustris_tz.npy", respectively. These files can be readed using using np.load. The colums of the array saved in *_sfh.npy are:

SFH (nmass, ntype, nobj, nsnap)

nmass: different halo masses, 4. Their values are 11.5, 12, 12.5, 13 Msun/h for UniverseMachine and 11.4, 11.8, 12.3, and 12.8 Msun/h for IllustrisTNG.
ntype: 0 for star-forming galaxies, 1 for quenched galaxies.
nobj: total number of galaxies of each type, 1000
nsnap: snapshots at which SFHs are tabulated (different for UniverseMachine and IllustrisTNG)
