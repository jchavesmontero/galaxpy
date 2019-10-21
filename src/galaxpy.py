import os
import subprocess
import glob
import sys

import numpy as np
from scipy.integrate import simps
from natsort import natsorted
from astropy import units, constants

path_program = os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) + '/'

class galaxy_sed(object):

  def __init__(self,
         params,
         resolution = 'lr',
         model_name = 'Padova1994',
         library = 'BaSeL',
         library_vs = 2003,
         units = 'flambda',
         w1lambda = '700',
         w2lambda = '20000',
         SFH_tt = 0,
         SFH_sf = 0):

    '''
    metallicity:   Initial metallicity of the galaxy. Options
             supported:
             0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05, 0.1 [Padova1994]
             0.0004, 0.001, 0.004, 0.008, 0.019, 0.03 [Padova2000]
    imf:       IMF options:
              salpeter, chabrier, and kroupa
    resolution:    Resolution of BC03 templates
              - 'hr':     High resolution
              - 'lr':     Low resolution
    library:     Specify specific library (only valid for 2012 version) -- 'stelib','BaSeL'
    library_vs: Specify which version of BC03 -- 2003, 2012
    dust:      Dust attenuation embedded in GALAXEV or post-processing
             Dictionary: 
             flag : 0 GALAXEV 1 post-processing
             GALAXEV option (Bruzual & Fall 2000, BC03 eq.6)
              tau_V : grid
              tau_V from ISM : 0.3 with scatter
             Post-processing
              ext_law : calzetti, fitzpatrick07, fitzpatrick, odonnell, cardelli
              reddening : grid
    sed:       Return SED
             flag Y or N
              units Units to return the SEDs
                'lnu':     ergs/s/Hz
                'llambda':   ergs/s/Angs

              W1, W2 wavelength range to compute the SED
    '''

    # check that all required keys are in params
    required_keys = ['metallicity', 
                     'IMF',  
                     'dust', 
                     'survey_filt',
                     'flag_mag',
                     'em_lines',
                     'flag_IGM_ext',
                     'cosmo',
                     'seed',
                     'work_dir']

    for key in required_keys:
      try:
        val = params[key]
      except KeyError:
        raise Exception(key + ' is not defined in params')

    path_one_lvl   = os.path.dirname(os.path.dirname(__file__)) + '/'
    self.seed    = params['seed']
    self.galaxev_dir = path_one_lvl + 'bc03/'
    self.em_line_dir = path_one_lvl + 'gutkin/'
    # self.work_dir  = path_one_lvl + 'runs/' + params['seed'] + '/'
    self.work_dir  = params['work_dir']
    self.survey_name = params['survey_filt']
    self.survey_filt = path_one_lvl + 'filters/' + params['survey_filt'] + '/'
         
    self.metallicity   = params['metallicity']
    self.imf       = params['IMF']
    self.dust      = params['dust']
    self.flag_mag    = params['flag_mag']
    self.em_lines    = params['em_lines']
    self.flag_IGM_ext  = params['flag_IGM_ext']

    self.cosmo     = params['cosmo']
    self.resolution  = resolution
    self.model_name  = model_name
    self.library   = library
    self.library_vs  = library_vs
    self.units     = units
    self.w1lambda  = w1lambda
    self.w2lambda  = w2lambda
      
    self.check_input()

    if(self.imf == 'chab'):
      self.imf2 = 'chabrier'
    if(self.imf == 'kroup'):
      self.imf2 = 'kroupa'
    if(self.imf == 'salp'):
      self.imf2 = 'salpeter'

    self.input_ised = (self.galaxev_dir 
               + '/models/' 
               + self.model_name 
               + '/'
               + self.imf2
               +'/bc' 
               + str(library_vs) 
               + '_' 
               + self.resolution 
               + '_' 
               + self.library 
               + '_' 
               + self.metallicity_key[self.metallicity] 
               + '_' 
               + self.imf 
               + '_ssp.ised')

    self.csp_output = self.work_dir + self.seed + '_csp_out'
    self.gpl_output = self.work_dir + self.seed + '_gpl_out'

    # select filters old_version
    # self.select_filters()

    # export everything that we need to run galaxev
    self.define_env()

    if(np.isscalar(SFH_tt)):

      import pickle
      file_SFH  = (path_one_lvl 
            + 'runs/SFHs/' 
            + 'SFHn_'
            + 'gal_type_' 
            + params['gal_type']
            + '.pkl')

      dict_save =  pickle.load( open( file_SFH, "rb" ) )
      aa = np.argmin( abs(params['mpeak'] - dict_save['arr_mpeak']) )
      self.age_SFH = dict_save['arr_tt']
      self.SFH = dict_save['arr_SFH'][aa,:].flatten()

    else:
      self.age_SFH = SFH_tt
      self.SFH = SFH_sf
    # st_mass = simps(self.SFH, self.age_SFH)
    # print( format(st_mass, '.3e') )

    self.file_SFH = self.work_dir + self.seed + '_SFH.txt'
    np.savetxt(self.file_SFH, np.transpose([self.age_SFH, self.SFH]))

    # run CSP GALAXEV
    self.run_galaxev_csp()


  def check_input(self):

    if(self.model_name == 'Padova1994'):
      self.metallicity_key = {0.0001:'m22',
                  0.0004:'m32',
                  0.004:'m42',
                  0.008:'m52',
                  0.02:'m62',
                  0.05:'m72',
                  0.1:'m82'}
    elif(self.model_name == 'Padova2000'):
      self.metallicity_key = {0.0004:'m122',
                  0.001:'m132',
                  0.004:'m142',
                  0.008:'m152',
                  0.019:'m162',
                  0.03:'m72'}

    if self.metallicity not in self.metallicity_key.keys():
      raise Exception('Incorrect metallicity provided: ' 
              + str(self.metallicity) + '\n' +
              'Please choose from:' + str(self.metallicity_key.keys()))

    imf_keys = ['salp', 'chab', 'kroup']
    if self.imf not in imf_keys:
      raise Exception( 'Incorrect IMF provided: '+ self.imf +'\n'
              + 'Please choose from:' + imf_keys )

    # gal_type_keys = ['sf', 'qs']
    # if self.gal_type not in gal_type_keys:
    #   raise Exception( 'Incorrect gal_type provided: '+ self.gal_type +'\n'
    #           + 'Please choose from:' + gal_type_keys )

    ext_keys = ['N', 'galaxev', 'calzetti', 'fitzpatrick07', 
      'fitzpatrick', 'odonnell', 'cardelli']
    if self.dust['flag'] not in ext_keys:
      raise Exception('Dust flag not correct \n' +
              'Please choose from: ' + ext_keys)

    dust_keys = ['tau_V', 'etau_V']
    if(self.dust['flag'] == 'galaxev'):
      if not all (key in self.dust.keys() for key in dust_keys):
        raise Exception('Incorrect keys in dust,' +
          'you have to include tau_V and etau_V')

    dust_keys = ['Av']
    if( (self.dust['flag'] != 'galaxev') & (self.dust['flag'] != 'N') ):
      if not all (key in self.dust.keys() for key in dust_keys):
        raise Exception('Incorrect keys in dust,' +
          'you have to include Av')

    if not os.path.isdir(self.survey_filt):
      raise Exception('Filter directory not found at :' 
        + str(self.survey_filt) )

    if os.path.isdir(self.work_dir) == False:
      try:
        os.makedirs(self.work_dir)
      except:
        raise(self.work_dir + ' cannot be created')

    model_name_keys = ['Padova1994', 'Padova2000']
    if self.model_name not in model_name_keys:
      raise Exception('Incorrect model provided: '
              + self.model_name +'\n' +
              'Please choose from:' + model_name_keys )

    resolution_keys = ['hr', 'lr']
    if self.resolution not in resolution_keys:
      raise Exception('Incorrect resolution provided: '
              +str(self.resolution)+'\n' +
              'Please choose from: ' + resolution_keys )

    units_keys = ['flambda', 'fnu']
    if self.units not in units_keys:
      raise Exception('Incorrect flux units provided: ' 
              + str(self.units) + '\n' +
              'Please choose from: ' + units_keys )

    library_vs_keys = [2003, 2012]
    if self.library_vs not in library_vs_keys:
      raise Exception('Invalid library_vs: '+ self.library_vs 
              + '\n' + 'Please choose from: ' 
              + library_vs_keys)

    library_keys = ['stelib', 'BaSeL']
    if self.library not in library_keys:
      raise Exception('Incorrect library choice: '
              + library_keys +'\n' +
              'Please choose from: ' + library_keys)


  def define_env(self):
    ''' Needed to run GALAXEV. 
      Note: this is only working in Linux'''

    self.env_string = ('export FILTERS=' 
              + self.galaxev_dir 
              + 'src/FILTERBIN.RES;' 
              + 'export A0VSED='
              + self.galaxev_dir
              + 'src/A0V_KURUCZ_92.SED;'
              + 'export RF_COLORS_ARRAYS='
              + self.galaxev_dir
              + 'src/RF_COLORS.filters;' 
              + 'export SUNSED='
              + self.galaxev_dir
              + 'src/SUN_KURUCZ_92.SED;')


  def run_galaxev_csp(self):

    # In the presence of dust attenuation and emission lines, we need
    # to run the code twice. This is because we introduce emission lines
    # a posteriori, and we need to know the impact of dust attenuation
    # on them. This is only true for the galaxev dust attenuation model,
    # as it depends on the SFH. For other models it is not necessary

    if((self.dust['flag'] == 'galaxev') & (self.em_lines['flag'] == 'Y')):
      it = 2
    else:
      it = 1

    for ii in range(it):
      if((ii == 0) & (self.dust['flag'] == 'galaxev')):
        flag_dust = ('Y' 
              + '\n' 
              + self.dust['tau_V'] 
              + '\n' 
              + self.dust['etau_V'])
        name_out = self.csp_output
      else:
        flag_dust = 'N'
        name_out = self.csp_output + '_nd'

      csp_input = (self.input_ised 
             + '\n'
             + flag_dust 
             + '\n'
             + '0\n'
             + '6\n'
             + self.file_SFH
             + '\n'
             + name_out
             + '\n')
    
      csp_input_file = self.work_dir + self.seed + '_csp.in'
      with open(csp_input_file, 'w') as file: 
        file.write(csp_input)

      call_string = (self.env_string 
              + self.galaxev_dir 
              +'src/csp_galaxev < ' 
              + csp_input_file)

      # call csp_galaxev
      subprocess.call(call_string,
              cwd = self.work_dir, 
              shell=True, 
              # stdout = open(self.work_dir+'aa.txt', 'w'), 
              # stderr = open(self.work_dir+'ab.txt', 'w'))
              stdout=open(os.devnull,'w'), 
              stderr=open(os.devnull,'w'))



  def run_galaxev_gpl(self, zz0):

    # output zz, age, wav_em, lum_em

    self.zz = []
    self.age = []

    if(len(zz0) <= 50):
      zz = zz0
      it = 1
    else:
      # 50 is the maximum number of outputs from galaxevpl
      num = np.ceil(len(zz0)/50.)
      zz_arr = np.array_split(zz0, num)
      it = len(zz_arr)

    if((self.dust['flag'] == 'galaxev') & (self.em_lines['flag'] == 'Y')):
      it2 = 2
    else:
      it2 = 1

    for ii in range(it):
      if(len(zz0) > 50):
        zz = zz_arr[ii]

      age = np.array(self.cosmo.age(zz).value)
      self.zz.append(zz)
      self.age.append(np.around(age, decimals=3))
      ages = ','.join(age.astype(str))

      for jj in range(it2):
        if(jj == 0):
          infile = self.csp_output
          outfile = self.gpl_output+str(ii)+'.dat'
        else:
          infile = self.csp_output + '_nd'
          outfile = self.gpl_output+str(ii)+'_nd.dat'

        # call galaxevpl        
        gpl_input = (infile
               + '\n'
               + ages
               + '\n'
               + self.w1lambda
               + ','
               + self.w2lambda
               + '\n'
               + outfile)

        gpl_input_file = self.work_dir + self.seed + '_gpl.in'
        with open(gpl_input_file, 'w') as file: 
          file.write(gpl_input)

        verbose = 1
        if(verbose == 0):
          subprocess.call(self.galaxev_dir
                  + 'src/galaxevpl < '
                  + gpl_input_file,
                  cwd = self.work_dir, 
                  shell = True, 
                  # stdout = open(self.work_dir+'aa.txt', 'w'), 
                  # stderr = open(self.work_dir+'ab.txt', 'w'))
                  stdout = open(os.devnull, 'w'), 
                  stderr = open(os.devnull, 'w'))
        else:
          subprocess.call(self.galaxev_dir
                  + 'src/galaxevpl < '
                  + gpl_input_file,
                  cwd = self.work_dir, 
                  shell = True, 
                  stdout = open(self.work_dir+'aa.txt', 'w'), 
                  stderr = open(self.work_dir+'ab.txt', 'w'))

      self.read_gpl(it=ii)

    # once we have read everything, we convert luminosities to
    # fluxes and magnitudes
    self.zz = np.concatenate(self.zz)
    self.n_zz = len(self.zz)
    self.age = np.concatenate(self.age)
    self.lum_em = np.concatenate(self.lum_em, axis=1)
    self.lum2fluxmag()


  def read_gpl(self, it=0):

    # read gpl output (luminosity), introduce emission lines, and
    # apply IGM extinction
    # output wav_em, lum_em

    infile = self.gpl_output + str(it) + '.dat'
    # I want to get F_lambda erg cm^-2 s^-1 A^-1
    # The code is in Lsun/Angstrom
    fil = np.loadtxt(infile)
    if(it == 0):
      # Angstrom
      self.wav_em = fil[:, 0]
      self.lum_em = []

    # units Lsun/Angstrom
    self.lum_em.append(fil[:, 1:])

    # Introduce emission lines
    if(self.em_lines['flag'] == 'Y'):
      if(self.dust['flag'] == 'galxev'):
        infile_nd = self.gpl_output + str(it) + '_nd.dat'
        fil_em = np.loadtxt(infile_nd)
        dust_att = fil[:, 1:]/fil_em[:, 1:]
      else:
        dust_att = 0

      self.add_emlines(dust_att = dust_att,
                       it = it,
                       pmetal = self.em_lines['metal_line'],
                       plogio = self.em_lines['log_io'])

    # Apply IGM extinction
    if(self.flag_IGM_ext == 'Y'):
      self.igm_ext(it=it)


  def lum2fluxmag(self, ini_wav=2500, out_wav=12500):

    ind = np.where( (self.wav_em >= ini_wav) & (self.wav_em <= out_wav) )[0]
    self.wav = self.wav_em[ind]
    self.flux_obs = np.zeros((len(self.wav), self.n_zz))    
    
    # flux_lambda_obs = Lum_lambda_em / (4 pi D_L^2 (1+z))
    for ii in range(0, self.n_zz):
      if(self.zz[ii] > 0):
        # move sed to obs wav, interpolate, get results at the
        # self.wav (we want all spectra with the same wav)
        lum_at_obs = np.interp(self.wav, 
                     self.wav_em*(1+self.zz[ii]), 
                     self.lum_em[:, ii])

        dist_lum = self.cosmo.luminosity_distance(self.zz[ii]).value #Mpc        
        num = lum_at_obs * units.L_sun.to('erg/s') # units lum erg s^-1 A^-1
        # BC03 Eq. 8 
        #dist lum units Mpc -> cm
        den = 4 * np.pi * (dist_lum * units.Mpc.to('cm'))**2 * (1 + self.zz[ii])
        self.flux_obs[:, ii] = num/den        

    # apply dust attenuation now if model != galaxev
    if( (self.dust['flag'] != 'galaxev') & (self.dust['flag'] != 'N') ):
      self.apply_dust()

    if(self.flag_mag == 'Y'):    
    # read the filters where we are going to compute magnitudes
      self.read_filters()
      self.compute_obs_mags()


  def compute_obs_mags(self):

    # mag = -2.5 * log10 (i1/i2)
    # i1 = dlambda lambda flux_obs * R(lambda)
    # i2 = dlambda lambda C_lambda R(lambda)

    clight = constants.c.to('Angstrom/s').value # A s^-1
    sysAB_nu = 3.631e-20 # ergs s^-1 cm^-2 Hz^-1
    # F_lambda erg cm^-2 s^-1 A^-1

    self.obs_flux = np.zeros( (self.n_filters, self.n_zz) )
    self.obs_mag = np.zeros( (self.n_filters, self.n_zz) )

    for jj in range(0, self.n_filters):
      xx = self.filters['wav_'+str(jj)]
      yy0 = xx * self.filters['trans_'+str(jj)]

      den_f = simps(yy0, xx)

      sysAB_lambda = sysAB_nu * clight / xx**2
      yy = sysAB_lambda * yy0
      den_m = simps(yy, xx)

      for ii in range(0, self.n_zz):
        if(self.zz[ii] > 0):     
          flux_at_band_wav = np.interp(self.filters['wav_'+str(jj)], 
                         self.wav, 
                         self.flux_obs[:, ii])
                        
          yy = flux_at_band_wav * yy0
          num = simps(yy, xx)

          self.obs_mag[jj, ii] = -2.5 * np.log10(num/den_m)
          self.obs_flux[jj, ii] = num/den_f


  # best-fit parameters to SDSS 7 data (0.04<z<0.2)
  # standard model
  # vary pmetal
  def add_emlines(self,
          it = 0,
          dust_att = 0,
          pmetal = 0.014, 
          plogio = -3.5,
          pd2m = 0.3,
          phden = 100, 
          pC20 = 1., # C0 solar
          pIMF = 100):

    # col 1 log ionization param
    # keys_logio = -(np.arange(7)/2. + 1)
    # -1, -4
    # plogio = -1
    # very important .44, .1, .2, .05

    # col 2 dust-to-metal ratio
    # keys_dust2metal = np.array([0.1, 0.3, 0.5])
    # pd2m = 0.3
    # important .2, .05

    # col 3 hydrogen gas density (per cubic cm)
    # keys_hden = np.array([100, 1000])
    # phden = 100
    # not very important .05, .06

    # col 4 C/O ratio in units of solar value
    # keys_C2O = np.array([0.10, 0.14, 0.20, 0.27, 0.38, 0.52, 
    #   0.72, 1.00, 1.40])
    # pC20 = 0.38
    # not important, less than .05

    # col 5 cutoff IMF
    # keys_IMF = np.array([100, 300])
    # pIMF = 100
    # important .24, .14, .1 

    # col 6-23:

    #[OII]3727 
    # Hbeta 
    #[OIII]4959 
    #[OIII]5007 
    #[NII]6548 
    #Halpha 
    #[NII]6584 
    #[SII]6717 
    #[SII]6731 
    #NV1240 
    #CIV1548 
    #CIV1551 
    #HeII1640 
    #OIII]1661 
    #OIII]1666 
    #[SiIII]1883 
    #SiIII]1888 
    #CIII]1908

    if(it == 0):
      # Using line ratios from Gutkin et al. 2016
      keys_metal = np.array([0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.004, 
        0.006, 0.008, 0.010, 0.014, 0.017, 0.020, 0.030, 0.040])

      ii = np.argmin( abs(pmetal - keys_metal) )
      frac, _ = np.modf(keys_metal[ii])
      str0 = str(frac)[2:]
      if(len(str0) < 3):
        str0 += '0'

      self.line_wav = np.array([3727, 4862, 4959, 5007, 6548, 6564, 
        6584, 6717, 6731, 1240, 1548, 1551, 1640, 1661, 1666,
        1883, 1888, 1908])
      
      file_in = 'nebular_emission_Z' + str0 + '.txt'
      with open(self.em_line_dir + file_in, 'r') as file:
        fil = np.loadtxt(file)
      aa = np.where( (fil[:,0] == plogio) &
               (fil[:,1] == pd2m) &
               (fil[:,2] == phden) &
               (fil[:,3] == pC20) &
               (fil[:,4] == pIMF) )

      # Lsun per unit SFR [Msun/yr]
      self.line_rat = fil[aa, 5:].flatten()

      

    # Lyman alpha 1216
    # Ly alpha to H alpha ratio 8.7 (Hu et al. 1998)
    # From 9.1 to 11.6 Hummer & Storey et al. 1987
    # due to absorption in the position of the line, we have to 
    # multiply the line intensity by a factor of a few to see a line!
    # the problem is that we need to include nebular emission
    # this is for the future
    # lyabs = 1.e3
    # self.line_wav = np.append(self.line_wav, [1216])
    # self.line_rat = np.append(self.line_rat, [self.line_wav[5] * 8.7 * lyabs])
    
    
    # position of lines
    ind_lines = np.zeros(self.line_wav.shape[0], dtype=int)
    # account for resolution of the spectrum
    res_lambda = np.zeros(self.line_wav.shape[0])

    # number of lines
    nlines = len(self.line_wav)
    for jj in range(0, nlines):
      ind_lines[jj] = np.argmin( abs(self.line_wav[jj] - self.wav_em) )
      res_lambda[jj] = abs(self.wav_em[ind_lines[jj]] - self.wav_em[ind_lines[jj]+1])

    flag0 = np.isscalar(dust_att)
    # introduce lines
    len_arr = self.lum_em[it].shape[1]
    for ii in range(0, len_arr):
      ind = np.argmin( abs(self.age_SFH - self.age[it][ii]*1.e9) )
      lum_lines = self.line_rat * self.SFH[ind]
      if(flag0 == True):
        self.lum_em[it][ind_lines, ii] += lum_lines / res_lambda
      else:
        self.lum_em[it][ind_lines, ii] += lum_lines * dust_att[ind_lines, ii] / res_lambda


    # introduce properly lines only in high resolution spectra 
    # line_width 4 A
    # sigma_line = 4
    # for ii in range(0, self.n_zz):
    #   ind = np.argmin( abs(self.age_SFH - self.age[ii]) )
    #   SFR = self.SFH[ind]
    #   lum_lines = self.line_rat * SFR
    #   for jj in range(0, nlines):
    #     norm = (1./np.sqrt(2*np.pi)/sigma_line) * lum_lines[jj]
    #     line_prof = norm * np.exp(-0.5 * (self.wav_em - self.line_wav[jj])**2 / sigma_line**2)
    #     self.lum_em[:, ii] += line_prof
    


  def read_filters(self):
    # read filters
    filter_files = glob.glob(self.survey_filt 
                 + self.survey_name 
                 + "*")
    filter_files = natsorted(filter_files)

    self.n_filters = len(filter_files)
    self.filter_pivot = np.zeros(self.n_filters)
    self.filters = {}

    for ii, tempfile in enumerate(filter_files):
      with open(tempfile, 'r') as file_temp:
        fil = np.loadtxt(file_temp)
        self.filters['wav_'+ str(ii)] = fil[:,0]
        self.filters['trans_'+ str(ii)] = fil[:,1]
        # pivot wavelength Eq. A11 Tokunaga & Vacca 2005
        num = simps(fil[:,0] * fil[:,1], fil[:,0])
        den = simps(fil[:,1] / fil[:,0], fil[:,0])
        self.filter_pivot[ii] = np.sqrt(num/den)



  def igm_ext(self, it=0):
    # Becker et al. 2015
    # Extinction shortward Lyman alpha
    wav_LA = 1216.
    sig_LA = 44.88

    wav_LB = 1026.
    sig_LB = 7.18

    wav_LG = 972.
    sig_LG = 2.50

    len_arr = self.lum_em[it].shape[1]
    for ii in range(0, len_arr):

      # Barnett et al. 2017 Eq. 1
      if(self.zz[it][ii] <= 5.5):
        tauLA = 0.85 * ( (1.+self.zz[it][ii])/5. )**4.3
      else:
        tauLA = 2.63 * ( (1.+self.zz[it][ii])/6.5 )**11.
      tauLB = sig_LB/sig_LA*tauLA
      tauLG = sig_LG/sig_LA*tauLA
      
      extLA = np.exp(-tauLA)
      extLB = np.exp(-tauLB)
      extLG = np.exp(-tauLG)
      
      ind_LA = np.where(self.wav_em < wav_LA)[0]
      ind_LB = np.where(self.wav_em < wav_LB)[0]
      ind_LG = np.where(self.wav_em < wav_LG)[0]

      self.lum_em[it][ind_LA, ii] *= extLA
      self.lum_em[it][ind_LB, ii] *= extLB
      self.lum_em[it][ind_LG, ii] *= extLG




  def apply_dust(self):

    import extinction

    for ii in range(0, self.n_zz):
      if self.dust['flag'] == 'calzetti':
        self.flux_obs[:, ii] = extinction.apply(extinction.calzetti00(self.wav, 
          self.dust['Av'], 4.05), self.flux_obs[:, ii])
      elif self.dust['flag'] == 'cardelli':
        self.flux_obs[:, ii] = extinction.apply(extinction.ccm89(self.wav, 
          self.dust['Av'], 4.05), self.flux_obs[:, ii])
      elif self.dust['flag'] == 'odonnell':
        self.flux_obs[:, ii] = extinction.apply(extinction.odonnell94(self.wav, 
          self.dust['Av'], 4.05), self.flux_obs[:, ii])
      elif self.dust['flag'] == 'fitzpatrick':
        self.flux_obs[:, ii] = extinction.apply(extinction.fitzpatrick99(self.wav, 
          self.dust['Av'], 3.1), self.flux_obs[:, ii])
      elif self.dust['flag'] == 'fitzpatrick07':
        self.flux_obs[:, ii] = extinction.apply(extinction.fm07(self.wav, 
          self.dust['Av']), self.flux_obs[:, ii])


  # def select_filters_old(self):
  #   # create new file by appending new filters at the end
  #   # of filterfrm.res

  #   file_name = 'filterfrm.res'

  #   file_all_filt = self.galaxev_dir + 'src/' + file_name
  #   file_all_filt_temp = self.work_dir + file_name
  #   if os.path.isfile(file_all_filt):
  #     shutil.copyfile(file_all_filt, file_all_filt_temp)
  #   else:
  #     raise Exception('The file ' + file_all_filt + ' does not exist')

  #   filter_files = glob.glob(self.survey_filt + self.survey_name + "*")
  #   filter_files = natsorted(filter_files)

  #   with open(file_all_filt_temp, 'a+') as file:
  #     for tempfile in filter_files:
  #       with open(tempfile, 'r') as file_temp:
  #         tempfile_name_ext = os.path.basename(tempfile)
  #         t_name, t_ext = os.path.splitext(tempfile_name_ext)
  #         file.write('# ' + t_name + '\n')
  #         file.write(file_temp.read())

  #   input_string = (file_all_filt_temp 
  #           + '\nn\ny\n'
  #           + self.galaxev_dir 
  #           + 'src/FILTERBIN.RES')
  #   input_file = self.work_dir + self.seed + '_filt.in'
  #   with open(input_file, 'w') as file: 
  #     file.write(input_string)

  #   subprocess.call(self.galaxev_dir 
  #           + 'src/add_filters < ' 
  #           + input_file, 
  #           cwd = self.work_dir, 
  #           shell = True)

  #   file_name = 'filters.log'
  #   shutil.copyfile(self.work_dir + file_name, 
  #           self.galaxev_dir + 'src/' + file_name)



