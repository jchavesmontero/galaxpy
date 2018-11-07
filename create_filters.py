import os
import numpy as np

def create_top_hat_filters(params):
  r""" 
    Routine to create top-hat filters

    name_survey: survey name for this set of filters
    lambda_res: spectral resolution R
    lambda_in: starting wavelength of the first filter
    lambda_out: final wavelength of the last filter
    num_first: number associated to the first filter created here.
      It is useful if you want to create filters with different 
      resolutions

  """

  # check that all required keys are in params
  required_keys = ['name_survey', 
                   'lambda_res', 
                   'lambda_in', 
                   'lambda_out', 
                   'num_first']

  for key in required_keys:
    try:
      val = params[key]
    except KeyError:
      raise(key + ' is not defined in params')

  #mandatory keys in params
  name_survey = params['name_survey']
  lambda_res = float(params['lambda_res'])
  lambda_in = float(params['lambda_in'])
  lambda_out = float(params['lambda_out'])
  num_first = int(params['num_first'])
  #optional keys in params
  response = params.get('response', 1.)

  # if filter folder does not exist, create it
  path_one_lvl = os.path.dirname(os.path.dirname(__file__))
  folder_filters = path_one_lvl + '/filters/' + name_survey + '/'
  if os.path.isdir(folder_filters) == False:
    try:
      os.makedirs(folder_filters)
    except:
      raise(folder_filters+' cannot be created')


  lambda_in_0 = lambda_in*1

  n_filt = 0
  while(lambda_in_0 < lambda_out):
    ss = lambda_in_0/(2.*lambda_res-1.)
    xc = lambda_in_0 + ss
    wc = xc/lambda_res
    lambda_in2 = lambda_in_0 + wc
    lambda_in_0 = lambda_in2
    n_filt += 1

  for i in range(0, n_filt):
    ss = lambda_in/(2.*lambda_res-1.)
    xc = lambda_in + ss
    wc = xc/lambda_res
    lambda_in2 = lambda_in + wc
    step = int(wc)/100.

    wav  = np.arange(lambda_in-step*2, lambda_in2+step*2, step)

    resp = np.zeros(len(wav)) + response
    resp[0]  = 0
    resp[-1] = 0

    filename = (folder_filters + name_survey 
                + '-band_' + str(i + num_first) + '.filt')
    with open(filename, 'w') as file:
      np.savetxt(file, np.transpose([wav, resp]))

    lambda_in  = lambda_in2

  print(str(n_filt) + ' bands were created')