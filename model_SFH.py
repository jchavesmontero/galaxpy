import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
import os

def sSFH(tc, sm, pMS, pQS, pTS):
    ''' Model to compute sSFH for central galaxies. 
        It consists of three parts: 
        
        MS (main sequence), third order polynomium as a function of cosmic time. 
        Four free parameters.
        
        QS (quenched sequence), first order polynomium as a function of cosmic time. 
        The y-intercep is a first order polynomium that depends on galaxy stellar mass 
        at redshift zero.
        Three free parameters.
        
        TS (transition sequence), smooth transition between MS and QS. 
        Two first order polynomia as a function of galaxy stellar mass.
        Four free parameters
        
        Returns:
        
        sSFR: for galaxies of stellar mass sm at a cosmic time tc
        tdeparture: first time at which the sSFR of a galaxy is 10% smaller than 
        the MS
        
        Model fitted using data for galaxies within 10 < log10( Mstar[Msun]) < 12
        It should work for log10( Mstar[Msun]) < 12. I checked that all galaxies with
        log10( Mstar[Msun]) < 9 are in the MS
        Do not trust the results for tc[Gyr]>0.4 
        
        '''
    
    def comp_MS(log10_tc, pMS):
        p30 = np.poly1d(pMS)
        return p30(log10_tc)
    
    def comp_QS(log10_tc, log10_sm, pQS):
        p10 = np.poly1d(pQS[1:])
        res = log10_tc * pQS[0] + p10(log10_sm)
        return res
    
    def comp_TS(log10_sm, pTS):
        p10 = np.poly1d(pTS[0:2])
        pa = p10(log10_sm)
        if(pa > 0):
            pa = 0
        p10 = np.poly1d(pTS[2:4])
        pb = p10(log10_sm)
        if(pb < 0):
            pb = 0        
        return 10.**pa, pb
    
    log10_tc = np.log10(tc)
    log10_sm = np.log10(sm)
    
    MS = comp_MS(log10_tc, pMS)
    QS = comp_QS(log10_tc, log10_sm, pQS)
    TS1, TS2 = comp_TS(log10_sm, pTS)
#     plt.plot(tc, MS)
#     plt.plot(tc, QS)
    
    # The QS decreases as a power law by increasing the cosmic time. 
    # As a consequence, if we go back in time, it increases and eventually surpasses the MS.
    # To model the sSFH we compute at what time this happens, before this time we use MS 
    # and after it QS.
    
    # If the current mass of a galaxy is smaller than 10^9 Msun, main sequence
    
    res = np.zeros(len(tc))

    # tcosmic greater than 0.5 Gyr to avoid weird things 
    # (second crossing of QS for low values of tc, not physical)
    ind0 = np.where( (MS <= QS) | (tc < 0.5) )[0]
    ind1 = np.where( (MS > QS) & (tc > 0.5) )[0]

    if(len(ind1) == 0):
        sSFR = 10.**MS
        tdeparture = -1
    else:
        tdeparture = np.min(tc[ind1])
        res[ind0] = MS[ind0]
        res[ind1] = QS[ind1]

        # To avoid a sharp transition between both regimes, we use a TS depending on stellar mass.
        if(TS1 > 0):
            res[ind1] -= TS1 * MS[ind1] * np.exp(-0.5*(TS2-tc[ind1])**2)

        # If due to this transition the resulting sSFH is larger than that of the MS,
        # we use that of the MS
        a = np.where(res[ind1] > MS[ind1])[0]
        res[ind1[a]] = MS[ind1[a]]

        # We were working in logarithmic space, going back to non-logarithmic space
        sSFR = 10.**res

        # Computing departure time -- sSFR 10% smaller than that of MS
        ind2 = np.where(10.**MS[ind1] > sSFR[ind1]*1.1)[0]
        if(len(ind2) > 0):
            tdeparture = np.min(tc[ind1[ind2]])
        else:
            tdeparture = -1
    
    return sSFR, tdeparture

def SMH(tc, mass, pSMH):
    npar = len(pSMH[0,:])
    par = np.zeros(npar)
    for jj in range(0, npar):
        p_f = np.poly1d(pSMH[:, jj])
        par[jj] = p_f(np.log10(mass))
    p20 = np.poly1d(par)
    xx = np.log10(tc)
    res = par[0] * np.tanh( (xx-0.5)/par[1] ) + par[2] + par[3]/(0.5 + xx)**2
    return 10.**res
    


def SFH(mass, file_out):

  path_one_lvl = os.path.dirname(os.path.dirname(__file__))

  # it only works between 10^9 and 10^12

  # Read data
  # dict_save = {'pMS': pMS, 'pQS': pQS, 'pTS': pTS, 'pSMH': pSMH}
  file_Name = path_one_lvl + '/data/params_SFH.pkl'
  with open(file_Name, 'rb') as fileObject:
      res = pickle.load(fileObject)

  pMS = res['pMS']
  pQS = res['pQS']
  pTS = res['pTS']
  pSMH = res['pSMH']

  nn = 200
  tc = np.logspace(0, np.log10(13.5), nn)

  res1 = SMH(tc, mass, pSMH)
  res2, tdep = sSFH(tc, mass, pMS, pQS, pTS)
  SFH = res1 * res2

#  plt.plot(tc, SFH)
#  plt.show()
  # with open(path_one_lvl + '/runs/SFHs/' + file_out, 'w') as fileObject:
  with open(file_out, 'w') as fileObject:
    np.savetxt(fileObject, np.transpose([tc * 1.e9, SFH]))

  return tc, SFH

if __name__ == '__main__':
  SFH(float(sys.argv[1]))
