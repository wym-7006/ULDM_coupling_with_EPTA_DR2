import scipy.stats as ss

from enterprise.signals import (gp_signals,white_signals,
                                      utils,signal_base,parameter,selections, deterministic_signals)
from enterprise import constants as const
from numpy import array
import numpy as np

import scipy.constants as sc

#TODO a exponential prior equals to a gamma prior of a=b=1.
def GammaPrior(value, a, loc, scale):
    """Prior function for Gamma parameters."""

    return ss.gamma.pdf(value, a, loc, scale)



def GammaSampler(a, loc, scale, size=None):
    """Sampling function for Gamma parameters."""

    return ss.gamma.rvs(a, loc, scale, size=size)



def Gamma(a, loc, scale, size=None):

    class Gamma(parameter.Parameter):
        _size = size
        _prior = parameter.Function(GammaPrior, a=a, loc=loc, scale=scale)
        _sampler = staticmethod(GammaSampler)
        _typename = parameter._argrepr("Gamma", a=a, loc=loc, scale=scale)

    return Gamma



############Theoretical value of log10_Phi based on m
theo_phis={0: array([-13.18842499]),
 1: array([-13.26842499]),
 2: array([-13.34842499]),
 3: array([-13.42842499]),
 4: array([-13.50842499]),
 5: array([-13.58842499]),
 6: array([-13.66842499]),
 7: array([-13.74842499]),
 8: array([-13.82842499]),
 9: array([-13.90842499]),
 10: array([-13.98842499]),
 11: array([-14.06842499]),
 12: array([-14.14842499]),
 13: array([-14.22842499]),
 14: array([-14.30842499]),
 15: array([-14.38842499]),
 16: array([-14.46842499]),
 17: array([-14.54842499]),
 18: array([-14.62842499]),
 19: array([-14.70842499]),
 20: array([-14.78842499]),
 21: array([-14.86842499]),
 22: array([-14.94842499]),
 23: array([-15.02842499]),
 24: array([-15.10842499]),
 25: array([-15.18842499]),
 26: array([-15.26842499]),
 27: array([-15.34842499]),
 28: array([-15.42842499]),
 29: array([-15.50842499]),
 30: array([-15.58842499]),
 31: array([-15.66842499]),
 32: array([-15.74842499]),
 33: array([-15.82842499]),
 34: array([-15.90842499]),
 35: array([-15.98842499]),
 36: array([-16.06842499]),
 37: array([-16.14842499]),
 38: array([-16.22842499]),
 39: array([-16.30842499]),
 40: array([-16.38842499]),
 41: array([-16.46842499]),
 42: array([-16.54842499]),
 43: array([-16.62842499]),
 44: array([-16.70842499]),
 45: array([-16.78842499]),
 46: array([-16.86842499]),
 47: array([-16.94842499]),
 48: array([-17.02842499]),
 49: array([-17.10842499]),
 50: array([-17.18842499]),
 51: array([-17.26842499])}

eta=1/3
dt_eta=-16/3
C_n=0.048
zeta=1
K_A=0.83
C_A=0.110
xi_A=4+K_A
m_u=2.32
m_d=4.71
C_A_hat=C_A*(m_u+m_d)**2/(2*m_u*m_d)



@signal_base.function
def usdm_delay(toas, effect_type="grav_effect", coupl_type="gluon",log10_Phi=-15,
               log10_A_g=-15,log10_A_m=-15,log10_A_u=-15,log10_A_e=-15,log10_A_p=-15,
               log10_mass=-23, phase_e=0, phase_p=0, norm_Phi_p=1, norm_Phi_e=1, tref=0):
    """
    Ultralight scalar-filed dark matter delay term in TOAs.
    Example: J1810-03227 in Nataliya K. Porayko et al, 1810.03227.
    """
    ##### sensitive_params for gluon, muon(m_hat), mu, electron, photon 
    sensitive_params_p=np.array([eta+dt_eta, C_n*(eta+dt_eta), eta*6*10**(-3), eta*5*10**(-5),0])   ########y_mu=6*10**{-3},we scale it                                                                                                here for the prior of A_mu
    sensitive_params_e=np.array([zeta, zeta*(C_n+C_A_hat),0,1+zeta,xi_A])
    
   
    toas -= tref
    
    coupl_types=["gluon","quark","muon","electron","photon"]
        
    coupl_indx=coupl_types.index(coupl_type)
    
    
    # print(f"10**log10_As[0]:{log10_A_g}")
    
    freq_grav=2*10**log10_mass*sc.eV/sc.h
    freq_coup=10**log10_mass*sc.eV/sc.h
    
    if effect_type=="grav_effect":
        grav_eff_wf = norm_Phi_e * np.sin(2*np.pi*freq_grav* toas + 2 * phase_e) - norm_Phi_p * np.sin(2*np.pi*freq_grav* toas + 2 * phase_p)
        grav_eff= 10**log10_Phi/(2*np.pi)/freq_grav * grav_eff_wf 
        
        return grav_eff
    
    elif effect_type=="coupl_effect":
        As=np.array([10**log10_A_g,10**log10_A_m,10**log10_A_u,10**log10_A_e,10**log10_A_p])
        coupl_eff = np.sqrt(norm_Phi_e)*np.dot(As[coupl_indx]/(2*np.pi)/freq_coup, sensitive_params_e[coupl_indx]) * np.sin(2*np.pi*freq_coup * toas + phase_e)+np.sqrt(norm_Phi_p)* np.dot(As[coupl_indx]/(2*np.pi)/freq_coup, sensitive_params_p[coupl_indx]) * np.sin(2*np.pi*freq_coup* toas + phase_p)
        
        return coupl_eff
    
    elif effect_type=="total_effect":
        grav_eff_wf = norm_Phi_e * np.sin(2*np.pi*freq_grav* toas + 2 * phase_e) - norm_Phi_p * np.sin(2*np.pi*freq_grav* toas + 2 * phase_p)
        grav_eff= 10**log10_Phi/(2*np.pi)/freq_grav * grav_eff_wf 
        
        As=np.array([10**log10_A_g,10**log10_A_m,10**log10_A_u,10**log10_A_e,10**log10_A_p])
        coupl_eff = np.sqrt(norm_Phi_e)*np.dot(As[coupl_indx]/(2*np.pi)/freq_coup, sensitive_params_e[coupl_indx]) * np.sin(2*np.pi*freq_coup * toas + phase_e)+np.sqrt(norm_Phi_p)* np.dot(As[coupl_indx]/(2*np.pi)/freq_coup, sensitive_params_p[coupl_indx]) * np.sin(2*np.pi*freq_coup* toas + phase_p)
        
        return  grav_eff+coupl_eff
    else:
        print("Error: No effect_type matched!")
        
        


def scalar_dm_block(prior = 'uniform', effect_type="grav_effct",  coupl_type="gluon", log10_Phi_low =-20, log10_Phi_high=-12,log10_A_low=-20, log10_A_high=-10, masstype='bin', mass_indx=0, corr="uncorr", phase_0=0, phase_1=2*np.pi, tref=0, name="usdm"):
    """
    Returns ultralight scalar-filed dark matter delay:
    :param log10_Phi_low, log10_Phi_low:
        amplitude of Phi that encode the information of scalar mass.
    :param log10_freq_high, log10_freq_low:
        frequency scope, is associated with the scalar mass by f=2m c^2/h.
    :range_indx:
        the frequency to take when the frequency scope is devided into 100 pieces.
    :param phase_0, phase_1:
        oscilating phase scope on Earth and Pulsar.
    :param name: Name of signal
    :return usdm_delay:
        ultralight scalar-field dark matter delay waveform.
    """
    log10_mass_low=-24
    log10_mass_high=-22
    
    log10_mass_range = np.linspace(log10_mass_low, log10_mass_high, 51)
    
    if masstype=="bin":
        if mass_indx==50:
            log10_m_low=log10_mass_range[mass_indx]
            log10_m_high=-21.96
            
        else:
            log10_m_low=log10_mass_range[mass_indx]
            log10_m_high=log10_mass_range[mass_indx+1]
    
        log10_m = parameter.Uniform(log10_m_low, log10_m_high)("{}_log10_m".format(name)) 
        
        print(f"The log10 mass bin used in this scanning is: {log10_m_low, log10_m_high}.")
        
        log10_Phi_high=1/2*(theo_phis[mass_indx][0]+theo_phis[mass_indx+1][0])
        print(f"the upper limit of log10_Phi is {log10_Phi_high}." )
            
    
    elif masstype=="fix":
        log10_m = parameter.Constant(log10_mass_range[mass_indx])("{}_log10_m".format(name)) 
        
        print(f"The log10 mass is fixed at: {log10_m}.")
        
        log10_Phi_high=theo_phis[mass_indx][0]
        print(f"the upper limit of log10_Phi is {log10_Phi_high}." )
        
    else:
        print("No mass choice is matched!")
        
        
    
    ########## g for g, m for m_hat, u for \mu, e for e, p for \gamma
    log10_Phi=-15
    log10_A_g=-15
    log10_A_m=-15
    log10_A_u=-15
    log10_A_e=-15
    log10_A_p=-15
    
    coupl_types=["gluon","quark","muon","electron","photon"]
        
    coupl_indx=coupl_types.index(coupl_type)
    
    if prior == 'uniform':
        if effect_type=="total_effect":
            log10_Phi = parameter.LinearExp(log10_Phi_low, log10_Phi_high)("{}_log10_Phi".format(name))
            if coupl_type=="gluon":
                log10_A_g = parameter.LinearExp(log10_A_low, log10_A_high)("{}_log10_Ag".format(name))
            elif coupl_type=="quark":
                log10_A_m = parameter.LinearExp(log10_A_low, log10_A_high)("{}_log10_Am".format(name))
            elif coupl_type=="muon":
                log10_A_u = parameter.LinearExp(log10_A_low+2, log10_A_high+2)("{}_log10_Au".format(name))
            elif coupl_type=="electron":
                log10_A_e = parameter.LinearExp(log10_A_low, log10_A_high)("{}_log10_Ae".format(name))
            elif coupl_type=="photon":
                log10_A_p = parameter.LinearExp(log10_A_low, log10_A_high)("{}_log10_Ap".format(name))
            else:
                print("Error: No coupl_type matched!")
        elif effect_type=="grav_effect":
                log10_Phi = parameter.LinearExp(log10_Phi_low, -12)("{}_log10_Phi".format(name))
        elif effect_type=="coupl_effect":
            if coupl_type=="gluon":
                log10_A_g = parameter.LinearExp(log10_A_low, log10_A_high)("{}_log10_Ag".format(name))
            elif coupl_type=="quark":
                log10_A_m = parameter.LinearExp(log10_A_low, log10_A_high)("{}_log10_Am".format(name))
            elif coupl_type=="muon":
                log10_A_u = parameter.LinearExp(log10_A_low+1, log10_A_high+1)("{}_log10_Au".format(name))
            elif coupl_type=="electron":
                log10_A_e = parameter.LinearExp(log10_A_low+2, log10_A_high+2)("{}_log10_Ae".format(name))
            elif coupl_type=="photon":
                log10_A_p = parameter.LinearExp(log10_A_low+2, log10_A_high+2)("{}_log10_Ap".format(name))
            else:
                print("Error: No coupl_type matched!")
            
        else:
            print("Error: No effect_type matched!")
       
        
        
    if prior == 'log-uniform':
        if effect_type=="total_effect":
            log10_Phi = parameter.Uniform(log10_Phi_low, log10_Phi_high)("{}_log10_Phi".format(name))
            if coupl_type=="gluon":
                log10_A_g = parameter.Uniform(log10_A_low, log10_A_high)("{}_log10_Ag".format(name))
            elif coupl_type=="quark":
                log10_A_m = parameter.Uniform(log10_A_low, log10_A_high)("{}_log10_Am".format(name))
            elif coupl_type=="muon":
                log10_A_u = parameter.Uniform(log10_A_low+1, log10_A_high+1)("{}_log10_Au".format(name))
            elif coupl_type=="electron":
                log10_A_e = parameter.Uniform(log10_A_low+2, log10_A_high+2)("{}_log10_Ae".format(name))
            elif coupl_type=="photon":
                log10_A_p = parameter.Uniform(log10_A_low+2, log10_A_high+2)("{}_log10_Ap".format(name))
            else:
                print("Error: No coupl_type matched!")
            
        elif effect_type=="grav_effect":
            log10_Phi = parameter.Uniform(log10_Phi_low, -12)("{}_log10_Phi".format(name))
        elif effect_type=="coupl_effect":
            if coupl_type=="gluon":
                log10_A_g = parameter.Uniform(log10_A_low, log10_A_high)("{}_log10_Ag".format(name))
            elif coupl_type=="quark":
                log10_A_m = parameter.Uniform(log10_A_low, log10_A_high)("{}_log10_Am".format(name))
            elif coupl_type=="muon":
                log10_A_u = parameter.Uniform(log10_A_low+1, log10_A_high+1)("{}_log10_Au".format(name))
            elif coupl_type=="electron":
                log10_A_e = parameter.Uniform(log10_A_low+2, log10_A_high+2)("{}_log10_Ae".format(name))
            elif coupl_type=="photon":
                log10_A_p = parameter.Uniform(log10_A_low+2, log10_A_high+2)("{}_log10_Ap".format(name))
            else:
                print("Error: No coupl_type matched!")
            
        else:
            print("Error: No effect_type matched!")
             
    
    # the phase of the pulsar and the earth
    usdm_phase_p = parameter.Uniform(phase_0, phase_1)
    phase_e = parameter.Uniform(phase_0, phase_1)("{}_phase_e".format(name))
    
    #  normalized factor 
    if corr=="uncorr":
        norm_Phi_p = Gamma(1, 0, 1)
        norm_Phi_e = Gamma(1, 0, 1)("{}_normPhi_e".format(name))
    elif corr=="pulsar-corr":
        norm_Phi_e = Gamma(1, 0, 1)("{}_normPhi_e".format(name))
        norm_Phi_p = norm_Phi_e
    elif corr=="full-corr": 
        norm_Phi_p = parameter.Constant(1)
        norm_Phi_e = parameter.Constant(1)("{}_normPhi_e".format(name))
    else:
        print("No correlation pattern is given!")

    
    wf = usdm_delay(effect_type=effect_type, coupl_type=coupl_type, log10_Phi=log10_Phi,
                    log10_A_g=log10_A_g,log10_A_m=log10_A_m,log10_A_u=log10_A_u,log10_A_e=log10_A_e,log10_A_p=log10_A_p,
                    log10_mass=log10_m, phase_e=phase_e, phase_p=usdm_phase_p, norm_Phi_p=norm_Phi_p, norm_Phi_e=norm_Phi_e,
                    tref=tref)
    
    
    scalar_dm_delay = scalar_dm_Signal(wf, name=name)
    
    return scalar_dm_delay


def scalar_dm_Signal(wf, name='usdm'):

    BaseClass = deterministic_signals.Deterministic(wf, name=name)

    class scalar_dm_Signal(BaseClass):

        def __init__(self, psr):
            super(scalar_dm_Signal, self).__init__(psr)

    return scalar_dm_Signal