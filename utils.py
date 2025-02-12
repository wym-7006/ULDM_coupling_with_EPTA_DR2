from glob import glob
import json, os, pickle
from enterprise.signals import (gp_signals,white_signals,
                                      utils,signal_base,parameter,selections, deterministic_signals)
from enterprise_extensions import chromatic as chrom
from enterprise_extensions import model_utils, blocks
from enterprise_extensions.blocks import (bwm_block, bwm_sglpsr_block,common_red_noise_block,
                                          white_noise_block, red_noise_block,
                                          dm_noise_block,
                                          chromatic_noise_block)
from enterprise.pulsar import Pulsar
from enterprise import constants as const
from enterprise_extensions import deterministic
from enterprise_extensions import dropout as do
from enterprise_extensions.chromatic.solar_wind import solar_wind_block
from enterprise_extensions.timing import timing_block
from enterprise.signals.signal_base import LogLikelihood

import scipy.constants as sc
import numpy as np

# from scipy.stats import gamma as Gamma
from numpy import array


import functools
from collections import OrderedDict

from uldm_block import scalar_dm_block



eV=sc.electron_volt

h = sc.Planck

transfactor= 10**(-23)*eV/h

yr = sc.Julian_year

fyr = 1.0 / yr


# BASEDIR = os.path.expanduser("/public/home/wuym/projects/working/PTA_2023/epta-dr2/EPTA-DR2/scripts_gwb")
BASEDIR = os.path.expanduser("~/projects/working/PTA_2023/epta-dr2/EPTA-DR2/scripts_gwb")


def get_pulsars(psrlist, DATADIR, cache_dir,ephem='DE440', cache_file="psrs_cache.pkl", use_cache=True):
    """
    Reads in list of pulsar names and ephemeris version
    and returns list of instantiated enterprise Pulsar objects.
    """
    
    parfiles = sorted(glob(DATADIR + '/J*/*.par'))
    timfiles = sorted(glob(DATADIR + '/J*/*_all.tim'))
    
    parfiles = [x for x in parfiles if x.split('/')[-1].split('.')[0] in psrlist]
    timfiles = [x for x in timfiles if x.split('/')[-1].split('_')[0] in psrlist]
    

    # check for cached file
    file_dir= BASEDIR + cache_dir + DATADIR.split('/')[-2] + '/'
    os.makedirs(file_dir, exist_ok=True) 
    cache_file = os.path.join(file_dir, cache_file)
    print(f"cache_file:{ cache_file}")
    if use_cache and os.path.exists(cache_file):
        print('Reading pulsars from cached file.\n')
        with open(cache_file, 'rb') as fin:
            psrs = pickle.load(fin)
    else:
        psrs = []
        for par, tim in zip(parfiles, timfiles):
            pname = par.split('/')[-1].split('.')[0]
            psrs.append(Pulsar(par, tim, ephem=ephem))

        print('Writing pulsars to cache.\n')
        with open(cache_file, 'wb') as fout:
            pickle.dump(psrs, fout)  
    print(f"Finish loading {len(psrs)} pulsars.")

    return psrs








######### USDM block 

def model_ULDM(psrs, tm_var=False, tm_linear=False, tmparam_list=None,
                  tm_sigma=5., tm_svd=False, tm_norm=True, tm_marg=False,
                  dense_like=False, Tspan_common=None, common_components=30,
                  common_psd='powerlaw', common_idx=None, noisedict=None,
                  white_var=False, white_global=False, tnequad=False,
                  inc_ecorr=False, select='backend', ecorr_select='nanograv',
                  orf='crn', orf_names=None, orf_bins=None, orf_ifreq=0,
                  leg_lmax=5, log10_A_common=None, gamma_common=None,
                  gammamin_common=None, gammamax_common=None, delta_common=None,
                  common_modes=None, common_logmin=None, common_logmax=None,
                  common_logf=False, common_fmin=None, common_fmax=None,
                  upper_limit=False, upper_limit_red=None, upper_limit_dm=None,
                  upper_limit_chrom=None, upper_limit_common=None,
                  bayesephem=False, be_type='setIII_1980', sat_orb_elements=False,
                  is_wideband=False, use_dmdata=False, tndm=False,
                  dm_var=False, Tspan_dm=None, dm_components=30,
                  dm_type='gp', dm_kernel='diag', dm_psd='powerlaw',
                  dm_select=None, dm_modes=None, dm_annual=False,
                  num_dmdips=1, dmpsr_list=['J1713+0747'],
                  dm_expdip_sign='negative', dm_expdip_idx=2,
                  dm_expdip_tmin=None, dm_expdip_tmax=None,
                  dm_logmin=None, dm_logmax=None,
                  dm_logf=False, dm_fmin=None, dm_fmax=None,
                  gamma_dm=None, gammamin_dm=None, gammamax_dm=None,
                  dm_chrom=False, Tspan_chrom=None, chrom_components=30,
                  dmchrom_kernel='nondiag',
                  chrom_psd='powerlaw',
                  chrom_idx=4, chrom_select=None, chrom_modes=None,
                  chrom_logmin=None, chrom_logmax=None,
                  chrom_logf=False, chrom_fmin=None, chrom_fmax=None,
                  gamma_chrom=None, gammamin_chrom=None, gammamax_chrom=None,
                  red_var=True, Tspan_red=None, red_components=30,
                  red_psd='powerlaw', red_select=None, red_modes=None,
                  wgts=None, logfreq=False, nmodes_log=10,
                  logmin=None, logmax=None, tnfreq=False,
                  logf=False, fmin=None, fmax=None,
                  gamma=None, gammamin=None, gammamax=None,
                  red_breakflat=False, red_breakflat_fq=None,
                  coefficients=False, pshift=False, pseed=None,
                  psr_models=False, psrnoise_models=None,
                  dropout=False, dropout_psr='all', dropout_common=False,
                  dropbin=False, dropbin_psr='all', dropbin_common=False,
                  dropbin_min=10, k_threshold=0.5, common_select=None,
                  extra_sigs=None, flagname='group', flagval=None,
                  uldm_term=False,
                  uldm_type="Scalar",
                  uldm_effect_type="grav_effect",
                  uldm_coupl_type="gluon",
                  uldm_upper_limit=False,
                  # uldm_log10_freq=None,
                  uldm_mass="bin",
                  uldm_mass_indx=0,
                  uldm_corr="full-corr",
                  uldm_osc_orient=None):
    """
    Reads in list of enterprise Pulsar instances and returns a PTA
    object instantiated with user-supplied options.

    :param tm_var: boolean to vary timing model coefficients.
        [default = False]
    :param tm_linear: boolean to vary timing model under linear approximation.
        [default = False]
    :param tmparam_list: list of timing model parameters to vary.
        [default = None]
    :param tm_sigma: number of sigmas to vary the timing model parameters.
        [default = 5.]
    :param tm_svd: stabilize timing model designmatrix with SVD.
        [default = False]
    :param tm_norm: normalize the timing model design matrix, or provide custom
        normalization. Alternative to 'tm_svd'.
        [default = True]
    :param noisedict: Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
        [default = None]
    :param white_var: boolean for varying white noise or keeping fixed.
        [default = False]
    :param Tspan: timespan assumed for describing stochastic processes,
        in units of seconds. If None provided will find span of pulsars.
        [default = None]
    :param modes: list of frequencies on which to describe red processes.
        [default = None]
    :param wgts: sqrt summation weights for each frequency bin, i.e. sqrt(delta f).
        [default = None]
    :param logfreq: boolean for including log-spaced bins.
        [default = False]
    :param nmodes_log: number of log-spaced bins below 1/T.
        [default = 10]
    :param common_psd: psd of common process.
        ['powerlaw', 'spectrum', 'turnover', 'turnover_knee,', 'broken_powerlaw']
        [default = 'powerlaw']
    :param common_components: number of frequencies starting at 1/T for common process.
        [default = 30]
    :param log10_A_common: value of fixed log10_A_common parameter for
        fixed amplitude analyses.
        [default = None]
    :param gamma_common: fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
        [default = None]
    :param common_logmin: specify lower prior for common psd. This is a prior on log10_rho
        if common_psd is 'spectrum', else it is a prior on log amplitude
    :param common_logmax: specify upper prior for common psd. This is a prior on log10_rho
        if common_psd is 'spectrum', else it is a prior on log amplitude
    :param orf: comma de-limited string of multiple common processes with different orfs.
        [default = crn]
    :param orf_names: comma de-limited string of process names for different orfs. Manual
        control of these names is useful for embedding model_general within a hypermodel
        analysis for a process with and without hd correlations where we want to avoid
        parameter duplication.
        [default = None]
    :param orf_ifreq:
        Frequency bin at which to start the Hellings & Downs function with
        numbering beginning at 0. Currently only works with freq_hd orf.
        [default = 0]
    :param leg_lmax:
        Maximum multipole of a Legendre polynomial series representation
        of the overlap reduction function.
        [default = 5]
    :param upper_limit_common: perform upper limit on common red noise amplitude. Note
        that when perfoming upper limits it is recommended that the spectral index also
        be fixed to a specific value.
        [default = False]
    :param upper_limit: apply upper limit priors to all red processes.
        [default = False]
    :param red_var: boolean to switch on/off intrinsic red noise.
        Dictionary with pulsar name and number of red noise processes.
        [default = True]
    :param red_psd: psd of intrinsic red process.
        ['powerlaw', 'spectrum', 'turnover', 'tprocess', 'tprocess_adapt']
        [default = 'powerlaw']
    :param red_components: number of frequencies starting at 1/T for intrinsic red process.
        Dictionary with pulsar name and number of frequencies.
        [default = 30]
    :param upper_limit_red: perform upper limit on intrinsic red noise amplitude. Note
        that when perfoming upper limits it is recommended that the spectral index also
        be fixed to a specific value.
        [default = False]
    :param red_select: selection properties for intrinsic red noise.
        Dictionary with pulsar name and selection.
        ['backend', 'band', 'band+', None]
        [default = None]
    :param red_breakflat: break red noise spectrum and make flat above certain frequency.
        [default = False]
    :param red_breakflat_fq: break frequency for 'red_breakflat'.
        [default = None]
    :param bayesephem: boolean to include BayesEphem model.
        [default = False]
    :param be_type: flavor of bayesephem model based on how partials are computed.
        ['orbel', 'orbel-v2', 'setIII', 'setIII_1980']
        [default = 'setIII_1980']
    :param is_wideband: boolean for whether input TOAs are wideband TOAs. Will exclude
        ecorr from the white noise model.
        [default = False]
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if is_wideband.
        [default = False]
    :param dm_var: boolean for explicitly searching for DM variations.
        Dictionary with pulsar name and boolean.
        [default = False]
    :param dm_type: type of DM variations.
        Dictionary with pulsar name and type. Only 'gp' will add the DM variations.
        ['gp', other choices selected with additional options; see below]
        [default = 'gp']
    :param dm_psd: psd of DM GP.
        ['powerlaw', 'spectrum', 'turnover', 'tprocess', 'tprocess_adapt']
        [default = 'powerlaw']
    :param dm_components: number of frequencies starting at 1/T for DM GP.
        Dictionary with pulsar name and number of frequencies.
        [default = 30]
    :param upper_limit_dm: perform upper limit on DM GP. Note that when perfoming
        upper limits it is recommended that the spectral index also be
        fixed to a specific value.
        [default = False]
    :param dm_annual: boolean to search for an annual DM trend.
        [default = False]
    :param chrom_var: boolean to search for a generic chromatic GP.
        Dictionary with pulsar name and boolean.
        [default = False]
    :param chrom_psd: psd of generic chromatic GP.
        ['powerlaw', 'spectrum', 'turnover']
        [default = 'powerlaw']
    :param chrom_idx: spectral index of generic chromatic GP.
        [default = 4]
    :param chrom_components: number of frequencies starting at 1/T for chromatic GP.
        Dictionary with pulsar name and number of frequencies.
        [default = 30]
    :param white_global: string to search for a global white noise.
        ['gequad', 'gefac']
        [default = False]
    :param coefficients: boolean to form full hierarchical PTA object;
        (no analytic latent-coefficient marginalization)
        [default = False]
    :param pshift: boolean to add random phase shift to red noise Fourier design
        matrices for false alarm rate studies.
        [default = False]
    :param psr_models:
        Return list of psr models rather than signal_base.PTA object.
    :param extra_sigs: Any additional `enterprise` signals to be added to the
        model.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood

    Default PTA object composition:
        1. fixed EFAC per backend/receiver system (per pulsar)
        2. fixed EQUAD per backend/receiver system (per pulsar)
        3. fixed ECORR per backend/receiver system (per pulsar)
        4. Red noise modeled as a power-law with 30 sampling frequencies
           (per pulsar)
        5. Linear timing model (per pulsar)
        6. Common-spectrum uncorrelated process modeled as a power-law with
           30 sampling frequencies. (global)
    """

    if isinstance(upper_limit, str):
        amp_prior = upper_limit
    else:
        amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # timing model
    if not tm_var and not use_dmdata:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd, normed=tm_norm,
                                       coefficients=coefficients)
    elif not tm_var and use_dmdata:
        dmjump = parameter.Constant()
        if white_var:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           selection=selections.Selection(selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        # create new attribute for enterprise pulsar object
        for p in psrs:
            p.tmparams_orig = OrderedDict.fromkeys(p.t2pulsar.pars())
            for key in p.tmparams_orig:
                p.tmparams_orig[key] = (p.t2pulsar[key].val,
                                        p.t2pulsar[key].err)
        s = timing_block(tmparam_list=tmparam_list, simga=tm_sigma, linear=tm_linear)

    # find the maximum time span to set GW frequency sampling
    if Tspan_common is None:
        Tspan_common = model_utils.get_tspan(psrs)
    if Tspan_red == 'common':
        Tspan_red = Tspan_common
    if Tspan_dm == 'common':
        Tspan_dm = Tspan_common
    if Tspan_chrom == 'common':
        Tspan_chrom = Tspan_common

    if logfreq:
        fmin = 10.0
        red_modes, wgts = model_utils.linBinning(Tspan_red, nmodes_log,
                                                 1.0 / fmin / Tspan_red,
                                                 red_components, nmodes_log)
        wgts = wgts**2.0

    # common red noise block
    if orf is not None:
        crn = []
        counter = 0
        if orf_names is None:
            orf_names = orf
        for elem, elem_name in zip(orf.split(','), orf_names.split(',')):
            if isinstance(upper_limit_common, list):
                ul = upper_limit_common[counter]
            else:
                ul = upper_limit_common
            if ul is None:
                amp_prior_common = amp_prior
            elif isinstance(ul, str):
                amp_prior_common = ul
            else:
                amp_prior_common = 'uniform' if ul else 'log-uniform'
            if isinstance(log10_A_common, list):
                log10_A_val = log10_A_common[counter]
            else:
                log10_A_val = log10_A_common
            if isinstance(gamma_common, list):
                gamma_val = gamma_common[counter]
            else:
                gamma_val = gamma_common
            if isinstance(gammamin_common, list):
                gammamin_val = gammamin_common[counter]
            else:
                gammamin_val = gammamin_common
            if isinstance(gammamax_common, list):
                gammamax_val = gammamax_common[counter]
            else:
                gammamax_val = gammamax_common
            if isinstance(delta_common, list):
                delta_val = delta_common[counter]
            else:
                delta_val = delta_common
            if isinstance(common_modes, list):
                modes_val = common_modes[counter]
            else:
                modes_val = common_modes
            if isinstance(common_select, list):
                select_val = common_select[counter]
            else:
                select_val = common_select
            if isinstance(common_logmin, list):
                logmin_val = common_logmin[counter]
            else:
                logmin_val = common_logmin
            if isinstance(common_logmax, list):
                logmax_val = common_logmax[counter]
            else:
                logmax_val = common_logmax
            if isinstance(common_idx, list):
                idx_val = common_idx[counter]
            else:
                idx_val = common_idx
            #if 'zero_diag' in elem:
            #    log10_A_val = log10_A_common
            #else:
            #    log10_A_val = None
            crn.append(common_red_noise_block(psd=common_psd, prior=amp_prior_common,
                                              tnfreq=tnfreq, Tspan=Tspan_common,
                                              components=common_components, logf=common_logf,
                                              fmin=common_fmin, fmax=common_fmax,
                                              log10_A_val=log10_A_val, gamma_val=gamma_val,
                                              delta_val=delta_val, modes=modes_val,
                                              name='gw_{}'.format(elem_name),
                                              orf=elem, orf_bins=orf_bins,
                                              orf_ifreq=orf_ifreq, leg_lmax=leg_lmax,
                                              coefficients=coefficients, select=select_val,
                                              pshift=pshift, pseed=pseed,
                                              logmin=logmin_val, logmax=logmax_val,
                                              dropout=dropout, dropout_psr=dropout_psr,
                                              dropout_common=dropout_common,
                                              dropbin=dropbin, dropbin_psr=dropbin_psr,
                                              dropbin_common=dropbin_common,
                                              dropbin_min=dropbin_min,
                                              k_threshold=k_threshold,
                                              idx=idx_val, tndm=tndm,
                                              flagname=flagname, flagval=flagval))
            # orf_ifreq only affects freq_hd model.
            # leg_lmax only affects (zero_diag_)legendre_orf model.
            counter += 1
        crn = functools.reduce((lambda x, y: x+y), crn)
        #s += crn
        
    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True, model=be_type,
                                                           sat_orb_elements=sat_orb_elements)
        
    # uldm model
    if uldm_term:
        uldm_tmin = np.min([p.toas.min() for p in psrs])
        uldm_amp_prior = 'uniform' if uldm_upper_limit else 'log-uniform'
        if uldm_type=="Scalar":
            s += scalar_dm_block(prior= uldm_amp_prior, effect_type=uldm_effect_type, coupl_type=uldm_coupl_type,
                                 # log10_freq=uldm_log10_freq,
                                 masstype=uldm_mass,
                                 mass_indx=uldm_mass_indx,corr=uldm_corr,tref=uldm_tmin)
        

    # adding white-noise, and acting on psr objects
    models = []

    for p in psrs:
        if psrnoise_models is None:
            s0 = s
            
            # red noise
            if isinstance(red_var, dict):
                if isinstance(red_var[p.name], list):
                    n_red = red_var[p.name][0]
                else:
                    n_red = red_var[p.name]
            else:
                n_red = red_var
            for i in range(n_red):
                if isinstance(upper_limit_red, dict):
                    if isinstance(upper_limit_red[p.name], list):
                        ul = upper_limit_red[p.name][i]
                    else:
                        ul = upper_limit_red[p.name]
                else:
                    if isinstance(upper_limit_red, list):
                        ul = upper_limit_red[i]
                    else:
                        ul = upper_limit_red
                if ul is None:
                    amp_prior_red = amp_prior
                elif isinstance(ul, str):
                    amp_prior_red = ul
                else:
                    amp_prior_red = 'uniform' if ul else 'log-uniform'
                if isinstance(gamma, dict):
                    if isinstance(gamma[p.name], list):
                        gamma_val = gamma[p.name][i]
                    else:
                        gamma_val = gamma[p.name]
                else:
                    if isinstance(gamma, list):
                        gamma_val = gamma[i]
                    else:
                        gamma_val = gamma
                if isinstance(gammamin, dict):
                    if isinstance(gammamin[p.name], list):
                        gammamin_val = gammamin[p.name][i]
                    else:
                        gammamin_val = gammamin[p.name]
                else:
                    if isinstance(gammamin, list):
                        gammamin_val = gammamin[i]
                    else:
                        gammamin_val = gammamin
                if isinstance(gammamax, dict):
                    if isinstance(gammamax[p.name], list):
                        gammamax_val = gammamax[p.name][i]
                    else:
                        gammamax_val = gammamax[p.name]
                else:
                    if isinstance(gammamax, list):
                        gammamax_val = gammamax[i]
                    else:
                        gammamax_val = gammamax
                if isinstance(red_select, dict):
                    if isinstance(red_select[p.name], list):
                        red_sel = red_select[p.name][i]
                    else:
                        red_sel = red_select[p.name]
                else:
                    if isinstance(red_select, list):
                        red_sel = red_select[i]
                    else:
                        red_sel = red_select
                if isinstance(red_modes, dict):
                    if isinstance(red_modes[p.name], list):
                        red_m = red_modes[p.name][i]
                    else:
                        red_m = red_modes[p.name]
                else:
                    if isinstance(red_modes, list):
                        red_m = red_modes[i]
                    else:
                        red_m = red_modes
                if isinstance(Tspan_red, dict):
                    if isinstance(Tspan_red[p.name], list):
                        T_red = Tspan_red[p.name][i]
                    else:
                        T_red = Tspan_red[p.name]
                else:
                    if isinstance(Tspan_red, list):
                        T_red = Tspan_red[i]
                    else:
                        T_red = Tspan_red
                if isinstance(red_components, dict):
                    if isinstance(red_components[p.name], list):
                        red_cp = red_components[p.name][i]
                    else:
                        red_cp = red_components[p.name]
                else:
                    if isinstance(red_components, list):
                        red_cp = red_components[i]
                    else:
                        red_cp = red_components
                if isinstance(logmin, dict):
                    if isinstance(logmin[p.name], list):
                        logmin_val = logmin[p.name][i]
                    else:
                        logmin_val = logmin[p.name]
                else:
                    if isinstance(logmin, list):
                        logmin_val = logmin[i]
                    else:
                        logmin_val = logmin
                if isinstance(logmax, dict):
                    if isinstance(logmax[p.name], list):
                        logmax_val = logmax[p.name][i]
                    else:
                        logmax_val = logmax[p.name]
                else:
                    if isinstance(logmax, list):
                        logmax_val = logmax[i]
                    else:
                        logmax_val = logmax
                if isinstance(fmin, dict):
                    if isinstance(fmin[p.name], list):
                        fmin_val = fmin[p.name][i]
                    else:
                        fmin_val = fmin[p.name]
                else:
                    if isinstance(fmin, list):
                        fmin_val = fmin[i]
                    else:
                        fmin_val = fmin
                if isinstance(fmax, dict):
                    if isinstance(fmax[p.name], list):
                        fmax_val = fmax[p.name][i]
                    else:
                        fmax_val = fmax[p.name]
                else:
                    if isinstance(fmax, list):
                        fmax_val = fmax[i]
                    else:
                        fmax_val = fmax
                if i == 0:
                    red_name = 'red_noise'
                else:
                    red_name = 'red_noise_'+str(i)
                if red_m is not None or red_cp is not None:
                    s0 += red_noise_block(psd=red_psd, prior=amp_prior_red, Tspan=T_red,
                                          name=red_name, components=red_cp, logmin=logmin_val,
                                          logmax=logmax_val, logf=logf, fmin=fmin_val,
                                          fmax=fmax_val, tnfreq=tnfreq, modes=red_m, wgts=wgts,
                                          gamma_val=gamma_val, coefficients=coefficients,
                                          select=red_sel, break_flat=red_breakflat,
                                          break_flat_fq=red_breakflat_fq)

            # DM variations
            if isinstance(dm_var, dict):
                if isinstance(dm_var[p.name], list):
                    n_dm = dm_var[p.name][0]
                else:
                    n_dm = dm_var[p.name]
            else:
                n_dm = dm_var
            if n_dm:
                if isinstance(upper_limit_dm, dict):
                    if isinstance(upper_limit_dm[p.name], list):
                        ul = upper_limit_dm[p.name][0]
                    else:
                        ul = upper_limit_dm[p.name]
                else:
                    ul = upper_limit_dm
                if ul is None:
                    amp_prior_dm = amp_prior
                elif isinstance(ul, str):
                    amp_prior_dm = ul
                else:
                    amp_prior_dm = 'uniform' if ul else 'log-uniform'
                if isinstance(gamma_dm, dict):
                    if isinstance(gamma_dm[p.name], list):
                        gamma_val = gamma_dm[p.name][0]
                    else:
                        gamma_val = gamma_dm[p.name]
                else:
                    gamma_val = gamma_dm
                if isinstance(gammamin_dm, dict):
                    if isinstance(gammamin_dm[p.name], list):
                        gammamin_val = gammamin_dm[p.name][0]
                    else:
                        gammamin_val = gammamin_dm[p.name]
                else:
                    gammamin_val = gammamin_dm
                if isinstance(gammamax_dm, dict):
                    if isinstance(gammamax_dm[p.name], list):
                        gammamax_val = gammamax_dm[p.name][0]
                    else:
                        gammamax_val = gammamax_dm[p.name]
                else:
                    gammamax_val = gammamax_dm
                if isinstance(dm_modes, dict):
                    if isinstance(dm_modes[p.name], list):
                        dm_m = dm_modes[p.name][0]
                    else:
                        dm_m = dm_modes[p.name]
                else:
                    dm_m = dm_modes
                if isinstance(Tspan_dm, dict):
                    if isinstance(Tspan_dm[p.name], list):
                        T_dm = Tspan_dm[p.name][0]
                    else:
                        T_dm = Tspan_dm[p.name]
                else:
                    T_dm = Tspan_dm
                if isinstance(dm_components, dict):
                    if isinstance(dm_components[p.name], list):
                        dm_cp = dm_components[p.name][0]
                    else:
                        dm_cp = dm_components[p.name]
                else:
                    dm_cp = dm_components
                if isinstance(dm_logmin, dict):
                    if isinstance(dm_logmin[p.name], list):
                        dm_logmin_val = dm_logmin[p.name][0]
                    else:
                        dm_logmin_val = dm_logmin[p.name]
                else:
                    dm_logmin_val = dm_logmin
                if isinstance(dm_logmax, dict):
                    if isinstance(dm_logmax[p.name], list):
                        dm_logmax_val = dm_logmax[p.name][0]
                    else:
                        dm_logmax_val = dm_logmax[p.name]
                else:
                    dm_logmax_val = dm_logmax
                if isinstance(dm_fmin, dict):
                    if isinstance(dm_fmin[p.name], list):
                        dm_fmin_val = dm_fmin[p.name][0]
                    else:
                        dm_fmin_val = dm_fmin[p.name]
                else:
                    dm_fmin_val = dm_fmin
                if isinstance(dm_fmax, dict):
                    if isinstance(dm_fmax[p.name], list):
                        dm_fmax_val = dm_fmax[p.name][0]
                    else:
                        dm_fmax_val = dm_fmax[p.name]
                else:
                    dm_fmax_val = dm_fmax
                if isinstance(dm_type, dict):
                    if isinstance(dm_type[p.name], list):
                        dm_tp = dm_type[p.name][0]
                    else:
                        dm_tp = dm_type[p.name]
                else:
                    dm_tp = dm_type
                if dm_tp == 'gp':
                    if dm_modes is not None or dm_cp is not None:
                        s0 += dm_noise_block(gp_kernel=dm_kernel, psd=dm_psd, prior=amp_prior_dm,
                                             modes=dm_m, Tspan=T_dm, components=dm_cp,
                                             logf=dm_logf, fmin=dm_fmin_val, fmax=dm_fmax_val,
                                             tnfreq=tnfreq, tndm=tndm, gamma_val=gamma_val,
                                             coefficients=coefficients, select=dm_select,
                                             logmin=dm_logmin_val, logmax=dm_logmax_val)
                if dm_annual:
                    s0 += chrom.dm_annual_signal()
                if isinstance(dm_chrom, dict):
                    if isinstance(dm_chrom[p.name], list):
                        n_chrom = dm_chrom[p.name][0]
                    else:
                        n_chrom = dm_chrom[p.name]
                else:
                    n_chrom = dm_chrom 
    
                if n_chrom:
                    if isinstance(upper_limit_chrom, dict):
                        if isinstance(upper_limit_chrom[p.name], list):
                            ul = upper_limit_chrom[p.name][0]
                        else:
                            ul = upper_limit_chrom[p.name]
                    else:
                        if isinstance(upper_limit_chrom, list):
                            ul = upper_limit_chrom[0]
                        else:
                            ul = upper_limit_chrom
                    if ul is None:
                        amp_prior_chrom = amp_prior
                    elif isinstance(ul, str):
                        amp_prior_chrom = ul
                    else:
                        amp_prior_chrom = 'uniform' if ul else 'log-uniform'
                    if gamma_chrom == 'dm':
                        gamma_val = gamma_val
                    elif isinstance(gamma_chrom, dict):
                        if isinstance(gamma_chrom[p.name], list):
                            gamma_val = gamma_chrom[p.name][0]
                        else:
                            gamma_val = gamma_chrom[p.name]
                    else:
                        gamma_val = gamma_chrom
                    if gammamin_chrom == 'dm':
                        gammamin_val = gammamin_val
                    elif isinstance(gammamin_chrom, dict):
                        if isinstance(gammamin_chrom[p.name], list):
                            gammamin_val = gammamin_chrom[p.name][0]
                        else:
                            gammamin_val = gammamin_chrom[p.name]
                    else:
                        gammamin_val = gammamin_chrom
                    if gammamax_chrom == 'dm':
                        gammamax_val = gammamax_val
                    elif isinstance(gammamax_chrom, dict):
                        if isinstance(gammamax_chrom[p.name], list):
                            gammamax_val = gammamax_chrom[p.name][0]
                        else:
                            gammamax_val = gammamax_chrom[p.name]
                    else:
                        gammamax_val = gammamax_chrom
                    if chrom_modes == 'dm':
                        chrom_m = dm_m
                    elif isinstance(chrom_modes, dict):
                        if isinstance(chrom_modes[p.name], list):
                            chrom_m = chrom_modes[p.name][0]
                        else:
                            chrom_m = chrom_modes[p.name]
                    else:
                        chrom_m = chrom_modes
                    if Tspan_chrom == 'dm':
                        T_chrom = T_dm
                    elif isinstance(Tspan_chrom, dict):
                        if isinstance(Tspan_chrom[p.name], list):
                            T_chrom = Tspan_chrom[p.name][0]
                        else:
                            T_chrom = Tspan_chrom[p.name]
                    else:
                        T_chrom = Tspan_chrom
                    if chrom_components == 'dm':
                        chrom_cp = dm_cp
                    elif isinstance(chrom_components, dict):
                        if isinstance(chrom_components[p.name], list):
                            chrom_cp = chrom_components[p.name][0]
                        else:
                            chrom_cp = chrom_components[p.name]
                    else:
                        chrom_cp = chrom_components
                    if chrom_logmin == 'dm':
                        chrom_logmin_val = dm_logmin_val
                    elif isinstance(chrom_logmin, dict):
                        if isinstance(chrom_logmin[p.name], list):
                            chrom_logmin_val = chrom_logmin[p.name][0]
                        else:
                            chrom_logmin_val = chrom_logmin[p.name]
                    else:
                        chrom_logmin_val = chrom_logmin
                    if chrom_logmax == 'dm':
                        chrom_logmax_val = dm_logmax_val
                    elif isinstance(chrom_logmax, dict):
                        if isinstance(chrom_logmax[p.name], list):
                            chrom_logmax_val = chrom_logmax[p.name][0]
                        else:
                            chrom_logmax_val = chrom_logmax[p.name]
                    else:
                        chrom_logmax_val = chrom_logmax
                    if chrom_fmin == 'dm':
                        chrom_fmin_val = dm_fmin_val
                    elif isinstance(chrom_fmin, dict):
                        if isinstance(chrom_fmin[p.name], list):
                            chrom_fmin_val = chrom_fmin[p.name][0]
                        else:
                            chrom_fmin_val = chrom_fmin[p.name]
                    else:
                        chrom_fmin_val = chrom_fmin
                    if chrom_fmax == 'dm':
                        chrom_fmax_val = dm_fmax_val
                    elif isinstance(chrom_fmax, dict):
                        if isinstance(chrom_fmax[p.name], list):
                            chrom_fmax_val = chrom_fmax[p.name][0]
                        else:
                            chrom_fmax_val = chrom_fmax[p.name]
                    else:
                        chrom_fmax_val = chrom_fmax
                    if chrom_modes is not None or chrom_cp is not None:
                        s0 += chromatic_noise_block(gp_kernel=dmchrom_kernel, psd=chrom_psd,
                                                    idx=chrom_idx, prior=amp_prior_chrom,
                                                    modes=chrom_m, Tspan=T_chrom,
                                                    components=chrom_cp, logf=chrom_logf,
                                                    fmin=chrom_fmin_val, fmax=chrom_fmax_val,
                                                    tnfreq=tnfreq, tndm=tndm, gamma_val=gamma_val,
                                                    gammamin=gammamin_val, gammamax=gammamax_val,
                                                    coefficients=coefficients, select=chrom_select,
                                                    logmin=chrom_logmin_val, logmax=chrom_logmax_val)
                if p.name in dmpsr_list:
                    if dm_expdip_tmin is None and dm_expdip_tmax is None:
                        tmin = [p.toas.min() / const.day for ii in range(num_dmdips)]
                        tmax = [p.toas.max() / const.day for ii in range(num_dmdips)]
                    else:
                        tmin = (dm_expdip_tmin if isinstance(dm_expdip_tmin, list)
                                else [dm_expdip_tmin])
                        tmax = (dm_expdip_tmax if isinstance(dm_expdip_tmax, list)
                                else [dm_expdip_tmax])
                    dm_expdip_idx = (dm_expdip_idx if isinstance(dm_expdip_idx,list)
                                                   else [dm_expdip_idx]*int(num_dmdips))
                    dm_expdip_sign = (dm_expdip_sign if isinstance(dm_expdip_sign,list)
                                                     else [dm_expdip_sign]*int(num_dmdips))
                    for dd in range(num_dmdips):
                        s0 += chrom.dm_exponential_dip(tmin=tmin[dd], tmax=tmax[dd],
                                                       idx=dm_expdip_idx[dd],
                                                       sign=dm_expdip_sign[dd],
                                                       name='dmexp_{0}'.format(dd+1))
        
        else:
            s0 = psrnoise_models[p.name]
                                                           
        if extra_sigs is not None:
            extra_sigs = np.atleast_1d(extra_sigs)
            for esig in extra_sigs:
                s0 += esig
            
        if (p.name == dropout_psr and not dropout) or orf is None:
            s1 = s0
        else:
            s1 = s0 + crn
        no_select = selections.Selection(selections.no_selection)
        if white_global == 'gequad':
            s1 += white_signals.TNEquadNoise(log10_equad=parameter.Uniform(-9, -5),
                                             selection=no_select, name=white_global)
        if white_global == 'gefac':
            s1 += white_signals.MeasurementNoise(efac=parameter.Uniform(0.1, 5.0),
                                                 selection=no_select, name=white_global)
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s1 + white_noise_block(vary=white_var, inc_ecorr=True, tnequad=tnequad,
                                        select=select, ecorr_select=ecorr_select)
        else:
            s2 = s1 + white_noise_block(vary=white_var, inc_ecorr=inc_ecorr, tnequad=tnequad,
                                        select=select, ecorr_select=ecorr_select)
        models.append(s2(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    if psr_models:
        return models
    else:
        # set up PTA
        if dense_like:
            pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
        else:
            pta = signal_base.PTA(models)

        # set white noise parameters
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

        return pta

