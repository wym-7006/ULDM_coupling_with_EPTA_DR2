import hypermodel

import numpy as np
import os, glob, json
import optparse

import utils as ut

import enterprise
from enterprise.pulsar import Pulsar

from enterprise.signals.parameter import function
from enterprise.signals.deterministic_signals import Deterministic
from enterprise.signals import parameter, selections
from enterprise.signals.selections import Selection

# from enterprise_extensions import models, sampler, hypermodel

from enterprise_extensions import models

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

parser = optparse.OptionParser()
parser.add_option('--datadir', action='store', dest='datadir', default='/DR2new/', type='string')
parser.add_option('--outdir', action='store', dest='outdir', default='/local_data/', type='string')
parser.add_option('--noisedir', action='store', dest='noisedir', default='/noisefiles/', type='string')
parser.add_option('--cachedir', action='store', dest='cachedir', default='/psrs_cache/', type='string')
parser.add_option('--iteration', action='store', dest='iteration', default=1, type='int')
parser.add_option('--orf', action='store', dest='orf', default='crn', type='string')
parser.add_option('--common_psd', action='store', dest='common_psd', default='powerlaw', type='string')
parser.add_option('--gamma_common', action='store', dest='gamma_common', default=13/3, type='float')
parser.add_option('--red_components', action='store', dest='red_components', default=0, type='int')
parser.add_option('--dm_components', action='store', dest='dm_components', default=0, type='int')
parser.add_option('--chrom_components', action='store', dest='chrom_components', default=0, type='int')
parser.add_option('--bayesephem', action='store_true', dest='bayesephem', default=False)
parser.add_option('--common_sin1', action='store_true', dest='sin_wave1', default=False)
parser.add_option('--common_sin2', action='store_true', dest='sin_wave2', default=False)
parser.add_option('--lnweight1', action='store', dest='weight1', default=0, type='float')
parser.add_option('--lnweight2', action='store', dest='weight2', default=0, type='float')
parser.add_option('--resume', action='store_true', dest='resume', default=True)
parser.add_option('--emp', action='store', dest='emp', default=None, type='string')
parser.add_option('--number', action='store', dest='num', default=5e6, type='float')
parser.add_option('--thin', action='store', dest='thin', default=10, type='int')
parser.add_option('--bayestype', action='store', dest='bayestype', default='upper', type='string')
parser.add_option('--effect', action='store', dest='effect', default='total_effect', type='string')
parser.add_option('--epcorr', action='store', dest='epcorr', default="uncorr")
parser.add_option('--coupltype', action='store', dest='coupltype', default='gluon', type='string')
parser.add_option('--masschoice', action='store', dest='masschoice', default='bin', type='string')
parser.add_option('--sectindx', action='store',dest='sectindx', default=1,  type='int') 
parser.add_option('--massindx', action='store', dest='massindx', default=0, type='int') 
(options,args) = parser.parse_args()


# load par and tim files
DataDir = options.datadir
OutDir = options.outdir
NoiseDir = options.noisedir
CacheDir = options.cachedir
iteration = options.iteration

orf=options.orf

# print(f"the bayesephem is :{options.bayesephem}")

EPcorr = options.epcorr
print(f"the EPcorr is :{options.epcorr}")
      
bayestype = options.bayestype
effecttype = options.effect
coupltype = options.coupltype 
masstype = options.masschoice
mass_indx_sect = options.sectindx
mass_indx = options.massindx



DATADIR = os.path.expanduser("~/projects/working/PTA_2023/epta-dr2/EPTA-DR2")
BASEDIR = os.path.expanduser("~/projects/working/PTA_2023/epta-dr2/EPTA-DR2/scripts_gwb")

if DataDir == '/DR2new/':
    common_components=9
    num_dmdips=1
    OUTDIR = os.path.expanduser("~/projects/working/PTA_2023/epta-dr2/EPTA-DR2/ULDM_total_tests/local_data/DR2new/")
else:
    common_components=24
    num_dmdips=2
    OUTDIR = os.path.expanduser("~/projects/working/PTA_2023/epta-dr2/EPTA-DR2/ULDM_total_tests/local_data/DR2full/")

if "DR2new" in DataDir:
    datadir = DATADIR + '/DR2new/'
    noisedir = DATADIR + NoiseDir + '/DR2new/'
    
elif "DR2full" in DataDir:
    datadir = DATADIR + '/DR2full/'
    noisedir = DATADIR + NoiseDir + '/DR2full/'


    
print(f"noisedir is {noisedir}!")


parfiles = sorted(glob.glob(datadir + '/J*/*.par'))
timfiles = sorted(glob.glob(datadir + '/J*/*_all.tim'))
noisefiles = sorted(glob.glob(noisedir + '/*.json'))

# filter to one set of par+tim+noisefile per pulsar
PsrList = np.loadtxt(BASEDIR + '/psrlist_epta_dr2_25psrs.txt',dtype=str)
psrs=ut.get_pulsars(PsrList, datadir, cache_dir=CacheDir, ephem='DE440', cache_file="psrs_cache.pkl", use_cache=True)
noisefiles = [x for x in noisefiles if x.split('/')[-1].split('_')[0] in PsrList]
    
# set reference time for the sin wave to the earliest TOA in the data set
dataset_tmin = np.min([p.toas.min() for p in psrs])
dataset_tmax = np.max([p.toas.max() for p in psrs])


# load noise models and files

params = {}
for nf in noisefiles:
    with open(nf, 'r') as fin:
        params.update(json.load(fin))

if not options.red_components:
    try:
        red_dict = {}
        with open(noisedir + '/red_dict.json','r') as rd:
            red_dict.update(json.load(rd))
    except:
        raise UserWarning('Custom pulsar red noise frequency components not set.')
else:
    red_dict = options.red_components

if not options.dm_components:
    try:
        dm_dict = {}
        with open(noisedir + '/dm_dict.json','r') as dd:
            dm_dict.update(json.load(dd))
    except:
        raise UserWarning('Custom pulsar DM noise frequency components not set.')
else:
    dm_dict = options.dm_components

if not options.chrom_components:
    try:
        chrom_dict = {}
        with open(noisedir + '/chrom_dict.json','r') as cd:
            chrom_dict.update(json.load(cd))
    except:
        raise UserWarning('Custom pulsar scattering noise frequency components not set.')
else:
    chrom_dict = options.chrom_components

try:
    gamma_common = float(options.gamma_common)
except:
    gamma_common = None

# setup model

if bayestype=="bayes":
    uldm_upper_limit=False
    
    pta = dict.fromkeys(np.arange(0, 2))
    pta[0] = ut.model_ULDM(psrs, noisedict=params, orf=orf,
                              common_psd=options.common_psd,
                              common_components=common_components,
                              gamma_common=gamma_common,
                              bayesephem=options.bayesephem,
                              sat_orb_elements=True,     #### need to be done
                              tnequad=True,
                              tm_svd=True, tm_marg=True,
                              red_var=True, red_components=red_dict,
                              dm_var=True, dm_components=dm_dict,
                              # dm_chrom=True, 
                              chrom_components=chrom_dict,
                              dmchrom_kernel='diag', tndm=True,
                              num_dmdips=num_dmdips,
                              dmpsr_list=['J1713+0747'], dm_expdip_idx=[1,4],
                              dm_expdip_tmin=[57490,54650],
                              dm_expdip_tmax=[57530,54850], 
                              uldm_term=False, 
                              uldm_type="Scalar",
                              uldm_effect_type=effecttype,
                              uldm_coupl_type=coupltype,
                              uldm_upper_limit=uldm_upper_limit,
                              uldm_mass=masstype,
                              uldm_mass_indx=mass_indx,
                              uldm_corr=EPcorr,
                              uldm_osc_orient=None)
    
    pta[1] = ut.model_ULDM(psrs, noisedict=params, orf=orf,
                              common_psd=options.common_psd,
                              common_components=common_components,
                              gamma_common=gamma_common,
                              bayesephem=options.bayesephem,
                              sat_orb_elements=True,     #### need to be done
                              tnequad=True,
                              tm_svd=True, tm_marg=True,
                              red_var=True, red_components=red_dict,
                              dm_var=True, dm_components=dm_dict,
                              # dm_chrom=True, 
                              chrom_components=chrom_dict,
                              dmchrom_kernel='diag', tndm=True,
                              num_dmdips=num_dmdips,
                              dmpsr_list=['J1713+0747'], dm_expdip_idx=[1,4],
                              dm_expdip_tmin=[57490,54650],
                              dm_expdip_tmax=[57530,54850], 
                              uldm_term=True, 
                              uldm_type="Scalar",
                              uldm_effect_type=effecttype,
                              uldm_coupl_type=coupltype,
                              uldm_upper_limit=uldm_upper_limit,
                              uldm_mass=masstype,
                              uldm_mass_indx=mass_indx,
                              uldm_corr=EPcorr,
                              uldm_osc_orient=None)
    
else:
    uldm_upper_limit=True
    
    pta = ut.model_ULDM(psrs, noisedict=params, orf=orf,
                              common_psd=options.common_psd,
                              common_components=common_components,
                              gamma_common=gamma_common,
                              bayesephem=options.bayesephem,
                              sat_orb_elements=True,     #### need to be done
                              tnequad=True,
                              tm_svd=True, tm_marg=True,
                              red_var=True, red_components=red_dict,
                              dm_var=True, dm_components=dm_dict,
                              # dm_chrom=True, 
                              chrom_components=chrom_dict,
                              dmchrom_kernel='diag', tndm=True,
                              num_dmdips=num_dmdips,
                              dmpsr_list=['J1713+0747'], dm_expdip_idx=[1,4],
                              dm_expdip_tmin=[57490,54650],
                              dm_expdip_tmax=[57530,54850], 
                              uldm_term=True, 
                              uldm_type="Scalar",
                              uldm_effect_type=effecttype,
                              uldm_coupl_type=coupltype,
                              uldm_upper_limit=uldm_upper_limit,
                              uldm_mass=masstype,
                              uldm_mass_indx=mass_indx,
                              uldm_corr=EPcorr,
                              uldm_osc_orient=None)

if effecttype=="grav_effect":
    outdir= OUTDIR + '/uldm/' +'/' + effecttype + '/'+ masstype + '/' + EPcorr + '/'+ bayestype + f"/mass_{mass_indx_sect}_sec/{mass_indx}/"+f"{iteration}"
    
else:
    outdir= OUTDIR + '/uldm/'  +'/' + effecttype + '/'+ coupltype + '/'+ masstype + '/' + EPcorr + '/'+ bayestype + f"/mass_{mass_indx_sect}_sec/{mass_indx}/"+f"{iteration}"
    
    

# get initial sample and run sampler
super_model = hypermodel.HyperModel(pta,log_weights=[options.weight1,options.weight2])
sp = super_model.setup_sampler(resume=options.resume, outdir=outdir, empirical_distr=options.emp)
x0 = super_model.initial_sample()
sp.sample(x0, int(options.num), SCAMweight=30, AMweight=10, DEweight=50, thin=options.thin)

