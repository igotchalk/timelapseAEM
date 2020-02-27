#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
from SimPEG import Mesh, Utils, EM, Maps
from pymatsolver import Pardiso
from scipy.constants import mu_0
import numpy as np
import matplotlib.pyplot as plt
from pyMKL import mkl_set_num_threads
from multiprocessing import Pool
from SimPEG import Mesh
import pandas as pd

from pathlib import Path
import pandas as pd
import fileinput
# import cartopy



it = int(sys.argv[1])
# In[2]:

stds =  (.05,.05,.03,.05,.05)
s = (1e-2, 1., 1e-2, 1., 1e-2)
x = (10.,  10.,10.,  10.,1.)
y = (1.,   1., 1.,   1., 1.)  
m0_vals =(10.,  10.,10.,  20.,10.)

std,alpha_s,alpha_x,alpha_y,m0_val = (stds[it],s[it],x[it],y[it],m0_vals[it])


# In[3]:


datadir = Path('../data/raw_data')
data17dir = datadir.joinpath('AEM_data_2017_avg')
data19dir = datadir.joinpath('AEM_data_2019_avg')

data17 = datadir.joinpath('AEM_data_2017','MCWD3_SCI1i_MOD_dat.xyz')
data19 = datadir.joinpath('AEM_data_2019','MCWD19_SCI8i_MOD_dat.xyz')
df17 = pd.read_csv(data17,header=20,delim_whitespace=True)
df19 = pd.read_csv(data19,header=20,delim_whitespace=True)

df17 = df17.assign(skytem_type=304)
df19 = df19.assign(skytem_type=312)
df = pd.read_pickle(datadir.joinpath('processed_df_1719.pkl'))


# In[6]:


line = [l for l in df.LINE_NO.unique()]
# line = (100501,)
# line = (206800,206801)
df =  df.loc[df.LINE_NO.isin(line),:]


ch1_cols = [c for c in df.columns if c.startswith('DBDT_Ch1')]
ch2_cols = [c for c in df.columns if c.startswith('DBDT_Ch2')]


#Remove soundings with few time gates
thresh = 15
msk = df.loc[df.CHANNEL_NO==1,ch1_cols]==9999.
rm_mask = (df.loc[df.CHANNEL_NO==1,ch1_cols]==9999.).sum(axis=1) > thresh 
remove_inds = rm_mask[rm_mask==True].index
df = df.drop(index=np.r_[remove_inds,remove_inds+1])


# In[ ]:





# In[7]:


# df = pd.read_pickle(datadir.joinpath('processed_df_1719.pkl'))

# (df.loc[df.CHANNEL_NO==1,ch1_cols]==9999.).sum(axis=1).hist()
# plt.title('Distribution of removed time gates')
# plt.xlabel('No. removed time gates')
# plt.ylabel('No. soundings')


# In[8]:


#TAKE THE LOCATION OF CHANNEL 1 FOR EACH SOUNDING
msk = np.logical_and(df.CHANNEL_NO==1, df.LINE_NO.isin(line))

xy = df.loc[msk,['UTMX', 'UTMY']].values
Line = df.loc[msk,['LINE_NO']].values
dem = df.loc[msk,'ELEVATION'].values[:]
height = df.loc[msk,'INVALT'].values[:]
# height = df.loc[msk,'TX_ALTITUDE'].values[:]
system = df.loc[msk,'skytem_type'].values[:]
msk_312 = system==312


#Shift the 312 system spatially to create the "temporal" constraint
shift = 0
xy[msk_312] = xy[msk_312]+shift/np.sqrt(2)
rx_locations = np.c_[xy[:,:], height+dem+2.]
src_locations = np.c_[xy[:,:], height+dem]
topo = np.c_[xy[:,:], dem]


# In[ ]:





# In[9]:


# f,ax = plt.subplots(1,figsize=(6,7))
# plt.scatter(xy[:,0][msk_312],xy[:,1][msk_312],s=3,c='k',label='312 (shifted)')
# plt.scatter(xy[:,0][~msk_312],xy[:,1][~msk_312],s=1,c='r',label='304 (unshifted)')
# plt.gca().grid(True)
# plt.legend()
# plt.gca().set_aspect(1)


# In[10]:


print ( 'msk',msk.shape,'\n'
'xy',xy.shape,'\n'
'Line',Line.shape,'\n'
'dem',dem.shape,'\n'
'height',height.shape,'\n'
'system',system.shape,'\n'
'rx_locations',rx_locations.shape,'\n'
'src_locations',src_locations.shape,'\n'
'topo',topo.shape,'\n')


# In[11]:


### 312 Waveform###

area_312 = 342
unit_conversion = 1e-12

i_start_hm = 16
i_start_lm = 10
i_end_hm = None
i_end_lm = None

# i_end_hm = -1
# i_end_lm = -2


sl_hm_312 = slice(i_start_hm,i_end_hm)
sl_lm_312 = slice(i_start_lm,i_end_lm)

waveform_hm_312 = np.loadtxt(datadir.parent.joinpath('aem_waveform_marina/hm_312.txt'))
waveform_lm_312 = np.loadtxt(datadir.parent.joinpath('aem_waveform_marina/lm_312.txt'))
time_input_currents_HM_312 = waveform_hm_312[:,0] 
input_currents_HM_312 = waveform_hm_312[:,1]
time_input_currents_LM_312 = waveform_lm_312[:,0] 
input_currents_LM_312 = waveform_lm_312[:,1]

time_gates = np.loadtxt(datadir.parent.joinpath('aem_waveform_marina/time_gates.txt'))
# GateTimeShift=-2.1E-06
GateTimeShift=0
MeaTimeDelay=0.000E+00
NoGates=28
t0_lm_312 = waveform_lm_312[:,0].max()
times_LM_312 = (time_gates[:NoGates,0] + GateTimeShift + MeaTimeDelay)[sl_lm_312] - t0_lm_312

GateTimeShift=-1.5E-06
MeaTimeDelay=3.5E-04
NoGates=37
t0_hm_312 = waveform_hm_312[:,0].max()
times_HM_312 = (time_gates[:NoGates,0] + GateTimeShift + MeaTimeDelay)[sl_hm_312] - t0_hm_312    


# In[12]:


### 304 Waveform ###
area_304 = 337.04
unit_conversion = 1e-12

i_start_hm = 10
i_start_lm = 10
i_end_hm = None
i_end_lm = None

# i_end_hm = -1
# i_end_lm = -2


sl_hm_304 = slice(i_start_hm,i_end_hm)
sl_lm_304 = slice(i_start_lm,i_end_lm)

waveform_hm_304 = np.loadtxt(datadir.parent.joinpath('aem_waveform_marina/hm_304.txt'))
waveform_lm_304 = np.loadtxt(datadir.parent.joinpath('aem_waveform_marina/lm_304.txt'))
time_input_currents_HM_304 = waveform_hm_304[:,0] 
input_currents_HM_304 = waveform_hm_304[:,1]
time_input_currents_LM_304 = waveform_lm_304[:,0] 
input_currents_LM_304 = waveform_lm_304[:,1]

time_gates = np.loadtxt(datadir.parent.joinpath('aem_waveform_marina/time_gates.txt'))
GateTimeShift=-1.8E-06
MeaTimeDelay=0.000E+00
NoGates=28
t0_lm_304 = waveform_lm_304[:,0].max()
# times_LM_304 = (time_gates[:NoGates,0] + GateTimeShift + MeaTimeDelay)[i_start_lm:] - t0_lm_304
times_LM_304 = (time_gates[:NoGates,0] + GateTimeShift + MeaTimeDelay)[sl_lm_304] - t0_lm_304

GateTimeShift=-1.4E-06
MeaTimeDelay=6.000E-05
NoGates=37
t0_hm_304 = waveform_hm_304[:,0].max()
# times_HM_304 = (time_gates[:NoGates,0] + GateTimeShift + MeaTimeDelay)[i_start_hm:] - t0_hm_304    
times_HM_304 = (time_gates[:NoGates,0] + GateTimeShift + MeaTimeDelay)[sl_hm_304] - t0_hm_304    


# In[13]:


lm_TEM_312 = pd.read_csv(datadir.parent.joinpath('aem_waveform_marina','lm_312.tem'),
            header=14,delim_whitespace=True,names=np.arange(7)).values[:,:2]
hm_TEM_312 = pd.read_csv(datadir.parent.joinpath('aem_waveform_marina','hm_312.tem'),
            header=20,delim_whitespace=True,names=np.arange(7)).values[:,:2]

lm_times_TEM_312= lm_TEM_312[:,0]
lm_dbdt_TEM_312 =lm_TEM_312[:,1]
hm_times_TEM_312= hm_TEM_312[:,0]
hm_dbdt_TEM_312 =hm_TEM_312[:,1]

# plt.loglog(lm_times_TEM_312[10:],lm_dbdt_TEM_312[10:])
# plt.loglog(hm_times_TEM_312[10:],hm_dbdt_TEM_312[10:])

# plt.title('From 312 TEM files')


# In[14]:


hm_times_TEM_304 = [5.93150e-05,
6.08150e-05,
6.28150e-05,
6.48150e-05,
6.68150e-05,
6.88200e-05,
7.08100e-05,
7.33200e-05,
7.68100e-05,
8.13100e-05,
8.68100e-05,
9.38200e-05,
1.02810e-04,
1.14310e-04,
1.28810e-04,
1.46810e-04,
1.69300e-04,
1.97300e-04,
2.32800e-04,
2.78300e-04,
3.35300e-04,
4.07300e-04,
4.98300e-04,
6.12300e-04,
7.56300e-04,
9.37800e-04,
1.16660e-03,
1.45460e-03,
1.81860e-03,
2.27760e-03,
2.85560e-03,
3.57460e-03,
4.45460e-03,
5.53160e-03,
6.84960e-03,
8.46360e-03,
1.04386e-02]

lm_times_TEM_304 = [1.00000e-07,
4.15000e-07,
2.41500e-06,
4.41500e-06,
6.41500e-06,
8.42000e-06,
1.04100e-05,
1.29200e-05,
1.64100e-05,
2.09100e-05,
2.64100e-05,
3.34200e-05,
4.24100e-05,
5.39100e-05,
6.84100e-05,
8.64100e-05,
1.08900e-04,
1.36900e-04,
1.72400e-04,
2.17900e-04,
2.74900e-04,
3.46900e-04,
4.37900e-04,
5.51900e-04,
6.95900e-04,
8.77400e-04,
1.10620e-03,
1.39420e-03]


# In[15]:


# plt.plot(time_input_currents_LM_304, input_currents_LM_304,label='304')
# plt.plot(time_input_currents_LM_312, input_currents_LM_312,label='312')
# plt.legend()


# # In[16]:


# plt.plot(time_input_currents_HM_304, input_currents_HM_304,label='304')
# plt.plot(time_input_currents_HM_312, input_currents_HM_312,label='312')
# plt.legend()


# In[17]:


from SimPEG import (
    Regularization, Directives, Inversion, 
    InvProblem, Optimization, DataMisfit, Utils, Maps
)
from simpegEM1D import (
    GlobalEM1DProblemTD, GlobalEM1DSurveyTD, 
    get_vertical_discretization_time, EM1DSurveyTD,
    get_2d_mesh, LateralConstraint, 
)
from pymatsolver import PardisoSolver
hz_312 = np.loadtxt(datadir.joinpath('AEM_data_2019','thickness.txt'))
hz_304 = np.loadtxt(datadir.joinpath('AEM_data_2017','thickness.txt'))


hz = hz_304
# hz = np.unique(hz_312.values[inds_312,:])* 0.3048
# hz = np.unique(hz_304.values[inds_304,:])* 0.3048
n_sounding = xy.shape[0]
mesh = get_2d_mesh(n_sounding, hz)


# In[18]:


# n_sounding = n_sounding_312 + n_sounding_304
mesh = get_2d_mesh(n_sounding, hz)
# rx_locations = np.vstack((rx_locations_312, rx_locations_304))
# src_locations = np.vstack((src_locations_312, src_locations_304))
# topo = np.vstack((topo_312, topo_304))

times = []
time_dual_moment = []
input_currents = []
time_input_currents = []
input_currents_dual_moment = []
time_input_currents_dual_moment = []

for i_sounding in range(n_sounding):
    if system[i_sounding]==312:
        times.append(times_HM_312)
        time_dual_moment.append(times_LM_312)
        time_input_currents.append(time_input_currents_HM_312)
        input_currents.append(input_currents_HM_312)
        time_input_currents_dual_moment.append(time_input_currents_LM_312)
        input_currents_dual_moment.append(input_currents_LM_312)  
    elif system[i_sounding]==304:
        times.append(times_HM_304)
        time_dual_moment.append(times_LM_304)
        time_input_currents.append(time_input_currents_HM_304)
        input_currents.append(input_currents_HM_304)
        time_input_currents_dual_moment.append(time_input_currents_LM_304)
        input_currents_dual_moment.append(input_currents_LM_304)      





mapping = Maps.ExpMap(mesh)

survey = GlobalEM1DSurveyTD(
    rx_locations = rx_locations[:,:],
    src_locations = src_locations[:,:],    
    topo = topo[:,:],
    time = times,
    time_dual_moment = time_dual_moment,
    src_type = np.array(["VMD"], dtype=str).repeat(n_sounding),
    rx_type = np.array(["dBzdt"], dtype=str).repeat(n_sounding),    
    offset = np.array([13.25], dtype=float).repeat(n_sounding).reshape([-1,1]),    
    wave_type = np.array(["general"], dtype=str).repeat(n_sounding),    
    field_type = np.array(["secondary"], dtype=str).repeat(n_sounding),    
    input_currents=input_currents,
    time_input_currents=time_input_currents,
    base_frequency = np.array([30.]).repeat(n_sounding),
    input_currents_dual_moment=input_currents_dual_moment,
    time_input_currents_dual_moment=time_input_currents_dual_moment,
    base_frequency_dual_moment = np.array([210.]).repeat(n_sounding),
    moment_type=np.array(["dual"], dtype=str).repeat(n_sounding)
)

prob = GlobalEM1DProblemTD(
    [], sigmaMap=mapping, hz=hz, parallel=True, n_cpu=6,
    Solver=PardisoSolver
)
prob.pair(survey)


# In[21]:


mesh.vectorCCx


# In[23]:


# plt.scatter(xy[system==312,0], xy[system==312,1], s=5, c='k',label='312')
# plt.scatter(xy[system==304,0], xy[system==304,1], s=2, c='r',label='304')
# plt.legend()
# plt.gca().grid(True)


# In[ ]:





# In[24]:


ch1_cols = [c for c in df.columns if c.startswith('DBDT_Ch1')]
ch2_cols = [c for c in df.columns if c.startswith('DBDT_Ch2')]

lm_header_304 = ch1_cols[sl_lm_304]
hm_header_304 = ch2_cols[sl_hm_304]
lm_header_312 = ch1_cols[sl_lm_312]
hm_header_312 = ch2_cols[sl_hm_312]

ch1_cols.append('skytem_type')
ch2_cols.append('skytem_type')

data_hm_all = df.loc[df.CHANNEL_NO==2,ch2_cols]
data_lm_all = df.loc[df.CHANNEL_NO==1,ch1_cols]



floor_hm = 0.
floor_lm = 0.
# std = std


data = []
uncert = []
for idx in data_lm_all.index:
    if data_lm_all.loc[idx,'skytem_type']==312:
        data.append(np.r_[data_hm_all.loc[idx+1,hm_header_312].values/area_312,
                          data_lm_all.loc[idx,lm_header_312].values/area_312])
        uncert.append(np.r_[abs(data_hm_all.loc[idx+1,hm_header_312].values)/area_312*std + floor_hm,
                            abs(data_lm_all.loc[idx,lm_header_312].values)/area_312*std + floor_lm])
        
    elif data_lm_all.loc[idx,'skytem_type']==304:
        data.append(np.r_[data_hm_all.loc[idx+1,hm_header_304].values/area_304,
                          data_lm_all.loc[idx,lm_header_304].values/area_304])
                      
        uncert.append(np.r_[abs(data_hm_all.loc[idx+1,hm_header_304].values)/area_304*std + floor_hm,
                            abs(data_lm_all.loc[idx,lm_header_304].values)/area_304*std + floor_lm])
    else:
        print(data_lm_all.loc[idx,'skytem_type'])
        
        
dobs = np.concatenate(data).ravel().astype('float')
uncert = np.concatenate(uncert).ravel().astype('float')
dobs[np.isnan(dobs)] = 9999./area_312
inactive_inds = np.logical_or(dobs==9999./area_304,dobs==9999./area_312)
survey.dobs = -dobs.copy() 
uncert[inactive_inds] = np.Inf



print('survey.nD size',survey.nD)
print('uncert size',uncert.size)


# ### Run the inversion 

print('alpha_s',alpha_s,'\n','alpha_x',alpha_x,'\n','alpha_y',alpha_y,'\n','m0_val',m0_val,'\n')
output_dir = Path("/scratch/users/ianpg/timelapseAEM/line{line}-sep{sep}-s{alpha_s}x{alpha_x}y{alpha_y}m{m0_val}/"
                  .format(line=line[0],
                          sep=shift,
                          alpha_s=int(100*alpha_s),
                          alpha_x=int(100*alpha_x),
                          alpha_y=int(100*alpha_y),
                          m0_val=int(m0_val)))
print('ouput dir: {}'.format(output_dir))
mesh = get_2d_mesh(n_sounding, hz)
m0 = np.ones(mesh.nC) * np.log(1/m0_val)
regmap = Maps.IdentityMap(mesh)

# mapping is required ... for IRLS
reg = LateralConstraint(
    mesh, mapping=regmap,
    alpha_s =  alpha_s,
    alpha_x =  alpha_x,
    alpha_y =  alpha_y,    
)
reg.get_grad_horizontal(xy[:,:], hz, dim=3, use_cell_weights=True, minimum_distance=1e3)

np.random.seed(1)
dmisfit = DataMisfit.l2_DataMisfit(survey)
dmisfit.W = 1./uncert

opt = Optimization.ProjectedGNCG(maxIter = 10, maxIterCG=20)
# opt.upper = m_upper
# opt.lower = m_lower
invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)
beta = Directives.BetaSchedule(coolingFactor=2, coolingRate=1)
betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1.)
target = Directives.TargetMisfit(chifact=1.)
save_model = Directives.SaveOutputDictEveryIteration(directory=output_dir.as_posix())
inv = Inversion.BaseInversion(invProb, directiveList=[beta, betaest, target, save_model])
prob.counter = opt.counter = Utils.Counter()
opt.LSshorten = 0.5
opt.remember('xc')
try:
    save_model.outDict = {}
except:
    pass

print('Target misfit:',target.target)

## Run it!
mopt = inv.run(m0)
import local_utils
local_utils.save_obj(output_dir,save_model.outDict,'outDict')
