# load, rotate and save HMI ic files to a pickle

import os
import numpy as np
import sunpy.map
import _pickle as cPickle

hmi_dir = '/data/slam/sinjan/arlongterm_hmi/ic_45/'

hmi_files = []

dates = [12,13,14,15,16,17]

for i in dates:
    hmi_files += [j for j in os.listdir(hmi_dir) if f'202310{i}' in j]
    
hmi_files.sort()

hmi_ic_arr=np.zeros((4102,4102,len(hmi_files)))

with open('/data/slam/sinjan/arlongterm_pickles/hrt_arr_first_ar.pickle','rb') as f:
    hrt_arr = cPickle.load(f)
    
for i,hfile in enumerate(hmi_files):
    smap = sunpy.map.Map(hmi_dir+hfile).rotate()
    norm = smap.data[1500:2500,1500:2500].mean()
    nans = np.isnan(hrt_arr[:,:,i])
    tmp = smap.data
    tmp[nans] = np.nan
    hmi_ic_arr[...,i] = tmp/norm
    
with open('/data/slam/sinjan/arlongterm_pickles/hmi_ic_cutout_20231012_20231017.pickle','wb') as f:
    cPickle.dump(hmi_ic_arr,f)