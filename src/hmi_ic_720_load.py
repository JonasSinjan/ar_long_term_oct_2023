import os
import numpy as np
import sunpy.map
import _pickle as cPickle

hmi_dir = '/scratch/slam/sinjan/arlongterm_hmi/ic_720/'

hmi_files = []

dates = [12,13,14,15,16,17]

for i in dates:
    hmi_files += [j for j in os.listdir(hmi_dir) if f'202310{i}' in j]
    
hmi_files.sort()

hmi_ic_arr=np.zeros((4098,4098,len(hmi_files)))

    
for i,hfile in enumerate(hmi_files):
    smap = sunpy.map.Map(hmi_dir+hfile).rotate()
    norm = smap.data[1500:2500,1500:2500].mean()
    tmp = smap.data
    hmi_ic_arr[...,i] = tmp/norm
    
with open('/data/slam/sinjan/arlongterm_pickles/hmi_ic_720_cutout_20231012_20231017.pickle','wb') as f:
    cPickle.dump(hmi_ic_arr,f)