import os
import numpy as np
import sunpy.map
import _pickle as cPickle
from astropy.io import fits

hmi_dir = '/scratch/slam/sinjan/arlongterm_hmi/ic_720_nolimbdark/'

hmi_files = []

dates = [12,13,14,15,16,17]

#for i in dates:
#    hmi_files += [j for j in os.listdir(hmi_dir) if f'202310{i}' in j]
hmi_files = os.listdir(hmi_dir) 

hmi_files.sort()

hmi_ic_arr=np.zeros((4098,4098,len(hmi_files)))

ic_720_files = os.listdir('/scratch/slam/sinjan/arlongterm_hmi/ic_720/')

ic_720_files.sort()
    
for i,hfile in enumerate(hmi_files):
    data = fits.getdata(hmi_dir+hfile)  
    header = fits.open('/scratch/slam/sinjan/arlongterm_hmi/ic_720/'+ic_720_files[i])[1].header   
    # Note that it is case insensitive for the keys
    updated_map = sunpy.map.Map(data, header).rotate()  
    hmi_ic_arr[...,i] = updated_map.data
    
with open('/data/slam/sinjan/arlongterm_pickles/hmi_ic_720_nolimbdark_20231012_20231017.npy','wb') as f:
    np.save(f,hmi_ic_arr)