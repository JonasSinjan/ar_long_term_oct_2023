import drms
import os

client = drms.Client(email='jonassinjan8@gmail.com')
out_dir = '/scratch/slam/sinjan/arlongterm_hmi/'

#blos_45 = 'hmi.m_45s[2023.03.29_11:36_TAI-2023.03.29_15:48_TAI]'
#blos_720 = 'hmi.m_720s[2023.03.29_11:36_TAI-2023.03.29_15:48_TAI]'
#b_720 = 'hmi.b_720s[2023.03.29_11:36_TAI-2023.03.29_15:48_TAI]'

#ic_45 = 'hmi.ic_45s[2023.03.29_11:36_TAI-2023.03.29_15:48_TAI]'
#ic_720 = 'hmi.ic_720s[2023.03.29_11:36_TAI-2023.03.29_15:48_TAI]'

b_720_dconS = 'hmi.b_720s_dconS[2023.10.12_00:20_TAI/5.5d@1h]'

m = client.export(b_720_dconS, protocol='fits')
m.download(out_dir+'b_720_dconS/')