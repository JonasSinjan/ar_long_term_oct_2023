from sunpy.map.header_helper import make_fitswcs_header
import sunpy.map
from sunpy.coordinates import propagate_with_solar_surface
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
import sunpy.coordinates
from sunpy.coordinates import frames
import numpy as np
import os
import json
import datetime
from scipy.fftpack import fft2, ifft2, fftshift

from utils import phi_disambig, largest_rectangle_in_mask, shift_array_multi, fits_get_sampling
from wcs_correction import image_register
from apply_hmi_psf import make_psf_hmi_th
from phi_b2ptr import phi_b2ptr


def load_phi_files(date):
    fdir = f'/scratch/solo/phi/AR_Long_Term_2023_SL/l2/2023-10-{date}/'
    phi_files = os.listdir(fdir)

    phi_bmag_files = [fdir + f for f in phi_files if 'bmag' in f]
    phi_binc_files = [fdir + f for f in phi_files if 'binc' in f]
    phi_bazi_files = [fdir + f for f in phi_files if 'bazi' in f]

    phi_bmag_files.sort()
    phi_binc_files.sort()
    phi_bazi_files.sort()

    assert len(phi_bmag_files) == len(phi_binc_files) == len(phi_bazi_files), "Mismatch in number of files"
    assert phi_bmag_files == [f.replace('binc','bmag') for f in phi_binc_files], "Mismatch in file names of Bmag with Binc"
    assert phi_bmag_files == [f.replace('bazi','bmag') for f in phi_bazi_files], "Mismatch in file names of Bmag with Bazi"

    return phi_bmag_files, phi_binc_files, phi_bazi_files


def run_phi_disambig(bazi, bazi_fp):
    #get bamb_fp
    date = bazi_fp.split('bazi_202310')[1].split('T')[0]
    bamb_filename = bazi_fp.split('/')[-1].replace("bazi","bamb")
    bamb_fp = f'../bamb_phi/2023_AR_LT_Jonas_bamb/2023-10-{date}/{bamb_filename}'
    return phi_disambig(bazi, bamb_fp, method=1)


def apply_cross_calibration_correction(bmag_fp, binc_fp, bazi_fp, hmi_cadence=720):
    bmag = fits.getdata(bmag_fp)
    binc = fits.getdata(binc_fp)
    bazi = fits.getdata(bazi_fp)

    stokes_fp = bmag_fp.replace('bmag','stokes')
    stokes = fits.getdata(stokes_fp)

    _,_,_,cpos = fits_get_sampling(stokes_fp)
    umbra_mask = np.where(stokes[cpos,0,:,:]<=0.55)
    penumbra_mask = np.where((stokes[cpos,0,:,:]>0.55) & (stokes[cpos,0,:,:]<=0.85))

    poltot = (stokes[:cpos,1,:,:]**2 + stokes[:cpos,2,:,:]**2 + stokes[:cpos,3,:,:]**2)/stokes[:cpos,0,:,:]**2
    poltot = np.mean(poltot, axis=0)

    poltot[umbra_mask] = 0
    poltot[penumbra_mask] = 0
    plage_mask = np.where((poltot>=0.005))
    

    if hmi_cadence == 720:
        bmag[umbra_mask] = bmag[umbra_mask] / 0.93
        bmag[penumbra_mask] = (bmag[penumbra_mask] - 183) / 0.877
        bmag[plage_mask] = (bmag[plage_mask] + 373) / 1.3

        shifted_binc = binc - 90

        shifted_binc[umbra_mask] = shifted_binc[umbra_mask] / 0.97
        shifted_binc[penumbra_mask] = shifted_binc[penumbra_mask] / 0.85
        shifted_binc[plage_mask] = shifted_binc[plage_mask] / 0.90

        #clip any values that are now outside of the range [-90, 90]
        shifted_binc = np.clip(shifted_binc, -90, 90)

        binc = shifted_binc + 90

        bazi[umbra_mask] = bazi[umbra_mask] - 5
        bazi[penumbra_mask] = bazi[penumbra_mask] - 5
    
    return bmag, binc, bazi


def create_bvec(bmag, binc, disambig):
    if type(bmag) == str and bmag.endswith('.fits.gz'):
        bmag = fits.getdata(bmag)
    if type(binc) == str and binc.endswith('.fits.gz'):
        binc = fits.getdata(binc)

    bvec = np.zeros((3, bmag.shape[1], bmag.shape[0]), dtype=np.float32)
    bvec[0, :, :] = bmag
    bvec[1, :, :] = binc
    bvec[2, :, :] = disambig
    return bvec


def apply_hmi_psf_on_phi_bptr(bmag_fp, bptr):
    h = fits.getheader(bmag_fp)
    phi_dsun = h['DSUN_OBS']
    hmi_psf = make_psf_hmi_th(bptr.shape[1],phi_dsun)
    hmi_psf /= hmi_psf.max()

    bp = bptr[:,:,0]
    bt = bptr[:,:,1]
    br = bptr[:,:,2]

    bp_psf = fftshift(ifft2(fft2(bp)/bp.size * fft2(hmi_psf/hmi_psf.sum())).real * bp.size) 
    bt_psf = fftshift(ifft2(fft2(bt)/bt.size * fft2(hmi_psf/hmi_psf.sum())).real * bt.size) 
    br_psf = fftshift(ifft2(fft2(br)/br.size * fft2(hmi_psf/hmi_psf.sum())).real * br.size) 
    
    bptr_psf = bptr.copy()
    bptr_psf[:,:,0] = bp_psf
    bptr_psf[:,:,1] = bt_psf
    bptr_psf[:,:,2] = br_psf
    return bptr_psf


def update_hdr(filepath, date):
    hdr = fits.getheader(filepath)
    phidatid = hdr['PHIDATID']
    with open(f'../hrt_wcs/phi_hrt_wcs_headers_arlongterm_202310{date}.json') as f:
        new_wcs = json.load(f)[phidatid]
    hdr['CRVAL1'] = new_wcs['CRVAL1']
    hdr['CRVAL2'] = new_wcs['CRVAL2']
    hdr['CRPIX1'] = new_wcs['CRPIX1']
    hdr['CRPIX2'] = new_wcs['CRPIX2']
    hdr['PC1_1'] = new_wcs['PC1_1']
    hdr['PC1_2'] = new_wcs['PC1_2']
    hdr['PC2_1'] = new_wcs['PC2_1']
    hdr['PC2_2'] = new_wcs['PC2_2']
    hdr['CROTA'] = new_wcs['CROTA']
    return hdr


def calculate_hmi_br_filetimes():
    fdir = '/scratch/slam/sinjan/arlongterm_hmi/sharp_cea_720s/'
    hmi_cea_files = os.listdir(fdir)
    hmi_br_files = sorted([fdir + f for f in hmi_cea_files if 'TAI.Br.fits' in f])
    dtai = datetime.timedelta(seconds=37)
    t_obs_hmi = [datetime.datetime.strptime(fits.getheader(i,1)['T_OBS'],'%Y.%m.%d_%H:%M:%S.%f_TAI') - dtai for i in hmi_br_files]
    return t_obs_hmi, hmi_br_files


def get_hmi_cea_br_map(hmi_br_filetimes, hmi_br_files, bmag_fp):
    phi_date_ear = datetime.datetime.strptime(fits.getheader(bmag_fp)['DATE_EAR'],'%Y-%m-%dT%H:%M:%S.%f')
    diff = [np.abs((phi_date_ear - t).total_seconds()) for t in hmi_br_filetimes]
    ind = np.argmin(diff)
    hmi_br_map = sunpy.map.Map(hmi_br_files[ind])
    print('HMI Br file: ', hmi_br_files[ind])
    return hmi_br_map


def make_cea_hdr_for_phi(hmi_br_map, phi_bptr_map):
    cea_hdr_phi = make_fitswcs_header(hmi_br_map.data.shape, hmi_br_map.reference_coordinate.replicate(rsun=phi_bptr_map.reference_coordinate.rsun), projection_code='CEA',scale=u.Quantity(hmi_br_map.scale))

    cea_hdr_phi['dsun_obs'] = hmi_br_map.coordinate_frame.observer.radius.to(u.m).value
    cea_hdr_phi['hglt_obs'] = hmi_br_map.coordinate_frame.observer.lat.value
    cea_hdr_phi['hgln_obs'] = hmi_br_map.coordinate_frame.observer.lon.value
    cea_hdr_phi['crpix1'] = hmi_br_map.fits_header['CRPIX1']
    cea_hdr_phi['crpix2'] = hmi_br_map.fits_header['CRPIX2']
    cea_hdr_phi['crval1'] = hmi_br_map.fits_header['CRVAL1']
    cea_hdr_phi['crval2'] = hmi_br_map.fits_header['CRVAL2']    
    cea_hdr_phi['PC1_1'] = 1
    cea_hdr_phi['PC1_2'] = 0
    cea_hdr_phi['PC2_1'] = 0
    cea_hdr_phi['PC2_2'] = 1
    cea_hdr_phi['cdelt1'] = hmi_br_map.fits_header['cdelt1']
    cea_hdr_phi['cdelt2'] = hmi_br_map.fits_header['cdelt2']
    return cea_hdr_phi


def reproject_phi_2_hmi_cea(phi_ptr_map, cea_hdr_phi):
    with propagate_with_solar_surface():
        outmap = phi_ptr_map.reproject_to(cea_hdr_phi,algorithm='adaptive', kernel='Hann')
    return outmap


def calculate_xy_shifts(phi_cea_br_map, hmi_br_map):
    mask = ~np.isnan(phi_cea_br_map.data)
    _, br, tl = largest_rectangle_in_mask(mask)

    slice_x = slice(br[1],tl[1],1)
    slice_y = slice(br[0],tl[0],1)
    #print(slice_x, slice_y)
    #print(hmi_br_map.meta)
    #print(phi_cea_br_map.meta)
    s = image_register(hmi_br_map.data[slice_y,slice_x], phi_cea_br_map.data[slice_y,slice_x], subpixel = False)
    print(s[1])
    return s[1]


def correct_shifts(s, phi_cea_bp_map,phi_cea_bt_map, phi_cea_br_map):
    phi_cea_bp_shifted = shift_array_multi(phi_cea_bp_map.data, s, fill_value=np.nan)
    phi_cea_bt_shifted = shift_array_multi(phi_cea_bt_map.data, s, fill_value=np.nan)
    phi_cea_br_shifted = shift_array_multi(phi_cea_br_map.data, s, fill_value=np.nan)

    phi_cea_bp_shifted_map = sunpy.map.Map((phi_cea_bp_shifted, phi_cea_bp_map.fits_header))
    phi_cea_bt_shifted_map = sunpy.map.Map((phi_cea_bt_shifted, phi_cea_bt_map.fits_header))
    phi_cea_br_shifted_map = sunpy.map.Map((phi_cea_br_shifted, phi_cea_br_map.fits_header))

    return phi_cea_bp_shifted_map, phi_cea_bt_shifted_map, phi_cea_br_shifted_map


def get_phi_cea_bptr_maps(bptr_psf, phi_updated_hdr, bmag_fp, hmi_br_filetimes, hmi_br_files):
    hmi_br_map = get_hmi_cea_br_map(hmi_br_filetimes, hmi_br_files, bmag_fp)

    
    bp_map = sunpy.map.Map((bptr_psf[:,:,0], phi_updated_hdr))
    bt_map = sunpy.map.Map((bptr_psf[:,:,1], phi_updated_hdr))
    br_map = sunpy.map.Map((bptr_psf[:,:,2], phi_updated_hdr))

    cea_hdr_phi = make_cea_hdr_for_phi(hmi_br_map, br_map)

    phi_cea_bp_map = reproject_phi_2_hmi_cea(bp_map, cea_hdr_phi)
    phi_cea_bt_map = reproject_phi_2_hmi_cea(bt_map, cea_hdr_phi)
    phi_cea_br_map = reproject_phi_2_hmi_cea(br_map, cea_hdr_phi)
    #out_dir = '/scratch/slam/sinjan/arlongterm/phi_cea_maps/'
    #save_phi_cea_maps_to_files(out_dir, bmag_fp, phi_cea_bp_map, phi_cea_bt_map, phi_cea_br_map, intermediate=True)
    shift = calculate_xy_shifts(phi_cea_br_map, hmi_br_map)
    
    return correct_shifts(shift, phi_cea_bp_map, phi_cea_bt_map, phi_cea_br_map)


def save_phi_cea_maps_to_files(out_dir, bmag, phi_cea_bp_map, phi_cea_bt_map, phi_cea_br_map, intermediate=False, extension=None):
    # Save the maps to files
    bmag_filename = bmag.split('/')[-1]
    phi_cea_bp_filename = bmag_filename.replace('bmag','cea_Bp').replace('.gz','')
    phi_cea_bt_filename = bmag_filename.replace('bmag','cea_Bt').replace('.gz','')
    phi_cea_br_filename = bmag_filename.replace('bmag','cea_Br').replace('.gz','')

    if extension is not None:
        phi_cea_bp_filename = phi_cea_bp_filename.replace('.fits', f'_{extension}.fits')
        phi_cea_bt_filename = phi_cea_bt_filename.replace('.fits', f'_{extension}.fits')
        phi_cea_br_filename = phi_cea_br_filename.replace('.fits', f'_{extension}.fits')

    if intermediate:
        phi_cea_bp_filename = phi_cea_bp_filename.replace('.fits','_intermediate.fits')
        phi_cea_bt_filename = phi_cea_bt_filename.replace('.fits','_intermediate.fits')
        phi_cea_br_filename = phi_cea_br_filename.replace('.fits','_intermediate.fits')
    
    phi_cea_bp_map.save(out_dir + phi_cea_bp_filename, overwrite=True)
    phi_cea_bt_map.save(out_dir + phi_cea_bt_filename, overwrite=True)
    phi_cea_br_map.save(out_dir + phi_cea_br_filename, overwrite=True)


def main(out_dir, date, num_files=None):

    phi_bmag_files, phi_binc_files, phi_bazi_files = load_phi_files(date)
    hmi_br_filetimes, hmi_br_files = calculate_hmi_br_filetimes()

    for bmag_fp, binc_fp, bazi_fp in zip(phi_bmag_files[:num_files], phi_binc_files[:num_files], phi_bazi_files[:num_files]):
        print(f'Processing {bmag_fp}')
        bmag, binc, bazi = apply_cross_calibration_correction(bmag_fp, binc_fp, bazi_fp, hmi_cadence=720)
        disambig = run_phi_disambig(bazi, bazi_fp)
        bvec = create_bvec(bmag, binc, disambig)
        phi_updated_hdr = update_hdr(bmag_fp, date)
        bptr, _, _ = phi_b2ptr(phi_updated_hdr,bvec)
        bptr_psf = apply_hmi_psf_on_phi_bptr(bmag_fp,bptr)
        phi_cea_bp_map, phi_cea_bt_map, phi_cea_br_map = get_phi_cea_bptr_maps(bptr_psf, phi_updated_hdr, bmag_fp, hmi_br_filetimes, hmi_br_files)
        print('Saving maps to files')
        save_phi_cea_maps_to_files(out_dir, bmag_fp, phi_cea_bp_map, phi_cea_bt_map, phi_cea_br_map, intermediate=False, extension='cross-calib')


if __name__ == "__main__":
    out_dir = '/scratch/slam/sinjan/arlongterm/phi_cea_maps/'
    main(out_dir, '13', 24)
    main(out_dir, '14', 24)
    main(out_dir, '15', 21)
    main(out_dir, '16', 24)
    main(out_dir, '17', 11)