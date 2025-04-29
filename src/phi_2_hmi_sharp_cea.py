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
from scipy.fftpack import fft2, ifft2, fftshift

from utils import phi_disambig
from stereo_help import image_register
from apply_hmi_psf import make_psf_hmi_th
from phi_b2ptr import phi_b2ptr

def run_phi_disambig(bazi_fp):
    #get bamb_fp
    date = bazi_fp.split('bazi_202310')[1].split('T')[0]
    bamb_fp = f'../bamb_phi/2023_AR_LT_Jonas_bamb/2023-10-{date}/{bazi_fp.replace("bazi","bamb")}'
    return phi_disambig(bazi_fp, bamb_fp, method=2)


def create_bvec(bmag_fp, binc_fp, disambig):
    bmag = fits.getdata(bmag_fp)
    binc = fits.getdata(binc_fp)

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


def update_hdr(hdr, date):
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


def correct_shifts(phi_cea_map, hmi_map):
    #TODO:
    #- need to set nans to a value or use a mask/slice
    s = image_register(hmi_map, phi_cea_map, subpixel = False)

    phi_cea_shifted = np.roll(phi_cea_map.data, s[1], axis=(0,1))
    return sunpy.map.Map((phi_cea_shifted, phi_cea_map.fits_header))


def get_phi_cea_bptr_maps(bptr_psf, phi_updated_hdr):
    #get_hmi_cea_br_map
    #make_cea_hdr_for_phi
    #reproject_phi_2_hmi_cea (for Bp,Bt,Br separately)
    #find shifting comparing PHI_Br map with HMI Br map
    #apply shift to all 3 maps
    #return maps
    pass


def get_hmi_cea_br_map():
    #needed to provide WCS information for reprojection target and for finding any residual shifts
    pass


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


def save_phi_cea_maps_to_files(out_dir, bmag, phi_cea_bp_map, phi_cea_bt_map, phi_cea_br_map):
    # Save the maps to files
    bmag_filename = bmag.split('/')[-1]
    phi_cea_bp_filename = bmag_filename.replace('bmag','cea_Bp')
    phi_cea_bt_filename = bmag_filename.replace('bmag','cea_Bt')
    phi_cea_br_filename = bmag_filename.replace('bmag','cea_Br')
    
    phi_cea_bp_map.save(out_dir + phi_cea_bp_filename)
    phi_cea_bt_map.save(out_dir + phi_cea_bt_filename)
    phi_cea_br_map.save(out_dir + phi_cea_br_filename)


def main(out_dir, date):

    phi_bmag_files, phi_binc_files, phi_bazi_files = load_phi_files(date)

    for bmag_fp, binc_fp, bazi_fp in zip(phi_bmag_files, phi_binc_files, phi_bazi_files):
        disambig = run_phi_disambig(bazi_fp)
        bvec = create_bvec(bmag_fp, binc_fp, disambig)
        bptr, _, _ = phi_b2ptr(bvec)
        bptr_psf = apply_hmi_psf_on_phi_bptr(bmag_fp,bptr)
        phi_updated_hdr = update_hdr(bmag_fp, date)
        phi_cea_bp_map, phi_cea_bt_map, phi_cea_br_map = get_phi_cea_bptr_maps(bptr_psf, phi_updated_hdr)
        save_phi_cea_maps_to_files(out_dir, bmag_fp, phi_cea_bp_map, phi_cea_bt_map, phi_cea_br_map)