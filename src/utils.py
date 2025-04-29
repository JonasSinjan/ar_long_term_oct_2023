from astropy.io import fits
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize as spo
import scipy.signal as sps
from datetime import datetime as dt
import datetime
import time
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

from scipy.ndimage import map_coordinates
from astropy import units as u
from astropy.wcs import WCS

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\u001b[0m'

def printc(*args, color = bcolors.RESET, **kwargs):
    """My custom print() function.

    Parameters 
    ----------
    *args:
        arguments to be printed
    color: string
        color of the text
    **kwargs:
        keyword arguments to be passed to print()

    Returns
    -------
    None

    From SPGPyLib PHITools
    """
    print(u"\u001b"+f"{color}", end='\r')
    print(*args, **kwargs)
    print(u"\u001b"+f"{bcolors.RESET}", end='\r')
    return 


def load_fits(path):
    """load a fits file

    Parameters
    ----------
    path: string
    location of the fits file

    Output
    ------
    data: numpy array, of stokes images in (row, col, wv, pol) 
    header: hdul header object, header of the fits file
    """
    with fits.open(f'{path}') as hdul_tmp:
        data = np.asarray(hdul_tmp[0].data, dtype = np.float32)
        header = hdul_tmp[0].header

    return data, header 


def get_data(path, scaling = True, bit_convert_scale = True, scale_data = True):
    """load science data from path and scale it if needed

    Parameters
    ----------
    path: string
        location of the fits file
    scaling: bool
        if True, divide by number of accumulations
    bit_convert_scale: bool
        if True, divide by 256 if the data is in 24.8bit format
    scale_data: bool
        if True, scale the data to the maximum range of the detector
    
    Returns
    -------
    data: numpy array
        stokes images in (row, col, wv, pol)
    header: hdul header object
        header of the fits file
    """
    try:
        hdr = fits.open(path)
        data = hdr[0].data
#         data, header = load_fits(path)
        if 'L2' in hdr[0].header['LEVEL']:
            return hdr[0].data, hdr[0].header
        if np.size(hdr) > 9:
            ex = 9
        else:
            ex = 7
        
        if bit_convert_scale: #conversion from 24.8bit to 32bit
            try:
                IMGformat = hdr[ex].data['PHI_IMG_format'][-1]
            except:
                print("Most likely file does not have 9th Image extension")
                IMGformat = 'IMGFMT_16'
            if IMGformat != 'IMGFMT_24_8':
                data /=  256.
            else:
                print("Dataset downloaded as raw: no bit convert scaling needed")
        if scaling:
            
            accu = hdr[0].header['ACCACCUM']*hdr[0].header['ACCROWIT']*hdr[0].header['ACCCOLIT'] #getting the number of accu from header

            data /= accu

            printc(f"Dividing by number of accumulations: {accu}",color=bcolors.OKGREEN)

        if scale_data: #not for commissioning data

            try:    
                maxRange = hdr[ex].data['PHI_IMG_maxRange']
            
                data *= int(maxRange[0])/int(maxRange[-1])
            except IndexError:
                data *= 81920/128
                
        return data, hdr[0].header

    except Exception:
        printc("ERROR, Unable to open fits file: {}",path,color=bcolors.FAIL)
        raise ValueError()
       

def fits_get_sampling(file,num_wl = 6, TemperatureCorrection = True, TemperatureConstant = 40.323e-3, verbose = False):
    '''Open fits file, extract the wavelength axis and the continuum position, from Voltages in header

    Parameters
    ----------
    file: string
        location of the fits file
    num_wl: int
        number of wavelength
    TemperatureCorrection: bool
        if True, apply temperature correction to the wavelength axis
    TemperatureConstant: float
        Temperature constant to be used when TemperatureCorrection is True. Default: 40.323e-3 Å/K. Suggested (old) value: 36.46e-3 Å/K
    verbose: bool
        if True, print the continuum position
    
    Returns
    -------
    wave_axis: numpy array
        wavelength axis
    voltagesData: numpy array
        voltages of the wavelength axis
    tunning_constant: float
        tunning constant of the etalon
    cpos: int
        continuum position
    
    Adapted from SPGPyLib

    Usage: wave_axis,voltagesData,tunning_constant,cpos = fits_get_sampling(file,num_wl = 6, TemperatureCorrection = False, verbose = False)
    No S/C velocity corrected!!!
    cpos = 0 if continuum is at first wavelength and = num_wl - 1 (usually 5) if continuum is at the end
    '''
    fg_head = 3
    with fits.open(file) as hdu_list:
        header = hdu_list[fg_head].data
        tunning_constant = float(header[0][4])/1e9
        ref_wavelength = float(header[0][5])/1e3
        Tfg = hdu_list[0].header['FGOV1PT1'] # ['FGH_TSP1'] #temperature of the FG
        
        try:
            voltagesData = np.zeros(num_wl)
            hi = np.histogram(header['PHI_FG_voltage'],bins=num_wl+1)
            yi = hi[0]; xi = hi[1]
            j = 0        
            for i in range(num_wl + 1):
                if yi[i] != 0 :
                    if i < num_wl:
                        idx = np.logical_and(header['PHI_FG_voltage']>=xi[i],header['PHI_FG_voltage']<xi[i+1])
                    else:
                        idx = np.logical_and(header['PHI_FG_voltage']>=xi[i],header['PHI_FG_voltage']<=xi[i+1])
                    voltagesData[j] = int(np.median(header['PHI_FG_voltage'][idx]))
                    j += 1
        except:
            printc('WARNING: Running fits_get_sampling_SPG',color=bcolors.WARNING)
            return fits_get_sampling_SPG(file, False)
    
    d1 = voltagesData[0] - voltagesData[1]
    d2 = voltagesData[num_wl-2] - voltagesData[num_wl-1]
    if np.abs(d1) > np.abs(d2):
        cpos = 0
    else:
        cpos = num_wl-1
    if verbose:
        print('Continuum position at wave: ', cpos)
    wave_axis = voltagesData*tunning_constant + ref_wavelength  #6173.341
    
    if TemperatureCorrection:
        if verbose:
            printc('-->>>>>>> If FG temperature is not 61, the relation wl = wlref + V * tunning_constant is not valid anymore',color=bcolors.WARNING)
            printc('          Use instead: wl =  wlref + V * tunning_constant + temperature_constant_new*(Tfg-61)',color=bcolors.WARNING)
        # temperature_constant_old = 40.323e-3 # old temperature constant, still used by Johann
        # temperature_constant_new = 37.625e-3 # new and more accurate temperature constant
        # temperature_constant_new = 36.46e-3 # value from HS
        wave_axis += TemperatureConstant*(Tfg-61) # 20221123 see cavity_maps.ipynb with example
        
    return wave_axis,voltagesData,tunning_constant,cpos


def fits_get_sampling_SPG(file,verbose = False):
    '''
    Obtains the wavelength and voltages from  fits header

    Parameters
    ----------
    file : str
        fits file path
    verbose : bool, optional
        More info printed. The default is False.

    Returns
    -------
    wave_axis : array
        wavelength axis
    voltagesData : array
        voltages
    tunning_constant : float
        tunning constant of etalon (FG)
    cpos : int
        continuum position
    
    From SPGPylibs PHITools
    '''
    fg_head = 3
    with fits.open(file) as hdu_list:
        header = hdu_list[fg_head].data
        j = 0
        dummy = 0
        voltagesData = np.zeros((6))
        tunning_constant = 0.0
        ref_wavelength = 0.0
        for v in header:
            #print(v)
            if (j < 6):
                if tunning_constant == 0:
                    tunning_constant = float(v[4])/1e9
                if ref_wavelength == 0:
                    ref_wavelength = float(v[5])/1e3
                if np.abs(np.abs(float(v[2])) - np.abs(dummy)) > 5: #check that the next voltage is more than 5 from the previous, as voltages change slightly
                    voltagesData[j] = float(v[2])
                    dummy = voltagesData[j] 
                    j += 1

    d1 = voltagesData[0] - voltagesData[1]
    d2 = voltagesData[4] - voltagesData[5]
    if np.abs(d1) > np.abs(d2):
        cpos = 0
    else:
        cpos = 5
    if verbose:
        print('Continuum position at wave: ', cpos)
    wave_axis = voltagesData*tunning_constant + ref_wavelength  #6173.3356

    return wave_axis,voltagesData,tunning_constant,cpos


def check_filenames(data_f):
    """checks if the science scans have the same DID - this would otherwise cause an issue for naming the output demod files

    Parameters
    ----------
    data_f : list
        list of science scan file names
    
    Returns
    -------
    scan_name_list : list
        list of science scan file names with unique DIDs
    """
    try:
        scan_name_list = [fits.getheader(scan)['PHIDATID'] for scan in data_f]
    except:
        scan_name_list = [str(scan.split('.fits')[0][-10:]) for scan in data_f]

    seen = set()
    uniq_scan_DIDs = [x for x in scan_name_list if x in seen or seen.add(x)] #creates list of unique DIDs from the list

    #print(uniq_scan_DIDs)
    #print(scan_name_list)S
    if uniq_scan_DIDs == []:
        print("The scans' DIDs are all unique")

    else:

        for x in uniq_scan_DIDs:
            number = scan_name_list.count(x)
            if number > 1: #if more than one
                print(f"The DID: {x} is repeated {number} times")
                i = 1
                for index, name in enumerate(scan_name_list):
                    if name == x:
                        scan_name_list[index] = name + f"_{i}" #add _1, _2, etc to the file name, so that when written to output file not overwriting
                        i += 1

        print("The New DID list is: ", scan_name_list)

    return scan_name_list


def check_size(data_arr):
    """check if science scans have same dimensions

    Parameters
    ----------
    data_arr : list
        list of science scan data arrays
    
    Returns
    -------
    None
    """
    first_shape = data_arr[0].shape
    result = all(element.shape == first_shape for element in data_arr)
    if (result):
        print("All the scan(s) have the same dimension")

    else:
        print("The scans have different dimensions! \n Ending process")

        exit()


def check_cpos(cpos_arr):
    """checks if the science scans have the same continuum positions

    Parameters
    ----------
    cpos_arr : list
        list of continuum positions

    Returns
    -------
    None
    """
    first_cpos = cpos_arr[0]
    result = all(c_position == first_cpos for c_position in cpos_arr)
    if (result):
        print("All the scan(s) have the same continuum wavelength position")

    else:
        print("The scans have different continuum_wavelength postitions! Please fix \n Ending Process")

        exit()


def compare_cpos(flat,cpos,cpos_ref):
    """checks if flat continuum same as data, if not try to move flat around - this assumes that there was a mistake with the continuum position in the flat

    Parameters
    ----------
    flat : array
        flat field data array
    cpos : int
        continuum position of flat field
    cpos_ref : int
        continuum position of science scan

    Returns
    -------
    flat : array
        flat field data array with continuum position corrected
    """
    if cpos != cpos_ref:
        print("The flat field continuum position is not the same as the data, trying to correct.")

        if cpos == 5 and cpos_ref == 0:

            return np.roll(flat, 1, axis = -1)

        elif cpos == 0 and cpos_ref == 5:

            return np.roll(flat, -1, axis = -1)

        else:
            print("Cannot reconcile the different continuum positions. \n Ending Process.")

            exit()
    else:
        return flat


def check_pmp_temp(hdr_arr):
    """check science scans have same PMP temperature set point

    Parameters
    ----------
    hdr_arr : list
        list of science scan header arrays
    
    Returns
    -------
    pmp_temp : str
    """
    first_pmp_temp = int(hdr_arr[0]['HPMPTSP1'])
    result = all(hdr['HPMPTSP1'] == first_pmp_temp for hdr in hdr_arr)
    if (result):
        t0 = time.strptime('2023-03-28T00:10:00','%Y-%m-%dT%H:%M:%S')
        t1 = time.strptime('2023-03-30T00:10:00','%Y-%m-%dT%H:%M:%S')
        tobs = time.strptime(hdr_arr[0]['DATE-OBS'][:-4],'%Y-%m-%dT%H:%M:%S')
        
        if (tobs > t0 and tobs < t1):
            first_pmp_temp = 50
            printc('WARNING: Data acquired on 2023-03-28 and 2023-03-29 have a PMP temperature setting to 40 deg, but the PMP are fluctuating at ~45 deg \nException to HRT pipeline to use the 50 deg demodulation matrix.',color=bcolors.WARNING)
        print(f"All the scan(s) have the same PMP Temperature Set Point: {first_pmp_temp}")
        pmp_temp = str(first_pmp_temp)
        return pmp_temp
    else:
        print("The scans have different PMP Temperatures! Please fix \n Ending Process")

        exit()


def check_IMGDIRX(hdr_arr):
    """check if all scans contain imgdirx keyword

    Parameters
    ----------
    hdr_arr : list
        list of science scan header arrays
    
    Returns
    -------
    header_imgdirx_exists : bool
    imgdirx_flipped : str or bool
        OPTIONS: 'YES' or 'NO' or False
    """
    if all('IMGDIRX' in hdr for hdr in hdr_arr):
        header_imgdirx_exists = True
        first_imgdirx = hdr_arr[0]['IMGDIRX']
        result = all(hdr['IMGDIRX'] == first_imgdirx for hdr in hdr_arr)
        if (result):
            print(f"All the scan(s) have the same IMGDIRX keyword: {first_imgdirx}")
            imgdirx_flipped = str(first_imgdirx)
            
            return header_imgdirx_exists, imgdirx_flipped
        else:
            print("The scans have different IMGDIRX keywords! Please fix \n Ending Process")
            exit()
    else:
        header_imgdirx_exists = False
        print("Not all the scan(s) contain the 'IMGDIRX' keyword! Assuming all not flipped - Proceed with caution")
        return header_imgdirx_exists, False


def compare_IMGDIRX(flat,header_imgdirx_exists,imgdirx_flipped,header_fltdirx_exists,fltdirx_flipped):
    """returns flat that matches the orientation of the science data

    Parameters
    ----------
    flat : array
        flat field data array
    header_imgdirx_exists : bool
        if all scans contain imgdirx keyword
    imgdirx_flipped : str or bool
        OPTIONS: 'YES' or 'NO' or False
    header_fltdirx_exists : bool
        if flat contains fltdirx keyword
    fltdirx_flipped : str or bool
        OPTIONS: 'YES' or 'NO' or False

    Returns
    -------
    flat : array
        flat field data array with orientation corrected
    """
    if header_imgdirx_exists and imgdirx_flipped == 'YES': 
        #if science is flipped
        if header_fltdirx_exists:
            if fltdirx_flipped == 'YES':
                return flat
            else:
                print('Flipping the calibration dataset')
                return flat[:,:,::-1]
        else:
            print('Flipping the calibration dataset')
            return flat[:,:,::-1]
    elif (header_imgdirx_exists and imgdirx_flipped == 'NO') or not header_imgdirx_exists: 
        #if science is not flipped, or keyword doesnt exist, then assumed not flipped
        if header_fltdirx_exists:
            if fltdirx_flipped == 'YES':
                print('Flipping the calibration dataset')
                return flat[:,:,::-1] #flip flat back to match science
            else:
                return flat
        else:
            return flat
    else:
        return flat


def stokes_reshape(data):
    """converting science to [y,x,pol,wv,scans]
    
    Parameters
    ----------
    data : array
        science data array
    
    Returns
    -------
    data : array
        science data array with shape [y,x,pol,wv,scans]
    """
    data_shape = data.shape
    if data_shape[0] == 25:
        data = data[:24]
        data_shape = data.shape
    if data.ndim == 4: # [24,y,x,scans]
        data = np.moveaxis(data,0,2).reshape(data_shape[1],data_shape[2],6,4,data_shape[-1]) #separate 24 images, into 6 wavelengths, with each 4 pol states
        data = np.moveaxis(data, 2,3)
    elif data.ndim == 3: # [24,y,x]
        data = np.moveaxis(data,0,2).reshape(data_shape[1],data_shape[2],6,4) #separate 24 images, into 6 wavelengths, with each 4 pol states
        data = np.moveaxis(data, 2,3)
    elif data.ndim == 5: # it means that it is already [y,x,pol,wv,scans]
        pass
    return data
    

def fix_path(path,dir='forward',verbose=False):
    """This function is used to fix the path for windows and linux systems

    Parameters
    ----------
    path : str
        path to be fixed
    dir : str, optional
        direction of the path, by default 'forward'
    verbose : bool, optional
        print the path, by default False

    Returns
    -------
    path : str
        fixed path

    From SPGPylibs PHITools
    """
    path = repr(path)
    if dir == 'forward':
        path = path.replace(")", r"\)")
        path = path.replace("(", r"\(")
        path = path.replace(" ", r"\ ")
        path = os.path.abspath(path).split("'")[1]
        if verbose == True:
            print('forward')
            print(path)
        return path
    elif dir == 'backward':
        path = path.replace("\\\\", "")
        path = path.split("'")[1]
        if verbose == True:
            print('backward')
            print(path)
        return path
    else:
        pass   


def filling_data(arr, thresh, mode, axis = -1):
    """filling the data with cubic spline interpolation

    Parameters
    ----------
    arr : array
        array to be filled
    thresh : float
        threshold for filling
    mode : str
        mode for filling, 'max', 'min', 'abs', 'exact rows', 'exact columns'
    axis : int, optional
        axis to be filled, by default -1

    Returns
    -------
    array
        filled array
    """
    from scipy.interpolate import CubicSpline
    
    a0 = np.zeros(arr.shape)
    a0 = arr.copy()
    if mode == 'max':
        a0[a0>thresh] = np.nan
    if mode == 'min':
        a0[a0<thresh] = np.nan
    if mode == 'abs':
        a0[np.abs(a0)>thresh] = np.nan
    if 'exact rows' in mode.keys():
        rows = mode['exact rows']
        for r in rows:
            a0[r] = np.nan
        axis = 1
    if 'exact columns' in mode.keys():
        cols = mode['exact columns']
        for c in cols:
            a0[:,r] = np.nan
        axis = 0
    
    N = arr.shape[axis]; n = arr.shape[axis-1]
    
    with np.errstate(divide='ignore'):
        for i in range(N):
            a1 = a0.take(i, axis=axis)
            nans, index = np.isnan(a1), lambda z: z.nonzero()[0]
            if nans.sum()>0:
                a1[nans] = CubicSpline(np.arange(n)[~nans], a1[~nans])(np.arange(n))[nans]
                if axis == 0:
                    a0[i] = a1
                else:
                    a0[:,i] = a1
    return a0
    
def ARmasking(stk, initial_mask, cpos = 0, bin_lim = 7, mask_lim = 5, erosion_iter = 3, dilation_iter = 3):
    """Creates a mask to cover active parts of the FoV
    Parameters
    ----------
    stk : array
        Stokes Vector
    Initial_mask : array
        Mask with off-limb or field_Stop excluded
    cpos : int
        continuum position (DEFAULT: 0)
    bin_lim : float
        number to be multiplied to the polarized std to se the maximum limit of the bins (DEFAULT: 7)
    mask_lim : float
        number of std that defines the contour of the mask (DEFAULT: 5)
    erosion_iter : int
        number of iterations for the erosion of the mask (DEFAULT: 3)
    dilation_iter : int
        number of iterations for the dilation of the mask (DEFAULT: 3)    

    Returns
    -------
    array
        AR_mask
    """

    AR_mask = initial_mask.copy()
    # automatic bins looking at max std of the continuum polarization
    if stk.shape[1] == 4:
        stk = np.einsum('lpyx->yxpl',stk.copy())
    lim = np.max((stk[:,:,1:,cpos]).std(axis=(0,1)))*bin_lim
    bins = np.linspace(-lim,lim,150)

    for p in range(1,4):
        hi = np.histogram(stk[:,:,p].flatten(),bins=bins)
        gval = gaussian_fit(hi, show = False)
        AR_mask *= np.max(np.abs(stk[:,:,p] - gval[1]),axis=-1) < mask_lim*abs(gval[2])

    AR_mask = np.asarray(AR_mask, dtype=bool)

    # erosion and dilation to remove small scale masked elements
    AR_mask = ~binary_dilation(binary_erosion(~AR_mask.copy(),generate_binary_structure(2,2), iterations=erosion_iter),
                               generate_binary_structure(2,2), iterations=dilation_iter)
    
    return AR_mask

def auto_norm(file_name):
    """This function is used to normalize the data from the fits extensions

    Parameters
    ----------
    file_name : str
        path to file

    Returns
    -------
    norm : float
        normalization factor
    """
    d = fits.open(file_name)
    try:
        print('PHI_IMG_maxRange 0:',d[9].data['PHI_IMG_maxRange'][0])
        print('PHI_IMG_maxRange -1:',d[9].data['PHI_IMG_maxRange'][-1])
        norm = d[9].data['PHI_IMG_maxRange'][0]/ \
        d[9].data['PHI_IMG_maxRange'][-1]/256/ \
        (d[0].header['ACCCOLIT']*d[0].header['ACCROWIT']*d[0].header['ACCACCUM'])
    except:
        norm = 1/256/ \
        (d[0].header['ACCCOLIT']*d[0].header['ACCROWIT']*d[0].header['ACCACCUM'])
    print('accu:',(d[0].header['ACCCOLIT']*d[0].header['ACCROWIT']*d[0].header['ACCACCUM']))
    return norm


# new functions by DC ######################################
def circular_mask(h, w, center, radius):
    """create a circular mask

    Parameters
    ----------
    h : int
        height of the mask
    w : int
        width of the mask
    center : [x,y]
        center of the mask
    radius : float
        radius of the mask

    Returns
    -------
    mask: 2D array
        mask with 1 inside the circle and 0 outside
    """
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def limb_fitting(img, hdr, field_stop, verbose=True, percent=False, fit_results=False):
    """Fits limb to the image using least squares method.

    Parameters
    ----------
    img : numpy.ndarray
        Image to fit limb to.
    hdr : astropy.io.fits.header.Header
        header of fits file
    field_stop : array
        field stop array
    verbose : bool, optional
        Print limb fitting results, by default True
    percent : bool, optional
        return mask with 96% of the readius, by default False
    fit_results : bool, optional
        return results of the circular fit, by default False

    Returns
    -------
    mask100: numpy.ndarray
        masked array (ie off disc region) with 100% of the radius
    sly: slice
        slice in y direction to be used for normalisation (ie good pixels on disc)
    slx: slice
        slice in x direction to be used for normalisation (ie good pixels on disc)
    side: str
        limb side
    mask96: numpy.ndarray
        masked array (ie off disc region) with 96% of the radius (only if percent = True)
    """
    def _residuals(p,x,y):
        """
        Finding the residuals of the fit
        
        Parameters
        ----------
        p : list
            [xc,yc,R] - coordinates of the centre and radius of the circle
        x : float
            test x coordinate
        y : float
            test y coordinate

        Returns
        -------
        residual = R**2 - (x-xc)**2 - (y-yc)**2
        """
        xc,yc,R = p
        residual = R**2 - (x-xc)**2 - (y-yc)**2
        return residual
    
    def _is_outlier(points, thresh=2):
        """Returns a boolean array with True if points are outliers and False otherwise
        
        Parameters
        ----------
        points : numpy.ndarray
            1D array of points
        thresh : int, optional
            threshold for outlier detection, by default 2
        """
        if len(points.shape) == 1:
            points = points[:,None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh
        
    def _image_derivative(d):
        """Calculates the image derivative in x and y using a 3x3 kernel
        
        Parameters
        ----------
        d : numpy.ndarray
            image to calculate derivative of
        
        Returns
        -------
        SX : numpy.ndarray
            derivative in x direction
        SY : numpy.ndarray
            derivative in y direction
        """
        import numpy as np
        from scipy.signal import convolve
        kx = np.asarray([[1,0,-1], [1,0,-1], [1,0,-1]])
        ky = np.asarray([[1,1,1], [0,0,0], [-1,-1,-1]])

        kx=kx/3.
        ky=ky/3.

        SX = convolve(d, kx,mode='same')
        SY = convolve(d, ky,mode='same')

        return SX, SY

    from scipy.optimize import least_squares
    from scipy.ndimage import binary_erosion

    side, center, Rpix, sly, slx, finder_small = limb_side_finder(img,hdr,verbose=verbose,outfinder=True)
    f = 16
    fract = int(img.shape[0]//f)
    finder = np.zeros(img.shape)
    for i in range(f):
        for j in range(f):
            finder[fract*i:fract*(i+1),fract*j:fract*(j+1)] = finder_small[i,j]
        
    if side == '':
        output = [None,sly,slx,side]
        
        if percent:
            output += [None]
        if fit_results:
            output += [None]    

        return output
    
    if 'N' in side or 'S' in side:
        img = np.moveaxis(img,0,1)
        finder = np.moveaxis(finder,0,1)
        center = center[::-1]
    
    s = 5
    thr = 3
    
    diff = _image_derivative(img)[0][s:-s,s:-s]
    rms = np.sqrt(np.mean(diff[field_stop[s:-s,s:-s]>0]**2))
    yi, xi = np.where(np.abs(diff*binary_erosion(field_stop,np.ones((2,2)),iterations=20)[s:-s,s:-s])>rms*thr)
    tyi = yi.copy(); txi = xi.copy()
    yi = []; xi = []
    for i,j in zip(tyi,txi):
        if finder[i,j]:
            yi += [i+s]; xi += [j+s]
    yi = np.asarray(yi); xi = np.asarray(xi)
    
    out = _is_outlier(xi)

    yi = yi[~out]
    xi = xi[~out]

    p = least_squares(_residuals,x0 = [center[0],center[1],Rpix], args=(xi,yi),
                              bounds = ([center[0]-150,center[1]-150,Rpix-50],[center[0]+150,center[1]+150,Rpix+50]))
        
    mask100 = circular_mask(img.shape[0],img.shape[1],[p.x[0],p.x[1]],p.x[2])

    mask96 = circular_mask(img.shape[0],img.shape[1],[p.x[0],p.x[1]],p.x[2]*.96)
    
    output = [sly,slx,side]
    if 'N' in side or 'S' in side:
        output = [np.moveaxis(mask100,0,1)] + output
        if percent:
            output += [np.moveaxis(mask96,0,1)]
    else:
        output = [mask100] + output
        if percent:
            output += [mask96]
    if fit_results:
        output += [p]    

    return output

def fft_shift(img,shift):
    """Shift an image in the Fourier domain and return the shifted image (non fourier domain)

    Parameters
    ----------
    img : 2D-image
        2D-image to be shifted
    shift : list
        [dy,dx] shift in pixel

    Returns
    -------
    img_shf : 2D-image
        shifted image
    """
    try:
        import pyfftw.interfaces.numpy_fft as fft
    except:
        import numpy.fft as fft
    sz = img.shape
    ky = fft.ifftshift(np.linspace(-np.fix(sz[0]/2),np.ceil(sz[0]/2)-1,sz[0]))
    kx = fft.ifftshift(np.linspace(-np.fix(sz[1]/2),np.ceil(sz[1]/2)-1,sz[1]))

    img_fft = fft.fft2(img)
    shf = np.exp(-2j*np.pi*(ky[:,np.newaxis]*shift[0]/sz[0]+kx[np.newaxis]*shift[1]/sz[1]))
    
    img_fft *= shf
    img_shf = fft.ifft2(img_fft).real
    
    return img_shf
    
def SPG_shifts_FFT(data,norma=True,prec=100,coarse_prec = 1.5,sequential = False):
    """FFT shifting function from SPGPylibs as used in FDT pipeline.

    Parameters
    ----------
    data : 3D-array
        [z,y,x] 3D-array of images to be shifted, images stacked along the first (z) axis.
    norma : bool, optional
        If True, the images are normalized before shifting. The default is True.
    prec : int, optional
        Precision of the shift. The default is 100.
    coarse_prec : float, optional
        Coarse precision of the shift. The default is 1.5.
    sequential : bool, optional
        if True, adds shifts to the previous one. The default is False.

    Returns
    -------
    row_shift : 1D-array
        row shifts
    column_shift : 1D-array
        column shifts
    shifted_image: 3D-array
        shifted images

    From SPGPylibs. Same function used for FDT pipeline, adapted by DC
    At least two images should be provided!
    usage: s_y, s_x, simage = PHI_shifts_FFT(image_cropped,prec=500,verbose=True,norma=False)
    (row_shift, column_shift) defined as  center = center + (y,x) 
    """
    def sampling(N):
        """
        From SPGPylibs. Same function used for FDT pipeline.
        This function creates a grid of points with NxN dimensions for calling the
        Zernike polinomials.
        Output:
            X,Y: X and Y meshgrid of the detector
        """
        if N%2 != 0:
            print('Number of pixels must be an even integer!')
            return
        x=np.linspace(-N/2,N/2,N)
        y=np.copy(x)
        X,Y=np.meshgrid(x,y)
        return X,Y 

    def aperture(X,Y,N,R):
        """
        From SPGPylibs. Same function used for FDT pipeline.
        This function calculates a simple aperture function that is 1 within
        a circle of radius R, takes and intermediate value between 0
        and 1 in the edge and 0 otherwise. The values in the edges are calculated
        according to the percentage of area corresponding to the intersection of the
        physical aperture and the edge pixels.
        http://photutils.readthedocs.io/en/stable/aperture.html
        Input:
            X,Y: meshgrid with the coordinates of the detector ('sampling.py')
            R: radius (in pixel units) of the mask
        Output:
            A: 2D array with 0s and 1s
        """
        from photutils import CircularAperture
        A=CircularAperture((N/2,N/2),r=R) #Circular mask (1s in and 0s out)
        A=A.to_mask(method='exact') #Mask with exact value in edge pixels
        A=A.to_image(shape=(N,N)) #Conversion from mask to image
        return A
        
    def dft_fjbm(F,G,kappa,dftshift,nr,nc,Nr,Nc,kernr,kernc):
        """
        From SPGPylibs. Same function used for FDT pipeline.
        Calculates the shift between a couple of images 'f' and 'g' with subpixel
        accuracy by calculating the IFT with the matrix multiplication tecnique.
        Shifts between images must be kept below 1.5 'dftshift' for the algorithm
        to work.
        Input: 
            F,G: ffts of images 'f' and 'g' without applying any fftshift
            kappa: inverse of subpixel precision (kappa=20 > 0.005 pixel precision)
        Output:
        """
        #DFT by matrix multiplication
        M=F*np.conj(G) #Cross-correlation
        CC=kernr @ M @ kernc
        CCabs=np.abs(CC)
        ind = np.unravel_index(np.argmax(CCabs, axis=None), CCabs.shape)
        CCmax=CC[ind]
        rloc,cloc=ind-dftshift
        row_shift=-rloc/kappa
        col_shift=-cloc/kappa
        rg00=np.sum(np.abs(F)**2)
        rf00=np.sum(np.abs(G)**2)
        error=np.sqrt(1-np.abs(CCmax)**2/(rg00*rf00))
        Nc,Nr=np.meshgrid(Nc,Nr)

        Gshift=G*np.exp(1j*2*np.pi*(-row_shift*Nr/nr-col_shift*Nc/nc)) 
        return error,row_shift,col_shift,Gshift


    #Normalization for each image
    sz,sy,sx = data.shape
    f=np.copy(data)
    if norma == True:
        norm=np.zeros(sz)
        for i in range(sz):
            norm[i]=np.mean(data[i,:,:])
            f[i,:,:]=data[i,:,:]/norm[i]

    #Frequency cut
    wvl=617.3e-9
    D = 0.14  #HRT
    foc = 4.125 #HRT
    fnum = foc / D
    nuc=1/(wvl*fnum) #Critical frequency (1/m)
    N=sx #Number of pixels per row/column (max. 2048)
    deltax = 10e-6 #Pixel size
    deltanu=1/(N*deltax)
    R=(1/2)*nuc/deltanu
    nuc=2*R#Max. frequency [pix]

    #Mask
    X,Y = sampling(N)
    mask = aperture(X,Y,N,R)

    #Fourier transform
    f0=f[0,:,:]
    #pf.movie(f0-f,'test.mp4',resol=1028,axis=0,fps=5,cbar='yes',cmap='seismic')
    F=np.fft.fft2(f0)

    #Masking
    F=np.fft.fftshift(F)
    F*=mask
    F=np.fft.ifftshift(F)

    #FJBM algorithm
    kappa=prec
    n_out=np.ceil(coarse_prec*2.*kappa)
    dftshift=np.fix(n_out/2)
    nr,nc=f0.shape
    Nr=np.fft.ifftshift(np.arange(-np.fix(nr/2),np.ceil(nr/2)))
    Nc=np.fft.ifftshift(np.arange(-np.fix(nc/2),np.ceil(nc/2)))
    kernc=np.exp((-1j*2*np.pi/(nc*kappa))*np.outer(\
    np.fft.ifftshift(np.arange(0,nc).T-np.floor(nc/2)),np.arange(0,n_out)-dftshift))
    kernr=np.exp((-1j*2*np.pi/(nr*kappa))*np.outer(\
    np.arange(0,n_out)-dftshift,np.fft.ifftshift(np.arange(0,nr).T-np.floor(nr/2))))

    row_shift=np.zeros(sz)
    col_shift=np.zeros(sz)
    shifted_image = np.zeros_like(data)

    if sequential == False:
        for i in np.arange(1,sz):
            g=f[i,:,:]
            G=np.fft.fft2(g)
            #Masking
            G=np.fft.fftshift(G)
            G*=mask
            G=np.fft.ifftshift(G)

            error,row_shift[i],col_shift[i],Gshift=dft_fjbm(F,G,kappa,dftshift,nr,\
            nr,Nr,Nc,kernr,kernc)
            shifted_image[i,:,:] = np.real(np.fft.ifft2(Gshift)) 
    if sequential == True:
        print('No fastidies')
        for i in np.arange(1,sz):
            g=f[i,:,:]
            G=np.fft.fft2(g)
            #Masking
            G=np.fft.fftshift(G)
            G*=mask
            G=np.fft.ifftshift(G)

            error,row_shift[i],col_shift[i],Gshift=dft_fjbm(F,G,kappa,dftshift,nr,\
            nr,Nr,Nc,kernr,kernc)
            shifted_image[i,:,:] = np.real(np.fft.ifft2(Gshift)) 
            F = np.copy(G) #Sequencial
            row_shift[i] = row_shift[i] + row_shift[i-1]
            col_shift[i] = col_shift[i] + col_shift[i-1]
 
    return row_shift,col_shift,shifted_image

#plotting functions for quick data analysis for communal use

def find_nearest(array, value):
    """return index of nearest value in array to the desired value

    Parameters
    ----------
    array : array
        array to search
    value : float
        value to search for

    Returns
    -------
    idx : int
        index of nearest value in array to the desired value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def gaus(x,a,x0,sigma):
    """return Gauss function

    Parameters
    ----------
    x : array
        x values
    a : float
        amplitude
    x0 : float
        mean x value
    sigma : float
        standard deviation

    Returns
    -------
    Gauss Function : array
    """
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def gaussian_fit(a,show=True):
    """Gaussian fit for data 'a' from np.histogram or plt.hist

    Parameters
    ----------
    a : array
        output from np.histogram
    show : bool, optional
        show plot of fit, by default True
    
    Returns
    -------
    p : array
        fitted coefficients for Gaussian function
    """
    xx=a[1][:-1] + (a[1][1]-a[1][0])/2
    y=a[0][:]
    p0=[0.,sum(xx*y)/sum(y),np.sqrt(sum(y * (xx - sum(xx*y)/sum(y))**2) / sum(y))] #weighted avg of bins for avg and sigma inital values
    p0[0]=y[find_nearest(xx,p0[1])-5:find_nearest(xx,p0[1])+5].mean() #find init guess for ampltiude of gauss func
    try:
        p,cov=spo.curve_fit(gaus,xx,y,p0=p0)
        if show:
            lbl = '{:.2e} $\pm$ {:.2e}'.format(p[1],p[2])
            plt.plot(xx,gaus(xx,*p),'r--', label=lbl)
            plt.legend(fontsize=9)
        return p
    except:
        printc("Gaussian fit failed: return initial guess",color=bcolors.WARNING)
        return p0
        

def iter_noise(temp, p = [1,0,1e-1], eps = 1e-6):
    """Iterative Gaussian fit for noise estimate

    Parameters
    ----------
    temp : array
        data to fit
    p : array, optional
        initial guess for Gaussian fit, by default [1,0,1e-1]
    eps : float, optional
        convergence criteria, by default 1e-6
    
    Returns
    -------
    p : array
        fitted coefficients for Gaussian function
    hi : array
        output from np.histogram
    """
    p_old = [1,0,100]; count = 0
    it = 0
    while np.abs(p[2] - p_old[2])>eps:
        p_old = p; count += 1
        hi = np.histogram(temp, bins=np.linspace(p[1] - 3*p[2],p[1] + 3*p[2],200),density=False);
        p = gaussian_fit(hi, show=False)
        if it == 50:
            break
        it += 1
    return p, hi

  
def blos_noise(blos_file, iter=True, fs = None):
    """plot blos on left panel, and blos hist + Gaussian fit (w/ iterative fit option - only shown in legend)

    Parameters
    ----------
    blos_file : str
        path to blos file
    iter : bool, optional
        performs iterative Gaussian fit, by default True
    fs : array, optional
        field stop mask, by default None

    Returns
    -------
    p or p_iter: fit coefficients for Gaussian function
    """
    blos = fits.getdata(blos_file)
    hdr = fits.getheader(blos_file)
    #get the pixels that we want to consider (central 512x512 and limb handling)
    _, _, _, sly, slx = limb_side_finder(blos, hdr)
    values = blos[sly,slx]

    fig, ax = plt.subplots(1,2, figsize = (14,6))
    if fs is not None:
        idx = np.where(fs<1)
        blos[idx] = -300
    im1 = ax[0].imshow(blos, cmap = "gray", origin = "lower", vmin = -200, vmax = 200)
    fig.colorbar(im1, ax = ax[0], fraction=0.046, pad=0.04)
    hi = ax[1].hist(values.flatten(), bins=np.linspace(-2e2,2e2,200))
    tmp = [0,0]
    tmp[0] = hi[0].astype('float64')
    tmp[1] = hi[1].astype('float64')

    #guassian fit + label
    p = gaussian_fit(tmp, show = False)    
    xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
    lbl = f'{p[1]:.2e} $\pm$ {p[2]:.2e} G'
    
    if iter:
        ax[1].plot(xx,gaus(xx,*p),'r--', label=lbl)
        try:
            p_iter, hi_iter = iter_noise(values,[1.,0.,10.],eps=1e-4); p_iter[0] = p[0]
            ax[1].plot(xx,gaus(xx,*p_iter),'g--', label= f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e} G")
            # ax[1].scatter(0,0, color = 'white', s = 0, label = lbl) #also display the original fit in legend
        except:
            print("Iterative Gauss Fit failed")
            p_iter = p

    else:
        ax[1].plot(xx,gaus(xx,*p),'r--', label=lbl)

    ax[1].legend(fontsize=15)

    date = blos_file.split('blos_')[1][:15]
    dt_str = dt.strptime(date, "%Y%m%dT%H%M%S")
    fig.suptitle(f"Blos {dt_str}")

    plt.tight_layout()
    plt.show()

    if iter:
        return p_iter
    else:
        return p


def blos_noise_arr(blos, fs = None):
    """
    plot blos on left panel, and blos hist + Gaussian fit (w/ iterative option)

    DEPRACATED - use blos_noise instead
    """

    fig, ax = plt.subplots(1,2, figsize = (14,6))
    if fs is not None:
        idx = np.where(fs<1)
        blos[idx] = -300
    im1 = ax[0].imshow(blos, cmap = "gray", origin = "lower", vmin = -200, vmax = 200)
    fig.colorbar(im1, ax = ax[0], fraction=0.046, pad=0.04)
    hi = ax[1].hist(blos.flatten(), bins=np.linspace(-2e2,2e2,200))
    #print(hi)
    tmp = [0,0]
    tmp[0] = hi[0].astype('float64')
    tmp[1] = hi[1].astype('float64')


    #guassian fit + label
    p = gaussian_fit(tmp, show = False)  
    try:  
        p_iter, hi_iter = iter_noise(blos,[1.,0.,1.],eps=1e-4)
        ax[1].scatter(0,0, color = 'white', s = 0, label = f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e} G")
    except:
        print("Iterative Gauss Fit failed")
    xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
    lbl = f'{p[1]:.2e} $\pm$ {p[2]:.2e} G'
    ax[1].plot(xx,gaus(xx,*p),'r--', label=lbl)
    ax[1].legend(fontsize=15)

    plt.tight_layout()
    plt.show()
    

def stokes_noise(stokes_file, iter=True):
    """plot stokes V on left panel, and Stokes V hist + Gaussian fit (w/ iterative option)

    Parameters
    ----------
    stokes_file : str
        path to stokes file
    iter : bool, optional
        whether to use iterative Gaussian fit, by default True

    Returns
    -------
    p or p_iter: array
        Gaussian fit parameters
    """
    stokes = fits.getdata(stokes_file)
    if stokes.shape[0] == 6:
        stokes = np.einsum('lpyx->yxpl',stokes)
    hdr = fits.getheader(stokes_file)
    out = fits_get_sampling(stokes_file)
    cpos = out[3]
    #first get the pixels that we want (central 512x512 and limb handling)
    _, _, _, sly, slx = limb_side_finder(stokes[:,:,3,cpos], hdr)
    values = stokes[sly,slx,3,cpos]

    fig, ax = plt.subplots(1,2, figsize = (14,6))
    im1 = ax[0].imshow(stokes[:,:,3,cpos], cmap = "gist_heat", origin = "lower", vmin = -1e-2, vmax = 1e-2)
    fig.colorbar(im1, ax = ax[0], fraction=0.046, pad=0.04)
    hi = ax[1].hist(values.flatten(), bins=np.linspace(-1e-2,1e-2,200))
    #print(hi)
    tmp = [0,0]
    tmp[0] = hi[0].astype('float64')
    tmp[1] = hi[1].astype('float64')

    #guassian fit + label
    p = gaussian_fit(tmp, show = False)    
    xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
    lbl = f'{p[1]:.2e} $\pm$ {p[2]:.2e}'
    
    if iter:
        ax[1].plot(xx,gaus(xx,*p),'r--', label=lbl)
        try:
            p_iter, hi_iter = iter_noise(values,[1.,0.,.1],eps=1e-6); p_iter[0] = p[0]
            ax[1].plot(xx,gaus(xx,*p_iter),'g--', label= f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e}")
            # ax[1].scatter(0,0, color = 'white', s = 0, label = lbl) #also display the original fit in legend
        except:
            print("Iterative Gauss Fit failed")
            ax[1].plot(xx,gaus(xx,*p),'r--', label=lbl)
            p_iter = p

    else:
        ax[1].plot(xx,gaus(xx,*p),'r--', label=lbl)

    ax[1].legend(fontsize=15)

    date = stokes_file.split('stokes_')[1][:15]
    dt_str = dt.strptime(date, "%Y%m%dT%H%M%S")
    fig.suptitle(f"Stokes {dt_str}")

    plt.tight_layout()
    plt.show()

    if iter:
        return p_iter
    else:
        return p


########### new WCS script 3/6/2022 ###########
def image_derivative(d):
    """Calculates the total image derivative (x**2 + y**2) using a 3x3 kernel
    
    Parameters
    ----------
    d : numpy.ndarray
        image to calculate derivative of
    
    Returns
    -------
    A : numpy.ndarray
        image derivative (combined X and Y)
    """
    kx = np.asarray([[1,0,-1], [1,0,-1], [1,0,-1]])
    ky = np.asarray([[1,1,1], [0,0,0], [-1,-1,-1]])
    kx=kx/3.
    ky=ky/3.

    SX = sps.convolve(d, kx,mode='same')
    SY = sps.convolve(d, ky,mode='same')

    A=SX**2+SY**2

    return A

def Inv2(x_c,y_c,x_u,y_u,k):
    """
    undistortion model
    by F. Kahil (MPS)
    """
    r_u = np.sqrt((x_u-x_c)**2+(y_u-y_c)**2) 
    x_d = x_c+(x_u-x_c)*(1-k*r_u**2)
    y_d = y_c+(y_u-y_c)*(1-k*r_u**2)
    return x_d,y_d

def und(hrt, order=1, flip = True):
    """
    spherical undistortion function 
    by F. Kahil (MPS)
    """
    if flip:
        hrt = hrt[:,::-1]
    Nx = Ny=2048
    x = y = np.arange(Nx)
    X,Y = np.meshgrid(x,y)
    x_c =1016
    y_c =982
    k=8e-09
    hrt_und = np.zeros((Nx,Ny))
    x_d, y_d = Inv2(x_c,y_c,X,Y,k)
    hrt_und = map_coordinates(hrt,[y_d,x_d],order=order)
    if flip:
        return hrt_und[:,::-1]
    else:
        return hrt_und


###############################################

def cavity_shifts(cavity_f, wave_axis,rows,cols,returnWL = True):
    """applies cavity shifts to the wave axis for use in RTE

    Parameters
    ----------
    cavity_f : str or array
        path to cavity map fits file or cavity array (already cropped)
    wave_axis : array
        wavelength axis
    rows : array
        rows of the pixels in the image, where the respective wavelength is shifted
    cols : array
        columns of the pixels in the image, where the respective wavelength is shifted

    Returns
    -------
    new_wave_axis[rows, cols]: array
        wavelength axis with the cavity shifts applied to the respective pixels
    """
    if isinstance(cavity_f,str):
        cavityMap, _ = load_fits(cavity_f) # cavity maps
        if cavityMap.ndim == 3:
            cavityWave = cavityMap[:,rows,cols].mean(axis=0)
        else:
            cavityWave = cavityMap[rows,cols]
    else:
        cavityMap = cavity_f
        if cavityMap.ndim == 3:
            cavityWave = cavityMap.mean(axis=0)
        else:
            cavityWave = cavityMap
        
    if returnWL:
        new_wave_axis = wave_axis[np.newaxis,np.newaxis] - cavityWave[...,np.newaxis]
        return new_wave_axis
    else:
        return cavityWave

def load_l2_stk(directory,did,version=None):
    import glob
    file_n = os.listdir(directory)
    key = 'stokes'
    if version is None:
        version = '*'
    datfile = glob.glob(os.path.join(directory, f'solo_L2_phi-hrt-{key}_*_{version}_{did}.fits.gz'))
    if not(datfile):
        print('No data found')
        return np.empty([]), np.empty([])
    else:
        if len(datfile) == 1:
            return fits.getdata(datfile[0],header=True)
        else:
            print('More than one file found:')
            print(datfile)
            print('Please specify the Version')
            return np.empty([]), np.empty([])

def load_l2_rte(directory,did,version=None):
    file_n = os.listdir(directory)
    if type(did) != str:
        did = str(did)
    if version is None:
        did_n = [directory+i for i in file_n if did in i]
    else:
        did_n = [directory+i for i in file_n if (did in i and version in i)]
    rte_n = ['icnt','bmag','binc','bazi','vlos','blos','chi2']
    rte_out = []
    for n in rte_n:
        try:
            rte_out += [fits.getdata([i for i in did_n if n in i][0])]
        except:
            print(n+' not found')
    
    rte_out = np.asarray(rte_out)
    
    return rte_out


def phi_disambig(bazi,bamb,method=2):
    """
    input
    bazi: magnetic field azimut. Type: str or array
    bamb: disambiguation fits. Type: str or array
    method: method selected for the disambiguation (0, 1 or 2). Type: int (2 as Default)
    
    output
    disbazi: disambiguated azimut. Type: array
    """
    # from astropy.io import fits
    if type(bazi) is str:
        bazi = fits.getdata(bazi)
    if type(bamb) is str:
        bamb = fits.getdata(bamb)
    
    disambig = bamb[0]/2**method
    disbazi = bazi.copy()
    disbazi[disambig%2 != 0] += 180
    
    return disbazi

def largest_rectangle_area_in_histogram(heights):
    """
    Helper function to compute largest rectangle area in a histogram row.
    Returns: (area, start_col, end_col, height)
    """
    stack = []
    max_area = 0
    start_col = end_col = max_height = 0
    heights = list(heights) + [0]  # Add sentinel

    for i, h in enumerate(heights):
        last_index = i
        while stack and stack[-1][1] > h:
            index, height = stack.pop()
            area = height * (i - index)
            if area > max_area:
                max_area = area
                start_col = index
                end_col = i - 1
                max_height = height
            last_index = index
        stack.append((last_index, h))

    return max_area, start_col, end_col, max_height

def largest_rectangle_in_mask(mask):
    """
    Find the largest rectangle containing only 1s in a binary mask.
    
    Args:
        mask: 2D numpy array of 0s and 1s.
    
    Returns:
        area: int, area of the largest rectangle
        top_left: tuple (row, col) of the top-left corner
        bottom_right: tuple (row, col) of the bottom-right corner
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array of 0s and 1s.")

    rows, cols = mask.shape
    heights = np.zeros(cols, dtype=int)
    max_area = 0
    top_left = bottom_right = None

    for row in range(rows):
        for col in range(cols):
            heights[col] = heights[col] + 1 if mask[row, col] == 1 else 0

        area, start_col, end_col, height = largest_rectangle_area_in_histogram(heights)

        if area > max_area:
            max_area = area
            top_left = (row - height + 1, start_col)
            bottom_right = (row, end_col)

    return max_area, top_left, bottom_right


def shift_array_multi(arr, shift, fill_value=0):
    """
    Shift a NumPy array along multiple axes without wrapping around edges.

    Parameters:
    - arr: np.ndarray
    - shift: int or tuple of ints (one per axis to shift)
    - fill_value: value to fill the emptied positions (default: 0)

    Returns:
    - shifted array
    """
    if isinstance(shift, int):
        shift = (shift,)
    if len(shift) > arr.ndim:
        raise ValueError("Shift tuple longer than array dimensions")

    # Pad the shift tuple with zeros if needed
    shift = tuple(shift[i] if i < len(shift) else 0 for i in range(arr.ndim))

    result = np.full_like(arr, fill_value)

    src_slices = []
    dst_slices = []

    for ax, sh in enumerate(shift):
        axis_len = arr.shape[ax]
        if sh > 0:
            src_slice = slice(0, axis_len - sh)
            dst_slice = slice(sh, axis_len)
        elif sh < 0:
            src_slice = slice(-sh, axis_len)
            dst_slice = slice(0, axis_len + sh)
        else:
            src_slice = slice(0, axis_len)
            dst_slice = slice(0, axis_len)

        src_slices.append(src_slice)
        dst_slices.append(dst_slice)

    # Convert slices to tuple of slices for advanced indexing
    result[tuple(dst_slices)] = arr[tuple(src_slices)]
    return result