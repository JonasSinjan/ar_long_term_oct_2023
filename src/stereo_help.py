import pyfftw.interfaces.numpy_fft as fft
import numpy as np
from scipy.signal import convolve
from scipy.optimize import curve_fit

def image_register(ref: np.ndarray,im: np.ndarray,subpixel: bool = True,deriv: bool = True) -> tuple[np.ndarray, list]:
    """get shift between two images using Fourier cross-correlation at pixel level and optional at subpixil

    Parameters  
    ----------
    ref : numpy.ndarray
        reference image
    im : numpy.ndarray
        image to be registered
    subpixel : bool, optional
        whether to use subpixel registration (default: True)
    deriv : bool, optional
        whether to use image derivative for registration (default: True)
    
    Returns
    -------
    r : numpy.ndarray
        cross-correlation image
    shifts : list
        shift in x and y directions
    """
    
    def _image_derivative(d):
        kx = np.asarray([[1,0,-1], [1,0,-1], [1,0,-1]])
        ky = np.asarray([[1,1,1], [0,0,0], [-1,-1,-1]])
        kx = kx/3.
        ky = ky/3.
        SX = convolve(d, kx,mode='same')
        SY = convolve(d, ky,mode='same')
        A = SX**2+SY**2
        return A
    
    def _g2d(X, offset, amplitude, sigma_x, sigma_y, xo, yo, theta):
        (x, y) = X
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                                + c*((y-yo)**2)))
        return g.ravel()
    
    def _gauss2dfit(a):
        sz = np.shape(a)
        X,Y = np.meshgrid(np.arange(sz[1])-sz[1]//2,np.arange(sz[0])-sz[0]//2)
        try:
            X = X[~X.mask]
            Y = Y[~Y.mask]
            a = a[~a.mask]
        except:
            pass
        c = np.unravel_index(a.argmax(),sz)
        y = a[c[0],:]
        x = X[c[0],:]
        stdx = 5 #np.sqrt(abs(sum(y * (x - sum(x*y)/sum(y))**2) / sum(y)))
        y = a[:,c[1]]
        x = Y[:,c[1]]
        stdy = 5 #np.sqrt(abs(sum(y * (x - sum(x*y)/sum(y))**2) / sum(y)))
        initial_guess = [np.median(a), np.max(a), stdx, stdy, c[1] - sz[1]//2, c[0] - sz[0]//2, 0]
        popt, pcov = curve_fit(_g2d, (X, Y), a.ravel(), p0=initial_guess)
        return np.reshape(_g2d((X,Y), *popt), sz), popt
    
    def one_power(array):
        return array/np.sqrt((np.abs(array)**2).mean())

    if deriv:
        ref = _image_derivative(ref)
        im = _image_derivative(im)

    shifts = np.zeros(2)
    FT1 = fft.fftn(ref - np.mean(ref))
    FT2 = fft.fftn(im - np.mean(im))
    ss = np.shape(ref)
    r = np.real(fft.ifftn(one_power(FT1) * one_power(FT2.conj())))
    r = fft.fftshift(r)
    #ppp = np.unravel_index(np.argmax(r),ss)
    r_sub=r[ss[0]//2-500:ss[0]//2+500,ss[1]//2-500:ss[1]//2+500] #restrict region to find argmax in a 1000,1000 box - might not work for 1k x 1k cropped images/or have any impact
    ppp = np.unravel_index(np.argmax(r_sub),r_sub.shape)
    ss=np.shape(r_sub)
    shifts = [(ss[0]//2-(ppp[0])),(ss[1]//2-(ppp[1]))]
    if subpixel:
        g, A = _gauss2dfit(r)
        ss = np.shape(g)
        shifts[0] = A[5]
        shifts[1] = A[4]
        del g
    del FT1, FT2
    return r, shifts