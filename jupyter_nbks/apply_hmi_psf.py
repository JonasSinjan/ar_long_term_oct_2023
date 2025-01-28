import astropy.units as u
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
from scipy.optimize import leastsq
from scipy import signal as sig
import numpy as np

def Zernike_polar(coefficients, r, u):
   #Z= np.insert(np.array([0,0,0]),3,coefficients)  
   Z =  coefficients
   #Z1  =  Z[0]  * 1*(np.cos(u)**2+np.sin(u)**2)
   #Z2  =  Z[1]  * 2*r*np.cos(u)
   #Z3  =  Z[2]  * 2*r*np.sin(u)

   Z4  =  Z[0]  * np.sqrt(3)*(2*r**2-1)  #defocus

   Z5  =  Z[1]  * np.sqrt(6)*r**2*np.sin(2*u) #astigma
   Z6  =  Z[2]  * np.sqrt(6)*r**2*np.cos(2*u)

   Z7  =  Z[3]  * np.sqrt(8)*(3*r**2-2)*r*np.sin(u) #coma
   Z8  =  Z[4]  * np.sqrt(8)*(3*r**2-2)*r*np.cos(u)

   Z9  =  Z[5]  * np.sqrt(8)*r**3*np.sin(3*u) #trefoil
   Z10=  Z[6] * np.sqrt(8)*r**3*np.cos(3*u)

   Z11 =  Z[7] * np.sqrt(5)*(1-6*r**2+6*r**4) #secondary spherical

   Z12 =  Z[8] * np.sqrt(10)*(4*r**2-3)*r**2*np.cos(2*u)  #2 astigma
   Z13 =  Z[9] * np.sqrt(10)*(4*r**2-3)*r**2*np.sin(2*u)

   Z14 =  Z[10] * np.sqrt(10)*r**4*np.cos(4*u) #tetrafoil
   Z15 =  Z[11] * np.sqrt(10)*r**4*np.sin(4*u)

   Z16 =  Z[12] * np.sqrt(12)*(10*r**4-12*r**2+3)*r*np.cos(u) #secondary coma
   Z17 =  Z[13] * np.sqrt(12)*(10*r**4-12*r**2+3)*r*np.sin(u)

   Z18 =  Z[14] * np.sqrt(12)*(5*r**2-4)*r**3*np.cos(3*u) #secondary trefoil
   Z19 =  Z[15] * np.sqrt(12)*(5*r**2-4)*r**3*np.sin(3*u)

   Z20 =  Z[16] * np.sqrt(12)*r**5*np.cos(5*u) #pentafoil
   Z21 =  Z[17] * np.sqrt(12)*r**5*np.sin(5*u)

   Z22 =  Z[18] * np.sqrt(7)*(20*r**6-30*r**4+12*r**2-1) #spherical

   Z23 =  Z[19] * np.sqrt(14)*(15*r**4-20*r**2+6)*r**2*np.sin(2*u) #astigma
   Z24 =  Z[20] * np.sqrt(14)*(15*r**4-20*r**2+6)*r**2*np.cos(2*u)

   Z25 =  Z[21] * np.sqrt(14)*(6*r**2-5)*r**4*np.sin(4*u)#trefoil
   Z26 =  Z[22] * np.sqrt(14)*(6*r**2-5)*r**4*np.cos(4*u)

   Z27 =  Z[23] * np.sqrt(14)*r**6*np.sin(6*u) #hexafoil 
   Z28 =  Z[24] * np.sqrt(14)*r**6*np.cos(6*u)

   Z29 =  Z[25] * 4*(35*r**6-60*r**4+30*r**2-4)*r*np.sin(u) #coma
   Z30 =  Z[26] * 4*(35*r**6-60*r**4+30*r**2-4)*r*np.cos(u)

   Z31 =  Z[27] * 4*(21*r**4-30*r**2+10)*r**3*np.sin(3*u)#trefoil
   Z32 =  Z[28] * 4*(21*r**4-30*r**2+10)*r**3*np.cos(3*u)

   Z33 =  Z[29] * 4*(7*r**2-6)*r**5*np.sin(5*u) #pentafoil
   Z34 =  Z[30] * 4*(7*r**2-6)*r**5*np.cos(5*u)

   Z35 =  Z[31] * 4*r**7*np.sin(7*u) #heptafoil
   Z36 =  Z[32] * 4*r**7*np.cos(7*u)

   Z37 =  Z[33] * 3*(70*r**8-140*r**6+90*r**4-20*r**2+1) #spherical

   ZW = Z4+Z5+Z6+Z7+Z8+Z9+Z10+Z11+Z12+Z13+Z14+Z15+Z16+ Z17+Z18+Z19+Z20+Z21+Z22+Z23+ Z24+Z25+Z26+Z27+Z28+ Z29+ Z30+ Z30+ Z31+ Z32+ Z33+ Z34+ Z35+ Z36+ Z37
   return ZW

def pupil_size(D,lam,pix,size):
    pixrad = pix*np.pi/(180*3600)  # Pixel-size in radians
    nu_cutoff = D/lam      # Cutoff frequency in rad^-1
    deltanu = 1./(size*pixrad)     # Sampling interval in rad^-1
    rpupil = nu_cutoff/(2*deltanu) #pupil size in pixels
    return int(rpupil)

## function for making the phase in a unit circle
def phase(coefficients,rpupil):
   r = 1
   x = np.linspace(-r, r, 2*rpupil)
   y = np.linspace(-r, r, 2*rpupil)

   [X,Y] = np.meshgrid(x,y) 
   R = np.sqrt(X**2+Y**2)
   theta = np.arctan2(Y, X)
      
   Z = Zernike_polar(coefficients,R,theta)
   Z[R>1] = 0
   return Z

def pupil_foc(coefficients,size,rpupil):
    #rpupil = pupil_size(D,lam,pix,size)
    A = np.zeros([size,size])
    A[size//2-rpupil+1:size//2+rpupil+1,size//2-rpupil+1:size//2+rpupil+1]= phase(coefficients,rpupil)
    aberr =  np.exp(1j*A)
    return aberr

def mask(rpupil, size):
   r = 1
   x = np.linspace(-r, r, 2*rpupil)
   y = np.linspace(-r, r, 2*rpupil) 

   [X,Y] = np.meshgrid(x,y) 
   R = np.sqrt(X**2+Y**2)
   theta = np.arctan2(Y, X)
   M = 1*(np.cos(theta)**2+np.sin(theta)**2)
   M[R>1] = 0
   Mask =  np.zeros([size,size])
   Mask[size//2-rpupil+1:size//2+rpupil+1,size//2-rpupil+1:size//2+rpupil+1]= M
   return Mask

def PSF(mask,abbe):
   ## making zero where the aberration is equal to 1 (the zero background)
   abbe_z = np.zeros((len(abbe),len(abbe)),dtype=complex)
   abbe_z = mask*abbe
   PSF = ifftshift(fft2(fftshift(abbe_z))) #from brandon
   PSF = (np.abs(PSF))**2 #or PSF*PSF.conjugate()
   #PSF = PSF/PSF.sum()
   return PSF
    
def make_psf_hmi_th(size, phi_dsun):
    """
    Giving the size of the array, this function computes the theoretical telescope PSF
    """
    coe = np.zeros(38)
    D = 140
    lam = 617.3341e-6
    pix = 0.5 / (1*u.au).to(u.m).value * phi_dsun
    print(pix)

    rpupil = pupil_size(D,lam,pix,size)
    print(rpupil)
    Mask = mask(rpupil,size)
    A_f = pupil_foc(coe,size,rpupil)
    psf_foc = PSF(Mask,A_f)
    # t0 = OTF(psf_foc)
    return psf_foc 