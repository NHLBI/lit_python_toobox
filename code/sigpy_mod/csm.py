import sigpy as sp
import numpy as np
import sigpy.mri as mr
from tqdm.auto import tqdm
import cupyx.scipy.ndimage

def quick_gridding(ksp, coord, dcf, matrix_r):
    """Quick gridding reconstruction for Nc, Nx, Ny, Nz image data.

    Args:
        ksp (array): Complex kspace measurements (Nc, Nexc, Nread).
        coord (array): Kspace coordinates (Nexc, Nread, Ndim).
        dcf (array): Density compensation (Nexc, Nread).
        matrix_r (tuple): Image shape to reconstruct to (Nx, Ny, Nz).

    Returns:
        x_gridding: Gridded image (Nc, Nx, Ny, Nz). 
    """
    
    Nc, Nexc, Nread = ksp.shape
    xp= sp.get_array_module(ksp)
    F = sp.linop.NUFFT(ishape=matrix_r,
                    coord=coord)
    D = sp.linop.Multiply(F.oshape, dcf)
    A = D * F
    
    x_gridded = xp.zeros((Nc,) + matrix_r, dtype=ksp.dtype)
    for channel in range(Nc): # Painfully slow, but must do this batched method to minimize GPU load :( 
        x_gridded[channel,...] = A.H(ksp[channel,...])
        
    # If you are blessed with lots of spare room on your GPU, skip coil batching using this code:
    # F = sp.linop.NUFFT(ishape=(Nc,) + matrix_r,
    #                 coord=coord)
    # D = sp.linop.Multiply(F.oshape, dcf)
    # A = D * F
    # x_gridded= A.H(ksp)
    
    return x_gridded


def jsense_csm(ksp, coord, dcf, matrix_r, device=sp.Device(-1),
                mps_ker_width=12,
                ksp_calib_width=32,
                lamda=0,):
    
    # Compute a gridded image from the non-Cartesian data, then FFT back into k-space on a Cartesian grid
    ksp = sp.to_device(ksp, device)
    img_s = quick_gridding(ksp, coord, dcf, matrix_r)    
    ksp = sp.fft(input=img_s, axes=(1, 2, 3))
    mps = mr.app.JsenseRecon(ksp,
                             mps_ker_width=mps_ker_width,
                             ksp_calib_width=ksp_calib_width,
                             lamda=lamda,
                             device=device,
                             comm=sp.Communicator(),
                             max_iter=30,
                             max_inner_iter=10).run()
    return mps

def jsense_csm_non_cartesian(ksp, coord, dcf, matrix_r, device=sp.Device(-1),
                mps_ker_width=5,
                ksp_calib_width=20,
                lamda=1e-2,):
    # Read in non-Cartesian ksp data and a non-Cartesian preconditioner (i.e. your DCF weights or a k-space preconditioner)
    mps = mr.app.JsenseRecon(ksp,
                             coord,
                             weights=dcf,
                             mps_ker_width=mps_ker_width,
                             ksp_calib_width=ksp_calib_width,
                             lamda=lamda,
                             img_shape=matrix_r,
                             device=device,
                             comm=sp.Communicator(),
                             max_iter=30,
                             max_inner_iter=20,
                             show_pbar=True).run()
    return mps


def espirit_csm(ksp, coord, dcf, matrix_r, device = sp.Device(-1), 
                crop=0, 
                thresh=0.02,
                kernel_width=6, 
                calib_width=24,
                max_iter=100,
                downscale_factor=1):
    
    # Reconstruct a lower res image, like BART
    downscaled_matrix_r = (int(matrix_r[0]/downscale_factor), int(matrix_r[1]/downscale_factor), int(matrix_r[2]/downscale_factor))
    
    # Compute a nufft image
    Nc, Nexc, Nread = ksp.shape
    ksp = sp.to_device(ksp, device)
    img_s = quick_gridding(ksp, coord, dcf, downscaled_matrix_r)
    # sp.plot.ImagePlot(img_s, x=1, y=2, z=0,
    #                         title=f"gridding image", colormap='gray', mode='p')
    ksp = sp.fft(input=img_s, axes=(1,2,3))
        
    # Shrink ksp matrix size to reduce memory usage of ESPIRIT, otherwise resize to full resolution
    NcNxNyNz = Nc * matrix_r[0] * matrix_r[1] * matrix_r[2]
    if NcNxNyNz > 10e6: # Need to find the true threshold, depends on GPU size.
        shrink_ksp = True
        print('The total number of elements in the kspace data is too large, so we will downsample the kspace data and reconstruct a smaller ESPIRIT map, then resize at the end.')
    else:
        shrink_ksp = False # If False, then ksp will be zero padded and mps will be calculated at full resolution.
        
    if shrink_ksp is False:
        # Pad the k-space data back to full resolution
        ksp = sp.util.resize(ksp, (Nc, ) + matrix_r)
        
    # Calculate CSM
    mps = mr.app.EspiritCalib(ksp,
                             thresh=thresh,
                             kernel_width=kernel_width,
                             calib_width=calib_width,
                             device=device,
                             crop=crop,
                             output_eigenvalue=False,
                             max_iter=max_iter).run()
        
    # Interpolate back to full resolution if ksp was not already resized
    if shrink_ksp: # Only do this if ksp shape does not match matrix_r
        mps = cupyx.scipy.ndimage.zoom(mps, (1,downscale_factor,downscale_factor,downscale_factor), order=5, mode='nearest')
        
    # sp.plot.ImagePlot(mps, x=1, y=2, z=0, colormap='hsv', mode='p')
    return mps



# Usage:
#  mps = csm.espirit_csm(ksp, coord, dens, matrix_r, device, 
                        # crop=0,
                        # thresh=0.02,
                        #     kernel_width=6,
                        #     calib_width=24,
                        #     max_iter=100).get()