import opticalflow3D
import cupy as cp
from cupyx.scipy.ndimage import map_coordinates, zoom, gaussian_filter
from skimage.transform import warp
import numpy as np
import matplotlib.pyplot as plt
import gc
import torch
import sigpy.plot as pl
import time
from cupyx.scipy.signal.windows import tukey



def register_images(input_images, ref_index, filter_size=25, gpu_id=0):
    # Set the specific GPU device
    with cp.cuda.Device(gpu_id):
        # Convert images to CuPy arrays and initialize output arrays
        images = cp.asarray(input_images)
        output = cp.zeros(images.shape, dtype=cp.complex64)
        deformation_fields = np.zeros([images.shape[0], 3, images.shape[1], images.shape[2], images.shape[3]])
        nimages, nr, nc, nz = images.shape

        # Reference image
        ref_image = images[ref_index, ...].squeeze()
        
        # Smooth the input image:
        image_smooth = False # Consider removing as presmoothing already does the exact same thing
        if image_smooth:
            sigma_smoother = [0, 0.5, 0.5, 0.5] # Only smooth over the spatial dimensions
            images = cp.array(gaussian_filter(images, sigma=sigma_smoother, mode='reflect', truncate=4))
            ref_image = images[ref_index, ...].squeeze()

        cp.cuda.runtime.deviceSynchronize()
        for ind in range(0, nimages):
            mov_image = images[ind, ...].squeeze()
            farneback = opticalflow3D.Farneback3D(iters=3,
                                    num_levels=3,
                                    scale=0.5,
                                    filter_size=filter_size,
                                    presmoothing=None, # Default, none
                                    filter_type="gaussian",
                                    sigma_k=0.05)

            output_vx, output_vy, output_vz, _ = farneback.calculate_flow(
                0.05 * cp.abs(ref_image / cp.max(ref_image.ravel())),
                0.05 * cp.abs(mov_image / cp.max(mov_image.ravel())),
                start_point=(ref_image.shape[0]//2, ref_image.shape[1]//2, ref_image.shape[2]//2),
                total_vol=(ref_image.shape[0], ref_image.shape[1], ref_image.shape[2]),
                sub_volume=(ref_image.shape[0], ref_image.shape[1], ref_image.shape[2]),
                overlap=(ref_image.shape[0], ref_image.shape[1], ref_image.shape[2]),
                threadsperblock=(8, 8, 8),
            )
            output_vx = output_vx.get()
            output_vy = output_vy.get()
            output_vz = output_vz.get()

            row_coords, col_coords, slice_coords = np.meshgrid(np.arange(nr), np.arange(nc), np.arange(nz),
                                                               indexing='ij')

            rcomp = cp.real(mov_image)
            icomp = cp.imag(mov_image)

            x = cp.array([row_coords + output_vx, col_coords + output_vy, slice_coords + output_vz])
            y = cp.array([row_coords + output_vx, col_coords + output_vy, slice_coords + output_vz])

            cp.cuda.runtime.deviceSynchronize()

            # Apply deformation to real and imaginary components
            output[ind, :, :, :] = map_coordinates(rcomp, x, mode="nearest") + 1j * map_coordinates(icomp, y, mode="nearest")
            cp.cuda.runtime.deviceSynchronize()

            deformation_fields[ind, ...] = np.concatenate([np.expand_dims(output_vx, 0), np.expand_dims(output_vy, 0), np.expand_dims(output_vz, 0)], axis=0)
            deformation_fields[ind, ...] = np.nan_to_num(deformation_fields[ind, ...])
                        
        # display_registered_images(input_images[ref_index, ...].squeeze(),
        #                           np.mean(input_images, axis=0).squeeze(),
        #                           np.mean(output, axis=0).squeeze(),
        #                           input_images.shape[3] / 2)
        
        # import sigpy.plot as pl
        # # pl.ImagePlot(images - output, x=1, y=2, z=0, hide_axes=True, colormap="magma") # Difference
        # pl.ImagePlot(deformation_fields[:,0,...], x=1, y=2, z=0, hide_axes=True, colormap="magma") # x component of deformation fields 
        # pl.ImagePlot(output, x=1, y=2, z=0, hide_axes=True) # Output

        # Return the output as a numpy array
        # output = cp.asnumpy(output)

        del rcomp
        del icomp
        del output_vx
        del output_vy
        del output_vz
        del farneback
        del x
        del y
        del images
        del ref_image
       
        gc.collect()
        return output, deformation_fields
    
 

def register_images_FAST(images, ref_index, filter_size=25, gpu_id=0, compute_jacobian=True):
    """A faster implementation of the register_images function.

    Args:
        images (cp.array): Cupy array containing images of shape: N, Nx, Ny, Nz.
        ref_index (int): Index of the fixed image to be registered to.
        filter_size (int, optional): Filter size for Farneback3D. Recommended: 9-50. 9 = very fine registrations (sensitive to noise); 50 = very broad registrations (not precise). Defaults to 25.
        gpu_id (int, optional): GPU device ID number. Defaults to 0.
        compute_jacobian (bool, optional: Multiply the registered image by the Jacobian Determinant.

    Returns:
        output: Registered images (cp.array).
        deformation_fields: Deformation fields (cp.array).
    """
    with cp.cuda.Device(gpu_id):
        
        tic_mem = time.time()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp._default_memory_pool.free_all_blocks()
        cp.cuda.Stream.null.synchronize()
        cp.cuda.stream.get_current_stream().synchronize()
        # gc.collect()
        toc_mem = time.time()
        print(f"Time taken to clear memory before registration... {toc_mem - tic_mem:.3f} seconds")
        
        # Convert images to CuPy arrays
        nimages, nr, nc, nz = images.shape

        # Preallocate output arrays
        output = cp.array(images, dtype=cp.complex64)
        deformation_fields = cp.zeros((nimages, 3, nr, nc, nz), dtype=cp.float32)
        output_vx = cp.zeros((nr, nc, nz), dtype=cp.float32)
        output_vy = cp.zeros((nr, nc, nz), dtype=cp.float32)
        output_vz = cp.zeros((nr, nc, nz), dtype=cp.float32)
        
        # Select a subvolume to perform registration over (to save time and memory)
        subfactor = 8 # 8 # Make a large value to not do any subvolumes, e.g. 8000
        subvolume = (slice(nr//subfactor, -nr//subfactor), slice(nc//subfactor, -nc//subfactor), slice(nz//subfactor, -nz//subfactor))
        subvolume_shape = (nr-(2*nr//subfactor),
                           nc-(2*nc//subfactor),
                           nz-(2*nz//subfactor))
        
        # Reference image and precompute max values
        ref_image = images[ref_index,...].squeeze()
        ref_image_max = cp.max(cp.abs(ref_image[subvolume]))
        images_max = cp.max(cp.abs(images[(slice(None),) + subvolume]), axis=(1, 2, 3))

        # Precompute meshgrid
        row_coords, col_coords, slice_coords = cp.meshgrid(cp.arange(nr), cp.arange(nc), cp.arange(nz), indexing='ij')        
        
        # Perform registration
        indices = [i for i in range(nimages) if i != ref_index] # Make a range that excludes the ref_index
        # indices = range(nimages) # Make a range that excludes the ref_index
        for ind in indices:
            mov_image = images[ind, ...].squeeze()
            # toc = time.time()
            
             # Initialize Farneback class
            farneback = opticalflow3D.Farneback3D(
            iters=4, num_levels=4, scale=0.5, filter_size=filter_size, spatial_size=9,
            presmoothing=None, filter_type="gaussian", sigma_k=0.05, device_id=gpu_id)
                 
            # Compute optical flow: NOTE: this requires Joey's modified optical flow that provides outputs on the GPU device specified above
            output_vx[subvolume], output_vy[subvolume], output_vz[subvolume], _ = farneback.calculate_flow(
                0.05 * cp.abs(ref_image[subvolume] / ref_image_max),
                0.05 * cp.abs(mov_image[subvolume] / images_max[ind]),
                start_point=(nr//2, nc//2, nz//2),
                total_vol=(nr, nc, nz),
                sub_volume=(nr, nc, nz),
                overlap=(nr, nc, nz),
                threadsperblock=(8, 8, 8),
            )        
            
            # Fix NaN's 
            output_vx = cp.nan_to_num(output_vx)
            output_vy = cp.nan_to_num(output_vy)
            output_vz = cp.nan_to_num(output_vz)
            
            # Apply Tukey filter to deformation fields (elementwise multiplication)
            tukey_3d = create_3d_tukey_window(*subvolume_shape, alpha=0.5)            
            output_vx[subvolume] *= tukey_3d
            output_vy[subvolume] *= tukey_3d
            output_vz[subvolume] *= tukey_3d
            
            # Clear memory
            tic_mem = time.time()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            cp._default_memory_pool.free_all_blocks()
            cp.cuda.Stream.null.synchronize()
            cp.cuda.stream.get_current_stream().synchronize()
            # gc.collect()
            toc_mem = time.time()
            print(f"Time taken to clear memory after flow measurement... {toc_mem - tic_mem:.3f} seconds")
                        
            # tic = time.time()   
            # print(f'Optical Flow registration completed for: Ref index: {ref_index}, Moving index: {ind}. Time taken: {tic - toc:.2f} seconds.')

            # Compute deformed coordinates
            x = cp.array((row_coords + output_vx, 
                          col_coords + output_vy, 
                          slice_coords + output_vz))

            # Separate real and imaginary components
            rcomp, icomp = cp.real(mov_image), cp.imag(mov_image)

            # Apply deformation to real and imaginary components
            output[ind, :, :, :] = map_coordinates(rcomp, x, mode="nearest") + 1j * map_coordinates(icomp, x, mode="nearest")
            del rcomp, icomp, x, mov_image

            # Store deformation fields directly
            deformation_fields[(ind, 0, ...)] = output_vx
            deformation_fields[(ind, 1, ...)] = output_vy
            deformation_fields[(ind, 2, ...)] = output_vz
            # deformation_fields[ind, ...] = cp.nan_to_num(deformation_fields[ind, ...])     
            
            tic_mem = time.time()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            cp._default_memory_pool.free_all_blocks()
            cp.cuda.Stream.null.synchronize()
            cp.cuda.stream.get_current_stream().synchronize()
            # gc.collect()
            toc_mem = time.time()
            print(f"Time taken to clear memory after registration... {toc_mem - tic_mem:.3f} seconds")       
            
        # # Optional: Multiply the output images by the Jacobian determinant
        # if compute_jacobian: # TODO: decide if having this boolean affects memory storage
        deformation_fields = cp.moveaxis(deformation_fields, 1, -1)
        jacobian = compute_jacobian_determinant(deformation_fields, ref_index=ref_index)
        limit = 0.85 # E.g. cap the jacobian at 0.9 and 1/0.9 
        jacobian[jacobian < limit] = limit
        jacobian[jacobian > 1/limit] = 1/limit
        # pl.ImagePlot(deformation_fields, x=1, y=2, z=0, hide_axes=True, colormap="magma", vmin=-1, vmax=1) # x component
        # pl.ImagePlot(jacobian, x=1, y=2, z=0, hide_axes=True, colormap="viridis", vmin=0.9, vmax=1/0.9) # x component
        output *= jacobian
        
        del output_vx
        del output_vy
        del output_vz
        del farneback
        del images
        del ref_image
        del jacobian

        return output, deformation_fields
    
    
    
    
    
    

def register_images_FOURIER(images, ref_index, filter_size=25, gpu_id=0, compute_jacobian=True):
    """A faster implementation of the register_images function.

    Args:
        images (cp.array): Cupy array containing images of shape: N, Nx, Ny, Nz.
        ref_index (int): Index of the fixed image to be registered to.
        filter_size (int, optional): Filter size for Farneback3D. Recommended: 9-50. 9 = very fine registrations (sensitive to noise); 50 = very broad registrations (not precise). Defaults to 25.
        gpu_id (int, optional): GPU device ID number. Defaults to 0.
        compute_jacobian (bool, optional: Multiply the registered image by the Jacobian Determinant.

    Returns:
        output: Registered images (cp.array).
        deformation_fields: Deformation fields (cp.array).
    """
    with cp.cuda.Device(gpu_id):
        
        tic_mem = time.time()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp._default_memory_pool.free_all_blocks()
        cp.cuda.Stream.null.synchronize()
        cp.cuda.stream.get_current_stream().synchronize()
        # gc.collect()
        toc_mem = time.time()
        print(f"Time taken to clear memory before registration... {toc_mem - tic_mem:.3f} seconds")
        
        # Convert images to CuPy arrays
        nimages, nr, nc, nz = images.shape

        # Preallocate output arrays
        output = cp.array(images, dtype=cp.complex64)
        deformation_fields = cp.zeros((nimages, 3, nr, nc, nz), dtype=cp.float32)
        output_vx = cp.zeros((nr, nc, nz), dtype=cp.float32)
        output_vy = cp.zeros((nr, nc, nz), dtype=cp.float32)
        output_vz = cp.zeros((nr, nc, nz), dtype=cp.float32)
        
        # Select a subvolume to perform registration over (to save time and memory)
        subfactor = 32 # 8 # Make a large value to not do any subvolumes, e.g. 8000
        subvolume = (slice(nr//subfactor, -nr//subfactor), slice(nc//subfactor, -nc//subfactor), slice(nz//subfactor, -nz//subfactor))
        subvolume_shape = (nr-(2*nr//subfactor),
                           nc-(2*nc//subfactor),
                           nz-(2*nz//subfactor))
        
        # Create a 3D Tukey window for smoothing
        tukey_3d = create_3d_tukey_window(*subvolume_shape, alpha=0.5)    
        
        # Reference image and precompute max values
        ref_image = images[ref_index,...].squeeze()
        ref_image_max = cp.max(cp.abs(ref_image[subvolume]))
        images_max = cp.max(cp.abs(images[(slice(None),) + subvolume]), axis=(1, 2, 3))

        # Precompute meshgrid
        row_coords, col_coords, slice_coords = cp.meshgrid(cp.arange(nr), cp.arange(nc), cp.arange(nz), indexing='ij')        
        
        # Perform registration
        indices = [i for i in range(nimages) if i != ref_index] # Make a range that excludes the ref_index
        # indices = range(nimages) # Make a range that excludes the ref_index
        for ind in indices:
            mov_image = images[ind, ...].squeeze()
            # toc = time.time()
            
             # Initialize Farneback class
            farneback = opticalflow3D.Farneback3D(
            iters=4, num_levels=4, scale=0.5, filter_size=filter_size, spatial_size=11,
            presmoothing=None, filter_type="gaussian", sigma_k=0.07, device_id=gpu_id)
                             
            # Compute optical flow: NOTE: this requires Joey's modified optical flow that provides outputs on the GPU device specified above
            output_vx[subvolume], output_vy[subvolume], output_vz[subvolume], _ = farneback.calculate_flow(
                0.05 * cp.abs(ref_image[subvolume] / ref_image_max),
                0.05 * cp.abs(mov_image[subvolume] / images_max[ind]),
                start_point=(nr//2, nc//2, nz//2),
                total_vol=(nr, nc, nz),
                sub_volume=(nr, nc, nz),
                overlap=(nr, nc, nz),
                threadsperblock=(8, 8, 8),
            )        
            
            # Fix NaN's 
            output_vx = cp.nan_to_num(output_vx)
            output_vy = cp.nan_to_num(output_vy)
            output_vz = cp.nan_to_num(output_vz)
            
            # Apply Tukey filter to deformation fields (elementwise multiplication)
            output_vx[subvolume] *= tukey_3d
            output_vy[subvolume] *= tukey_3d
            output_vz[subvolume] *= tukey_3d
            
            # Clear memory
            tic_mem = time.time()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            cp._default_memory_pool.free_all_blocks()
            cp.cuda.Stream.null.synchronize()
            cp.cuda.stream.get_current_stream().synchronize()
            # gc.collect()
            toc_mem = time.time()
            print(f"Time taken to clear memory after flow measurement... {toc_mem - tic_mem:.3f} seconds")
                        
            # tic = time.time()   
            # print(f'Optical Flow registration completed for: Ref index: {ref_index}, Moving index: {ind}. Time taken: {tic - toc:.2f} seconds.')
            
            # Store deformation fields directly
            deformation_fields[(ind, 0, ...)] = output_vx
            deformation_fields[(ind, 1, ...)] = output_vy
            deformation_fields[(ind, 2, ...)] = output_vz
            # deformation_fields[ind, ...] = cp.nan_to_num(deformation_fields[ind, ...])  
            
        # Smooth the deformation fields using Fourier smoothing
        deformation_fields_smooth = fourier_smooth_deformation_cupy(deformation_fields, n_harmonics=2)
        deformation_fields_smooth[ref_index,...] = 0 # Set the reference deformation field to zero
        
        # Register the images using the smoothed deformation fields
        for ind in indices:
            # Compute deformed coordinates
            x = cp.array((row_coords + deformation_fields_smooth[(ind, 0, ...)], 
                          col_coords + deformation_fields_smooth[(ind, 1, ...)], 
                          slice_coords + deformation_fields_smooth[(ind, 2, ...)]))

            # Separate real and imaginary components
            mov_image = images[ind, ...].squeeze()
            rcomp, icomp = cp.real(mov_image), cp.imag(mov_image)

            # Apply deformation to real and imaginary components
            output[ind, :, :, :] = map_coordinates(rcomp, x, mode="nearest") + 1j * map_coordinates(icomp, x, mode="nearest")
            del rcomp, icomp, x, mov_image

            tic_mem = time.time()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            cp._default_memory_pool.free_all_blocks()
            cp.cuda.Stream.null.synchronize()
            cp.cuda.stream.get_current_stream().synchronize()
            # gc.collect()
            toc_mem = time.time()
            print(f"Time taken to clear memory after registration... {toc_mem - tic_mem:.3f} seconds")       
            
        # # Optional: Multiply the output images by the Jacobian determinant
        # if compute_jacobian: # TODO: decide if having this boolean affects memory storage
        deformation_fields = cp.moveaxis(deformation_fields, 1, -1)
        deformation_fields_smooth = cp.moveaxis(deformation_fields_smooth, 1, -1)
        jacobian = compute_jacobian_determinant(deformation_fields_smooth, ref_index=ref_index)
        limit = 0.9 # E.g. cap the jacobian at 0.9 and 1/0.9 
        jacobian[jacobian < limit] = limit
        jacobian[jacobian > 1/limit] = 1/limit
        # pl.ImagePlot(deformation_fields, x=1, y=2, z=0, hide_axes=True, colormap="magma", vmin=-1, vmax=1) # x component
        # pl.ImagePlot(deformation_fields_smooth, x=1, y=2, z=0, hide_axes=True, colormap="magma", vmin=-1, vmax=1) # x component
        # pl.ImagePlot(jacobian, x=1, y=2, z=0, hide_axes=True, colormap="viridis", vmin=0.9, vmax=1/0.9) # x component
        output *= jacobian
        
        output_vx, output_vy, output_vz, farneback, images, ref_image, jacobian = None, None, None, None, None, None, None
        del output_vx
        del output_vy
        del output_vz
        del farneback
        del images
        del ref_image
        del jacobian

        return output, deformation_fields



def compute_jacobian_determinant(deformation_field, ref_index):
    """
    Compute the Jacobian determinant from a 3D deformation field, supporting both NumPy and CuPy arrays.

    Args:
        deformation_field (array-like): A 4D array of shape (N, H, W, D, 3),
                                        where the last dimension contains
                                        the displacement components (dx, dy, dz).
                                        Can be a NumPy or CuPy array.

    Returns:
        array-like: A 3D array of the same spatial shape (N, H, W, D) containing
                    the Jacobian determinant for each voxel. Returned on the
                    same device (NumPy or CuPy) as the input.
    """
    # Determine the library (NumPy or CuPy) based on the input
    xp = cp.get_array_module(deformation_field)
    N, H, W, D, ndim = deformation_field.shape 
    
    # Ensure the deformation field is in the correct shape
    if ndim != 3:
        print(f"The deformation field is incorrectly shaped with {deformation_field.shape}.")
        raise ValueError("Deformation field must have shape (N, H, W, D, 3).")
    
    # Center the deformation for an identity transform, like ANTs does (centers it about 1)
    deformation_field_copy = deformation_field.copy()
    identity = xp.meshgrid(
        *[xp.linspace(0, s-1, s) for s in deformation_field.shape[1:4]],
        indexing='ij'
    )
    deformation_field_copy += xp.stack(identity, axis=-1)
    # deformation_field_copy -= xp.stack(identity, axis=-1)

    # Compute gradients of the deformation field
    det_jacobian = xp.ones((N, H, W, D), dtype=float)
    indices = [i for i in range(N) if i != ref_index]
    for phase in indices:
        grad_x = xp.gradient(deformation_field_copy[phase, ..., 0], axis=(0, 1, 2))
        grad_y = xp.gradient(deformation_field_copy[phase, ..., 1], axis=(0, 1, 2))
        grad_z = xp.gradient(deformation_field_copy[phase, ..., 2], axis=(0, 1, 2))

        # Stack gradients into the Jacobian matrix
        J11, J12, J13 = grad_x # Gradients of dx
        J21, J22, J23 = grad_y # Gradients of dy
        J31, J32, J33 = grad_z # Gradients of dz

        # Compute the determinant of the Jacobian matrix for each voxel
        det_jacobian[phase,...] = (
            J11 * (J22 * J33 - J23 * J32)
            - J12 * (J21 * J33 - J23 * J31)
            + J13 * (J21 * J32 - J22 * J31)
        ) 


    return det_jacobian


def findGPUs():
    numdevices =  torch.cuda.device_count()
    memcap = list() 
    for devno in range(numdevices):
        with torch.cuda.device(devno):
            f,t = torch.cuda.mem_get_info()        
        #memcap.append(float(torch.cuda.get_device_properties(devno).total_memory)/float(1024**3))
        memcap.append(float(f)/float(1024**3))
        print(f'Memory: {memcap}')
    
    return np.argsort(np.array(memcap))

def create_3d_tukey_window(nx, ny, nz, alpha=0.5):
    """Create a 3D Tukey window on the GPU using CuPy.
        alpha=1 --> extreme Hanning filter
        alpha=0 --> rectangular
    
    """
    win_x = cp.asarray(tukey(nx, alpha=alpha))
    win_y = cp.asarray(tukey(ny, alpha=alpha))
    win_z = cp.asarray(tukey(nz, alpha=alpha))

    win_3d = cp.outer(win_x, win_y).reshape(nx, ny, 1) * win_z.reshape(1, 1, nz)
    return win_3d



def fourier_smooth_deformation_cupy(disp_field_cp, n_harmonics=3):
    """
    Smooth deformation field using real-valued Fourier basis.
    Shape: [T, 3, X, Y, Z]
    Returns: same shape, smoothed over time.
    """
    T, C, X, Y, Z = disp_field_cp.shape
    N = X * Y * Z
    B = 2 * n_harmonics + 1

    # Create real-valued Fourier basis [T, B]
    t = cp.linspace(0, 1, T, dtype=cp.float32)
    Phi = cp.empty((T, B), dtype=cp.float32)
    Phi[:, 0] = 1.0
    for k in range(1, n_harmonics + 1):
        Phi[:, 2 * k - 1] = cp.cos(2 * cp.pi * k * t)
        Phi[:, 2 * k]     = cp.sin(2 * cp.pi * k * t)
    Phi /= cp.linalg.norm(Phi, axis=0, keepdims=True)
    PhiT = Phi.T  # [B, T]

    # Flatten spatial dims: [T, C, N]
    disp_flat = disp_field_cp.reshape(T, C, N)

    # Project using einsum
    coeffs = cp.einsum('bt,tcn->bcn', PhiT, disp_flat)      # [B, C, N]
    smoothed = cp.einsum('tb,bcn->tcn', Phi, coeffs)         # [T, C, N]
    return smoothed.reshape(T, C, X, Y, Z).astype(cp.float32)