# -*- coding: utf-8 -*-
"""
Coil Sketching 4D reconstruction functions:

Originally written by Julio A. Oscanoa (joscanoa@stanford.edu), 2023.
Extended to 4D by Joseph Plummer (jplummer@stanford.edu), 2025.

This module contains MR iterative reconstruction apps for iterative sketching
based reconstruction. 
"""
import numpy as np
import time


from sketching_app import SketchedLinearLeastSquares
import sigpy as sp
from sigpy.mri import linop
from sigpy.mri.app import _estimate_weights
from scipy.linalg import hadamard
from sigpy import backend, util

import sys
sys.path.append('../sigpy_mod')
import prox_lr # Low rank proximal operator
import prox_llr # Locally Low rank proximal operator
import prox_ltr # Low rank proximal operator
import custom_linop # 4D NUFFT

import cupy as cp
import gc
import matplotlib.pyplot as plt
import os

class CoilSketching(SketchedLinearLeastSquares):

    def __init__(self, y, mps, reduced_ncoils, b0_map=None, moco=False, number_non_sketched_coils=None, 
                 solver=None,  sketch_type='Rademacher',
                 weights=None, coord=None, coil_batch_size=None, sketch_arrays=None,
                 device=sp.cpu_device, sketch_sigma=None, img_shape=None, 
                 toeplitz=False, **kwargs):

        self.img_shape = img_shape
        self.b0_map = b0_map
        self.mps = mps
        self.mps_S = None
        self.mps_S_array = None
        self.reduced_ncoils = reduced_ncoils
        self.solver = solver
        print(f"Double checking... self.solver = {solver}")
        
        self.toeplitz = toeplitz 
        self.moco = moco
        self.device = device
        print(f'self.device = {self.device}')
        
        if len(self.mps.shape) == 4:
            self.total_ncoils = mps.shape[0]
        elif len(self.mps.shape) == 5:
            self.total_ncoils = mps.shape[1]
        self.number_non_sketched_coils = number_non_sketched_coils
        if self.number_non_sketched_coils is None:
            self.number_non_sketched_coils = self.reduced_ncoils - 1

        self.coil_batch_size = coil_batch_size
        if self.coil_batch_size is None:
            self.coil_batch_size = self.total_ncoils
        else:
            print(f"Using a coil batch size of {self.coil_batch_size} in this reconstruction.")

        self.sketch_type = sketch_type
        self.sketch_arrays = sketch_arrays
        self.sketch_sigma = sketch_sigma

        weights = _estimate_weights(y, weights, coord)
        if weights is not None:
            y *= weights**0.5

        self.weights = weights
        self.coord = coord

        if len(self.coord.shape) == 3: # Assumes no respiratory phase dimension
            A = linop.Sense(self.mps, coord=self.coord, weights=self.weights,
                            coil_batch_size=coil_batch_size)
        else: # Assume the first dimension is respiratory phase (i.e. do a multi-phase reconstruction)
            A = custom_linop.Sense4D(self.mps, self.coord, self.weights, device, b0_map = self.b0_map, coil_batch_size=self.coil_batch_size)
            
            if not self.toeplitz:
                # Optional: move coord and weights to the device for faster NUFFT calculations (consumes more memory)
                print(f"Not using Toeplitz PSF, moving coord and weights to device {device} for faster NUFFT calculations.")
                print("WARNING: THIS WILL CONSUME MORE MEMORY, but should be faster.")
                self.coord = sp.to_device(self.coord, device=device) 
                self.weights = sp.to_device(self.weights, device=device) 
                print(f'self.coord device = {backend.get_device(self.coord)}')
                print(f'self.weights device = {backend.get_device(self.weights)}')
                
            else:
                tic = time.time()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                cp._default_memory_pool.free_all_blocks()
                cp.cuda.Stream.null.synchronize()
                cp.cuda.stream.get_current_stream().synchronize()
                gc.collect()
                toc = time.time()
                print(f"Time taken to clear memory before calculating the Toeplitz PSF: {toc - tic:.2f} seconds")
  
                # self.toeplitz_device = sp.Device(3)
                self.toeplitz_device = device # Override to use the same device as the reconstruction device
                with self.toeplitz_device:
                    print(f'self.toeplitz_device = {self.toeplitz_device} <-- the PSF will be calculated on this device')
                    print("Calculating Toeplitz PSF for all respiratory phases...")
                    nphase, nexc, nread, ndim = coord.shape
                    # self.psf = cp.zeros((nphase,2*mps.shape[-3],2*mps.shape[-2],2*mps.shape[-1]), dtype=y.dtype) # Only works for (Nt,Nx,Ny,Nz) (i.e. a 4D recon). Modify for (Nt,Nx,Ny)
                    self.psf = np.zeros((nphase,2*mps.shape[-3],2*mps.shape[-2],2*mps.shape[-1]), dtype=y.dtype) # Only works for (Nt,Nx,Ny,Nz) (i.e. a 4D recon). Modify for (Nt,Nx,Ny)
                    for resp in range(nphase):
                        coord_psf = backend.to_device(coord[resp,...], self.toeplitz_device)
                        weights_psf = backend.to_device(weights[resp,...], self.toeplitz_device)
                        self.psf[resp,...] = custom_linop.weighted_toeplitz_psf(coord=coord_psf, 
                                                                                weights=weights_psf, 
                                                                                shape=mps.shape[-ndim:], 
                                                                                oversamp=1.33, # Default 1.25, 
                                                                                width=3).get()
                        # Free memory for current iteration before proceeding to the next
                        coord_psf, weights_psf = None, None
                        del coord_psf, weights_psf
                        # cp.get_default_memory_pool().free_all_blocks()
                        # cp.cuda.stream.get_current_stream().synchronize()
                        # gc.collect()
                        
                    tic = time.time()
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                    cp._default_memory_pool.free_all_blocks()
                    cp.cuda.Stream.null.synchronize()
                    cp.cuda.stream.get_current_stream().synchronize()
                    gc.collect()
                    toc = time.time()
                    print(f"Time taken to clear memory after calculating the Toeplitz PSF: {toc - tic:.2f} seconds")
                    
                    # Plot the psf to check if it has severe ringing anywhere
                    folder = os.getcwd() + "/tmp"
                    results_exist = os.path.exists(folder + "/toeplitz")
                    if not results_exist:
                        os.makedirs(folder + "/toeplitz")
                        print("A new directory inside: " + folder +
                                " called 'toeplitz' has been created.")
                    psf_slice = np.abs(self.psf[0, :, :, self.psf.shape[-1] // 2]) # .get()
                    psf_log = np.log1p(psf_slice)  # log1p(x) = log(1 + x) to avoid log(0) errors
                    plt.figure(figsize=(6, 5))
                    plt.imshow(psf_log, cmap='magma', origin='lower', aspect='auto')
                    plt.colorbar(label="log(1 + |PSF|)")
                    plt.title("Log-Scaled Toeplitz PSF")
                    plt.savefig(folder + "/toeplitz/psf.png", dpi=300)
                    plt.show()
                
                # Move the psf back to original device
                self.psf = backend.to_device(self.psf, self.device)
                # self.psf = backend.to_device(self.psf, -1)  # Move to CPU
                           
        print(f'device = {device}')     
        tic = time.time()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp._default_memory_pool.free_all_blocks()
        cp.cuda.Stream.null.synchronize()
        cp.cuda.stream.get_current_stream().synchronize()
        gc.collect()
        toc = time.time()
        print(f"Time taken to clear memory before sending inputs to super class: {toc - tic:.2f} seconds")

        super().__init__(A, y, device=device,
                         solver=solver, **kwargs)

    def _load_sketched_model_A_S(self):
        nch_nsk = self.number_non_sketched_coils
        nch_sk = self.reduced_ncoils - nch_nsk
        print(f'nch_sk = {nch_sk}, nch_nsk = {nch_nsk}')

        isk1 = self.outer_iter * nch_sk
        isk2 = (self.outer_iter + 1) * nch_sk
        print(f'isk1 = {isk1}; isk2 = {isk2}')
        
        if len(self.img_shape) == 3: # 3D
            if self.mps_S is None:
                self.mps_S = self.device.xp.zeros([self.reduced_ncoils]+list(self.img_shape), dtype=self.mps.dtype)

            self.mps_S[:nch_nsk,...] = sp.to_device(self.mps[:nch_nsk, ...], device=self.device)
            self.mps_S[nch_nsk:,...] = sp.to_device(self.mps_S_array[isk1:isk2, ...], device=self.device)
            self.A_S = linop.Sense(self.mps_S, coord=self.coord, weights=self.weights)
        else: # Assuming 4D
            if len(self.mps.shape) == 5: # mps.shape = (Nt,Nc,Nx,Ny,Nz)
                # if self.mps_S is None:
                #     self.mps_S = self.device.xp.zeros([self.img_shape[0]] + [self.reduced_ncoils] + list(self.img_shape[1:]), dtype=self.mps.dtype)
                # print(f'self.mps.shape = {self.mps.shape}')
                # print(f'self.mps[:, :nch_nsk, ...].shape = {self.mps[:, :nch_nsk, ...].shape}')
                # print(f'self.mps_S.shape = {self.mps_S.shape}')
                # self.mps_S[:,:nch_nsk,...] = sp.to_device(self.mps[:, :nch_nsk, ...], device=self.device)
                # self.mps_S[:,nch_nsk:,...] = sp.to_device(self.mps_S_array[:, isk1:isk2, ...], device=self.device)
                # self.A_S = custom_linop.Sense4D(self.mps_S, coord=self.coord, weights=self.weights, device=self.device)
                
                if self.mps_S is None:
                    self.mps_S = np.zeros([self.img_shape[0]] + [self.reduced_ncoils] + list(self.img_shape[1:]), dtype=self.mps.dtype)
                print('WARNING: experimenting with keeping mps_S on the CPU. May be slower.')
                self.mps_S[:,:nch_nsk,...] = np.array(self.mps[:, :nch_nsk, ...], dtype=self.y.dtype)
                self.mps_S[:,nch_nsk:,...] = np.array(self.mps_S_array[:, isk1:isk2, ...], dtype=self.y.dtype)
                self.A_S = custom_linop.Sense4D(self.mps_S, coord=self.coord, weights=self.weights, device=self.device, b0_map=self.b0_map, coil_batch_size=self.coil_batch_size)
                
                if self.toeplitz:
                    print("Computing the Toeplitz Normal linear operator using the measured PSF...")
                    self.T_S = custom_linop.Toeplitz_Normal(self.mps_S, self.coord, psf=self.psf) # Force overwrite the normal operator feature
                    print("Toeplitz operator computed. Clearing variables to save space.")
                    
            elif len(self.mps.shape) == 4: # mps.shape = (Nc,Nx,Ny,Nz) -- i.e. one mps for all Nt
                if self.mps_S is None:
                    self.mps_S = self.device.xp.zeros([self.reduced_ncoils] + list(self.img_shape[1:]), dtype=self.mps.dtype)
                print(f'WARNING: experimenting with keeping mps_S on device {self.device}. May take up memory but should be faster.')
                print(f'self.mps.shape = {self.mps.shape}')
                print(f'self.mps[:, :nch_nsk, ...].shape = {self.mps[:nch_nsk, ...].shape}')
                print(f'self.mps_S.shape = {self.mps_S.shape}')
                self.mps_S[:nch_nsk,...] = sp.to_device(self.mps[:nch_nsk, ...], device=self.device)
                self.mps_S[nch_nsk:,...] = sp.to_device(self.mps_S_array[isk1:isk2, ...], device=self.device)
                self.A_S = custom_linop.Sense4D(self.mps_S, coord=self.coord, weights=self.weights, device=self.device, b0_map=self.b0_map, coil_batch_size=self.coil_batch_size)
                
                if self.toeplitz:
                    print("Computing the Toeplitz Normal linear operator using the measured PSF...")
                    self.T_S = custom_linop.Toeplitz_Normal(self.mps_S, self.coord, psf=self.psf) # Force overwrite the normal operator feature
                    print("Toeplitz operator computed. Clearing variables to save space.")   
            
        return

    def _make_sketched_models_A_S(self):
        mps = self.mps
        nch = self.reduced_ncoils
        nc = self.total_ncoils
        nch_nsk = self.number_non_sketched_coils
        sketch_type = self.sketch_type
        sigma = self.sketch_sigma
        img_shape = self.img_shape

        if nch_nsk == -1:
            nch_nsk = nch

        nch_sk = nch - nch_nsk
        nc_sk = nc - nch_nsk
        dim_c = nc-nch_nsk
        dim_ch = nch-nch_nsk
        print(f'nch_sk = {nch_sk}, nch_nsk = {nch_nsk}, dim_c = {dim_c}, dim_ch = {dim_ch}')
        if len(mps.shape) == 4: # (Nc,Nx,Ny,Nz)
            if len(img_shape) == 3:
                sk_size = [dim_c] + [1]*len(img_shape) + [self.max_outer_iter * dim_ch]
            else:
                sk_size = [dim_c] + [1]*len(img_shape[1:]) + [self.max_outer_iter * dim_ch]
        elif len(mps.shape) == 5: # (Nt,Nc,Nx,Ny,Nz)
            sk_size = [1] + [dim_c] + [1]*len(img_shape[1:]) + [self.max_outer_iter * dim_ch]

        print(f'sk_size (i.e. a shape) = {sk_size}')

        if nch_sk > 0:
            if sigma is None:
                if sketch_type == 'Gaussian':
                    sigma = np.random.normal(0,1, [dim_c, self.max_outer_iter * dim_ch])/np.sqrt(nch_sk)

                elif sketch_type == 'Rademacher':
                    sigma = (np.random.randint(0, 2, size=[dim_c, self.max_outer_iter * dim_ch])*2 - 1)/np.sqrt(nch_sk)
                print(f'Random coil indices selected using {sketch_type} distribution:')
                print(f'{sigma}')

            # Getting sketched coils
            sigma = sigma.astype(mps.dtype)
            sigma1 = np.reshape(sigma, sk_size)

            print(f'sigma.shape = {sigma.shape} and sigma1.shape = {sigma1.shape}')
            if len(mps.shape) == 4: # (Nc,Nx,Ny,Nz)
                print("Making mps_sk for 4D mps...")
                xp = backend.get_array_module(mps)
                mps_sk = xp.sum(xp.expand_dims(mps[nch_nsk:,...],-1) * sigma1, 0)
                mps_sk = xp.moveaxis(mps_sk, -1, 0)
                print(f'mps_sk.shape = {mps_sk.shape}')
                print("4D mps_sk completed. ")
                
                # print("Making mps_sk for 4D mps...")
                # mps_sk = np.sum(np.expand_dims(mps[nch_nsk:,...],-1) * sigma1, 0)
                # mps_sk = np.moveaxis(mps_sk, -1, 0)
                # print(f'mps_sk.shape = {mps_sk.shape}')
                # print("4D mps_sk completed. ")
            elif len(mps.shape) == 5: # (Nt,Nc,Nx,Ny,Nz)
                print("Calculating mps_sk inside of _make_sketched_models_A_S() (warning, this can be slow for large numbers of coils)")
                speedup = True # Batch respiratory phases to fit onto GPU 
                if speedup:
                    mps_sk_device=backend.Device(2)
                    mps_sk = np.zeros((mps.shape[0], sigma.shape[1], mps.shape[-3], mps.shape[-2], mps.shape[-1]), dtype=self.y.dtype)
                    sigma1 = np.squeeze(sigma1, axis=0) # Remove the first dimension again as we no longer have resp phases on inner loop
                    for resp in range(mps.shape[0]):
                        print(f"Estimating mps_sk using device {mps_sk_device} for respiratory phase {resp}...")
                        mps_tmp = backend.to_device(mps[resp,...], mps_sk_device) 
                        xp = backend.get_array_module(mps_tmp)
                        with mps_sk_device:   
                            mps_sk_tmp = xp.sum(xp.expand_dims(mps_tmp[nch_nsk:,...],-1) * xp.array(sigma1), 0)
                            mps_sk_tmp = xp.moveaxis(mps_sk_tmp, -1, 0)
                        mps_sk[resp,...] = backend.to_device(mps_sk_tmp, -1) # Move back to CPU
                        del mps_tmp, mps_sk_tmp
                        print(f'mps_sk.shape = {mps_sk.shape}')
                        time.sleep(2) # Wait a second to give it time to clear memory if not done already 
                else:
                    mps_sk = np.sum(np.expand_dims(mps[:,nch_nsk:,...],-1) * sigma1, 1)
                    mps_sk = np.moveaxis(mps_sk, -1, 1)
                    print(f'mps_sk.shape = {mps_sk.shape}')
                
        self.sketch_sigma = sigma
        self.mps_S_array = mps_sk
        return

    def _make_initial_sketched_problem(self):

    
        if len(self.y.shape) == 3: 
            self.mps_S = sp.to_device(self.mps[:self.reduced_ncoils, ...], device=self.device)
            self.A_S = linop.Sense(self.mps_S, coord=self.coord, weights=self.weights)
            self.y_S = sp.to_device(self.y[:self.reduced_ncoils,...], device=self.device) # Default, 3D 
        else:
            if len(self.mps.shape) == 4: # (Nc,Nx,Ny,Nz)
                print(f'WARNING: experimenting with keeping mps_S on device {self.device}. May take up memory but should be faster.')
                self.mps_S = sp.to_device(self.mps[:self.reduced_ncoils, ...], device=self.device)
                
                # print('WARNING: experimenting with keeping mps_S on the CPU. May be slower.')
                # self.mps_S = np.array(self.mps[:self.reduced_ncoils, ...], dtype=complex)
            elif len(self.mps.shape) == 5: # (Nt,Nc,Nx,Ny,Nz):
                # self.mps_S = sp.to_device(self.mps[:,:self.reduced_ncoils, ...], device=self.device)
                
                print('WARNING: experimenting with keeping mps_S on the CPU. May be slower.')
                self.mps_S = np.array(self.mps[:,:self.reduced_ncoils, ...], dtype=self.y.dtype)
                
            self.A_S = custom_linop.Sense4D(self.mps_S, coord=self.coord, weights=self.weights, device=self.device, b0_map=self.b0_map, coil_batch_size=self.coil_batch_size) # Switched from before mps_S--if bugged, move back
            # self.y_S = sp.to_device(self.y[:,:self.reduced_ncoils,...], device=self.device) # First dimension is resp phase in the 4D problem # TODO: do not declare this to try and reduce memory load - see option 1/2 in sketching_app.py
            
            if self.toeplitz:
                print("Computing the Toeplitz Normal linear operator using the measured PSF...")
                self.T_S  = custom_linop.Toeplitz_Normal(self.mps_S, self.coord, psf=self.psf) # Force overwrite the normal operator feature
                print("Toeplitz operator computed. Clearing variables to save space.")


        return

    def _get_AHy(self):
        if len(self.y.shape) == 3: 
            coil_batch_size = min(self.reduced_ncoils, self.coil_batch_size)
            print(f'_get_AHy() coil batch size = {coil_batch_size}')
            self.A_S = linop.Sense(self.mps, coord=self.coord, weights=self.weights,
                        coil_batch_size=coil_batch_size)
            self.AHy = self.A_S.H(sp.to_device(self.y, self.device))
        elif len(self.y.shape) == 4: # Assume 4D         
            print("USING JOEY'S ATTEMPT AT COIL BATCHING self.y")
            print("WARNING: currently only implemented for 4D mps or combine_csm = True")

            num_coil_batches = (self.total_ncoils + self.reduced_ncoils - 1) // self.reduced_ncoils # Note, this slows down recon by coil batching
            if num_coil_batches < 1:
                num_coil_batches = 1
            with self.device:  # Ensure all operations run on the correct device
                self.AHy = self.device.xp.zeros(self.img_shape, dtype=self.y.dtype)
                print(f'self.AHy.shape = {self.AHy.shape}')
                for c in range(num_coil_batches):
                    mps_cb = sp.to_device(self.mps[c * self.reduced_ncoils:((c + 1) * self.reduced_ncoils), ...], self.device)
                    self.A_S_cb = custom_linop.Sense4D(
                        mps_cb,  # WARNING: currently only implemented for 4D mps
                        coord=self.coord, 
                        weights=self.weights, 
                        device=self.device, 
                        b0_map=self.b0_map,
                        coil_batch_size=self.coil_batch_size
                    )                   
                    y_tmp = sp.to_device(self.y[:, c * self.reduced_ncoils:((c + 1) * self.reduced_ncoils), ...], self.device)
                    cp.add(self.AHy, self.A_S_cb.H(y_tmp), out=self.AHy)  # Safe in-place accumulation
                    self.mps_cb, self.A_S_cb, y_tmp = None, None, None
                    del self.A_S_cb, self.mps_cb, y_tmp
                    self._clear_gpu_memory()
        

        return

    def _get_true_gradient(self):
        brute_force=False # FORCE to use all coils at once, no batching
        if brute_force:
            print("WARNING: NOT BATCHING COILS TO CALCULATE TRUE GRADIENT. MAY CRASH GPU. ")
            print("Code supports 4D mps only... and no toeplitz...")
            with self.device:
                self.d = -self.AHy
                self.A_S_cb = custom_linop.Sense4D(self.mps, coord=self.coord, weights=self.weights, device=self.device, b0_map=self.b0_map, coil_batch_size=None)
                self.d += self.A_S_cb.N * self.x  
            
        else:
            print("Original loop: batching coils to calculate true gradient to save some memory.")
            # Memory efficient calculation of true gradient
            num_coil_batches = (self.total_ncoils + self.reduced_ncoils - 1) // self.reduced_ncoils 
            if num_coil_batches < 1:
                num_coil_batches = 1
            
            with self.device:
                self.d = -self.AHy

                for c in range(num_coil_batches):
                    # print("Joey was here, theory is that mps_S is being overwritten here with a coil-reduced size, causing bugs if num_coil_batches varies, so I renamed it")
                    if len(self.coord.shape) == 3:  # 3D
                        # self.mps_S = sp.to_device(self.mps[c*self.reduced_ncoils:((c+1)*self.reduced_ncoils)], self.device)
                        # self.A_S = linop.Sense(self.mps_S, coord=self.coord, weights=self.weights)
                        
                        self.mps_S_cb = sp.to_device(self.mps[c*self.reduced_ncoils:((c+1)*self.reduced_ncoils)], self.device)
                        self.A_S_cb = linop.Sense(self.mps_S_cb, coord=self.coord, weights=self.weights)
                    elif len(self.coord.shape) == 4: # Assume 4D
                        if len(self.mps.shape) == 4: # (Nc,Nx,Ny,Nz)
                            print(f'WARNING: experimenting with keeping mps_S on device {self.device}. May take up memory but should be faster.')
                            self.mps_S_cb = sp.to_device(self.mps[c*self.reduced_ncoils:((c+1)*self.reduced_ncoils),...], self.device)
                            # print("Keeping mps on cpu to help accelerate/reduce gpu load")
                            # self.mps_S_cb = np.array(self.mps[c*self.reduced_ncoils:((c+1)*self.reduced_ncoils),...])
                        elif len(self.mps.shape) == 5: # (Nt,Nc,Nx,Ny,Nz):
                            # self.mps_S_cb = sp.to_device(self.mps[:,c*self.reduced_ncoils:((c+1)*self.reduced_ncoils),...], self.device)
                            print("Keeping mps on cpu to help accelerate/reduce gpu load")
                            self.mps_S_cb = np.array(self.mps[:,c*self.reduced_ncoils:((c+1)*self.reduced_ncoils),...])

                        # TODO: this only applies for 4D problem, not 3D...
                        if self.toeplitz:
                            print("Computing the Toeplitz Normal linear operator using the measured PSF...")
                            T_S_cb = custom_linop.Toeplitz_Normal(self.mps_S_cb, self.coord, psf=self.psf) # Force overwrite the normal operator feature
                            print("Toeplitz operator computed. Clearing variables to save space.")
                            tmp = T_S_cb(self.x)
                            cp.add(self.d, tmp, out=self.d)  # Safe in-place accumulation
                            self.mps_S_cb, T_S_cb, tmp = None, None, None
                            del T_S_cb, self.mps_S_cb, tmp
                        else:
                            # self.A_S_cb_normal = custom_linop.Sense4D(self.mps_S_cb, coord=self.coord, weights=self.weights, device=self.device, b0_map=self.b0_map, coil_batch_size=self.coil_batch_size).N
                            # self.d += self.A_S_cb_normal * self.x  
                            # self.A_S_cb_normal = None
                            # self.mps_S_cb = None             
                            # del self.A_S_cb_normal, self.mps_S_cb     
                            A_S_cb = custom_linop.Sense4D(
                                self.mps_S_cb,
                                coord=self.coord,
                                weights=self.weights,
                                device=self.device,
                                b0_map=self.b0_map,
                                coil_batch_size=self.coil_batch_size
                                )

                            tmp = A_S_cb.N(self.x)
                            cp.add(self.d, tmp, out=self.d)  # Safe in-place accumulation
                            self.mps_S_cb, A_S_cb, tmp = None, None, None
                            del A_S_cb, self.mps_S_cb, tmp

                            cp.get_default_memory_pool().free_all_blocks()
                            cp.get_default_pinned_memory_pool().free_all_blocks()
                            cp._default_memory_pool.free_all_blocks()
                            cp.cuda.Stream.null.synchronize()
                            cp.cuda.stream.get_current_stream().synchronize()

                            gc.collect()  # Trigger garbage collection
        return
    
    def _clear_gpu_memory(self, *variables):
        """Clears GPU memory and optionally deletes specified variables."""
        with cp.cuda.Device(self.device):
            for var in variables:
                if isinstance(var, str) and hasattr(self, var):
                    setattr(self, var, None)  # Remove reference from self
                    delattr(self, var)  # Fully delete variable
                elif var is not None:
                    var = None  # Set local reference to None
                    del var  # Delete local reference
           
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            cp._default_memory_pool.free_all_blocks()
            cp.cuda.Stream.null.synchronize()
            cp.cuda.stream.get_current_stream().synchronize()
            gc.collect()
            print("Calling from sketching_app_4d.py: GPU memory cleared.")
            
            
            
    def _compute_max_eigenvalue_A(self):       
        import maxeig as me
        print("**** CALCULATING MAX EIGENVALUE OF A ****")
        subset_phases = self.coord.shape[0] // 3
        print(f"Warning: calculating max eigenvalue of A using only a subset ({subset_phases}) of the respiratory phases.")
        A_max_eig = custom_linop.Sense4D(self.mps, coord=self.coord[:subset_phases,...], weights=self.weights[:subset_phases,...], device=self.device, b0_map=self.b0_map, coil_batch_size=None)
        max_eig_app = me.MaxEig(A_max_eig.N, 
                                dtype=self.y.dtype, 
                                device=self.device,
                                max_iter=30,
                                show_pbar=True)
        max_eig = max_eig_app.run()
        max_eig = max_eig_app.plot_eigenvalues() # multiplies output by 1.1 
        max_eig_app = None
        self.max_eig_A = max_eig
        print(f"**** MAX EIGENVALUE OF A CALCULATED AS {self.max_eig_A} ****")        
        return

        
class SketchedL1WaveletRecon4D(CoilSketching):
    r"""L1 Wavelet regularized reconstruction.

    Solves the following problem efficiently using Coil Sketching:

    .. math::
        \min_x \frac{1}{2} \| P F S x - y \|_2^2 + \lambda \| W x \|_1

    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, W is the wavelet operator,
    x is the image, and y is the k-space measurements.

    Args:
        y (array): k-space measurements.
        mps (array): sensitivity maps.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        wave_name (str): wavelet name.
        device (Device): device to perform reconstruction.
        coil_batch_size (int): batch size to process coils.
        Only affects memory usage.
        comm (Communicator): communicator for distributed computing.
        **kwargs: Other optional arguments.

    References:
        Lustig, M., Donoho, D., & Pauly, J. M. (2007).
        Sparse MRI: The application of compressed sensing for rapid MR imaging.
        Magnetic Resonance in Medicine, 58(6), 1082-1195.

    """

    def __init__(self, y, mps, lamda, reduced_ncoils,
                 wave_name='db4', **kwargs):
        
        ndim = 3 # force 3D images for now
        if len(y.shape) == 3: # 3D, no respiratory frame
            img_shape = (mps.shape[-ndim:])
        else: # 4D
            print(f'y.shape = {y.shape}')
            img_shape = (y.shape[0],) + (mps.shape[-ndim:])
            
        W = sp.linop.Wavelet(img_shape, wave_name=wave_name, axes=(-3, -2, -1)) # Last 3 dimensions
        proxg = sp.prox.UnitaryTransform(sp.prox.L1Reg(W.oshape, lamda), W)

        def g(input):
            device = sp.get_device(input)
            xp = device.xp
            with device:
                return lamda * xp.sum(xp.abs(W(input))).item()

        super().__init__(y, mps, reduced_ncoils, proxg=proxg, g=g,
                                        img_shape=img_shape, **kwargs)

        
def SpatioTemporalFiniteDifference(ishape, axes=None, scales=None):
        """Linear operator that computes scaled finite difference gradient.

        Args:
            ishape (tuple of ints): Input shape.
            axes (tuple or list): Axes to compute finite difference on. All axes are used if None.
            scales (list or tuple): Scaling factor per axis. Must be same length as `axes`.
            
        Usage:
            # Temporal = axis 0 (full lambda), spatial = axes 1,2,3 (scaled by 0.1)
            G = FiniteDifference(img_shape, axes=(0,1,2,3), scales=(1.0, 0.1, 0.1, 0.1))

        Returns:
            Linop: A vertically stacked linear operator of scaled finite differences.
        """
        Id = sp.linop.Identity(ishape)
        ndim = len(ishape)
        axes = util._normalize_axes(axes, ndim)

        if scales is None:
            scales = [1.0] * len(axes)
        assert len(scales) == len(axes), "Length of scales must match number of axes."

        linops = []
        for i, scale in zip(axes, scales):
            D = Id - sp.linop.Circshift(ishape, [1], axes=[i])
            R = sp.linop.Reshape([1] + list(ishape), ishape)
            if scale != 1.0:
                D = scale * D
            linops.append(R * D)

        G = sp.linop.Vstack(linops, axis=0)
        return G
        
class SketchedTotalVariationRecon4D(CoilSketching):
    r"""Total variation regularized reconstruction.

    Solves the following problem efficiently using Coil Sketching:

    .. math::
        \min_x \frac{1}{2} \| P F S x - y \|_2^2 + \lambda \| G x \|_1

    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, G is the gradient operator,
    x is the image, and y is the k-space measurements.

    Args:
        y (array): k-space measurements.
        mps (array): sensitivity maps.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        device (Device): device to perform reconstruction.
        coil_batch_size (int): batch size to process coils.
        Only affects memory usage.
        comm (Communicator): communicator for distributed computing.
        **kwargs: Other optional arguments.

    References:
        Block, K. T., Uecker, M., & Frahm, J. (2007).
        Undersampled radial MRI with multiple coils.
        Iterative image reconstruction using a total variation constraint.
        Magnetic Resonance in Medicine, 57(6), 1086-1098.

    """

    def __init__(self, y, mps, lamda, reduced_ncoils,
                 **kwargs):

        ndim = 3 # force 3D images for now
        if len(y.shape) == 3: # 3D, no respiratory frame
            img_shape = (mps.shape[-ndim:])
        else: # 4D
            img_shape = (y.shape[0],) + (mps.shape[-ndim:])
        G = SpatioTemporalFiniteDifference(img_shape, axes=(0,1,2,3), scales=(1,0.5,0.5,0.5)) 
        proxg = sp.prox.L1Reg(G.oshape, lamda)

        def g(x):
            device = sp.get_device(x)
            xp = device.xp
            with device:
                return lamda * xp.sum(xp.abs(x)).item()

        super().__init__(y, mps, reduced_ncoils, proxg=proxg, g=g, G=G,
                                                img_shape=img_shape, **kwargs)
        
    
        
        
class SketchedLowRankRecon4D(CoilSketching):
    r"""Low Rank regularized reconstruction.

    Solves the following problem efficiently using Coil Sketching:

    .. math::
        \min_x \frac{1}{2} \| P F S x - y \|_2^2 + \lambda \| M x \|_*

    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, M is the motion field operator,
    x is the image, and y is the k-space measurements.

    Args:
        y (array): k-space measurements.
        mps (array): sensitivity maps.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        device (Device): device to perform reconstruction.
        coil_batch_size (int): batch size to process coils.
        Only affects memory usage.
        comm (Communicator): communicator for distributed computing.
        **kwargs: Other optional arguments.

    References:
        Block, K. T., Uecker, M., & Frahm, J. (2007).
        Undersampled radial MRI with multiple coils.
        Iterative image reconstruction using a total variation constraint.
        Magnetic Resonance in Medicine, 57(6), 1086-1098.

    """

    def __init__(self, y, mps, lamda, reduced_ncoils, moco=True, ref_index=0, solver=None,
                 device=sp.cpu_device,
                 **kwargs):

        ndim = 3 # force 3D images for now
        if len(y.shape) == 3: # 3D, no respiratory frame
            img_shape = (mps.shape[-ndim:])
        else: # 4D
            img_shape = (y.shape[0],) + (mps.shape[-ndim:])
        if moco:
            # M = custom_linop.RegisterImages(img_shape, ref_index, devnum=device.id) # Match y device
            M = custom_linop.RegisterImagesWithJacobian(img_shape, ref_index, devnum=device.id, compute_jacobian=True) # Use predeclared device
            print("Regularizing the motion compensated images.")
        else:
            M = sp.linop.Identity(img_shape) # Use axes argument to regularize over a custom dimension, default = all.
            print("Not regularizing the motion compensated images.")
        
        
        
        if solver == 'GradientMethod':
            G = None # Goes down the gradient method path, and computes G inside of proxg and g(x). This requires M to be unitary, of which RegisterImages is not.
            print("Using Gradient method to solve the reconstruction problem.")
            proxg = sp.prox.UnitaryTransform(prox_lr.GLRA(shape = M.oshape, 
                              lamda = lamda, plot=False, img_shape_for_plot=img_shape), M)
        elif solver == "PrimalDualHybridGradient":
            G = M # Goes down PDHG path, but warning--this may be slower, and require more iterations. Must be used if M is non unitary.
            print("Using PDHG to solve the reconstruction problem.")
            proxg = prox_lr.GLRA(shape = M.oshape, 
                              lamda = lamda, plot=False, img_shape_for_plot=img_shape)

        def g(x):
            device = sp.get_device(x)
            xp = device.xp
            with device:
                # x = np.reshape((x.shape[0], -1))
                print(f'x.shape inside g(x) = {x.shape}')
                x = np.reshape(x, img_shape)
                u,s,vh = np.linalg.svd(M(x),full_matrices=False)
                return lamda * xp.sum(xp.abs(s)).item()

        super().__init__(y, mps, reduced_ncoils, proxg=proxg, g=g, G=G, device=device,
                                                img_shape=img_shape, **kwargs)
        
        
        
class SketchedMotionCompensatedLowRankRecon4D(CoilSketching):
    r"""Low Rank regularized reconstruction.

    Solves the following problem efficiently using Coil Sketching:

    .. math::
        \min_x \frac{1}{2} \| P F S x - y \|_2^2 + \lambda \| M x \|_*

    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, M is the motion field operator,
    x is the image, and y is the k-space measurements.

    Args:
        y (array): k-space measurements.
        mps (array): sensitivity maps.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        device (Device): device to perform reconstruction.
        coil_batch_size (int): batch size to process coils.
        Only affects memory usage.
        comm (Communicator): communicator for distributed computing.
        **kwargs: Other optional arguments.

    References:
        Block, K. T., Uecker, M., & Frahm, J. (2007).
        Undersampled radial MRI with multiple coils.
        Iterative image reconstruction using a total variation constraint.
        Magnetic Resonance in Medicine, 57(6), 1086-1098.

    """

    def __init__(self, y, mps, lamda, reduced_ncoils, moco=True, ref_index=0, solver=None,
                 device=sp.cpu_device, N_frames_per_block=None,
                 **kwargs):

        ndim = 3 # force 3D images for now
        if len(y.shape) == 3: # 3D, no respiratory frame
            img_shape = (mps.shape[-ndim:])
        else: # 4D
            img_shape = (y.shape[0],) + (mps.shape[-ndim:])
        if N_frames_per_block is None:
            N_frames_per_block = np.max((img_shape[0]//2, 2)) # The smaller we make this number, the faster the recon, at the cost of less frames to perform SVD over
        if solver == 'GradientMethod':
            G = None # Goes down the gradient method path, and computes G inside of proxg and g(x). This requires M to be unitary, of which RegisterImages is not.
            print("Using Gradient method to solve the reconstruction problem.")
            proxg = prox_ltr.GLRA(shape = img_shape, 
                              lamda = lamda, plot=False, prox_moco=moco, N_frames_per_block=N_frames_per_block, compute_jacobian=True)
        elif solver == "PrimalDualHybridGradient":
            G = sp.linop.Identity(img_shape) # Goes down PDHG path, but warning--this may be slower and require more iterations. 
            print("Using PDHG to solve the reconstruction problem.")
            proxg = prox_ltr.GLRA(shape = img_shape, 
                              lamda = lamda, plot=False, prox_moco=moco, N_frames_per_block=N_frames_per_block, compute_jacobian=True)

        def g(x):
            device = sp.get_device(x)
            xp = device.xp
            with device:
                # x = np.reshape((x.shape[0], -1))
                print(f'x.shape inside g(x) = {x.shape}')
                x = np.reshape(x, img_shape)
                u,s,vh = np.linalg.svd(x,full_matrices=False)
                return lamda * xp.sum(xp.abs(s)).item()

        super().__init__(y, mps, reduced_ncoils, proxg=proxg, g=g, G=G, device=device,
                                                img_shape=img_shape, **kwargs)
        
    
class SketchedLocallyLowRankRecon4D(CoilSketching):
    r"""Locally Low Rank regularized reconstruction.

    Solves the following problem efficiently using Coil Sketching:

    .. math::
        \min_x \frac{1}{2} \| P F S x - y \|_2^2 + \lambda \| M x \|_*

    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, M is the motion field operator,
    x is the image, and y is the k-space measurements.

    Args:
        y (array): k-space measurements.
        mps (array): sensitivity maps.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        device (Device): device to perform reconstruction.
        coil_batch_size (int): batch size to process coils.
        Only affects memory usage.
        comm (Communicator): communicator for distributed computing.
        **kwargs: Other optional arguments.

    References:
        Block, K. T., Uecker, M., & Frahm, J. (2007).
        Undersampled radial MRI with multiple coils.
        Iterative image reconstruction using a total variation constraint.
        Magnetic Resonance in Medicine, 57(6), 1086-1098.

    """

    def __init__(self, y, mps, lamda, reduced_ncoils, moco=False, ref_index=0,  solver=None,
                 device=sp.cpu_device,
                 **kwargs):

        ndim = 3 # force 3D images for now
        if len(y.shape) == 3: # 3D, no respiratory frame
            img_shape = (mps.shape[-ndim:])
        else: # 4D
            img_shape = (y.shape[0],) + (mps.shape[-ndim:])    
        if moco:
            # M = custom_linop.RegisterImages(img_shape, ref_index, devnum=sp.get_device(y).id) # Match y device
            M = custom_linop.RegisterImages(img_shape, ref_index, devnum=device.id) # Use predeclared device
            print("Regularizing the motion compensated images.")
        else:
            M = sp.linop.Identity(img_shape) # Use axes argument to regularize over a custom dimension, default = all.
            print("Not regularizing the motion compensated images.")
            
        if solver == 'GradientMethod':
            G = None # Goes down the gradient method path, and computes G inside of proxg and g(x). This requires M to be unitary, of which RegisterImages is not.
            print("Using Gradient method to solve the reconstruction problem.")
            proxg = prox_llr.LLR(shape = img_shape, lamda = lamda, block=16) # Image size mush divide by block for now, as I have not implemented msk yet # TODO

        elif solver == "PrimalDualHybridGradient":
            G = M # Goes down PDHG path, but warning--this may be slower, and require more iterations. Must be used if M is non unitary.
            print("Using PDHG to solve the reconstruction problem.")
            proxg = prox_llr.LLR(shape = img_shape, lamda = lamda, block=16) # Image size mush divide by block for now, as I have not implemented msk yet # TODO


        def g(x):
            device = sp.get_device(x)
            xp = device.xp
            with device:
                # x = np.reshape((x.shape[0], -1))
                print(f'x.shape inside g(x) = {x.shape}')
                x = np.reshape(x, img_shape)
                u,s,vh = np.linalg.svd(M(x),full_matrices=False)
                return lamda * xp.sum(xp.abs(s)).item()

        super().__init__(y, mps, reduced_ncoils, proxg=proxg, g=g, G=G, device=device,
                                                img_shape=img_shape, **kwargs)



class SketchedSenseRecon(CoilSketching):
    r"""SENSE Reconstruction.

    Solves the following problem efficiently using Coil Sketching:

    .. math::
        \min_x \frac{1}{2} \| P F S x - y \|_2^2 +
        \frac{\lambda}{2} \| x \|_2^2

    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, x is the image, and y is the k-space measurements.

    Args:
        y (array): k-space measurements.
        mps (array): sensitivity maps.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        tseg (None or Dictionary): parameters for time-segmented off-resonance
            correction. Parameters are 'b0' (array), 'dt' (float),
            'lseg' (int), and 'n_bins' (int). Lseg is the number of
            time segments used, and n_bins is the number of histogram bins.
        coord (None or array): coordinates.
        device (Device): device to perform reconstruction.
        coil_batch_size (int): batch size to process coils.
            Only affects memory usage.
        comm (Communicator): communicator for distributed computing.
        **kwargs: Other optional arguments.

    References:
        Pruessmann, K. P., Weiger, M., Scheidegger, M. B., & Boesiger, P.
        (1999).
        SENSE: sensitivity encoding for fast MRI.
        Magnetic resonance in medicine, 42(5), 952-962.

        Pruessmann, K. P., Weiger, M., Bornert, P., & Boesiger, P. (2001).
        Advances in sensitivity encoding with arbitrary k-space trajectories.
        Magnetic resonance in medicine, 46(4), 638-651.

    """

    def __init__(self, y, mps, lamda, reduced_ncoils, solver=None,
                 **kwargs):
        
        ndim = 3 # force 3D images for now
        if len(y.shape) == 3: # 3D, no respiratory frame
            img_shape = (mps.shape[-ndim:])
        else: # 4D
            img_shape = (y.shape[0],) + (mps.shape[-ndim:])
            
            
        if solver == "PrimalDualHybridGradient":
            print("Using PDHG to solve the reconstruction problem.")
            G = sp.linop.Identity(img_shape)
            proxg = sp.prox.L2Reg(img_shape, lamda)

            def g(input):
                device = sp.get_device(input)
                xp = device.xp
                with device:
                    return lamda * xp.linalg.norm(input, ord='2').item()
        elif solver == "GradientMethod": 
            print(f"Using {solver} to solve the reconstruction problem.")
            G = None
            proxg = sp.prox.L2Reg(img_shape, lamda)

            def g(input):
                device = sp.get_device(input)
                xp = device.xp
                with device:
                    return lamda * xp.linalg.norm(input, ord='2').item()
        else:
            solver = "ConjugateGradient"
            print(f"Using {solver} to solve the reconstruction problem.")
            def g(input):
                device = sp.get_device(input)
                xp = device.xp
                with device:
                    return lamda * xp.linalg.norm(input, ord='2').item()
            G = None
            proxg = None
            
            
        super().__init__(y, mps, reduced_ncoils, lamda=lamda, g=g, proxg=proxg, G=G,
                                        solver=solver,
                                        img_shape=img_shape, **kwargs)
        