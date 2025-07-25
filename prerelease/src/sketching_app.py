# -*- coding: utf-8 -*-
"""
Coil Sketching 4D App.

Originally written by Julio A. Oscanoa (joscanoa@stanford.edu), 2023.
Extended to 4D by Joseph W. Plummer (joseph.plummer@nih.gov), 2025.

This module contains an abstract class App for sketched iterative reconstruction.
"""
import numpy as np
import sigpy as sp
import time

from tqdm.auto import tqdm
from sigpy import backend, linop, prox, util
from sigpy.alg import (PowerMethod, GradientMethod, ADMM,
                       ConjugateGradient, PrimalDualHybridGradient)
from sigpy.app import LinearLeastSquares
import cupy as cp
import gc

import sys
sys.path.append('../sigpy_mod')
import maxeig as me

import nibabel as nib
import os


class SketchedLinearLeastSquares(LinearLeastSquares):
    def __init__(self, A, y, max_init_iter=30, max_outer_iter=20, max_inner_iter=5, max_cg_iter_pdhg=4,
                 solver=None, save_objective_values=False,
                 alpha_init=None, sigma_init=None, tau_init=None, num_alphas=None,
                 device=sp.cpu_device, 
                 save_iterates=True,
                 **kwargs):
        
        self.A_S = None
        self.y_S= None
        self.sigma_t = None
        self.d = None #true gradient
        self.AHy = None
        self.solver=solver
        print(f"Triple checking... self.solver = {self.solver}")

        self.alpha_init = alpha_init
        self.sigma_init = sigma_init
        self.tau_init = tau_init
        self.num_alphas = num_alphas

        self.max_init_iter = max_init_iter
        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter - 1
        self.max_cg_iter_pdhg = max_cg_iter_pdhg # Number of CG iterations inside the PDHG inner loop (defaults to 4)
        self.iter = 0
        self.outer_iter = 0
        kwargs['max_iter'] = self.max_init_iter + self.max_outer_iter * self.max_inner_iter
        self.device = device
        
        # Manually turn on/off Nesterov's acceleration for FISTA
        self.accelerate = True # No discernable difference in performance, but it is on by default.

        # If num_alphas is not specified, set it to max_outer_iter // 2
        if self.num_alphas is None:
            self.num_alphas = self.max_outer_iter // 2

        self.y = y
        self._get_AHy()
        if self.max_outer_iter > 0:
            self._make_sketched_models_A_S()
            
        # Force initialize self.x so that it does not use the same device as self.y, which in this case may be on CPU
        self.x = self.device.xp.zeros(A.ishape, dtype=y.dtype)
        
        # Delete A if not needed due to memory constraints
        if save_objective_values is False:
            A = None # As we are using out own alg's, the only time we need A inside of LinearLeastSquares is when save_objective_values is True.

        # Optional: compute the max Eigenvalue for A and store it as self.max_eig_A
        # self._compute_max_eigenvalue_A()
        
        super().__init__(A, y, x=self.x, solver=solver, accelerate=self.accelerate, save_objective_values=save_objective_values, **kwargs)

        with self.device:
            print(f'Copying self.x on device {sp.get_device(self.x)} to new array self.x0 on device {self.device}.')
            self.x0 = self.device.xp.copy(self.x)
            print(f'self.x0 device = {sp.get_device(self.x0)}')
            
        if self.save_objective_values:
            self.y = sp.to_device(self.y, device=self.device)
            print(f'self.y device inside self.save_objective_values loop = {sp.get_device(self.y)}')
        else:
            self.y = None
            self.A = None
            
        self.nseed = 0

    def _get_alg(self):

        if self.solver is None:
            print("self.solver = None.")
            if self.proxg is None:
                self.solver = 'ConjugateGradient'

            elif self.G is None:
                self.solver = 'GradientMethod'

            else:
                self.solver = 'PrimalDualHybridGradient'
            print(f"self.solver automated to {self.solver}.")

        if self.solver == 'GradientMethod':
            print("INITIALIZING GRADIENT METHOD SOLVER SETTINGS.")
            
            # Free GPU memory
            self._clear_gpu_memory()

            # Alphas for each sketched problem
            if self.alpha is None and self.max_outer_iter > 0:
                print("Running down loop for: if self.alpha is None and self.max_outer_iter > 0:")
                self.alpha = np.ones((self.max_outer_iter,)) * np.inf

                for i in range(self.num_alphas):
                    self._load_sketched_model_A_S()                    
                    self.outer_iter +=1
                    
                    if self.toeplitz:
                        AHA = self.T_S
                        print("_load_sketched_model_A_S: AHA approximated using the Toeplitz PSF.")
                    else:
                        AHA = self.A_S.N 
                        
                    if self.lamda != 0:
                        print(f'self.lamda inside gradient method initialization = {self.lamda}')
                        I = linop.Identity(self.x.shape)
                        AHA += self.lamda * I
                        del I
                                            
                    # Use the specified GPU device
                    max_eig_device = self.device
                    with cp.cuda.Device(max_eig_device):
                        # Print memory usage before
                        memory_pool = cp.get_default_memory_pool()
                        print(f"{max_eig_device} before MaxEig: Allocated = {memory_pool.used_bytes()/1e6} MB, Reserved = {memory_pool.total_bytes()/1e6} MB")

                        # Run the MaxEig operation
                        if self.max_power_iter != 0:
                            # Custom MaxEig function:
                            max_eig_app = me.MaxEig(AHA, dtype=self.y.dtype, device=max_eig_device,
                                             max_iter=self.max_power_iter,show_pbar=self.show_pbar)
                            max_eig = max_eig_app.run()
                            max_eig = max_eig_app.plot_eigenvalues()
                            max_eig_app = None
                        else:
                            max_eig = self.max_eig_A
                            print(f"Maximum eigenvalue is set to {max_eig} as self.max_power_iter = 0.")

                        # Print memory usage after
                        self._clear_gpu_memory()
                        print(f"{max_eig_device} after MaxEig: Allocated = {memory_pool.used_bytes()/1e6} MB, Reserved = {memory_pool.total_bytes()/1e6} MB")

                    # Compute alpha
                    self.alpha[i] = 1 / max_eig if max_eig != 0 else 1
                    print(f'alpha[{i}] = {self.alpha[i]}')

                
                # Free GPU memory
                self._clear_gpu_memory(AHA)
                print(f"{max_eig_device} after freeing memory: Allocated = {memory_pool.used_bytes()/1e6} MB, Reserved = {memory_pool.total_bytes()/1e6} MB")
                print(f'alpha[{i}] = {self.alpha[i]} ----- num_alphas = {self.num_alphas} = max_outer_iter // 2')

                self.outer_iter = 0
                #Complete alphas with the min alpha
                if self.num_alphas < self.max_outer_iter:
                    min_alpha = min(self.alpha)
                    self.alpha[self.num_alphas:] = min_alpha

                if self.max_init_iter == 0:
                    self._update_GradientMethod()

            # Clear variables from _load_sketched_model_A_S() that are no longer needed
            self.mps_S = None
            self.A_S = None
            
            # Alpha for initialization
            if self.alpha_init is None and self.max_init_iter > 0:
                print("Running down loop for: if self.alpha_init is None and self.max_init_iter > 0:")
                self._make_initial_sketched_problem()
                
                # GPU optimization: we only need the encoding matrix A, so let's delete y_S from the GPU and collect it again after this step...
                print("Clearing self.y_S from the initial sketched problem as it is not needed during MaxEig normalization...")
                self.y_S = None
                self._clear_gpu_memory(self.y_S)
                
                if self.toeplitz:
                    AHA = self.T_S
                    print("_make_initial_sketched_problem: AHA approximated using the Toeplitz PSF.")
                else:
                    AHA = self.A_S.N 

                if self.lamda != 0:
                    I = linop.Identity(self.x.shape)
                    AHA += self.lamda * I
                    del I
                    
                # Remove elements from _make_initial_sketched_problem() that are no longer needed
                self._clear_gpu_memory()
                
                # Use the specified GPU device
                max_eig_device = self.device
                with cp.cuda.Device(max_eig_device):
                    # Print memory usage before
                    memory_pool = cp.get_default_memory_pool()
                    print(f"{max_eig_device} before freeing memory: Allocated = {memory_pool.used_bytes()/1e6} MB, Reserved = {memory_pool.total_bytes()/1e6} MB")

                    # Run the MaxEig operation
                    if self.max_power_iter != 0:                        
                        # Joey's custom MaxEig:
                        max_eig_app = me.MaxEig(AHA, dtype=self.y.dtype, device=max_eig_device,
                                            max_iter=self.max_power_iter,show_pbar=self.show_pbar)
                        max_eig = max_eig_app.run()
                        max_eig = max_eig_app.plot_eigenvalues()
                        max_eig_app = None
                    else:
                        max_eig = self.max_eig_A
                    self._clear_gpu_memory()
                    # Print memory usage after
                    print(f"{max_eig_device} after freeing memory: Allocated = {memory_pool.used_bytes()/1e6} MB, Reserved = {memory_pool.total_bytes()/1e6} MB")

                # Compute alpha
                self.alpha_init = 1 / max_eig if max_eig != 0 else 1
                print(f'alpha_init = {self.alpha_init}')
                
                # Free GPU memory
                self._clear_gpu_memory(AHA)
                AHA = None
                print(f"{max_eig_device} after freeing memory: Allocated = {memory_pool.used_bytes()/1e6} MB, Reserved = {memory_pool.total_bytes()/1e6} MB")
                print(f'alpha_init = {self.alpha_init}')
                
                # Clear out old variables
                print("Reloading self.y_S and self.A_S in the initial sketched problem...")
                self.y_S, self.A_S, self.mps_S = None, None, None
                gc.collect()
                
                with cp.cuda.Device(self.device):
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                    cp._default_memory_pool.free_all_blocks()
                    cp.cuda.Stream.null.synchronize()
                    cp.cuda.stream.get_current_stream().synchronize()
                    gc.collect()

                
                # After the initialization, run the first few iterations using _get_init_GradientMethod(), which is basically a non-sketched implementation of GradientMethod using a subset of coils.
                print(f"Memory before _get_init_GradientMethod(): Allocated = {memory_pool.used_bytes()/1e6} MB, Reserved = {memory_pool.total_bytes()/1e6} MB")
                self._make_initial_sketched_problem()
                # TODO: comment these out if recomputing AHy inside gradient_method, leave uncommented if using self.AHy
                # self.mps_S = None
                # self.y_S = None
                print("WARNING: deleting self.y_S before get_init_GradientMethod (might cause bugs with the AHy calculation inside the function)")
                gc.collect()
                self._get_init_GradientMethod()
                print(f"Memory after _get_init_GradientMethod(): Allocated = {memory_pool.used_bytes()/1e6} MB, Reserved = {memory_pool.total_bytes()/1e6} MB")
                
                # Remove remaining elements from _make_initial_sketched_problem() that are no longer needed
                self._clear_gpu_memory(self.A_S, self.y_S)
                # self.y_S, self.A_S, self.mps_S = None, None, None
                
                # Clear memory one last time
                print("Clearing memory after _get_init_GradientMethod()")
                with cp.cuda.Device(self.device):
                    tic = time.time()
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                    cp._default_memory_pool.free_all_blocks()
                    cp.cuda.Stream.null.synchronize()
                    cp.cuda.stream.get_current_stream().synchronize()
                    gc.collect()
                    toc = time.time()
                    print(f"Time taken to clear memory: {toc - tic:.2f} seconds")
                
        elif self.solver == 'ConjugateGradient': # Joey was here -- added this block as it did not exist before
            print("CONJUGATE GRADIENT SOLVER NOTICE:")
            print("It is assumed that AHA does not need to be normalized by the maximum eigenvalue.")
            print("A smarter mathematician than I may think differently.")
            print("This may alter the scale of lamda when applied to the L2 norm regularization term.")
            
            # Eigenvalue normalization for initialization
            if self.alpha_init is None and self.max_init_iter > 0:
                self._make_initial_sketched_problem()
                self._get_init_ConjugateGradient()
            
            
        elif self.solver == 'PrimalDualHybridGradient':
            print("INITIALIZING PDHG SOLVER SETTINGS.")
            
            # Free GPU memory
            self._clear_gpu_memory()
            
            # Additional factor applied to tau and sigma for the PDHG algorithm to improve stability and better satisfy the primal-dual condition: τσ||M|| ^2 < 1.
            self.primal_dual_scalar = 1 # Set to 1 for faster convergence, set to 0.5 for better stability, and set to 0.2 for even better stability but very slow convergence.
            print(f"Manually stabilizing the Primal-Dual Hybrid Gradient algorithm by setting self.primal_dual_scalar = {self.primal_dual_scalar}.")
            print(f"This is used to multiply the initial step sizes tau_init and sigma_init, and the step sizes tau and sigma, to ensure that the primal-dual condition is satisfied.")

            # Initial step size
            if  self.max_init_iter > 0:
                if self.sigma_init is None:
                    if self.tau_init is None:
                        self._make_initial_sketched_problem()
                        self.y_S = None # Joey's GPU optimization: we only need the encoding matrix A, so let's delete y_S from the GPU and collect it again after this step...
                        if self.toeplitz:
                            AHA = self.T_S
                            print("_make_initial_sketched_problem: AHA approximated using the Toeplitz PSF.")
                        else:
                            AHA = self.A_S.N                         
                        # Custom MaxEig calculation (uncomment for faster eigenvalue estimation):
                        # max_eig_app = me.MaxEig(AHA, dtype=self.y.dtype, device=self.device,
                        #                     max_iter=self.max_power_iter,show_pbar=self.show_pbar)
                        # max_eig = max_eig_app.run()
                        # max_eig = max_eig_app.plot_eigenvalues()
                        # max_eig_app = None
                        # Sigpy:
                        max_eig = sp.app.MaxEig(AHA, dtype=self.y.dtype, device=self.device,
                                                max_iter=self.max_power_iter, show_pbar=self.show_pbar).run()
                        
                        # Remove elements from _make_initial_sketched_problem() that are no longer needed
                        self.mps_S, self.A_S, self.y_S, AHA, self.T_S = None, None, None, None, None
                        self._clear_gpu_memory()

                        self.tau_init = self.primal_dual_scalar * 1 / max_eig

                    G = self.G
                    S = sp.linop.Multiply(G.oshape, self.tau_init)
                    GHG = G.H * S * G
                    
                    # Clear linops
                    self._clear_gpu_memory()

                    # Custom MaxEig:
                    # max_eig_app = me.MaxEig(GHG, dtype=self.y.dtype, device=self.device,
                    #                     max_iter=self.max_power_iter,show_pbar=self.show_pbar)
                    # max_eig = max_eig_app.run()
                    # max_eig = max_eig_app.plot_eigenvalues()
                    # max_eig_app = None
                    # Sigpy:
                    max_eig = sp.app.MaxEig(GHG, dtype=self.y.dtype, device=self.device,
                                            max_iter=self.max_power_iter, show_pbar=self.show_pbar).run()
                    
                    # Clear linops
                    G, S, GHG = None, None, None
                    del G, S, GHG
                    self._clear_gpu_memory()

                    self.sigma_init = self.primal_dual_scalar * 1 / max_eig

                elif self.tau_init is None:
                    self._make_initial_sketched_problem()
                    self.y_S = None # Joey's GPU optimization: we only need the encoding matrix A, so let's delete y_S from the GPU and collect it again after this step...
                    sigma = sp.linop.Multiply(self.A_S.oshape, self.sigma_init)
                    AHA = self.A_S.H * sigma * self.A_S

                    # Custom MaxEig:
                    # max_eig_app = me.MaxEig(AHA, dtype=self.y.dtype, device=self.device,
                    #                     max_iter=self.max_power_iter,show_pbar=self.show_pbar)
                    # max_eig = max_eig_app.run()
                    # max_eig = max_eig_app.plot_eigenvalues()
                    # max_eig_app = None
                    # Sigpy:
                    max_eig = sp.app.MaxEig(AHA, dtype=self.y.dtype, device=self.device,
                                            max_iter=self.max_power_iter, show_pbar=self.show_pbar).run()

                    # Remove elements from _make_initial_sketched_problem() that are no longer needed
                    self.mps_S, self.A_S, self.y_S, AHA, self.T_S, sigma = None, None, None, None, None, None
                    del self.mps_S, self.A_S, self.y_S, AHA, self.T_S, sigma
                    self._clear_gpu_memory()

                    self.tau_init = self.primal_dual_scalar * 1 / max_eig
                    
                # Reload self.y_S and self.A_S in the initial sketched problem as they were cleared before
                with cp.cuda.Device(self.device):
                    memory_pool = cp.get_default_memory_pool()
                print(f"Memory before _get_init_PrimalDualHybridGradient(): Allocated = {memory_pool.used_bytes()/1e6} MB, Reserved = {memory_pool.total_bytes()/1e6} MB")
                self._make_initial_sketched_problem()
                self.y_S = None # GPU optimization: we only need the encoding matrix A, so let's delete y_S from the GPU and collect it again after this step...
                print("Reloading self.y_S and self.A_S in the initial sketched problem...")
                
                self._get_init_PrimalDualHybridGradient()
                
                # Free GPU memory (but do not delete as they are reinitialized in the next iteration if equal to None)
                self.mps_S, self.A_S, self.y_S, AHA, self.T_S = None, None, None, None, None
                self._clear_gpu_memory()
                
            # Step sizes
            if self.max_outer_iter > 0:
                if self.sigma is None:
                    if self.tau is None:
                        # print("Joey was here")
                        # self._make_sketched_model_A()
                        self._load_sketched_model_A_S()
                        if self.toeplitz:
                            AHA = self.T_S
                            print("_load_sketched_model_A_S: AHA approximated using the Toeplitz PSF.")
                        else:
                            AHA = self.A_S.N 
                        # Custom MaxEig:
                        # max_eig_app = me.MaxEig(AHA, dtype=self.y.dtype, device=self.device,
                        #                     max_iter=self.max_power_iter,show_pbar=self.show_pbar)
                        # max_eig = max_eig_app.run()
                        # max_eig = max_eig_app.plot_eigenvalues()
                        # max_eig_app = None
                        # Sigpy:
                        max_eig = sp.app.MaxEig(AHA, dtype=self.y.dtype, device=self.device,
                                                max_iter=self.max_power_iter, show_pbar=self.show_pbar).run()
                        tau = self.primal_dual_scalar / max_eig 
                        self.tau = tau
                        
                        # Remove elements from _load_sketched_model_A_S() that are no longer needed
                        self.mps_S, self.A_S, self.y_S, AHA, self.T_S = None, None, None, None, None
                        self._clear_gpu_memory()

                    G = self.G
                    S = sp.linop.Multiply(G.oshape, tau)
                    GHG = G.H * S * G

                    # Joey's custom MaxEig:
                    # max_eig_app = me.MaxEig(GHG, dtype=self.y.dtype, device=self.device,
                    #                     max_iter=self.max_power_iter,show_pbar=self.show_pbar)
                    # max_eig = max_eig_app.run()
                    # max_eig = max_eig_app.plot_eigenvalues()
                    # max_eig_app = None
                    # Sigpy:
                    max_eig = sp.app.MaxEig(GHG, dtype=self.y.dtype, device=self.device,
                                            max_iter=self.max_power_iter, show_pbar=self.show_pbar).run()
                    sigma = self.primal_dual_scalar * 2 * 0.5 / max_eig # TODO: Check if making 1/x helps, default 0.5/x
                    print(f'sigma = self.primal_dual_scalar * 2 * 0.5/max_eig = {sigma}')
                    self.sigma = sigma
                    
                    # Free GPU memory
                    max_eig_app, G, S, GHG = None, None, None, None
                    del max_eig_app, G, S, GHG
                    self._clear_gpu_memory()


                elif self.tau is None:
                    # BUG: This code is not implemented properly. That said, it never gets called anyway. Need to remove.
                    print("self.tau is None... going down the corresponding loop (NOT IMPLEMENTED PROPERLY??)")
                    self._make_sketched_model_A()
                    S = sp.linop.Multiply(self.A_S.oshape, self.sigma)
                    AHA = self.A_S.H * S * self.A_S

                    # Joey's custom MaxEig:
                    # max_eig_app = me.MaxEig(AHA, dtype=self.y.dtype, device=self.device,
                    #                     max_iter=self.max_power_iter,show_pbar=self.show_pbar)
                    # max_eig = max_eig_app.run()
                    # max_eig = max_eig_app.plot_eigenvalues()
                    # max_eig_app = None
                    # Sigpy:
                    max_eig = sp.app.MaxEig(AHA, dtype=self.y.dtype, device=self.device,
                                            max_iter=self.max_power_iter, show_pbar=self.show_pbar).run()

                    tau = self.primal_dual_scalar * 1 / max_eig
                    self.tau = tau
                    
                    # Free GPU memory
                    max_eig_app, self.A_S, self.y_S, AHA = None, None, None, None
                    del max_eig_app, self.A_S, self.y_S, AHA
                    self._clear_gpu_memory()
                
                # Free GPU memory
                self._clear_gpu_memory()
                
                # Print the primal-dual step sizes
                print("primal-dual step sizes:")
                print(f"tau = {self.tau}")
                print(f"sigma = {self.sigma}")
                print(f"tau * sigma * ||M||^2 = {self.tau * self.sigma * max_eig}")
                print(f"tau_init = {self.tau_init}")
                print(f"sigma_init = {self.sigma_init}")
                print(f"tau_init * sigma_init * ||M||^2 = {self.tau_init * self.sigma_init * max_eig}")
                
                # Clear memory one last time
                print("Clearing memory after _get_init_PrimalDualHybridGradient()")
                with cp.cuda.Device(self.device):
                    tic = time.time()
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                    cp._default_memory_pool.free_all_blocks()
                    cp.cuda.Stream.null.synchronize()
                    cp.cuda.stream.get_current_stream().synchronize()
                    gc.collect()
                    toc = time.time()
                    print(f"Time taken to clear memory: {toc - tic:.2f} seconds")
                

                if self.alg is None:
                    self._get_PrimalDualHybridGradient()
        return
    
    def _make_sketched_models_A_S(self):
        raise NotImplementedError

    def _make_initial_sketched_problem(self):
        raise NotImplementedError
    
    def _clear_gpu_memory(self):
        raise NotImplementedError

    def _get_init_GradientMethod(self):
        
        # Free GPU memory
        self._clear_gpu_memory()
                   
        with self.device:
            AHy = self.A_S.H(self.y_S) # Adds more memory than just re-using self.AHy, but mathematically more accurate 
            
            def gradf(x):
                with self.device:
                    print(f'self.device inside get_init_GradientMethod = {self.device}')

                    if self.toeplitz:
                        gradf_x = self.T_S(x) - AHy 
                        # gradf_x = self.T_S(x) - self.AHy
                    else:
                        gradf_x = self.A_S.N(x) - AHy
                        # gradf_x = self.A_S.N(x) - self.AHy
                    
                    if self.lamda != 0:
                        print("self.lambda != 0")
                        if self.z is None:
                            util.axpy(gradf_x, self.lamda, x)
                        else:
                            util.axpy(gradf_x, self.lamda, x - self.z)
                            
                    return gradf_x
                        
                                
        # Free GPU memory
        self._clear_gpu_memory()
        
        self.alg = GradientMethod(
            gradf,
            self.x,
            alpha=self.alpha_init,
            proxg=self.proxg,
            max_iter=self.max_iter,
            accelerate=self.accelerate,
            tol=self.tol,
        )
            
        # Free GPU memory
        self._clear_gpu_memory()
            
            
        
     
        
    def _get_init_ConjugateGradient(self):
        if self.toeplitz:
            AHA = self.T_S
            print("_get_init_ConjugateGradient: AHA approximated using the Toeplitz PSF.")
        else:
            AHA = self.A_S.N 
        
        with self.device:
            AHy = self.A_S.H(self.y_S)
            print(f'linop: {self.A_S.H}')
        
        if self.lamda != 0:
            AHA += (self.lamda) * linop.Identity(self.x.shape)

        print(f'self.AHy device = {sp.get_device(self.AHy)}')
        print(f'self.x device = {sp.get_device(self.x)}')
        self.alg = ConjugateGradient(
            AHA, AHy, self.x, P=None, max_iter=self.max_iter, tol=self.tol
        )
        
        

    def _get_init_PrimalDualHybridGradient(self):
        
        # Free GPU memory
        self._clear_gpu_memory()
        
        gamma_primal = 0
        gamma_dual = 0

        with self.y_device:
            print(f'self.y_device = {self.y_device}')
        with self.x_device:
            print(f'self.x_device inside init_PrimalDualHybridGradient= {self.x_device}')
            if self.G is None:
                self.G = sp.linop.Identity(self.x0.shape)

            if self.toeplitz:
                AHA = self.T_S
                print("_get_init_PrimalDualHybridGradient: AHA approximated using the Toeplitz PSF.")
            else:
                AHA = self.A_S.N
            I = sp.linop.Identity(self.x.shape)
            # Option 1: use self.AHy (all coils) as b --> technically less accurate but may save memory as we can clear self.y_S
            print("WARNING: using self.AHy inside the PDHG call for b, in an attempt to save memory, at potential accuracy hit (or maybe improvement?)") # TODO
            AHy = self.AHy # Consider moving to inside the CG function call 
            
            # Option 2: use sketched AHy as b --> technically more accurate as we are solving the model for what it expects
            # AHy = self.A_S.H(self.y_S) # Adds more memory than just re-using self.AHy, but mathematically more accurate 
                
            def proxfc(sigma, v):
                return v - sigma*self.proxg(1/sigma, v/sigma)

            def proxg(tau, v):
                print(f"Forcing PDHG proxg iteration loop to have {self.max_cg_iter_pdhg} iterations.")
                sp.app.App(
                    sp.alg.ConjugateGradient(A = AHA + (1/tau)*I,  
                                            b = AHy + v/tau, 
                                            x = v,
                                            max_iter=self.max_cg_iter_pdhg), 
                    show_pbar=False).run()
                return v

        with self.x_device:
            u = self.x_device.xp.zeros(self.G.oshape, dtype=self.x.dtype)

        self.alg = sp.alg.PrimalDualHybridGradient(
            proxfc,
            proxg,
            self.G,
            self.G.H,
            self.x,
            u,
            self.tau_init,
            self.sigma_init,
            gamma_primal=gamma_primal,
            gamma_dual=gamma_dual,
            max_iter=self.max_iter,
            tol=self.tol)
        
        self._clear_gpu_memory()
        
        

    def _update_alg(self):
        if self.solver == 'GradientMethod':
            if self.G is not None:
                raise ValueError('GradientMethod cannot have G specified.')
            self._get_GradientMethod()
        elif self.solver == 'PrimalDualHybridGradient':
            self._get_PrimalDualHybridGradient()
        elif self.solver == 'ConjugateGradient':
            self._get_ConjugateGradient()
            print(f'Using solver: {self.solver}')
        else:
            raise ValueError('Invalid solver: {solver}.'.format(
                solver=self.solver))
        return
    
    
    
    def _get_ConjugateGradient(self):
        I = linop.Identity(self.x.shape)
        if self.toeplitz:
            AHA = self.T_S
            print("_get_ConjugateGradient: AHA approximated using the Toeplitz PSF.")
        else:
            AHA = self.A_S.N
        
        AHy = AHA(self.x0) - self.d

        if self.lamda != 0:
            AHA += (self.lamda)* I

        if self.alg is not None:
            self.iter = self.alg.iter
        
        self.alg = ConjugateGradient(
            AHA, AHy, self.x, P=None, max_iter=self.max_iter)
        self.alg.iter = self.iter

        return
    

    def _get_GradientMethod(self):
        cp.cuda.Device().synchronize()
        cp._default_memory_pool.free_all_blocks()
        print(gc.collect())
        
        # First iteration
        self.x -= np.min(self.alpha) * self.d
        if self.proxg is not None:
            print("******** WARNING: PERFORMING THE PROXIMAL OPERATOR PRIOR TO THE FIRST ITERATION.... POSSIBLY NOT NEEDED!!!!! *********")
            self.x = self.proxg(np.min(self.alpha), self.x)
                
        self._clear_gpu_memory()
        
        def gradf(x):
            with self.device:
                if self.toeplitz:
                    gradf_x = self.T_S(x - self.x0) + self.d
                else:
                    gradf_x = self.A_S.N(x - self.x0) + self.d

                if self.lamda != 0:
                    if self.z is None:
                        util.axpy(gradf_x, self.lamda, x)
                    else:
                        util.axpy(gradf_x, self.lamda, x - self.z)
                        
                return gradf_x

        if self.alg is not None:
            self.iter = self.alg.iter

        self.alg = GradientMethod(
                            gradf,
                            self.x,
                            np.min(self.alpha),
                            proxg=self.proxg,
                            max_iter=self.max_iter,
                            accelerate=self.accelerate)
        self.alg.iter = self.iter
        
        # Remove variables
        gradf = None
        del gradf
        
        cp.cuda.Device().synchronize()
        cp._default_memory_pool.free_all_blocks()
        print(gc.collect())
        return

    def _get_PrimalDualHybridGradient(self):
        self._clear_gpu_memory()

        with self.device:
            if self.G is None:
                self.G = sp.linop.Identity(self.x0.shape)

            if self.toeplitz:
                H_S = self.T_S
                print("_get_PrimalDualHybridGradient: AHA approximated using the Toeplitz PSF.")
            else:
                H_S = self.A_S.N
            I = sp.linop.Identity(self.x0.shape)
            b = H_S(self.x0) - self.d
            
            def proxFc(sigma, v):
                return v - sigma*self.proxg(1/sigma, v/sigma) 
            
            def proxG(tau, v):
                print(f"PDHG proxg iteration loop to have {self.max_cg_iter_pdhg} iterations.")
                sp.app.App(
                    sp.alg.ConjugateGradient(A = H_S + (1/tau)*I,  
                                            b = b + v/tau, 
                                            x = v,
                                            max_iter=self.max_cg_iter_pdhg), 
                    show_pbar=False).run()
                return v
        
        self._clear_gpu_memory()
        
        if self.alg is not None:
            self.iter = self.alg.iter

        gamma_primal = 0
        gamma_dual = 0
        with self.device:
            u = self.device.xp.zeros(self.G.oshape, dtype=self.x.dtype)
        self.alg = PrimalDualHybridGradient(
            proxFc,
            proxG,
            self.G,
            self.G.H,
            self.x,
            u,
            self.tau,
            self.sigma,
            gamma_primal=gamma_primal,
            gamma_dual=gamma_dual,
            max_iter=self.max_iter)
        self.alg.iter = self.iter # JWP + 1
        
        u = None
        del u
        self._clear_gpu_memory()
        
        # # Optional: save x after each iteration
        if self.save_iterates:
            folder = os.getcwd()
            results_exist = os.path.exists(folder + "/tmp")

            # Create a new directory because the results path does not exist
            if not results_exist:
                os.makedirs(folder + "/tmp")
                print("A new directory inside: " + folder +
                        " called 'tmp' has been created.")

            # Save images as Nifti files
            # Custom affine
            aff = np.array([[0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1]])
            ni_img = nib.Nifti1Image(abs(np.moveaxis(self.x[0:2,...].get(), 0, -1)), affine=aff) # First two phases only
            nib.save(ni_img, folder + '/tmp/tmp_pdhg')
            print(f'Saved the current iterate {self.alg.iter} to {folder}/tmp/tmp_pdhg.nii.gz')

        return

    def _pre_update(self):

        if self.alg.iter >= self.max_init_iter:
            if (self.alg.iter - self.max_init_iter) % self.max_inner_iter == 0 :
                backend.copyto(output=self.x0, input=self.x)
                self._get_true_gradient()
                
                self._load_sketched_model_A_S()
                self._update_alg()
                self.outer_iter += 1
                
                # Clear memory one last time
                print("Clearing memory after _get_true_gradient()")
                with cp.cuda.Device(self.device):
                    tic = time.time()
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                    cp._default_memory_pool.free_all_blocks()
                    cp.cuda.Stream.null.synchronize()
                    cp.cuda.stream.get_current_stream().synchronize()
                    gc.collect()
                    toc = time.time()
                    print(f"Time taken to clear memory: {toc - tic:.2f} seconds")
        return
    
