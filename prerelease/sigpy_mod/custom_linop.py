import sigpy as sp
import numpy as np
from sigpy.linop import Linop
from sigpy import backend, fourier
import sigpy.plot as pl
import registration_oflow3D
from cupyx.scipy.ndimage import map_coordinates
from cupyx.scipy.ndimage import gaussian_filter
import cupy as cp

import gc
import ants

class NFT(Linop):
    """NFT linear operator.

    Args:
        ishape (tuple of int): Input shape.
        coord (array): Coordinates, with values [-ishape / 2, ishape / 2]
        device (sp.Device): The device to perform the operation on.
        toeplitz (bool): Use toeplitz PSF to evaluate normal operator.
    """

    def __init__(self, ishape, coord, oversamp=1.5, width=4, device=sp.Device(-1), toeplitz=False):
        self.coord = coord
        self.device = device
        self.n_Channel = ishape[0]
        self.oversamp = oversamp
        self.width = width
        self.toeplitz=toeplitz
        
        oshape = list((self.n_Channel,)) + list(coord.shape[:-1])

        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = sp.backend.get_device(input)
        # device = sp.Device(6)

        with device:
            coord = sp.backend.to_device(self.coord, device)
            # coord = np.array(self.coord)
            nft = sp.linop.NUFFT(self.ishape[1:], coord=coord, oversamp=self.oversamp, width = self.width) 
            # nft_op = Diags([DLD(nft, device) for _ in range(self.n_Channel)], self.oshape, self.ishape)
            nft_op = Diags([nft for _ in range(self.n_Channel)], self.oshape, self.ishape)
            return nft_op(input)
  

    def _adjoint_linop(self):
        return NFTAdjoint(self.ishape, self.coord, oversamp=self.oversamp, width=self.width, device=self.device)
    

class NFTAdjoint(Linop):
    """NFT adjoint linear operator.

    Args:
        oshape (tuple of int): Output shape.
        coord (array): Coordinates, with values [-ishape / 2, ishape / 2].
        device (sp.Device): The device to perform the operation on.
    """

    def __init__(self, oshape, coord, oversamp=1.5, width=4, device=sp.Device(-1)):
        self.coord = coord
        self.device = device
        self.n_Channel = oshape[0]
        self.oversamp = oversamp
        self.width = width

        ishape = list((self.n_Channel,)) + list(coord.shape[:-1])

        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = sp.backend.get_device(input)
        # device = sp.Device(6)
        with device:
            coord = sp.backend.to_device(self.coord, device) # JWP moved the device switching to here to see if it can be made faster
            # coord = np.array(self.coord)
            nft_adj = sp.linop.NUFFTAdjoint(self.oshape[1:], coord=coord, oversamp=self.oversamp, width = self.width)
            # nft_adj_op = Diags([DLD(nft_adj, device) for _ in range(self.n_Channel)], self.oshape, self.ishape)
            nft_adj_op = Diags([nft_adj for _ in range(self.n_Channel)], self.oshape, self.ishape)
            return nft_adj_op(input)

    def _adjoint_linop(self):
        return NFT(self.oshape, self.coord, oversamp=self.oversamp, width = self.width, device=self.device)
    
    

    

def Sense4D(mps, 
            coord, 
            weights=None,
            device=sp.Device(-1),
            coil_batch_size=None, 
            b0_map = None,
            ):
    """Sense4D: A linear operator to apply sensitivity maps, Fourier encoding, and k-space preconditioning for 4D MRI. 

    Args:
        mps (np.ndarray): Coil sensitivity maps (Nt,Nc,Nx,Ny,Nz) or (Nc,Nx,Ny,Nz).
        coord (np.ndarray): k-space coordinates (Nt, Nexc, Nsamp, Ndim).
        weights (np.ndarray): k-space preconditioning weights (Nt, Nexc, Nsamp).
        device (CUDA device, optional): GPU/CPU device. Defaults to sp.Device(-1).
        coil_batch_size (int, optional): Coil batch size. Defaults to None.
        b0_map (np.ndarray): Off resonance map (Hz) of shape (Nt,Nx,Ny,Nz).

    Returns:
        A: 4D encoding operator. 
    """    
    import sys
    sys.path.append('../sigpy_mod')
    import linop_e
    import b_ct
    
    # Inputs
    As = []
    if len(mps.shape) == 4:
        nc=mps.shape[0]
        print("Using the same CSM for all respiratory phases.")
    elif len(mps.shape) == 5:
        print("Using a unique CSM for each respiratory phase.")
        nc=mps.shape[1]
    nphase, nexc, nread, ndim = coord.shape
    img_shape=mps.shape[-ndim:] 
    
    
    # Serialize linop if coil_batch_size is smaller than num_coils.
    if coil_batch_size is None:
        coil_batch_size = nc
        print(f"Automated coil batch size = {coil_batch_size} set to nc = {nc} inside Sense4D().")
    else:
        print(f"Coil batch size of {coil_batch_size} coils supplied to Sense4D().")

    if coil_batch_size < nc:
        print(f"Coil batching with {coil_batch_size} coils out of {nc} total coils.")
        num_coil_batches = (nc + coil_batch_size - 1) // coil_batch_size
        if len(mps.shape) == 4:
            A = sp.linop.Vstack(
                [
                    Sense4D(
                        mps[c * coil_batch_size : ((c + 1) * coil_batch_size), ...],
                        coord=coord,
                        weights=weights,
                        device=device
                    )
                    for c in range(num_coil_batches)
                ],
                axis=1,
            )
        elif len(mps.shape) == 5:
            A = sp.linop.Vstack(
                [
                    Sense4D(
                        mps[:, c * coil_batch_size : ((c + 1) * coil_batch_size), ...],
                        coord=coord,
                        weights=weights,
                        device=device
                    )
                    for c in range(num_coil_batches)
                ],
                axis=1,
            )
        return A
        
    for resp in range(nphase):
        # Homemade operator (consumes LESS memory than SigPy because it loops over coils rather than doing in one big batch)
        # F = NFT(ishape=(nc,) + img_shape,
        #                 coord=coord[resp,...],
        #                 oversamp=1.5,
        #                 width=4, # 4 default
        #                 device=device)
        
        # Homemade operator - slower but more accurate
        # F = NFT(ishape=(nc,) + img_shape,
        #                 coord=coord[resp,...],
        #                 oversamp=2,
        #                 width=3, # 4 default
        #                 device=device)
        
        # Sigpy operator (consumes more memory as it does not loop over channels like the custom operator. Slower in some cases, faster in others)
        F = sp.linop.NUFFT(ishape=(nc,) + img_shape,
                        coord=coord[resp,...],
                        oversamp=1.75,
                        width=3)
        
        # Apply B0 map (warning, slows down recon)
        if b0_map is not None:
            print("USING B0 MAP INSIDE RECON: WARNING: SLOW")
            F = b_ct.tseg_b_ct_gpu(F, 
                                b0 = b0_map[resp,...], 
                                bins = 10,
                                lseg = 10,
                                readout=4520-6,
                                devnum = device)
        

        if len(mps.shape) == 5: # (Nt,Nc,Nx,Ny,Nz)
            S = sp.linop.Multiply(ishape = (img_shape), 
                            mult=mps[resp,...])
        else: # (Nc,Nx,Ny,Nz)
            S = sp.linop.Multiply(ishape = (img_shape), 
                            mult=mps)
            
        if weights is not None:
            D = sp.linop.Multiply(F.oshape, weights[resp,...]**0.5) 
            A = D * F * S    
        else:
            A = F * S
        
        As.append(A)

    As_oshape = (nphase,) + (nc,) + (nexc, nread,)
    As = Diags(As, 
                    oshape=As_oshape,
                    ishape=(nphase,) + img_shape)
    return As



# def Toeplitz_Normal(mps, coord, psf):
#     """Perform the Toeplitz normal operator using your pre-measured Toeplitz PSF.

#     While fast, this is more memory intensive, and only correct to within +-1%.

#     Args:
#         mps (array): Sensitivity maps of size (nphase, nchannel, image_size) or (nchannel, image_size).
#         coord (array): Fourier domain coordinate array of shape (..., ndim).
#             ndim determines the number of dimension to apply nufft adjoint.
#             coord[..., i] should be scaled to have its range between
#             -n_i // 2, and n_i // 2.
#         psf (array): Toeplitz point spread function of size (nphase, image_size) (nchannel not included as it behaves the same for each channel)
 
#     Returns:
#         array: PSF to be used by the normal operator defined in
#             `sigpy.linop.NUFFT`

#     See Also:
#         :func:`sigpy.linop.NUFFT`

#     """
    
#     # Inputs
#     if len(mps.shape) == 4:
#         nc=mps.shape[0]
#     elif len(mps.shape) == 5:
#         nc=mps.shape[1]
#     nphase, nexc, nread, ndim = coord.shape
#     img_shape = mps.shape[-ndim:]
#     psf_resp_shape = psf.shape[-ndim:] # (2*Nx,2*Ny,2*Nz)
#     device = sp.backend.get_device(mps)

#     T_resp = []
#     for resp in range(nphase):    
        
#         # Temporarily store psf (for each phase) on the same device as mps for speed
#         psf_tmp = sp.backend.to_device(psf[resp,...], device=device)
#         # psf_tmp = psf[resp,...]
        
#         # Compute Toeplitz operator
#         # fft_axes = tuple(range(-1, -(ndim + 1), -1))
#         R = sp.linop.Resize(psf_resp_shape, img_shape)
#         # F = sp.linop.FFT(psf.shape, axes=fft_axes)
#         F = sp.linop.FFT(psf_resp_shape, axes=None)
#         # P = sp.linop.Multiply(psf_resp_shape, psf[resp,...])
#         P = sp.linop.Multiply(psf_resp_shape, psf_tmp)
#         if len(mps.shape) == 5: # (Nt,Nc,Nx,Ny,Nz)
#             S = sp.linop.Multiply(ishape = (img_shape), 
#                             mult=mps[resp,...])
#         else: # (Nc,Nx,Ny,Nz)
#             S = sp.linop.Multiply(ishape = (img_shape), 
#                             mult=mps)
#         T = R.H * F.H * P * F * R
#         T_op = Diags([T for _ in range(nc)], 
#                      (nc,) +  img_shape, 
#                      (nc,) + img_shape)
#         T_resp.append(S.H * T_op * S)
#     T_all = Diags(T_resp, 
#             oshape=(nphase,) + img_shape,
#             ishape=(nphase,) +img_shape)
    
#     return T_all



def Toeplitz_Normal(mps, coord, psf):
    """Perform the Toeplitz normal operator using your pre-measured Toeplitz PSF.

    While fast, this is more memory intensive, and only correct to within +-1%.

    Args:
        mps (array): Sensitivity maps of size (nphase, nchannel, image_size) or (nchannel, image_size).
        coord (array): Fourier domain coordinate array of shape (..., ndim).
            ndim determines the number of dimension to apply nufft adjoint.
            coord[..., i] should be scaled to have its range between
            -n_i // 2, and n_i // 2.
        psf (array): Toeplitz point spread function of size (nphase, image_size) (nchannel not included as it behaves the same for each channel)
 
    Returns:
        array: PSF to be used by the normal operator defined in
            `sigpy.linop.NUFFT`

    See Also:
        :func:`sigpy.linop.NUFFT`

    """
    
    # Inputs
    if len(mps.shape) == 4:
        nc=mps.shape[0]
    elif len(mps.shape) == 5:
        nc=mps.shape[1]
    nphase, nexc, nread, ndim = coord.shape
    img_shape = mps.shape[-ndim:]
    psf_resp_shape = (nc,) + psf.shape[-ndim:] # (2*Nx,2*Ny,2*Nz)
    device = sp.backend.get_device(mps)

    T_resp = []
    for resp in range(nphase):    
        
        # Temporarily store psf (for each phase) on the same device as mps for speed
        psf_tmp = sp.backend.to_device(psf[resp,...], device=device)
        # psf_tmp = psf[resp,...]
        
        # Compute Toeplitz operator
        # fft_axes = tuple(range(-1, -(ndim + 1), -1)) 
        R = sp.linop.Resize(psf_resp_shape, mps.shape)
        # F = sp.linop.FFT(psf.shape, axes=fft_axes)
        fft_axes = tuple(range(-1, -(ndim + 1), -1)) 
        F = sp.linop.FFT(psf_resp_shape, axes=fft_axes)
        # P = sp.linop.Multiply(psf_resp_shape, psf[resp,...])
        P = sp.linop.Multiply(psf_resp_shape, psf_tmp)
        if len(mps.shape) == 5: # (Nt,Nc,Nx,Ny,Nz)
            S = sp.linop.Multiply(ishape = (img_shape), 
                            mult=mps[resp,...])
        else: # (Nc,Nx,Ny,Nz)
            S = sp.linop.Multiply(ishape = (img_shape), 
                            mult=mps)
        T = R.H * F.H * P * F * R
        T_resp.append(S.H * T * S)
    T_all = Diags(T_resp, 
            oshape=(nphase,) + img_shape,
            ishape=(nphase,) +img_shape)
    
    return T_all
    

def weighted_toeplitz_psf(coord, weights, shape, oversamp=1.25, width=4):
    """Toeplitz PSF for fast Normal non-uniform Fast Fourier Transform.

    While fast, this is more memory intensive.

    Args:
        coord (array): Fourier domain coordinate array of shape (..., ndim).
            ndim determines the number of dimension to apply nufft adjoint.
            coord[..., i] should be scaled to have its range between
            -n_i // 2, and n_i // 2.
        weigths (array): k-space weights of same size as coord, minus the final dimension.
        shape (tuple of ints): shape of the form
            (..., n_{ndim - 1}, ..., n_1, n_0).
            This is the shape of the input array of the forward nufft.
        oversamp (float): oversampling factor. Make this match the NUFFT operator used by Sense4D.
        width (float): interpolation kernel full-width in terms of
            oversampled grid.

    Returns:
        array: PSF to be used by the normal operator defined in
            `sigpy.linop.NUFFT`

    See Also:
        :func:`sigpy.linop.NUFFT`

    """
    xp = backend.get_array_module(weights)
    with backend.get_device(weights):
        ndim = coord.shape[-1]

        new_shape = fourier._get_oversamp_shape(shape, ndim, 2)
        new_coord = fourier._scale_coord(coord, new_shape, 2)

        idx = [slice(None)] * len(new_shape)
        for k in range(-1, -(ndim + 1), -1):
            idx[k] = new_shape[k] // 2

        d = xp.zeros(new_shape, dtype=xp.complex64) # Default
        # d = xp.zeros(new_shape, dtype=xp.complex128) # Higher precision, more memory
        d[tuple(idx)] = 1

        psf_ksp = fourier.nufft(d, new_coord, oversamp, width)
        psf_img = fourier.nufft_adjoint(weights*psf_ksp, new_coord, d.shape, oversamp, width)
        psf_fft = fourier.fft(psf_img, axes=None, norm=None) * (2**ndim)

        return psf_fft
    
    
def Diags(L_Linop, oshape, ishape):
    # Reshape input, apply diagonalized linop, reshape output
    assert oshape[0] == ishape[0], 'First dim mismatch!'
    assert oshape[0] == len(L_Linop), 'Number of Linop mismatch!'
    Linops = sp.linop.Diag(L_Linop) # This vectorizes the output, so we need to do some additional reshaping
    
    # Additional reshaping, to make the output match the correct form
    i_vec_len = 1
    for tmp in ishape:
        i_vec_len = i_vec_len * tmp
    o_vec_len = 1
    for tmp in oshape:
        o_vec_len = o_vec_len * tmp

    R1 = sp.linop.Reshape(oshape=(o_vec_len,), ishape=oshape)
    R2 = sp.linop.Reshape(oshape=(i_vec_len,), ishape=ishape)
    Linops = R1.H*Linops*R2

    return Linops

def DLD(Linop, device=sp.Device(-1)):
    # Transform CPU input to GPU, perform Linop, then transform GPU output into CPU.
    B1 = sp.linop.ToDevice(Linop.ishape, idevice=sp.Device(-1), odevice=device)
    B2 = sp.linop.ToDevice(Linop.oshape, idevice=sp.Device(-1), odevice=device)
    Linop = B2.H*Linop*B1
    return Linop


class RegisterImages(Linop):
    def __init__(self, ishape, ref_index=0, filter_size=25, devnum=0):
        """Linear operator that computes the motion fields between respiratory states and applies them once called.

        Args:
            ishape (tuple of ints): Input shape (Nphase, Nx, Ny, Nz).
            ref_index (int): Reference index to register to.
            filter_size (int): Registration filter size.
            devnum (int): GPU device number.
        """
        self.ishape = ishape
        self.ref_index = ref_index
        self.filter_size = filter_size
        self.devnum = devnum
        
        # Initialize the operator
        oshape = ishape  # Output shape is the same as input shape
        super().__init__(oshape, ishape)
        
        # Initialize the counter variable
        self.counter = 0
        
        # Initialize the deformation fields
        self.deformation_fields = np.zeros((ishape + (3,)))
        
        # Use Jacobian
        self.jacobian_experiment = True # TODO: make robust for forward and adjoint operations
        self.jacobian = np.ones((ishape)) # Initial estimate
    
    def _apply(self, input):
        """Apply the motion fields operator on the input data."""
        
        img_shape = self.ishape[1:]
        nphase = self.ishape[0]
        
        # Compute the deformation fields using the input data
        counter_threshold = 0 # Warning, setting this to > 0 may throw off the MaxEig calculation and then damage the primal-dual algorithm stability.
        if self.counter >= counter_threshold: 
            _, deformation_fields = registration_oflow3D.register_images(
                input, ref_index=self.ref_index, filter_size=self.filter_size, gpu_id=self.devnum
            )
        else:
            print(f"Skipping registration as self.counter < {counter_threshold}...")
            deformation_fields = np.zeros((nphase, len(img_shape), img_shape[0], img_shape[1], img_shape[2]), dtype=float)
        deformation_fields = np.moveaxis(deformation_fields, 1, -1)  # Move the dimension axis to the end
        if self.jacobian_experiment:
            print(f"deformation_fields.shape: {deformation_fields.shape}")
            jacobian_input = abs(input.get())
            
        # Joey experiment: apply filtering to the deformation fields along the respiratory dimension to be robust to noisy measurements
        gaussian_smooth_motion = False
        if gaussian_smooth_motion:
            resp_dim = 0
            print(f"Gaussian filtering the deformation fields along dimension {resp_dim}...")
            # Sigma for each dimension: (temporal, Nx, Ny, Nz, dimensions)
            sd = 1 # Standard deviation
            sigma = (sd, 0, 0, 0, 0)  # Smooth only across temporal dimension
            deformation_fields = gaussian_filter(deformation_fields, sigma=sigma) # MIGHT NOT WORK WITH CUPY
            
        polynomial_smooth_motion = True    
        if polynomial_smooth_motion and self.counter >= counter_threshold:            
            resp_dim = 0
            print(f"Polynomial fitting the deformation fields along dimension {resp_dim}...")
            
            # Define the temporal axis and reshape deformation fields for polynomial fitting
            temporal_axis = np.arange(deformation_fields.shape[resp_dim])
            n_temporal = deformation_fields.shape[resp_dim]
            spatial_shape = deformation_fields.shape[1:]  # Nx, Ny, Nz, dimensions
            deformation_fields_flat = deformation_fields.reshape(n_temporal, -1)  # Flatten spatial dimensions
            
            # Fit a polynomial to each 1D deformation field along the temporal axis
            order = 3  # Change to 2 for a second-order fit. 3 is robust to cases where motion is not convex, e.g. if ref_index is not in the middle.
            coeffs = np.polyfit(temporal_axis, deformation_fields_flat, order)
            
            # Evaluate the polynomial for all temporal points
            poly_vals = np.polyval(coeffs, temporal_axis[:, None])
            
            # Optional: Preserve fixed points (e.g., where deformation is zero)
            fixed_points = (deformation_fields_flat == 0)
            poly_vals[fixed_points] = 0
            
            # Reshape the smoothed deformation fields back to the original shape
            deformation_fields = poly_vals.reshape(deformation_fields.shape)    
        
        # Create the list of motion field operators
        Ms = []
        for resp in range(nphase):
            if self.jacobian_experiment:
                self.jacobian[resp, ...] = calculate_jacobian_oflow(jacobian_input[self.ref_index,...], deformation_fields[(self.ref_index, resp), ...])
                # self.jacobian[resp, ...] = compute_jacobian_determinant(deformation_fields[(resp), ...])
                print(f'jacobian.shape: {self.jacobian.shape}')
                J = sp.linop.Multiply(ishape=img_shape, mult=self.jacobian[resp,...])
            
            M_resp = interp_op(ishape=img_shape, M_field=deformation_fields[resp, ...])
            # TODO: just use map coordinates instead
            if self.jacobian_experiment:
                Ms.append(J * M_resp)
                
            else:
                Ms.append(M_resp)
            
            

        # Combine them into a single operator
        M = Diags(Ms, oshape=[nphase] + img_shape, ishape=[nphase] + img_shape)
        
        self.counter += 1
        # print(f'self.counter = {self.counter}')
        
        # Save the deformation fields in self so that the adjoint can be called:
        self.deformation_fields = deformation_fields
        print("CAUTION: self.deformation_fields updated.")

        return M(input)  # Apply the combined operator to the input

    def _adjoint_linop(self):
        # Define the adjoint of the motion fields operator if needed
        # print("WARNING: The adjoint operator was called, meaning that the negative of the measurement motion fields is being applied. ")
        # print("RegisterImages does not behave like a unitary operator here, as it re-measures the deformation every time it is called.")
        # return -RegisterImages(self.ishape, self.ref_index, self.filter_size, self.devnum)

        print("WARNING: APPLYING THE INVERSE DEFORMATION FIELDS.")
        if self.jacobian_experiment:
            return ApplyDeformation(self.ishape, sp.to_device(-self.deformation_fields, self.devnum)) * sp.linop.Multiply(self.ishape, mult=1/self.jacobian)
        else:
            return ApplyDeformation(self.ishape, sp.to_device(-self.deformation_fields, self.devnum))

class interp_op(Linop):
    def __init__(self, ishape, M_field):
        ndim = M_field.shape[-1]
        assert list(ishape) == list(M_field.shape[:-1]), "Dimension mismatch!"
        oshape = ishape
        self.M_field = M_field
        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = backend.get_device(input)

        with device:
            return interp(input, self.M_field, device)

    def _adjoint_linop(self):
        device = backend.get_device(input)
        iM_field = -self.M_field
        M_field = None

        return interp_op(self.ishape, iM_field, M_field)
    

def interp(I, M_field, device=sp.Device(-1), k_id=1, deblur=True):
    # b spline interpolation
    N = 64
    if k_id == 0:
        kernel = [(3*(x/N)**3-6*(x/N)**2+4)/6 for x in range(0, N)] + \
            [(2-x/N)**3/6 for x in range(N, 2*N)]
        dkernel = np.array([-.2, 1.4, -.2])

        k_wid = 4
    else:
        kernel = [1-x/(2*N) for x in range(0, 2*N)]
        dkernel = np.array([0, 1, 0])
        deblur = False # False # TODO: check, default false
        if deblur:
            dkernel = sp.to_device(np.array([0, 1, 0], dtype=float), device)
        k_wid = 2
    kernel = np.asarray(kernel) # Does not appear to be used

    c_device = sp.get_device(I)
    ndim = M_field.shape[-1]

    # 2d/3d
    if ndim == 3:
        dkernel = dkernel[:, None, None] * \
            dkernel[None, :, None]*dkernel[None, None, :]
        Nx, Ny, Nz = I.shape
        my, mx, mz = np.meshgrid(np.arange(Ny), np.arange(Nx), np.arange(Nz))
        m = np.stack((mx, my, mz), axis=-1)
        M_field = M_field + m
    else:
        dkernel = dkernel[:, None]*dkernel[None, :]
        Nx, Ny = I.shape
        my, mx = np.meshgrid(np.arange(Ny), np.arange(Nx))
        m = np.stack((mx, my, mz), axis=-1)
        M_field = M_field + m
    # TODO remove out of range values

    # image warp
    g_device = device
    I = sp.to_device(input=I, device=g_device)
    I = sp.interp.interpolate(
        I, width=k_wid, kernel='spline', coord=sp.to_device(M_field.astype(np.float64), device))
    
    # Joey experiment
    # kernel = 'kaiser_bessel'
    # width = 2 # 4  # Typical width for Kaiser-Bessel
    # beta = 1 # 2.34 * width  # Rule of thumb: beta = 2.34 * width
    # I = sp.interp.interpolate(I, width=width, kernel=kernel, coord=sp.to_device(M_field.astype(np.float64), device), param=beta)

    # deconv
    if deblur is True:
        print("DEBLURRING")
        sp.conv.convolve(I, dkernel)
    I = sp.to_device(input=I, device=c_device)

    return I


class ApplyDeformation(sp.linop.Linop):
    """Linear operator that computes the motion fields between respiratory states. Here, the 
           operator requires deformation fields as an input, so that the inverse operator is simply the inverse deformation fields.

        Args:
            ishape (tuple of ints): Input shape (Nphase, Nx, Ny, Nz).
            deformation_fields (array): Deformation fields (Nphase, Nx, Ny, Nz, N_dimensions).
            ref_index (int): Reference index to register to.
            filter_size (int): Registration filter size.
    """
    def __init__(self, ishape, deformation_fields):
        # deformation_fields shape should be (nimages, nr, nc, nz, 3)
        self.deformation_fields = deformation_fields
        self.ishape = ishape
        self.oshape = ishape
        super().__init__(ishape, self.oshape)

    def _apply(self, images):
        print("WARNING: THIS IS ONLY BEING CALLED BECAUSE AN ADJOINT OPERATION WAS CALLED ELSEWHERE")
        
        # Apply deformation fields to images
        nimages, nr, nc, nz = images.shape
        output = cp.zeros_like(images)

        for i in range(nimages):
            vx = self.deformation_fields[i,:,:,:,0]
            vy = self.deformation_fields[i,:,:,:,1]
            vz = self.deformation_fields[i,:,:,:,2]
            x, y, z = cp.meshgrid(cp.arange(nr), cp.arange(nc), cp.arange(nz), indexing='ij')

            # Apply the deformation to the images (real values only)
            # coords = cp.array([x + vx, y + vy, z + vz])
            # output[i] = map_coordinates(images[i], coords, mode="nearest", order=3)
            
            # Apply deformation to both real and imaginary parts
            real_part = cp.real(images[i])
            imag_part = cp.imag(images[i])
            coords = cp.array([x + vx, y + vy, z + vz])
            real_output = map_coordinates(real_part, coords, mode="nearest", order=3)
            real_part = None
            imag_output = map_coordinates(imag_part, coords, mode="nearest", order=3)
            imag_part = None

            # Recombine real and imaginary parts
            output[i] = real_output + 1j * imag_output
        
        return output 

    # def _adjoint_linop(self):
    #     return ApplyDeformation(self.ishape, -self.deformation_fields)
        
        
        


def calculate_jacobian_oflow(fixed: np.ndarray, 
                             transform: np.ndarray, 
                             reference_index: int = 0) -> np.ndarray:
    """
    Calculate the Jacobian determinant of an image using a given forward transform (calculated by optical flow).

    Parameters:
    fixed (np.ndarray): The input image as a numpy array (fixed image in original transform).
    transform (np.ndarray): Deformation fields from Optical Flow (shape: Ntransforms, Nx, Ny, Nz, Ndim).
    reference_index (int): The reference index to register to.
    
    Returns:
    np.ndarray: The Jacobian determinant image as a numpy array.
    """
    # Convert the numpy array to an ANTsImage
    fixed_image = ants.from_numpy(fixed)
    
    # Create the Jacobian determinant image using the forward transform
    print("=" * 20 + f" Jacobian for reference index: {reference_index} " + "=" * 20)

    # Reshape into: (shape: Nx, Ny, Nz, Ndim, Ntransforms)
    Ntransforms, Nx, Ny, Nz, Ndim = transform.shape
    deform_fields=np.moveaxis(transform, [0, 1, 2, 3, 4], [4, 0, 1, 2, 3])
    # Explanation:
    # The axis originally at position 0 is moved to position 4.
    # The axis originally at position 1 is moved to position 0.
    # The axis originally at position 2 is moved to position 1.
    # The axis originally at position 3 is moved to position 2.
    # The axis originally at position 4 is moved to position 3.
    # print(f'Deformation field shape (after reshaping): {deform_fields.shape}')
    
    # n = transform index
    ants_deform_fields_oflow=[ants.from_numpy(np.squeeze(deform_fields[...,n]),has_components=True) for n in range(deform_fields.shape[-1])]
    jacobian_array = np.zeros((Ntransforms, Nx, Ny, Nz), dtype=float)
    for n in range(Ntransforms): # Iterate through number of transforms
        jacobian_array[n] = ants.create_jacobian_determinant_image(domain_image=fixed_image,
                                                                      tx=ants_deform_fields_oflow[n], 
                                                                      do_log=False, 
                                                                      geom=False).numpy()
        
    # Jacobian is the ratio of final volume to initial volume in a local sense. 
    # If the material is incompressible, all the motion (or deformation) the body undergoes should have J=1 In case of compressible materials, J>0; 
        
    return jacobian_array[-1]



def compute_jacobian_determinant(deformation_field):
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
    det_jacobian = xp.zeros((N, H, W, D), dtype=float)
    for phase in range(N):
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




class RegisterImagesWithJacobian(Linop):
    def __init__(self, ishape, ref_index=0, filter_size=25, devnum=0, compute_jacobian=True):
        # TODO: make more efficient by better handling GPU operations
        """Linear operator that computes the motion fields between respiratory states and applies them once called. 
            This operator also computes the Jacobian determinant of the motion fields and applies them if called.

        Args:
            ishape (tuple of ints): Input shape (Nphase, Nx, Ny, Nz).
            ref_index (int): Reference index to register to.
            filter_size (int): Registration filter size.
            devnum (int): GPU device number.
            compute_jacobian (bool): Compute Jacobian from the deformation fields (requires ANTs).
        """
        self.ishape = ishape
        self.ref_index = ref_index
        self.filter_size = filter_size
        self.devnum = devnum
        
        # Initialize the operator
        oshape = ishape  # Output shape is the same as input shape
        super().__init__(oshape, ishape)
        
        # Initialize the counter variable
        self.counter = 0
        
        # Initialize the deformation fields
        self.deformation_fields = np.zeros((ishape + (len(ishape)-1,))) # Assumes that len(ishape)-1 is the number of dimensions
        self.inverse_deformation_fields = np.zeros((ishape + (len(ishape)-1,))) # Assumes that len(ishape)-1 is the number of dimensions
        
        # Use Jacobian
        self.compute_jacobian = compute_jacobian
        self.jacobian = np.ones((ishape)) # Initial estimate
        self.inverse_jacobian = np.ones((ishape)) # Initial estimate
        
        # Use a polynomial fit to smooth the deformation fields
        self.polynomial_smooth_motion = False  
    
    def _apply(self, input):
        """Apply the motion fields operator on the input data."""
        
        img_shape = self.ishape[1:]
        nphase = self.ishape[0]
        ndim = len(img_shape)
        
        # Compute the deformation fields using the input data
        counter_threshold = 0 # Warning, setting this to > 0 may throw off the MaxEig calculation and then damage the primal-dual algorithm stability.
        if self.counter >= counter_threshold: 
            _, self.deformation_fields = registration_oflow3D.register_images(
                input, ref_index=self.ref_index, filter_size=self.filter_size, gpu_id=self.devnum
            )
            # OPTIONAL: smooth the deformation fields
            # print("WARNING: Smoothing the deformation fields along xyz dimensions.")
            # self.deformation_fields = gaussian_filter(sp.to_device(self.deformation_fields, self.devnum), sigma=(0, 0, 1.5, 1.5, 1.5)).get()
            
        else:
            print(f"Skipping registration as self.counter < {counter_threshold}...")
            self.deformation_fields = np.zeros((nphase, ndim, img_shape[0], img_shape[1], img_shape[2]), dtype=float)
            
        # Reshape the deformation fields to the correct shape
        self.deformation_fields = np.moveaxis(self.deformation_fields, 1, -1)  # Move the dimension axis to the end
            
        # Apply filtering/smoothing to the deformation fields to be robust to noisy measurements
        if self.polynomial_smooth_motion and self.counter >= counter_threshold:            
            print(f"Polynomial fitting the deformation fields along dimension 0...")
            
            # Define the temporal axis and reshape deformation fields for polynomial fitting
            temporal_axis = np.arange(self.deformation_fields.shape[0])
            n_temporal = self.deformation_fields.shape[0]
            deformation_fields_flat = self.deformation_fields.reshape(n_temporal, -1)  # Flatten spatial dimensions
            
            # Fit a polynomial to each 1D deformation field along the temporal axis
            order = 3  # Change to 2 for a second-order fit. 3 is robust to cases where motion is not convex, e.g. if ref_index is not in the middle.
            coeffs = np.polyfit(temporal_axis, deformation_fields_flat, order)
            
            # Evaluate the polynomial for all temporal points
            poly_vals = np.polyval(coeffs, temporal_axis[:, None])
            
            # Optional: Preserve fixed points (e.g., where deformation is zero, make sure the polynomial is zero)
            fixed_points = (deformation_fields_flat == 0)
            poly_vals[fixed_points] = 0
            
            # Reshape the smoothed deformation fields back to the original shape
            self.deformation_fields = poly_vals.reshape(self.deformation_fields.shape)    
            
        if self.compute_jacobian and self.counter >= counter_threshold:
            print(f"deformation_fields.shape: {self.deformation_fields.shape}")
            jacobian_input = abs(input.get())
            
            # Custom GPU version
            # self.jacobian = compute_jacobian_determinant_xp(sp.to_device(self.deformation_fields, self.devnum))
            self.jacobian = compute_jacobian_determinant(self.deformation_fields)
            # pl.ImagePlot(self.jacobian, x=1, y=2, z=0, hide_axes=True, colormap="viridis") # x component of deformation fields 
            
            # ANTs
            # self.jacobian = None
            # self.jacobian = calculate_jacobian_oflow(fixed = jacobian_input[self.ref_index,...], 
            #                                          transform = self.deformation_fields,
            #                                          reference_index = self.ref_index)
            # pl.ImagePlot(self.jacobian, x=1, y=2, z=0, hide_axes=True, colormap="viridis") # x component of deformation fields 
            
            # Ensure non-zero Jacobian values
            epsilon = 1e-6
            self.jacobian = np.maximum(self.jacobian, epsilon)               
            
        if self.counter >= counter_threshold:
            # Build linop
            J = sp.linop.Multiply(ishape=self.ishape, mult=self.jacobian)
            
            # Define the linear operator to apply the registration
            M = ApplyDeformation(self.ishape, sp.to_device(self.deformation_fields, self.devnum))
        else:
            J = sp.linop.Identity(self.ishape)
            M = sp.linop.Identity(self.ishape)
            
        # (OPTIONAL) plot adjoint test
        MHM_input = ApplyDeformation(self.ishape, sp.to_device(-self.deformation_fields, self.devnum)) * M(input)
        pl.ImagePlot(MHM_input - input, x=1, y=2, z=0, colormap="jet", title=f"{self.counter}: (GHG - I): NEGATIVE. mean = {cp.mean(abs(MHM_input - input))}", vmin=-1, vmax=1) # x component of deformation fields 
         
        # Update counter
        self.counter += 1
                
        if self.compute_jacobian:
            return J * M(input)  # Apply the combined operator to the input
        
        else:
            return M(input) 


    def _adjoint_linop(self):
        # Define the adjoint of the motion fields operator if needed
        # print("WARNING: The adjoint operator was called, meaning that the negative of the measurement motion fields is being applied. ")
        # print("RegisterImages does not behave like a unitary operator here, as it re-measures the deformation every time it is called.")
        # return -RegisterImages(self.ishape, self.ref_index, self.filter_size, self.devnum)

        print("WARNING: APPLYING THE INVERSE DEFORMATION FIELDS.")
        if self.compute_jacobian:
            return ApplyDeformation(self.ishape, sp.to_device(-self.deformation_fields, self.devnum)) * sp.linop.Multiply(self.ishape, mult=1/self.jacobian)
        else:
            return ApplyDeformation(self.ishape, sp.to_device(-self.deformation_fields, self.devnum))
        
    
def clear_gpu_memory(self, *variables):
    """Clears GPU memory and optionally deletes specified variables."""
    with cp.cuda.Device(self.device):
        for var in variables:
            var = None
            del var
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp._default_memory_pool.free_all_blocks()
        cp.cuda.Stream.null.synchronize()
        cp.cuda.stream.get_current_stream().synchronize()
        gc.collect()
        




