import numpy as np
from sigpy import backend, util, thresh, linop
from sigpy import prox
import matplotlib.pyplot as plt
import sigpy.plot as pl
import sys
sys.path.append('../sigpy_mod')
import custom_linop
import registration_oflow3D
import time
import gc

# The goal of this proximal operator is to perform global low rank thresholding on a block of respiratory frames registered to each other.
# We will select a small block of respiratory frames, register all together, and then perform the low rank thresholding, and extract only the "fixed" reference frame of that block after SVD thresholding. 
# Then, we will update to the next block of respiratory frames, and repeat the process.

# TODO: clean up naming conventions and reshaping structure

def GLRA(shape, lamda, A=None, sind_1=1, plot=False, prox_moco=True, N_frames_per_block=None, compute_jacobian=True):
    """
    Constructs the GLRA proximal operator for processing blocks of respiratory frames.

    Parameters:
    - shape: Shape of the input data.
    - lamda: Regularization parameter.
    - A: Optional linear operator applied to input before GLR.
    - sind_1: Number of leading dimensions flattened for matrix representation.
    - plot: Whether to plot the data.
    - prox_moco: Whether to perform motion correction within the proximal operator.
    - N_frames_per_block: Number of respiratory phases per block.
    - compute_jacobian: Whether to compute the Jacobian determinant for each registered image.

    Returns:
    - GLRA_prox: The constructed proximal operator.
    """
    original_image_shape = shape
    print( f"Original image shape: {original_image_shape}")
    u_len = 1
    for i in range(sind_1):
        u_len *= shape[i]

    v_len = 1
    for i in range(len(shape) - sind_1):
        v_len *= shape[-i - 1]

    ishape = (u_len, v_len)
    
    if N_frames_per_block is None:
        N_frames_per_block = original_image_shape[0]
        [print(f"Number of frames per block: {N_frames_per_block}")]
        
    GPR_prox = GLR(ishape, lamda, original_image_shape, plot, prox_moco, N_frames_per_block, compute_jacobian=compute_jacobian)
    R = linop.Reshape(oshape=ishape, ishape=shape)
    if A is None:
        RA = R
    else:
        RA = R * A

    GLRA_prox = prox.UnitaryTransform(GPR_prox, RA)
    return GLRA_prox


class GLR(prox.Prox):
    def __init__(self, shape, lamda, original_image_shape, plot, prox_moco=True, N_frames_per_block=None, compute_jacobian=True):
        """
        Global Low-Rank proximal operator initialization.

        Parameters:
        - shape: Shape of the matrix.
        - lamda: Regularization parameter.
        - original_image_shape: Shape of the original image (e.g. Nphase, Nx, Ny, Nz).
        - plot: Whether to plot intermediate steps.
        - prox_moco: Whether to perform motion correction.
        - N_frames_per_block: Number of respiratory phases per block.
        - compute_jacobian: Whether to compute the Jacobian determinant for each registered image.
        """
        self.lamda = lamda
        self.shape = shape
        self.original_image_shape = original_image_shape
        self.plot = plot
        self.prox_moco = prox_moco
        self.N_frames_per_block = N_frames_per_block
        self.compute_jacobian = compute_jacobian
        self.counter = 0 
        self.counter_threshold = 0 # Threshold for when to start performing motion correction
        # BUG: if self.counter_threshold is > 0, it messes up the memory loading with block, and will use up more GPU for some reason
        super().__init__(shape)

    def _prox(self, alpha, input):
        """
        Apply the proximal operator to the input, iterating over blocks of respiratory phases.
        """
        num_phases, Nx, Ny, Nz = self.original_image_shape  
        num_phases, NxNyNz = self.shape # Image shape collapsed into a matrix
        
        xp = backend.get_array_module(input)
        device = backend.get_device(input).id
        output = xp.zeros_like(input)
        xp.cuda.Stream.null.synchronize()
        
        with xp.cuda.Device(device):
            tic = time.time()
            xp.get_default_memory_pool().free_all_blocks()
            xp.get_default_pinned_memory_pool().free_all_blocks()
            xp._default_memory_pool.free_all_blocks()
            xp.cuda.Stream.null.synchronize()
            xp.cuda.stream.get_current_stream().synchronize()
            gc.collect()
            toc = time.time()
            print(f"Time taken to clear memory inside proximal operator function: {toc - tic:.2f} seconds")
        
        # Print a message when starting motion correction
        if self.counter_threshold == self.counter:
            print(f"Starting motion correction now from proxg() call number: {self.counter}")
        
        # Define fixed start and end range
        start_offset = -self.N_frames_per_block // 2
        end_offset = self.N_frames_per_block // 2

        # Iterate through blocks of size N
        for respiratory_phase in range(num_phases):
            tic = time.time()
            
            # Calculate start and end offsets
            start_offset = -(self.N_frames_per_block // 2)  
            end_offset = self.N_frames_per_block // 2 

            # Generate indices with wrapping
            # indices = [(respiratory_phase + offset) % num_phases for offset in range(start_offset, end_offset + 1)]
            indices = [(respiratory_phase + offset) % num_phases for offset in range(start_offset, end_offset)]
            indices = [idx if idx >= 0 else num_phases + idx for idx in indices] # Handle negative indices explicitly to wrap correctly
            
            # Compute the block indices with wrapping
            ref_index_within_block = self.N_frames_per_block//2 # The fixed reference frame for the current iteration is just the middle reference frame
            
            # Extract the block of respiratory phases
            block = input[indices,...]  # Select the block of respiratory phases based on wrapped indices
            # block = xp.copy(input[indices, ...])  # Force new allocation, select the block of respiratory phases based on wrapped indices
            # block = xp.ascontiguousarray(block)  # Ensure contiguous memory
            
            # Perform motion correction if enabled
            if self.prox_moco and self.counter >= self.counter_threshold:
                # Reshape the block to the original image shape (then undo the reshaping at the end)
                block = xp.reshape(block, (self.N_frames_per_block, Nx, Ny, Nz))
                # Compute the deformation fields for the block
                block, __ = registration_oflow3D.register_images_FOURIER(
                    block, ref_index=ref_index_within_block, filter_size=25+12, gpu_id=device, compute_jacobian=self.compute_jacobian
                )
                # pl.ImagePlot(block, x=1, y=2, z=0, hide_axes=True, colormap="gray") # x component
                # Revert the reshape
                block = xp.reshape(block, (self.N_frames_per_block, NxNyNz))
                           
            # Batched SVD processing for a 2D matrix (necessary for very large matrices)
            if block.shape[1] > 10e6: # Limit to be determined - max buffer size for cupy SVD
                batch_size = block.shape[1] // 2  # Run SVD on smaller batches if the matrix size is too large
            else:
                batch_size = block.shape[1]

            # Process in batches along the second dimension
            s_max = None  # Initialize s_max
            for batch_start in range(0, block.shape[1], batch_size):
                batch_end = min(batch_start + batch_size, block.shape[1])  # Ensure not to exceed bounds
                batch_input = block[:, batch_start:batch_end]  # Extract batch (subset of columns)

                # Perform SVD on the batch
                u, s, vh = xp.linalg.svd(batch_input, full_matrices=False)
                # if s_max is None:
                #     s_max = xp.max(s)  # Get the maximum singular value for the first batch only
                s_max = xp.max(s) # Dynamically update s_max for subsequent batches
                # print(f"Singular-values before thresholding (batch {batch_start}-{batch_end}): {S}")

                # Soft thresholding
                s_thresh = thresh.soft_thresh(self.lamda * alpha * s_max, s)
                print(f'Singular values after thresholding (batch {batch_start}-{batch_end}): {s_thresh[-3:]}')

                # Reconstruct the batch after thresholding
                batch_output = xp.matmul(u, s_thresh[..., None] * vh)

                # Assign the processed batch back to the corresponding columns in the output
                output[respiratory_phase, batch_start:batch_end] = batch_output[ref_index_within_block, ...]
                
            toc = time.time()
            print(f"Respiratory phase: {respiratory_phase}; block indices: {indices}; index within block: {ref_index_within_block}; registration & SVD time: {toc-tic:.2f} seconds")           
            
            if self.plot:
                try:
                    vmax = xp.percentile(abs(output.ravel()), 95)
                    pl.ImagePlot(output, x=2, y=1, z=0,
                                 title=f"Block [{indices}] after SVD Thresholding",
                                 colormap='gray', vmax=vmax)
                    plt.close()
                except Exception as e:
                    print(f"Image plotting failed: {e}")
                    
        # Update counter
        self.counter += 1

        return output.reshape(self.shape)  # Reshape back to the original shape
    
    
    