import numpy as np
from sigpy import backend, util, thresh, linop
from sigpy import prox
import matplotlib.pyplot as plt
import sigpy.plot as pl


def GLRA(shape,lamda,A = None,sind_1 = 1, plot=False, img_shape_for_plot=None, prox_moco=False):
    
    u_len = 1
    for i in range(sind_1):
        u_len = u_len * shape[i]
        
    v_len = 1
    for i in range(len(list(shape))-sind_1):
        v_len = v_len * shape[-i-1]    
    
    ishape = (u_len,v_len)
    
    GPR_prox = GLR(ishape, lamda, plot, img_shape_for_plot, prox_moco)
    R = linop.Reshape(oshape=ishape,ishape=shape)
    if A is None:
        RA = R        
    else:
        RA = R*A
    GLRA_prox = prox.UnitaryTransform(GPR_prox,RA)
    return GLRA_prox

class GLR(prox.Prox):
    def __init__(self, shape, lamda, plot, img_shape_for_plot, prox_moco=False):
        self.lamda = lamda
        self.shape = shape
        self.plot = plot
        self.img_shape_for_plot = img_shape_for_plot
        self.prox_moco = prox_moco
        super().__init__(shape)

    def _prox(self, alpha, input):
        
        xp = backend.get_array_module(input)
        device = backend.get_device(input).id
        
        if self.prox_moco:
            print("PERFORMING MOCO INSIDE OF PROXIMAL OPERATOR")
            import registration_oflow3D
            self.deformation_fields = None # Clear the previous deformation fields
            # Compute the deformation fields using the input data
            _, deformation_fields = registration_oflow3D.register_images(
                input.reshape(self.img_shape_for_plot), ref_index=0, filter_size=21, gpu_id=0
            )
            deformation_fields = np.moveaxis(deformation_fields, 1, -1)  # Move the dimension axis to the end
            import sys
            sys.path.append('../sigpy_mod')
            import custom_linop
            svd_shape = input.shape
            # M = custom_linop.MotionFieldsUnitary(self.img_shape_for_plot, deformation_fields=deformation_fields)
            deformation_fields = backend.to_device(deformation_fields, device=device)
            M = custom_linop.ApplyDeformation(self.img_shape_for_plot, deformation_fields=deformation_fields)
            input = M * input.reshape(self.img_shape_for_plot)
            input = input.reshape(svd_shape)     
        
        # Batched SVD processing for a 2D matrix (necessary for very large matrices)
        output = xp.zeros_like(input)  # Initialize output with the same shape as input
        if input.shape[1] > 10e6: # Limit to be determined
            batch_size = input.shape[1] // 2  # Run SVD on smaller batches if the matrix size is too large
        else:
            batch_size = input.shape[1]

        # Process in batches along the second dimension
        s_max = None  # Initialize s_max
        for batch_start in range(0, input.shape[1], batch_size):
            batch_end = min(batch_start + batch_size, input.shape[1])  # Ensure not to exceed bounds
            batch_input = input[:, batch_start:batch_end]  # Extract batch (subset of columns)

            # Perform SVD on the batch
            u, s, vh = xp.linalg.svd(batch_input, full_matrices=False)
            # if s_max is None:
            #     s_max = xp.max(s)  # Get the maximum singular value for the first batch only
            s_max = xp.max(s) # Dynamically update s_max for subsequent batches
            # print(f"Singular-values before thresholding (batch {batch_start}-{batch_end}): {S}")

            # Soft thresholding
            s_thresh = thresh.soft_thresh(self.lamda * alpha * s_max, s)
            # print(f"Singular-values after thresholding (batch {batch_start}-{batch_end}): {s_thresh}")
            print(f'Singular values after thresholding (batch {batch_start}-{batch_end}): {s_thresh[-3:]}')

            # Reconstruct the batch after thresholding
            batch_output = xp.matmul(u, s_thresh[..., None] * vh)

            # Assign the processed batch back to the corresponding columns in the output
            output[:, batch_start:batch_end] = batch_output
        
        
        if self.plot is True and self.img_shape_for_plot is not None:
            try:
                vmax=xp.percentile(abs(output.ravel()), 95)
                pl.ImagePlot(output.reshape(self.img_shape_for_plot),
                                    x=2, y=1, z=0, title="Spatial bases after soft thresholding",colormap='gray', vmax=vmax)
                plt.close()
                # pl.ImagePlot((np.matmul(u, s_t[...,None]*vh)-input).reshape(self.img_shape_for_plot),
                #                     x=2, y=1, z=0, title="difference",colormap='gray', vmin=-5, vmax=5)
                # plt.close()
            except:
                print("Image plotting failed.")
        if self.prox_moco:
            return (M.H * output.reshape(self.img_shape_for_plot)).reshape(svd_shape)
        else:
            return output
    
    