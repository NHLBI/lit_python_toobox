import matplotlib.pyplot as plt
from sigpy.app import App
from sigpy.alg import PowerMethod
from sigpy import util, backend
from sigpy.linop import Linop  # Ensure Linop is imported from sigpy
import numpy as np
from scipy.optimize import curve_fit

class MaxEig(App):
    """Computes maximum eigenvalue of a Linop and plots the eigenvalue convergence.

    Args:
        A (Linop): Hermitian linear operator.
        dtype (Dtype): Data type.
        device (Device): Device.
        max_iter (int): Maximum number of iterations.
        show_pbar (bool): Show progress bar.
        leave_pbar (bool): Leave progress bar after completion.

    Attributes:
        x (array): Eigenvector with largest eigenvalue.
        eigvals (list): List to store the maximum eigenvalue at each iteration.
    """

    def __init__(
        self,
        A,
        dtype=float,
        device=backend.cpu_device,
        max_iter=30,
        show_pbar=True,
        leave_pbar=True,
    ):
        self.x = util.randn(A.ishape, dtype=dtype, device=device)
        self.eigvals = []  # List to store eigenvalues at each iteration
        alg = PowerMethod(A, self.x, max_iter=max_iter)
        super().__init__(alg, show_pbar=show_pbar, leave_pbar=leave_pbar)

    def _summarize(self):
        if self.show_pbar:
            self.pbar.set_postfix(custom_max_eig="{0:.2E}".format(self.alg.max_eig))
        self.eigvals.append(self.alg.max_eig) # Hijack the _summarize function to save the eigval for each iteration

    def _output(self):
        return self.alg.max_eig
    
    

    def plot_eigenvalues(self):
        """Plot the maximum eigenvalues as a function of iteration and extrapolate up to X iterations."""
        
        def exponential_fit(x, a, b, c):
            """Exponential fitting function for convergence."""
            return a * np.exp(-b * x) + c 

        iterations = np.arange(1, len(self.eigvals))  # Iteration indices
        eigvals = np.array(self.eigvals[1:])  # Skip the first eigenvalue (usually noisy)

        # Plot the original eigenvalues
        plt.figure(figsize=(5, 3))
        label = r'$\lambda_{\text{max}}(A^H A) = \max_{\mathbf{x} \neq 0} \frac{\| A \mathbf{x} \|_2^2}{\| \mathbf{x} \|_2^2}$'
        plt.plot(iterations, eigvals, 'ro', label=label)

        # Perform curve fitting
        try:
            X_subset = int(1*len(eigvals))
            p0 = (-eigvals[-1], 0.01, 1.1*eigvals[-1])
            popt, _ = curve_fit(exponential_fit, iterations[:X_subset], eigvals[:X_subset], p0=p0)

            # Extrapolate the eigenvalue after X iterations and plot up to X iterations
            X = 50
            extended_iterations = np.arange(1, X + 1)  # Extend iterations to X
            extrapolated_vals = exponential_fit(extended_iterations, *popt)
            plt.plot(extended_iterations, extrapolated_vals, 'k--', label=f'extrapolation from {X_subset} to {X} iter')
            
            # Extrapolated value at iteration X
            estimated_value_X = exponential_fit(X, *popt)
            print(f"Estimated eigenvalue after {X} iterations: {estimated_value_X:.2E}")
        except:
            print("MAX EIGENVALUE CURVE FITTING FAILED - LIKELY DUE TO FAILED CONVERGENCE.")
            estimated_value_X = self.eigvals[-1]

        # Add titles and labels
        plt.xlabel('iteration')
        plt.ylabel('maximum eigenvalue')
        plt.title(f'eigenvalue convergence to: {self.eigvals[-1]:.2E}')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Force override
        scale_factor = 1.1
        print(f"Despite attempting a fit, we are just assuming that {scale_factor}x the {len(eigvals)}th eigen value is our max.")
        estimated_value_X = eigvals[-1]*scale_factor
        print(f"Max Eigenvalue assumed = {estimated_value_X:.2E}.")
        
        return estimated_value_X


# Example usage in your script:
# from your_script import MaxEig  # Ensure you import MaxEig from your script

# A = your Hermitian Linop here
# max_eig_app = MaxEig(A)
# max_eig_app.run()
# max_eig_app.plot_eigenvalues()
