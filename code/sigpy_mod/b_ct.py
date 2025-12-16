
import sigpy as sp
import numpy as np
import matplotlib.pyplot as plt


def tseg_b_ct_gpu(F, b0, t2star=None, bins=10, lseg=10, readout=10e-3, plot=False, devnum=0):
    """Creates B and Ct matrices needed for time-segmented compensation.

    Args:
        F (linop): Fourier encoding linear operator (e.g. NUFFT).
        b0 (array): inhomogeneity matrix of frequency offsets (Hz).
        t2star (array): inhomogeneity matrix of t2star (s).
        bins (int): number of histogram bins to use.
        lseg (int): number of time segments.
        readout (float): length of readout pulse (s).
        plot (Bool): plot basis.

    Returns:
        2-element tuple containing

        - **B** (*array*): temporal interpolator.
        - **Ct** (*array*): off-resonance phase at each time segment center.
    """
    
    # Set device X
    device = sp.Device(devnum)
    xp = device.xp
    
    # create time vector
    N_samp = F.oshape[-1]
    t = xp.linspace(0, readout, N_samp)
    
    # Convert b0 to cupy array
    b0_cp = xp.asarray(b0)
    
    hist_wt_b0, bin_edges_b0 = xp.histogram(
        xp.imag(2j * xp.pi * xp.ravel(b0_cp)), bins
    )
    # The minus sign is dealt with in the ct and b lines, near end of code
    if t2star is None:
        t2star_cp = xp.ones_like(b0_cp) * 1e21 # make large number so exp(-1/t2star) -> 1
    else:
        t2star_cp = xp.asarray(t2star)
    
    hist_wt_t2star, bin_edges_t2star = xp.histogram(
        xp.ravel(1/t2star_cp), bins
    )

    # Build B and Ct
    bin_centers_b0 = bin_edges_b0[1:] - bin_edges_b0[1] / 2
    bin_centers_t2star = bin_edges_t2star[1:] - bin_edges_t2star[1] / 2
    # Get total number of counts falling into each bin
    hist_wt = hist_wt_t2star + hist_wt_b0
    zk = bin_centers_t2star + 1j * bin_centers_b0
    tl = xp.linspace(0, lseg, lseg) / lseg * readout   # time seg centers
    # calculate off-resonance phase/t2star decay @ each time seg, for hist bins
    ch = xp.exp(-xp.expand_dims(tl, axis=1) @ xp.expand_dims(zk, axis=0))
    w = xp.diag(xp.sqrt(hist_wt))
    p = xp.linalg.pinv(w @ xp.transpose(ch)) @ w
    b = p @ xp.exp(
        -xp.expand_dims(zk, axis=1) @ xp.expand_dims(t, axis=0)
    )
    b = xp.transpose(b)
    b0_v = xp.expand_dims((2j * xp.pi * xp.ravel(b0_cp)) +
                          xp.ravel(1/t2star_cp), axis=0)
    ct = xp.transpose(xp.exp(-xp.expand_dims(tl, axis=1) @ b0_v))

    # Plot
    if plot:
        fig, (ax1, ax2) = plt.subplots(
            2, sharex=False, figsize=(6, 3), dpi=100)
        ax1.plot(xp.asnumpy(xp.real(b[:, :])).tolist(), color='g')
        ax1.plot(xp.asnumpy(xp.imag(b[:, :])).tolist(), color='r')
        ax1.set_ylabel('b')
        ax1.set_xlabel("sample number")
        ax2.plot(xp.asnumpy(xp.real(ct[:, 0].reshape(F.ishape[1:]))).ravel(), color='c')
        ax2.plot(xp.asnumpy(xp.imag(ct[:, 0].reshape(F.ishape[1:]))).ravel(), color='m')
        ax2.set_ylabel('ct')
        ax2.set_xlabel("sample number")
        plt.show()

    for ii in range(lseg):
        # Compute linops
        Bi = sp.linop.Multiply(F.oshape, xp.asnumpy(b[:, ii]))
        Cti = sp.linop.Multiply(F.ishape, xp.asnumpy(ct[:, ii].reshape(F.ishape[1:]))) # TODO: confirm this is correct - update, this works but does not match sigpy as we omit the need for an S input

        # Effectively, calculate A = F + Bi * F(Cti)
        if ii == 0:
            A = Bi * F * Cti
        else:
            A = A + Bi * F * Cti
            
        # TODO: OPTIONAL update to match sigpy:
        # # operation below is effectively A = A + Bi * F(Cti * S)
        # if ii == 0:
        #     A = Bi * F * S * Cti
        # else:
        #     A = A + Bi * F * S * Cti

    return A