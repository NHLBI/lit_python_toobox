import sigpy as sp

def _llr(x, lamda, N, L, w, shift):

    device = sp.get_device(x)
    xp = device.xp

    with device:
        for k in range(N):
            x = xp.roll(x, shift, axis=-(k + 1))
        mats = L(x)
        
        option_1 = True
        if option_1:
            # Option 1: iterate through each block to make a 2D problem... although beware as s_max varies with each block. Probably safe if you choose a medium size block.
            # Assuming mats is of shape (numblocks, resp_frames, Y)
            num_blocks = mats.shape[0]
            for i in range(num_blocks):  # Iterate over blocks
                (u, s, vh) = xp.linalg.svd(mats[i], full_matrices=False)
                s_max = xp.max(s)
                s_t = sp.thresh.soft_thresh(lamda * s_max, s)
                mats[i] = xp.matmul(u, s_t[..., None] * vh)
                
        else:    
            # Option 2: Perform svd over the 3D matrix, and rely on the fact that np.linalg.svd iterates over first dim and applies svd on last 2 dims. This computes a global s_max.
            (u, s, vh) = xp.linalg.svd(mats, full_matrices=False)
            s_max = xp.max(s)
            s_t = sp.thresh.soft_thresh(lamda * s_max, s)
            # print('Singular-values before thresholding:{}'.format(s))
            print(f'lamda: {lamda}, s_max: {s_max}')
            print(f'mats.shape = {mats.shape}')
            print(f'x.shape = {x.shape}')
            mats[...] = xp.matmul(u * s_t[...,None, :], vh)
            
        x = L.H(mats)
        if w is not None:
            x = x / w[None, ...]
        for k in range(N):
            x = xp.roll(x, -shift, axis=-(k + 1))
        return x


class LLR(sp.prox.Prox):
    def __init__(self, shape, lamda, block, msk=None, stride=None):
        self.N = len(shape[1:])
        assert self.N == 2 or self.N == 3

        self.lamda = lamda
        self.block = block
        self.msk = msk

        if stride is None:
            stride = block

        B = sp.linop.ArrayToBlocks(shape[1:], (block,) * self.N, (stride,) * self.N)

        if stride != block:
            dev = sp.Device(7) # HARD CODED FOR NOW
            xp = dev.xp
            self.w = (B.H * B)(xp.ones(B.ishape, dtype=xp.complex64))
        else:
            self.w = None

        B = sp.linop.ArrayToBlocks(shape, (block,) * self.N, (stride,) * self.N)
        if self.N == 3:
            T = sp.linop.Transpose(B.oshape, (1, 2, 3, 0, 4, 5, 6))
            n = T.oshape[0] * T.oshape[1] * T.oshape[2]
        else:
            T = sp.linop.Transpose(B.oshape, (1, 2, 0, 3, 4))
            n = T.oshape[0] * T.oshape[1]
        R = sp.linop.Reshape((n, shape[0], block ** self.N), T.oshape)
        # M = sp.linop.Multiply(shape, msk[None, ...]) # TODO: implement masking 
        self.L = R * T * B #  * M 

        self.c = 0
        super().__init__(shape)

    def _prox(self, alpha, input):
        self.c = self.c + 1
        return _llr(input, self.lamda * alpha, self.N, self.L, self.w, self.c)
    
    
    
    
