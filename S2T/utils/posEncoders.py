import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Don't worry of warning under Tensorflow it's ok.
from tensorflow.keras.layers import Layer, LayerNormalization
from tensorflow import cast, float32, newaxis

import numpy as np

    
class SinuSoidal(Layer):
    '''add position empedding vector to empeding vector and normlize'''
    def __init__(self, n: int=10000):
        super().__init__()
        self.n = n
        self.normalizer = LayerNormalization()
    
    def __call__(self, x_emp):
        assert len(x_emp.shape) > 1 and len(x_emp.shape) < 4, f"expected 2D tensor for unbatched tensor or 3d for batched but get {len(x_emp.shape)}D tensor"
        if len(x_emp.shape) == 2:
            T, E = x_emp.shape
            pos = sin_soidal(T, E, self.n)
            x_emp = x_emp + pos
            del pos
            return self.normalizer(x_emp)
        
        _, T, E = x_emp.shape
        pos = sin_soidal(T, E, self.n)
        x_emp = x_emp + pos[newaxis, :, :]
        del pos
        return self.normalizer(x_emp)
    
def sin_soidal(length: int, depth: int, n: int=10000):
    '''create positionalemppeding matrix
    @params:
            length:  Max number of tokens in as sentence that the model will deal with it during inference.
            depth:   Empeddingdim
            n:       Hyper-parameter from the paper 
    '''
    
    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)  [0, 1, 2, 3 ... length-1]

    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth) [0 / depth, 1 / depth, 2/depth, 3/depth ... length-1/depth]
    
    angle_rates = 1 / (n**depths)             # (1, depth)

    angle_rads = positions * angle_rates      # (pos, depth)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
#         print(angle_rads.shape)
    return cast(angle_rads, dtype=float32)


if __name__ == '__main__':
    from tensorflow import __version__
    from tensorflow.config import list_physical_devices
    import numpy as np
    
    print('tf version:', __version__)
    print('Available devices:', end='\n\t\t\t\t')
    print('\n\t\t\t\t'.join(map(str, list_physical_devices())))