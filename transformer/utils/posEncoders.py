'''
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import layers

from tensorflow.keras.layers import Layer, Embedding
from tensorflow import cast, float32, newaxis, Tensor, is_tensor, convert_to_tensor

from tensorflow.math import sqrt

import numpy as np


class SinuSoidal(Layer):
    '''
    Custom layer that get the emppeding of words with positons info
    as propoased on Attention is all you need paper -> https://arxiv.org/abs/1706.03762 
    
    '''
    
    def __init__(self, input_dim: int, output_dim: int, max_sent_lenght: int, mask_zero: bool=False, **kwargs) ->None:
        '''instantiate Empedding layer and static positional encoding matrix
        @params:
                input_dim:       Integer. Size of the vocabulary, i.e. maximum integer index + 1.
                output_dim:      Integer. Dimension of the dense embedding.
                mask_zero:       Bool.    Get the Mask of inputs or not, 
                                          more info are here -> https://www.tensorflow.org/guide/keras/masking_and_padding.
                max_sent_lenght: Integer. Max number of tokens in as sentence that the model will 
                                          deal with it during inference.
        '''
        
        super(SinuSoidal, self).__init__()

        self.depth = output_dim
        
        self.embedding = Embedding(input_dim, output_dim, mask_zero=mask_zero, **kwargs) 
        
        self.pos_encoding = self._get_positional_encoding(length=max_sent_lenght, depth=output_dim)

    def call(self, x: Tensor[Tensor[float32]], **kwargs)->Tensor[Tensor[Tensor[float32]]]:
        '''get the postional empedding of x tokens
        @params:
                x: 2D matrix of shape [batch_size, time_step], each row represent one sentenece,
                   each time step represent idx equivelant token.
                   
        @return:
                3D matrix of shape [batch_size, time_step, empedding _dim] 
        '''
        if not is_tensor(x):
            x = convert_to_tensor(x, dtype=float32)
        length = x.shape[1]   #[batch_size, timestep]
        
        x = self.embedding(x, **kwargs) #[batch_size, timestep, depth]
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= sqrt(cast(self.depth, float32))
        
        x = x + self.pos_encoding[newaxis, :length, :] 

        return x
    
    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    
    def _get_positional_encoding(self, length: int, depth: int, n: int=10000)->Tensor[Tensor[Tensor[float32]]]: 
        
        '''create positionalemppeding matrix
        @params:
                length:  Max number of tokens in as sentence that the model will deal with it during inference.
                depth:   Empeddingdim
                n:       Hyper-parameter from the paper 
        '''
        
        
        depth = depth

        positions = np.arange(length)[:, np.newaxis]     # (seq, 1)  [0, 1, 2, 3 ... length-1]

        depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth) [0 / depth, 1 / depth, 2/depth, 3/depth ... length-1/depth]
        
        angle_rates = 1 / (n**depths)         # (1, depth)

        angle_rads = positions * angle_rates      # (pos, depth)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
#         print(angle_rads.shape)
        return cast(angle_rads, dtype=float32)
    
    








if __name__ == '__main__':
    from tensorflow import __version__
    from tensorflow.config import list_physical_devices
    
    print('tf version:', __version__)
    print('Available devices:', end='\n\t\t\t\t')
    print('\n\t\t\t\t'.join(map(str, list_physical_devices())))
    
    print(SinuSoidal(20, 60, 100)._get_positional_encoding())