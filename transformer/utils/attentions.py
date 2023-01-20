import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import layers
from tensorflow import is_tensor, convert_to_tensor, float32, Tensor
from numpy import ndarray
from typing import Union




class BaseAttention(layers.Layer):
    def __init__(self, num_heads:int, key_dim:int, *args, **kwargs):
        '''
        @params:
                num_heads:                          int.    Number of attention heads. 
                key_dim:                            int.    Size of each attention head for query and key.
                value_dim=None:                     int.    Size of each attention head for value.
                dropout=0.0:                        float.  Dropout probability.
                use_bias=True:                      bool.   whether the dense layers use bias vectors/matrices.
                output_shape=None:                  int.  The expected shape of an output tensor, besides the 
                                                                batch and sequence dims. If not specified, 
                                                                projects back to the key feature dim.
                attention_axes=None:                int.    axes over which the attention is applied. None means 
                                                                attention over all axes, but batch, heads, 
                                                                and features.
                kernel_initializer='glorot_uniform':str     Initializer for dense layer kernels.
                bias_initializer='zeros':           str     Initializer for dense layer biases.
                kernel_regularizer=None:                    Regularizer for dense layer kernels.
                bias_regularizer=None:                      Regularizer for dense layer biases.
                activity_regularizer=None:                  Regularizer for dense layer activity.
                kernel_constraint=None:                     Constraint for dense layer kernels.
                bias_constraint=None:                       Constraint for dense layer kernels.
                **kwargs
        '''
        super(BaseAttention, self).__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, *args, **kwargs)
        self.layernorm = layers.LayerNormalization()
        self.add = layers.Add()
        
class SelfAttention(BaseAttention):
    def call(self, query:Union[Tensor, ndarray], key: Union[Tensor, ndarray],
             value: Union[Tensor, ndarray], causal_mask: bool=False, 
             return_score=False, training=False, **kwargs)->Tensor: #attention_mask: Union[Tensor, ndarray]=None, 
        '''
        @params
            query:          3D matrix.  Query Tensor of shape (B, T, dim).
            value:          3D matrix.  Value Tensor of shape (B, S, dim).
            key:            3D matrix.  Optional key Tensor of shape (B, S, dim). If not given, 
                                            will use value for both key and value, which is the most 
                                            common case.
            attention_mask:	bool.       Mask of shape (B, T, S), that prevents attention to certain positions. 
                                            The boolean mask specifies which query elements can attend to 
                                            which key elements, 1 indicates attention and 0 indicates no 
                                            attention. Broadcasting can happen for the missing batch 
                                            dimensions and the head dimension.
            return_scores:  bool.       indicate whether the output should be (attention_output, attention_scores) if True, 
                                            or attention_output if False. Defaults to False.
            training:	    bool.       Python boolean indicating whether the layer should behave in training 
                                            mode (adding dropout) or in inference mode (no dropout). 
                                            Defaults to either using the training mode of the parent 
                                            layer/model, or False (inference) if there is no parent layer.
            causal_mask:    bool        A boolean to indicate whether to apply a causal mask to prevent tokens 
                                            from attending to future tokens (e.g., used in a decoder Transformer).
            
            @return
                    The result of the computation, of shape (B, T, E), where T is for target sequence shapes 
                    and E is the query input last dimension if output_shape is None. Otherwise, the multi-head 
                    outputs are projected to the shape specified by output_shape.
                    
                    if return score is true [Optional] multi-head attention coefficients over attention axes.
            '''
        if not is_tensor(query):
            query = convert_to_tensor(query, dtype=float32)
        if not is_tensor(value):
            value = convert_to_tensor(value, dtype=float32)
        if not is_tensor(key):
            key = convert_to_tensor(key, dtype=float32)
            
        if return_score:
            attn_output, score = self.mha(query, value, key, use_causal_mask=causal_mask, training=training,
                                          return_attention_scores=return_score, #attention_mask=attention_mask, 
                                          **kwargs)
            query = self.add([query, attn_output])
            query = self.layernorm(query)
            return query, score
        attn_output = self.mha(query, value, key, use_causal_mask=causal_mask, training=training,
                                return_attention_scores=return_score, #attention_mask=attention_mask, 
                                    **kwargs)
        query = self.add([query, attn_output])
        query = self.layernorm(query)
        return query
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "mha": layers.serialize(self.mha)
        })
        return config    

    @classmethod
    def from_config(cls, config):
      config['mha'] = layers.deserialize(config['mha'])
      return super().from_config(config)
        

if __name__ == '__main__':
    from tensorflow import __version__
    from tensorflow.config import list_physical_devices 
    print(__version__)
    print(list_physical_devices())
    import numpy as np
    
    
    dummy = np.random.randn(3, 5, 10)
    # sa = SelfAttention(8, 10)
    # x = sa(dummy, dummy, dummy)
    # print(x.numpy().shape)
    
    sa = SelfAttention(8, 10)
    x = sa(dummy, dummy, dummy, causal_mask=True)
    print(x.numpy())