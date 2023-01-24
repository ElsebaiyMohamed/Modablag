import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Layer, MultiHeadAttention, LayerNormalization, 
                              Add, serialize, deserialize, Dropout, Dense)
from tensorflow import is_tensor, convert_to_tensor, float32, Tensor
from numpy import ndarray
from typing import Union, List




class BaseAttention(Layer):
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
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, *args, **kwargs)
        self.layernorm = LayerNormalization()
        self.add = Add()
        
class SelfAttention(BaseAttention):
    def call(self, query: Union[Tensor, ndarray, List], key: Union[Tensor, ndarray, List],
             value: Union[Tensor, ndarray, List], causal_mask: bool=False, 
             return_score=False, training=False, **kwargs)->Tensor: #attention_mask: Union[Tensor, ndarray, List]=None, 
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
            "mha": serialize(self.mha)
        })
        return config    

    @classmethod
    def from_config(cls, config):
      config['mha'] = deserialize(config['mha'])
      return super().from_config(config)
  
  
  
  
class FeedForward(Layer):
    def __init__(self, units: list, dropout=0.1):
        '''
        @params:
                units:   list.  Number of units at each dense layer, make sure that the number of
                                      units at the last layer same as the inputs to the layer.
                dropout: float. dropout ratio before Add&Normlize layer
        '''
        super(FeedForward, self).__init__()
        self.seq = Sequential([Dense(u) for u in units])
        self.seq.add(Dropout(dropout))
        self.add = Add()
        self.layer_norm = LayerNormalization()

    def call(self, x: Union[Tensor, ndarray], training: bool=False)->Tensor:
        '''
        @params:
                x       : 2D float32 matrix.
                training: bool.               behave in training or inference mode
        @return:
                2D float32 matrix with the same shape as the inputs
        '''
        if not is_tensor(x):
            x = convert_to_tensor(x, dtype=float32)
        
        x = self.add([x, self.seq(x, training=training)])
        x = self.layer_norm(x, training=training) 
        return x
    
    
class EncoderLayer(Layer):
    def __init__(self, key_dim, num_heads, output_shape, dropout=0.1):
        #TODO -> docstring
        super().__init__()

        self.self_attention = SelfAttention(num_heads=num_heads,
                                            key_dim=key_dim,
                                            dropout=dropout,
                                            output_shape=output_shape)

        self.ffn = FeedForward([key_dim, output_shape])

    def call(self, x, training=False, **kwargs):
        #TODO -> docstring
        x = self.self_attention(query=x, key=x, value=x, training=training, **kwargs)
        x = self.ffn(x, training=training)
        return x
  
  
  

class DecoderLayer(Layer):
    #TODO -> docstring
    def __init__(self, key_dim, num_heads, output_shape: int, dropout=0.1):
        #TODO -> docstring
        super().__init__()

        self.self_attention  = SelfAttention(num_heads=num_heads,
                                             key_dim=key_dim,
                                             dropout=dropout,
                                             output_shape=output_shape)
        
        self.cross_attention = SelfAttention(num_heads=num_heads,
                                             key_dim=key_dim,
                                             dropout=dropout,
                                             output_shape=output_shape)

        self.ffn = FeedForward([key_dim, output_shape])
        self.layernorm = LayerNormalization()
        self.add = Add()

    def call(self, x, context, training=False, causal_mask=True, **kwargs):
        #TODO -> docstring
        x = self.self_attention(query=x, key=x, value=x, training=training, causal_mask=causal_mask, **kwargs)
        x = self.cross_attention(query=x, key=context, value=context, training=training, 
                                 causal_mask=causal_mask, **kwargs)
        return self.ffn(x, training=training)

  
  
        

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