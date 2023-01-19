import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import layers




class BaseAttention(layers.Layer):
    def __init__(self, *args, **kwargs):
        '''
        num_heads,
        key_dim,
        value_dim=None,
        dropout=0.0,
        use_bias=True,
        output_shape=None,
        attention_axes=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
        '''
        super(BaseAttention, self).__init__()
        self.mha = layers.MultiHeadAttention(*args, **kwargs)
        self.layernorm = layers.LayerNormalization()
        self.add = layers.Add()
        
class SelfAttention(BaseAttention):
    def call(self, query, key, value, causal_mask=False, attention_mask=None, 
             return_score=False, training=False, **kwargs):
        '''
            query                   |  Query Tensor of shape (B, T, dim).
            value                   |  Value Tensor of shape (B, S, dim).
            key                     |  Optional key Tensor of shape (B, S, dim). If not given, will use value for both key and value, which is the most common case.
            attention_mask	        |  a boolean mask of shape (B, T, S), that prevents attention to certain positions. The boolean mask specifies which query elements can attend to which key elements, 1 indicates attention and 0 indicates no attention. Broadcasting can happen for the missing batch dimensions and the head dimension.
            return_scores |  A boolean to indicate whether the output should be (attention_output, attention_scores) if True, or attention_output if False. Defaults to False.
            training	            |  Python boolean indicating whether the layer should behave in training mode (adding dropout) or in inference mode (no dropout). Defaults to either using the training mode of the parent layer/model, or False (inference) if there is no parent layer.
            causal_mask	        |  A boolean to indicate whether to apply a causal mask to prevent tokens from attending to future tokens (e.g., used in a decoder Transformer).
            '''
        if return_score:
            attn_output, score = self.mha(query, value, key, use_causal_mask=causal_mask, training=training,
                                          attention_mask=attention_mask, return_attention_scores=return_score,
                                          **kwargs)
            query = self.add([query, attn_output])
            query = self.layernorm(query)
            return query, score
        attn_output = self.mha(query, value, key, use_causal_mask=causal_mask, training=training,
                                          attention_mask=attention_mask, return_attention_scores=return_score,
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