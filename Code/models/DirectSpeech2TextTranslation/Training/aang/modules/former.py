import torch.nn as nn
import torch.nn.functional as F



class EncoderLayer(nn.Module):
    def __init__(self, **kwargs):
        """@params:
                    d_model, nhead, nch, dropout=0.3, batch_first=True
        """
        super(EncoderLayer, self).__init__()
        self.config = kwargs        
        
        self.pre_norm = nn.LayerNorm(self.config['d_model'])
        
        self.mha      = nn.MultiheadAttention(embed_dim=self.config['d_model'], num_heads=self.config['nhead'], dropout=self.config['dropout'], 
                                               bias=True, add_bias_kv=True, batch_first=self.config['batch_first'])
        
        self.res_front = nn.Sequential(nn.Conv1d(1, 3, 7, 1, padding='same'),  
                                       nn.ReLU(),
                                       nn.Conv1d(3, 3, 5, 1, padding='same'),  
                                       nn.Conv1d(3, self.config['nch'], 3, 1, padding='same'),
                                       nn.ReLU())
        
        
        self.res_back  = nn.Linear(self.config['d_model']* self.config['nch'], self.config['d_model'], bias=False)


        self.ffn       = nn.Sequential(nn.LayerNorm(self.config['d_model']),
                                       nn.Linear(self.config['d_model'], self.config['d_model']*2),
                                       nn.ReLU(),
                                       nn.Dropout(self.config['dropout']),
                                       nn.Linear(self.config['d_model']*2, self.config['d_model']),
                                       nn.ReLU(),
                                       nn.Dropout(self.config['dropout']/2))
        
        
        
    def forward(self, x, padding_mask=None, need_weights=False, training=False):
        if x.dim() < 3:
            x = x.view(1, -1, self.config['d_model']).contiguous()
        x = self.pre_norm(x)
        att_out, att_weights = self.mha(query=x, key=x, value=x, key_padding_mask=padding_mask, 
                                        need_weights=need_weights, is_causal=False)
        
        
        B, T, D = x.size()
        x = x.view(B*T, 1, D)
        x = self.res_front(x)
        
        x = x.view(B, T, self.config['nch']*D)
        x = F.dropout(x, 0.15, training=training)
        x = self.res_back(x)
        
        att_out_pre = att_out + x; del x
        
        att_out = self.ffn(att_out_pre)
        att_out = att_out + att_out_pre
        
        if need_weights:
            return att_out, att_weights
        
        return att_out
    
    def init_weights(self):
        
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()

            else:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
                
        for m in self.modules():
            if isinstance(m, (nn.LayerNorm,)):
                m.reset_parameters()
                
                
    def freeze(self, state=True):
        self.requires_grad_(state)

                
    def get_config(self):
        return self.config
    
    
    
    
class Encoder(nn.Module):
    def __init__(self, **kwargs):
        '''d_model=512, nhead=4, nch=5, dropout=0.3, batch_first=True, size'''
        super().__init__()
        self.config = kwargs
        assert self.config['d_model'] % self.config['nhead'] == 0, 'd_model should be dvisible by num of heads'
        self.enc_stack = nn.ModuleList([EncoderLayer(d_model=self.config['d_model'], nch=self.config['nch'], nhead=self.config['nhead'], 
                                                     dropout=self.config['dropout'], batch_first=self.config['batch_first']) 
                                        for _ in range(self.config['size'])])
        
        
    def forward(self, x, padding_mask=None, need_weights=False, training=False):
        if need_weights:
            
            for layer in self.enc_stack:
                x, atten_weights = layer(x, padding_mask=padding_mask, need_weights=need_weights, training=training)
            return x, atten_weights
        
        for layer in self.enc_stack:
                x = layer(x, padding_mask=padding_mask, need_weights=need_weights, training=training)
        return x
    
    def init_weights(self):
        
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()

            else:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
                
        for m in self.modules():
            if isinstance(m, (nn.LayerNorm,)):
                m.reset_parameters()
                
    def freeze(self, state=True):
        self.requires_grad_(state)

                
    def get_config(self):
        return self.config
    
    
    
#####################################################################
#####################################################################


class DecoderLayer(nn.Module):
    def __init__(self, **kwargs):
        """@params:
                    d_model, nhead, nch, dropout=0.3, batch_first=True
        """
        super(DecoderLayer, self).__init__()
        self.config = kwargs
        
        self.pre_norm1 = nn.LayerNorm(self.config['d_model'])
        
        self.smha      = nn.MultiheadAttention(embed_dim=self.config['d_model'], num_heads=self.config['nhead'], dropout=self.config['dropout'], 
                                               bias=True, add_bias_kv=True, batch_first=self.config['batch_first'])
        
        self.res_front = nn.Sequential(nn.Conv1d(1, 3, 7, 1, padding='same'),  
                                       nn.ReLU(),
                                       nn.Conv1d(3, 3, 5, 1, padding='same'),  
                                       nn.Conv1d(3, self.config['nch'], 3, 1, padding='same'),  
                                       nn.ReLU())
        
        self.res_back  = nn.Linear(self.config['d_model']* self.config['nch'], self.config['d_model'], bias=False)


        self.pre_norm2 = nn.LayerNorm(self.config['d_model'])
        
        self.cmha      = nn.MultiheadAttention(embed_dim=self.config['d_model'], num_heads=self.config['nhead']*2, dropout=self.config['dropout'], 
                                               bias=True, add_bias_kv=True, batch_first=self.config['batch_first'])
        
        self.ffn       = nn.Sequential(nn.LayerNorm(self.config['d_model']),
                                       nn.Linear(self.config['d_model'], self.config['d_model']*2),
                                       nn.ReLU(),
                                       nn.Dropout(self.config['dropout']),
                                       nn.Linear(self.config['d_model']*2, self.config['d_model']),
                                       nn.ReLU(),
                                       nn.Dropout(0.1))
        
        
    def forward(self, query, key, query_mask=None, key_mask=None, need_weights=False, training=False):
        if query.dim() < 3:
            query = query.view(1, -1, self.config['d_model']).contiguous()
            
        if key.dim() < 3:
            key = key.view(1, -1, self.config['d_model']).contiguous()
            
        query = self.pre_norm1(query)
        x     = query.clone()
        
        att_out, _ = self.smha(query, query, query, key_padding_mask=query_mask, need_weights=False, is_causal=True)
        att_out = att_out + x; del x
        
        B, T, D = query.size()
        query = query.view(B*T, 1, D)
        query = self.res_front(query)
        
        query = query.view(B, T, self.config['nch']*D)
        query = F.dropout(query, 0.15, training=training)

        query = self.res_back(query)
        query = att_out + query; del att_out
        
        
        query = self.pre_norm2(query)
        att_out, att_weight = self.cmha(query, key, key, key_padding_mask=key_mask, need_weights=need_weights, is_causal=False)
        
        att_out_pre = att_out + query; del query
        
        att_out = self.ffn(att_out_pre)
        att_out = att_out + att_out_pre
        
        if need_weights:
            return att_out, att_weight
        
        return att_out
    
    def init_weights(self):
        
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()

            else:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
                
        for m in self.modules():
            if isinstance(m, (nn.LayerNorm, )):
                m.reset_parameters()
                
    def freeze(self, state=True):
        self.requires_grad_(state)

                
    def get_config(self):
        return self.config



class Decoder(nn.Module):
    def __init__(self, **kwargs):
        '''d_model=512, nhead=4, nch=5, dropout=0.3, batch_first=True, size'''
        super().__init__()
        self.config = kwargs
        assert self.config['d_model'] % self.config['nhead'] == 0, 'd_model should be dvisible by num of heads'
        
        self.dec_stack = nn.ModuleList([DecoderLayer(d_model=self.config['d_model'], nch=self.config['nch'], nhead=self.config['nhead'], 
                                                     dropout=self.config['dropout'], batch_first=self.config['batch_first']) 
                                        for _ in range(self.config['size'])])
        
        
    def forward(self, query, key, query_mask=None, key_mask=None, need_weights=False, training=False):
        if need_weights:
            for layer in self.dec_stack:
                query, atten_weight = layer(query, key, query_mask=query_mask, key_mask=key_mask, need_weights=need_weights,
                                            training=training)
            return query, atten_weight
        
        for layer in self.dec_stack:
            query = layer(query, key, query_mask=query_mask, key_mask=key_mask, need_weights=need_weights, training=training)
            
        return query
    
    def init_weights(self):
        
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()

            else:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
                
        for m in self.modules():
            if isinstance(m, (nn.LayerNorm,)):
                m.reset_parameters()
                
                
    def freeze(self, state=True):
        
        self.requires_grad_(state)
    

                
    def get_config(self):
        return self.config

if __name__ == '__main__':
    import rich
    rich.print("""
               'Encoder Layer for cabtureing wave contextual info'
               'Decoder Layer for aligen text modal with wave modal contextual info'
                """)