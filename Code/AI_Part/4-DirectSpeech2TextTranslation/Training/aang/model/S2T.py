from dataclasses import dataclass, field
import torch
import torch .nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..modules.wave import Wave2Chunk
from ..modules.former import Encoder, Decoder
from ..modules.heads import Head


@dataclass
class HPrams:
    wave_param: dict     = field(default_factory=dict)
    encoder_params: dict = field(default_factory=dict)
    decoder_params: dict = field(default_factory=dict)
    head_names: dict     = field(default_factory=dict)
    head_params: dict    = field(default_factory=dict)
    
    

class Model(nn.Module):  
    
    def __init__(self, wave_param: dict, encoder_params: dict, decoder_params: dict, head_names: dict, head_params: dict):
        super(Model, self).__init__()
        
        self.hparams = HPrams(wave_param=wave_param, encoder_params=encoder_params, decoder_params=decoder_params, 
                              head_names=head_names, head_params=head_params)
        
        self.wave_enc = Wave2Chunk(**self.hparams.wave_param)
        

        self.context_enc = Encoder(**self.hparams.encoder_params)
        
        self.context_dec = Decoder(**self.hparams.decoder_params)
        
        self.heads = nn.ModuleDict()
        for h, pad_idx in self.hparams.head_names.items():
            self.heads[h] = nn.ModuleDict({'emp_layer': nn.Embedding(num_embeddings=self.hparams.head_params[h]['voc_size'], 
                                                            embedding_dim=self.hparams.head_params[h]['d_model'], padding_idx=pad_idx),
                                   'output_layer': Head(**self.hparams.head_params[h])})
            
    def pe(self, length: int, depth: int, device, n=10000):
        '''create positionalemppeding matrix
        @params:
                length:  Max number of tokens in as sentence that the model will deal with it during inference.
                depth:   Empeddingdim
        '''
        
        positions = torch.arange(length, device=device).view(-1, 1)    # (seq, 1)  [0, 1, 2, 3 ... length-1]

        depths = torch.arange(depth, device=device).view(1, -1) / depth   # (1, depth) [0 / depth, 1 / depth, 2/depth, 3/depth ... length-1/depth]

        angle_rates = 1 / (n**depths)             # (1, depth)

        angle_rads = positions * angle_rates      # (pos, depth)

        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    #         print(angle_rads.shape)
        return angle_rads.float()   
    
    def forward(self, wave, target: dict, training=False, wave_mask=False, need_enc_weights=False,
                target_mask=False, need_dec_weights=False):
        
        model_output = dict() 
        for h, _ in self.hparams.head_names.items():
            model_output[h] = dict()
        
        
        wave = self.wave_enc(wave, mask=wave_mask, training=training)
        if wave_mask:
            wave, wave_mask = wave
        else:
            wave_mask = None

        B, T, D = wave.size()
        wave = self.pe(T, D, wave.device) + wave

        wave = self.context_enc(wave, padding_mask=wave_mask, need_weights=need_enc_weights, training=training)
        
        if need_enc_weights:
            wave, model_output['encoder_weights'] = wave[0], wave[1].detach()
        
        
        for h, _ in self.hparams.head_names.items():
            
            target_head = target[h]
            assert target_head.dim() < 3, f'Head: {h}, target size should be 1D tensor for unbatched and 2D for batched'
            if target_head.dim() < 2:
                target_head = target_head.view(1, -1)
                
            query_mask = None
            if target_mask:
                query_mask = (target_head == self.hparams.head_names.get(h)).to(target_head.device)
            
            target_head = self.heads[h]['emp_layer'](target_head)
            B, T, D = target_head.size()
            target_head = self.pe(T, D, target_head.device) + target_head
                        
            
            target_head = self.context_dec(query=target_head, key=wave, query_mask=query_mask, key_mask=wave_mask, 
                                              need_weights=need_dec_weights, training=training)

            if need_dec_weights:
                target_head, model_output[h]['attention_weights'] = target_head[0], target_head[1].detach()
            
            model_output[h]['predection'] = self.heads[h]['output_layer'](target_head); del target_head
            
                
        return model_output
    
    def look_ahead_mask(self, tgt_len:int, src_len:int, device):
        mask = torch.triu(torch.ones(tgt_len, src_len, device=device), diagonal=1).type(torch.bool)
        
        return mask
    
    def init_weights(self):
        self.wave_enc.init_weights()
        self.context_enc.init_weights()
        self.context_dec.init_weights()
        for h, _ in self.hparams.head_names.items():
            self.heads[h]['output_layer'].init_weights()
            
            
class Speech2TextArcht(pl.LightningModule):
    def __init__(self, wave_param: dict, encoder_params: dict, decoder_params: dict, head_names: list, head_params: dict, lr: float):
        super(Speech2TextArcht, self).__init__()
        self.save_hyperparameters()
        self.model = Model(wave_param, encoder_params, decoder_params, head_names, head_params)
        
        self.model.init_weights()
#         torch._dynamo.reset()
#         self.model = torch.compile(self.model, dynamic=True)
        
    def forward(self, wave, target: dict, training=False, wave_mask=False, need_enc_weights=False, enc_mask=False,
                target_mask=False, need_dec_weights=False, dec_mask=False):
        
        results = self.model(wave, target, training, wave_mask, need_enc_weights, enc_mask,
                             target_mask, need_dec_weights, dec_mask)
        
        return results
    
    def training_step(self, batch, batch_idx):
        at = 'train'
        wave, en, ar = batch
        ground_truth = {'en': en[:, 1:], 'ar': ar[:, 1:]}
        results = self(wave=wave, target={'en': en[:, :-1], 'ar': ar[:, :-1]}, training=True, wave_mask=True, 
                       target_mask=True)
        loss = 0
        named_loss = dict()
        for h, pad_idx in self.hparams.head_names.items():
            h_loss = F.cross_entropy(results[h]['predection'].transpose(2, 1), ground_truth[h], reduction='sum')
            loss += h_loss
            
            named_loss[f'{h}_Loss'] = h_loss
                
        
#         self.log_dict(named_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        at = 'val'
            
        wave, en, ar = batch
        ground_truth = {'en': en[:, 1:], 'ar': ar[:, 1:]}
        results = self(wave=wave, target={'en': en[:, :-1], 'ar': ar[:, :-1]}, training=False, wave_mask=True, 
                       target_mask=True)
        loss = 0
        named_loss = dict()
        for h, pad_idx in self.hparams.head_names.items():
            h_loss = F.cross_entropy(results[h]['predection'].transpose(2, 1), ground_truth[h], reduction='sum')
            loss += h_loss
            named_loss[f'{h}_{at}_Loss'] = h_loss
        
#         self.log_dict(named_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.1)
        scheduler = {
            "scheduler": CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-4),
            "interval": "epoch",
            "frequency": 1,}
        return [optimizer], [scheduler]

    
        
if __name__ == '__main__':
    import rich
    rich.print("""
               'model for getting distribution over vocublary to represent the next token'
                """)