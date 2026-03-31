# %% [code]
# %% [code]
# %% [code]
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class CompactWave(nn.Module):
    def __init__(self, sr, frame_size, frame_stride, out_dim=512):
        assert frame_size >= 2000, "the frame size minimum should be greater than or equal 2000"
        assert sr == 16000, "suported sample rate is 16000"
        super(CompactWave, self).__init__()
        self.sr = sr
        self.frame_size  = frame_size                                           
        self.frame_stride = frame_stride
        self.out_dim = out_dim
        pram_per_layer = [(13, 1, 0, 3), (11, 1, 0, 2), (11, 1, 0, 2), (2, 2, 0, 1),
                          (7, 1, 0, 1), (7, 2, 0, 1), (7, 1, 0, 2), (3, 3, 0, 1),
                          (5, 2, 0, 1), (5, 2, 0, 1), (5, 2, 0, 2), (2, 1, 0, 1),   
                          (3, 1, 0, 1), (3, 2, 0, 1), (3, 1, 0, 1), (2, 2, 0, 1)]
    
        self.conv_block_1 = nn.Sequential(nn.Conv1d(1, 15, 11, 1, dilation=3),  
                                          nn.ReLU(True),
                                          nn.Conv1d(15, 15, 11, 1, dilation=2),  
                                          nn.Conv1d(15, 15, 11, 1, dilation=2),  
                                          nn.ReLU(True),
                                          nn.MaxPool1d(2, 2),                   
                                          nn.BatchNorm1d(15))
        
               
        self.conv_block_2 = nn.Sequential(nn.Conv1d(15, 25, 7, 1, dilation=1),  
                                          nn.ReLU(True),
                                          nn.Conv1d(25, 25, 7, 2, dilation=1),  
                                          nn.Conv1d(25, 25, 7, 1, dilation=2),  
                                          nn.ReLU(True),
                                          nn.MaxPool1d(3, 3),                  
                                          nn.BatchNorm1d(25))
        
               
        self.conv_block_3 = nn.Sequential(nn.Conv1d(25, 35, 5, 2, dilation=1),  
                                          nn.ReLU(True),
                                          nn.Conv1d(35, 35, 5, 2, dilation=1),  
                                          nn.Conv1d(35, 35, 5, 2, dilation=2), 
                                          nn.ReLU(True),
                                          nn.MaxPool1d(2, 1),                  
                                          nn.BatchNorm1d(35))
        
               
        self.conv_block_4 = nn.Sequential(nn.Conv1d(35, 45, 3, 1, dilation=1),  
                                          nn.ReLU(True),
                                          nn.Conv1d(45, 45, 3, 2, dilation=1),  
                                          nn.Conv1d(45, 45, 3, 1, dilation=1),  
                                          nn.ReLU(True),
                                          nn.MaxPool1d(2, 2),                   
                                          nn.BatchNorm1d(45))
        
        out_shape = self.frame_size
        for p in pram_per_layer:
            out_shape = self._conv_output_length(out_shape, *p)
        
        self.projection = nn.Sequential(nn.Linear(out_shape, out_dim//2),
                                        nn.Dropout(0.1, True),
                                        nn.LeakyReLU(0.5, True),
                                        nn.Linear(out_dim//2, out_dim))
        
        self.norm = nn.BatchNorm1d(out_dim)
        
        
        
    def _conv_output_length(self, length_in, kernel_size, stride=1, padding=0, dilation=1):
        return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        
       
    def forward(self, waves, training=False):
        # ensure that all chunks will be the same size
        
        waves = self.__pad_wave(waves)
        # convert the the wave into overlabing chunks
        
        waves = waves.unfold(dimension = 1, size=self.frame_size, step=self.frame_stride).unsqueeze(2)
        
        B, chunks, c, seq_len = waves.size()
        waves = waves.contiguous().view(B*chunks, c, seq_len)
        
        waves = self.conv_block_1(waves)
        waves = self.conv_block_2(waves)
        waves = self.conv_block_3(waves)
        waves = self.conv_block_4(waves)
        waves = waves.view(B, -1, waves.size(-1)) #view(B, chunks, -1, waves.size(-1)).contiguous().
        
        waves = self.projection(waves).transpose(2, 1)
        waves = self.norm(waves).transpose(1, 2)
            
        return waves
        
    def __pad_wave(self, wave: torch.TensorType):
        B, w_len = 1, None
        if not isinstance(wave, torch.Tensor):
            wave = torch.tensor(wave, dtype=torch.float16)
        wave = wave.type(torch.float16)
        if wave.dim() == 2:
            B, w_len = wave.size()
        elif wave.dim() == 1:
            w_len, = wave.size()
            wave = wave.unsqueeze(0)
        else:
            raise ValueError(f'expected number of dims is 1 for unbatched and 2 for batched but you provide {wave.dim()}')
        
        if w_len < self.frame_size:
            raise ValueError(f'vrey short wave lenght. minimum wave lenght is {self.frame_size}')
        
        num_frames = int(math.ceil(float(abs(w_len - self.frame_size)) / self.frame_stride))
        plus_len = (num_frames * self.frame_stride + self.frame_size) - w_len
        wave = F.pad(wave, [0, plus_len], "constant", 0) 
        return wave
    
    
    

class Textizer(nn.Module):
    def __init__(self, voc_size, special: dict, sr=16000, frame_size=2000, frame_stride=1600, out_dim=512):
        super().__init__()
        self.voc      = voc_size
        self.special  = special
        self.out_dim  = out_dim
        self.wave_enc = CompactWave(sr, frame_size, frame_stride, out_dim)
        self.wave_pos = nn.GRU(self.out_dim, self.out_dim, batch_first=True)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=out_dim, nhead=16, dim_feedforward=850, dropout=0.3, batch_first=True)
        self.enc      = nn.TransformerEncoder(encoder_layer, num_layers=5)
        
        self.word2vec = nn.Embedding(num_embeddings=voc_size, embedding_dim=out_dim, padding_idx=self.special['pad'], scale_grad_by_freq=True)
        self.word_pos = nn.GRU(self.out_dim, self.out_dim, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=16, dim_feedforward=850, dropout=0.15, batch_first=True)
        self.dec      = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        self.out_head = nn.Sequential(nn.Linear(out_dim, voc_size//2),
                                      nn.LayerNorm(voc_size//2),
                                      nn.Dropout(0.2, True),
                                      nn.Linear(voc_size//2, voc_size))
        
    def forward(self, waves, target, teacher_force=0.65, training=False):
        waves  = self.wave_enc(waves, training)
        h_o    = torch.zeros((1, waves.size(0), self.out_dim)).to(waves.device)
        pos, _ = self.wave_pos(waves, h_o)
        waves  = waves + pos / torch.sqrt(torch.tensor(pos.size(1))); del pos
        waves  = self.enc(waves)
        
        pad_mask = target != self.special['pad']
        mask     = torch.triu(torch.ones(target.size(1), target.size(1)) * float('-inf'), diagonal=1).to(waves.device)
        h_o    = torch.zeros((1, target.size(0), self.out_dim), dtype=torch.float16).to(waves.device)
        
        if random.random() < teacher_force:
            target = self.word2vec(target)
            pos, _ = self.word_pos(target, h_o)
            target  = target + pos / torch.sqrt(torch.tensor(pos.size(1))).type(torch.float16)
            target = self.dec(target, waves, tgt_mask=mask, tgt_key_padding_mask=pad_mask)
            target = self.out_head(target)
#             target = F.log_softmax(target, -1)
            return target
            
        else: 
            out = torch.zeros((target.size(0), target.size(1), self.voc)).to(target.device)
            for time_step in range(1, target.size(1)-1):
                _mask = pad_mask[:, :time_step]
                
                dec_start = self.word2vec(target[:, :time_step])
                pos, _ = self.word_pos(dec_start, h_o)
                le = pos.size(1) if len(pos.size()) > 1 else 1
                dec_start = dec_start + pos[:, :time_step] / torch.sqrt(torch.tensor(le))
                dec_start = self.dec(dec_start, waves, tgt_key_padding_mask=_mask)
                dec_start = self.out_head(dec_start[:, -1])
                out[:, time_step, :] = dec_start
                dec_start = F.log_softmax(dec_start, dim=-1)
                out[:, time_step, :] = dec_start
                dec_start = torch.argmax(dec_start, -1)
                target[:, time_step] = dec_start
            
            return out
    
    @torch.inference_mode()
    def transcribe(self, wave):
        assert waves.dim == 1, "this function works only with one record at a time."
        wave   = self.wave_enc(wave, False)
        h_o    = torch.zeros((1,  self.out_dim)).to(wave.device)
        pos, _ = self.wave_pos(wave, h_o)
        wave   = wave + pos / torch.sqrt(torch.tensor(pos.size(1))); del pos
        wave   = self.enc(wave)
        ids    = self.greedy_decoding(wave)
        return ids
    
    def greedy_decoding(self, context):
        ids = self.special['cls'] * torch.ones((1)).to(context.device)
        
        for time_step in range(0, self.special['max']):
            dec_start = self.word2vec(ids)
            h_o    = torch.zeros((1,  self.out_dim)).to(context.device)
            pos, _ = self.word_pos(dec_start, h_o)
            le = pos.size(0)
            dec_start = dec_start + pos / torch.sqrt(torch.tensor(le))
            dec_start = self.dec(dec_start, context)
            dec_start = self.out_head(dec_start[-1])
            dec_start = F.log_softmax(dec_start, dim=-1)
            dec_start = torch.argmax(dec_start, -1)
            ids       = torch.cat((ids, dec_start))
            if dec_start == self.special['end']:
                return ids
            
        return ids
            

    
    
if __name__ == '__main__':
    device  = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    from  torch.cuda.amp import autocast
    with autocast():
        waves   = torch.randn((3, 16000*5)).half().to(device)
        targets = torch.randint(0, 20000, (3, 200), dtype=torch.int32).to(device)
        print(summary(Textizer(20000, {'pad': 4}), input_data=[waves, targets], batch_dim=0))