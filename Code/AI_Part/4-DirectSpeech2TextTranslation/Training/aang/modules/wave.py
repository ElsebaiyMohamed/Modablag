import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class Wave2Chunk(nn.Module):
    
    def __init__(self, **kwargs):
        '''frame_size, frame_stride, b1, b2, b3, b4, out_dim=512'''
        
        super(Wave2Chunk, self).__init__()
        
        self.config = kwargs
        assert self.config['frame_size'] >= 2000, "the frame size minimum should be greater than or equal 2000"
        assert not self.config['frame_size'] % self.config['b4'], 'frame size should be divisible by num of fiinal chanels'
        
        pram_per_layer = [(19, 5, 0, 3), (19, 5, 0, 2), (2, 2, 0, 1),
                          (7, 1, 0, 1), (7, 2, 0, 1), (3, 3, 0, 1),
                          (5, 2, 0, 1), (5, 2, 0, 1), (2, 1, 0, 1),   
                          (3, 1, 0, 1), (3, 2, 0, 1), (2, 2, 0, 1)]
        
        out_shape = self.config['frame_size']
        for p in pram_per_layer:
            out_shape = self._conv_output_length(out_shape, *p)
        
        self.conv_block_1 = nn.Sequential(nn.Conv1d(1, self.config['b1'] // 2, 19, 5, dilation=3),  
                                          nn.ReLU(),
                                          nn.Conv1d(self.config['b1'] // 2, self.config['b1'], 19, 5, dilation=2),  
                                          nn.ReLU(),
                                          nn.MaxPool1d(2, 2),                   
                                          nn.BatchNorm1d(self.config['b1']))
        
               
        self.conv_block_2 = nn.Sequential(nn.Conv1d(self.config['b1'], self.config['b2'] // 2, 7, 1, dilation=1),  
                                          nn.ReLU(),
                                          nn.Conv1d(self.config['b2'] // 2, self.config['b2'], 7, 2, dilation=1),  
                                          nn.ReLU(),
                                          nn.MaxPool1d(3, 3),                  
                                          nn.BatchNorm1d(self.config['b2']))
        
               
        self.conv_block_3 = nn.Sequential(nn.Conv1d(self.config['b2'], self.config['b3'] // 2, 5, 2, dilation=1),  
                                          nn.ReLU(),
                                          nn.Conv1d(self.config['b3'] // 2, self.config['b3'], 5, 2, dilation=1),  
                                          nn.ReLU(),
                                          nn.MaxPool1d(2, 1),                  
                                          nn.BatchNorm1d(self.config['b3']))
        
               
        self.conv_block_4 = nn.Sequential(nn.Conv1d(self.config['b3'], self.config['b4'] // 2, 3, 1, dilation=1),  
                                          nn.ReLU(),
                                          nn.Conv1d(self.config['b4'] // 2, self.config['b4'], 3, 2, dilation=1),  
                                          nn.ReLU(),
                                          nn.MaxPool1d(2, 2),                   
                                          nn.BatchNorm1d(self.config['b4']))
        
        
        
        self.projection = nn.Sequential(nn.Linear(out_shape, self.config['out_dim']//2),
                                        nn.Dropout(0.1),
                                        nn.ReLU(),
                                        nn.Linear(self.config['out_dim']//2, self.config['out_dim']))
        
        self.norm = nn.BatchNorm1d(self.config['out_dim'])
        
    def _conv_output_length(self, length_in, kernel_size, stride=1, padding=0, dilation=1):
        return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    def forward(self, waves, mask=False, training=False):
        # ensure that all chunks will be the same size
        if mask:
#             waves, mask = waves
            mask = self.__get_mask(waves)

        else:
            mask = None
            # waves = self.__pad_wave(waves, mask)

#         # convert the the wave into overlabing chunks
        # waves = waves.unfold(dimension=2, size=self.config['frame_size'], step=self.config['frame_stride']).transpose(2, 1)
        
#         B, chunks, c, seq_len = waves.size()
        
        B, c, seq_len = waves.size()

#         waves = waves.contiguous().view(B*chunks, c, seq_len)
        
        waves = self.conv_block_1(waves)
        waves = self.conv_block_2(waves)
        waves = self.conv_block_3(waves)
        waves = self.conv_block_4(waves)
        waves = waves.view(-1, waves.size(-1)) #view(B * chunks * c, waves.size(-1))
        
        waves = self.projection(waves)
        waves = self.norm(waves).view(B, -1, waves.size(-1))
        if mask is not None:
            return waves, mask
        return waves
        
        
    def __get_mask(self, wave):
        mask = (wave == 0).squeeze(1).to(wave.device)
        return mask
        
        
        
        
        
        
    def __pad_wave(self, wave, mask=False):
        """Only supported shapes (B, C, Wave_size) or (B, Wave_size) or (Wave_size, ) for unbatched
        """
        
        B, C, w_len = 1, 1, None
        if not isinstance(wave, torch.Tensor):
            wave = torch.tensor(wave)
        wave = wave
        if wave.dim() == 3:
            B, C, w_len = wave.size()
        elif wave.dim() == 2:
            B, w_len = wave.size()
        elif wave.dim() == 1:
            w_len, = wave.size()
            wave = wave.unsqueeze(0)

        else:
            raise ValueError(f'expected number of dims is 1 for unbatched and 2 for batched but you provide {wave.dim()}')
    
        assert w_len >= self.config['frame_size'], f"vrey short wave lenght. minimum wave lenght is {self.config['frame_size']}"
        assert C == 1, f'only support mono wave at this time. Expected num of channels is {1} but given {C}'
        
        wave = wave.reshape(B, C, w_len)                   
        num_frames = int(math.ceil(float(abs(w_len - self.config['frame_size'])) / self.config['frame_stride']))
        plus_len = (num_frames * self.config['frame_stride'] + self.config['frame_size']) - w_len
        wave = F.pad(wave, [0, plus_len], "constant", 0) 
        
        if mask:
            mask = (wave == 0).squeeze(1).to(wave.device)
#             mask = F.pad(mask, [0, plus_len], "constant", True)
            mask = mask.unfold(dimension=1, size=self.config['frame_size'], step=self.config['frame_stride'])
            mask = mask.contiguous().view(B, mask.size(1)*self.config['b4'], -1)
            mask = torch.all(mask, 2)
            return wave, mask
        return wave
    
    def init_weights(self):
        
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()

            else:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
                
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                m.reset_parameters()
                
    def get_config(self):
        return self.config
    
    
if __name__ == '__main__':
    import rich
    rich.print("""
               'Sub module for converting wave into overlaping chunks'
                'and apply conv op on each chunk to extract features.'
                """)