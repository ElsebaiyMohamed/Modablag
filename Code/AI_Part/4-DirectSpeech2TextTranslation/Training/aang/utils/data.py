import torch
import torchaudio
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import math
import pandas as pd
import yaml
from yaml import CLoader
import os

from os.path import join
from aang.utils.lang import TokenHandler

class DFMap(Dataset):
    def __init__(self, df, ar_config: dict, en_config: dict, wav_config, **kwargs):

        self.df = df
        self.ar_config  = ar_config.copy()
        self.en_config  = en_config.copy()
        self.wav_config = wav_config.copy()
        
        self.ar_config['tokenizer'] = TokenHandler(self.ar_config['tokenizer'], 'ar')
        self.en_config['tokenizer'] = TokenHandler(self.en_config['tokenizer'], 'en')
        
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        du, of, speaker_id, wav_path, en, ar = self.df.iloc[idx]
        of = int(of * self.wav_config['sr'])
        du = int(du * self.wav_config['sr'])

        wave, _ = torchaudio.load(wav_path, frame_offset=of, num_frames=du)

        if wave.size(1) < self.wav_config['sr'] * self.wav_config['wave_size']:
            wave = F.pad(wave,[0, self.wav_config['sr'] * self.wav_config['wave_size'] - wave.size(1)])
        elif wave.size(1) > self.wav_config['sr'] * self.wav_config['wave_size']:
            wave    = wave[:, :self.wav_config['sr'] * self.wav_config['wave_size']]
        else: None
#         en = self.en_token.encode_line(en)
        ar = self.ar_config['tokenizer'](ar, self.ar_config['size'])
        ar = torch.tensor(ar, dtype=torch.int64)

        en = self.en_config['tokenizer'](en, self.en_config['size'])
        en = torch.tensor(en, dtype=torch.int64)
        # return wave, en, ar
        wave, mask = self.__pad_wave(wave, mask=True)
        wave = wave.unfold(dimension=1, size=self.wav_config['frame_size'], step=self.wav_config['frame_stride']).transpose(0, 1)
        return (wave, mask), en, ar
    
    
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
    
        assert w_len >= self.wav_config['frame_size'], f"vrey short wave lenght. minimum wave lenght is {self.config['frame_size']}"
        assert C == 1, f'only support mono wave at this time. Expected num of channels is {1} but given {C}'
        
        wave = wave.reshape(C, w_len)                   
        num_frames = int(math.ceil(float(abs(w_len - self.wav_config['frame_size'])) / self.wav_config['frame_stride']))
        plus_len = (num_frames * self.wav_config['frame_stride'] + self.wav_config['frame_size']) - w_len
        wave = F.pad(wave, [0, plus_len], "constant", 0) 
        
        if mask:
            mask = (wave == 0).squeeze(0)#.to(wave.device)
#             mask = F.pad(mask, [0, plus_len], "constant", True)
            mask = mask.unfold(dimension=0, size=self.wav_config['frame_size'], step=self.wav_config['frame_stride'])
            mask = mask.contiguous().view(mask.size(0)*self.wav_config['b4'], -1)
            mask = torch.all(mask, 1)
            return wave, mask
        return wave

        

class MuSTCDataset(pl.LightningDataModule):
    def __init__(self, data_config, loader_config=None):
        super().__init__()
        self.data_config = data_config
        self.loader_config = loader_config
        
        
    def setup(self, stage: str):
        self.data = self.load_from_dir(self.data_config['dir_path'])
                
    def load_from_dir(self, root):
        # split_name = root.split(os.sep)[-1]
        split_names = ['train', 'dev']
        final_data = dict()
        for spl in split_names:
            txt_dir = os.listdir(join(root, spl, 'txt'))
            wav_dir = join(root, spl, 'wav')
            files   = dict()
            for f in txt_dir:
                if 'yaml' in f:
                    files['yaml'] = join(root, spl, 'txt', f)
                elif 'ar' in f:
                    files['ar'] = join(root, spl, 'txt', f)
                elif 'en' in f:
                    files['en'] = join(root, spl, 'txt', f)
                else: None
                
            data = self.get_yaml_data(files['yaml'])
            data = pd.DataFrame.from_records(data)
            
            data['wav'] = data['wav'].apply(lambda x: os.path.join(wav_dir, x))

            data['en'] = self.get_text_data(files['en'])
            
            data['ar'] = self.get_text_data(files['ar'])
            data = data.sample(frac=1).reset_index(drop=True)
            data = data.sample(frac=1).reset_index(drop=True)
            final_data[spl] = DFMap(data, **self.data_config)
        return final_data
    
    def get_yaml_data(self, path):
        with open(path) as f:
            data = yaml.load(f, Loader=CLoader)
            return data    
    
    def get_text_data(self, path):
        with open(path, 'rt', encoding='utf-8', errors='ignors') as f:
            return f.readlines()
        
    def train_dataloader(self):
        return DataLoader(self.data['train'], batch_size=self.loader_config['batch_size'], num_workers=self.loader_config['worker'], drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.data['dev'], batch_size=self.loader_config['batch_size'], num_workers=10, drop_last=True)
    
if __name__ == "__main__":
    pass
