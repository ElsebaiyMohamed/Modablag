import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from aang.utils.data import MuSTCDataset
from aang.utils.callback import *
from aang.model.S2T import Speech2TextArcht

import os
import multiprocessing as mp
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--en_json", default=None)
    parser.add_argument("--ar_json", default=None)
    
    parser.add_argument("--batch_size", default=8)
    parser.add_argument("--worker", default=2)
    
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--strategy", default='auto')
    parser.add_argument("--epochs", default=2)
    
    
    args = parser.parse_args()
    
    # strategy = DDPStrategy(process_group_backend="nccl", start_method='fork', find_unused_parameters=False) if str(args.strategy) == 'ddp' else str(args.strategy)
    strategy = str(args.strategy)
        
    if args.data_dir is not None:
        ar_config  = {'tokenizer': args.ar_json,
                      'size': 110}
        en_config  = {'tokenizer': args.en_json,
                      'size': 110}
        wav_config = {'sr': 16000,
                      'wave_size': 30,
                      'frame_size': 20000,
                      'frame_stride': 16000,
                      'b4': 20}
        
        datamodel = MuSTCDataset(loader_config= {'batch_size': int(args.batch_size), 'worker': int(args.worker)}, data_config=dict(dir_path=args.data_dir, ar_config=ar_config.copy(), en_config=en_config.copy(), wav_config=wav_config))

        
        pred = Predictions({'ar': ar_config['tokenizer'], 'en': en_config['tokenizer']})
        
        wave_param      = dict(frame_size=20000, frame_stride=16000, b1=5, b2=10, b3=15, b4=20, out_dim=512)
        encoder_params  = dict(d_model=512, nhead=8, nch=20, dropout=0.3, batch_first=True, size=6)
        decoder_params  = dict(d_model=512, nhead=16, nch=25, dropout=0.5, batch_first=True, size=6)

        head_params     = dict(en=dict(d_model=512, voc_size=500), ar=dict(d_model=512, voc_size=500))
        
        tokenizers      = dict(en=TokenHandler(en_config['tokenizer'], 'en'),
                               ar=TokenHandler(ar_config['tokenizer'], 'ar'))
        
        head_names      = dict(en=tokenizers['en'].get_id("<PAD>"), ar=tokenizers['ar'].get_id("<PAD>"))
        hyper_parameter = dict(lr=3e-3, wave_param=wave_param, encoder_params=encoder_params, decoder_params=decoder_params, 
                            head_names=head_names, head_params=head_params)

        model = Speech2TextArcht(**hyper_parameter)
        devices = args.devices if args.devices == 'auto' else int(args.devices)
        trainer = pl.Trainer(accelerator=args.accelerator, devices=devices, 
                             max_epochs=int(args.epochs), #sync_batchnorm=True, 
                             log_every_n_steps=200,
                             callbacks=[progress_bar, pred, ckp], #, ,],# swa,],  #
                            #  accumulate_grad_batches=2,
                             strategy=strategy,
                             enable_model_summary=True, enable_checkpointing=True, benchmark=True, 
                             default_root_dir=os.getcwd())

    
        trainer.fit(model, datamodel)
    else:
        print('invalid data dir')
    
    
    
    
        
