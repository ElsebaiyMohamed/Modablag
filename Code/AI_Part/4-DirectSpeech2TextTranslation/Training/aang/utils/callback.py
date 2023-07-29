from pytorch_lightning.callbacks import Callback, RichProgressBar, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

import torch
import torchmetrics.functional as MF

from rich import print as rprint

from .lang import TokenHandler


class Predictions(Callback):
    def __init__(self, config):
        self.tokenizers = dict()
        for k, v in config.items():
            self.tokenizers[k] = TokenHandler(v, k)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        if not batch_idx % 100:
            wave, en, ar = batch
            ground_truth = {'en': en[:, 1:], 'ar': ar[:, 1:]}
            with torch.no_grad():
                results = pl_module(wave=wave, target={'en': en[:, :-1], 'ar': ar[:, :-1]}, training=False, wave_mask=True, 
                            target_mask=True, dec_mask=True)
            pred = ''
            r = torch.randint(0, en.size(0), (1, )).item()
            for h, pad_idx in pl_module.hparams.head_names.items():

                t = self.tokenizers[h].decode_batch(results[h]['predection'].detach().argmax(-1).tolist())
                j = self.tokenizers[h].decode_batch(ground_truth[h].detach().tolist())
                blue = MF.bleu_score(t, j)
                
                pred += f"'{h} guess: with blue score: {blue:0.4f} ' \n\t {t[r]}\n\n"

            rprint(f'\nGround Truth {batch_idx}: \n\t {j[r]} \n\n {pred}')
                
    def load_state_dict(self, state_dict):
        self.tokenizers.update(state_dict)

    def state_dict(self):
        return self.tokenizers.copy()

    
    

# create your own theme!
progress_bar = RichProgressBar(theme=RichProgressBarTheme(description="green_yellow",
                                                          progress_bar="green1",
                                                          progress_bar_finished="green1",
                                                          progress_bar_pulse="#6206E0",
                                                          batch_progress="green_yellow",
                                                          time="grey82",
                                                          processing_speed="grey82",
                                                          metrics="green1",
                                                        ))


ckp = ModelCheckpoint(every_n_train_steps=1000, save_last=True, auto_insert_metric_name=False)
swa = StochasticWeightAveraging(swa_lrs=1e-2, annealing_epochs=2)
if __name__ == '__main__':
    pass
