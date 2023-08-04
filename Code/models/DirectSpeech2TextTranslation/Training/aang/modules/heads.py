import torch.nn as nn



class Head(nn.Module):
    def __init__(self, **kwargs):
        '''d_model, voc_size'''
        super().__init__()
        self.config = kwargs
        
        self.layer1 =  nn.Sequential(nn.Linear(self.config['d_model'], self.config['voc_size'] // 3),
                                     nn.Tanhshrink())
        self.norm1 = nn.BatchNorm1d(self.config['voc_size'] // 3)
        
        self.layer2 = nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(self.config['voc_size'] // 3, self.config['voc_size']))

       
    def forward(self, x, **kwargs):
        x = self.layer1(x)
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.layer2(x)        
        return x
        
    def init_weights(self):
        
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()

            else: 
                nn.init.xavier_uniform_(p.data)
                
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d,)):  
                m.reset_parameters()
                
    def freeze(self, state=True):
        self.requires_grad_(state)

                
    def get_config(self):
        return self.config
    
    
if __name__ == '__main__':
    import rich
    rich.print("""
               'Sub module for getting distribution over vocublary to represent the next token'
                """)