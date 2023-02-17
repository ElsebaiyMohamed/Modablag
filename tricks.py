#decorator

class Dog:
    def __init__(self, name) -> None:
        self.name = name
        
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        assert name.isalpha()
        self._name = name
    
    @name.deleter
    def name(self):
        self.name = None
        
import h5py

with h5py.File(r"D:\Study\GitHub\dev\btached\batch_0.h5", 'r') as f:
    print(f['en'][()])