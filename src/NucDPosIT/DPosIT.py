from exo_model import ExoModel
from Alignment_file_manager import AlignmentFileManager
from abc import ABC, abstractmethod
import numpy as np


class AbstractDPosIT(ABC):
    @abstractmethod
    def alignemnt(self):
        pass
    
    @abstractmethod
    def fit_res(self):
        pass
    
    @abstractmethod
    def fit_exo(self):
        pass
    
    @abstractmethod
    def predict_exo(self):
        pass
    
    @abstractmethod
    def digest_nucs(self, size):
        pass
    
    @abstractmethod
    def fit_exo(self, probs_path=None):
        pass
    
    @abstractmethod
    def tlens(self):
        pass
    
    @abstractmethod
    def iterover(self, contig, iterover='template', win_len=300, start=None, stop=None):
        pass



class DPosIT(AbstractDPosIT):
    def __init__(self, input_path, file_format, out_dir=None, l0=147, reg_koef = 0.12, max_iter=2000):
        self.file_manager = AlignmentFileManager(input_path, file_format, out_dir)
        self.exo_model = ExoModel(l0, reg_koef, max_iter)
    
    @property
    def alignemnt(self):
        return self.file_manager.alignment_file
    
    @property
    def fit_res(self):
        return self.exo_model.optimization_.x
    
    @property
    def tlens(self):
        tlens = []
        for read in self.alignemnt:
            if read.is_paired and not read.is_unmapped and not read.mate_is_unmapped and read.tlen > 0:
                tlens.append(read.tlen) 
        return np.array(tlens)
    
    def digest_nucs(self, size, dyad=0, id_='nuc'):
        return self.exo_model.digest_nucs(size, dyad, id_)
    
    def predict_exo(self, X):
        return self.exo_model.predict(X)
    
    def fit_exo(self, probs_path=None, *args, **kwargs):
        if probs_path is None:
            tlens = self.tlens.reshape(-1, 1)
            return self.exo_model.fit(tlens)
        try:
            loaded_probs = np.loadtxt(probs_path, *args, **kwargs)
            self.exo_model.load_probs(loaded_probs)
            return self.exo_model
        except Exception as e:
            warnings.warn(f"error, {e}")
            tlens = self.tlens.reshape(-1, 1)
            return self.exo_model.fit(tlens)
        
    def iterover(self, contig, iterover='template', win_len=300, start=None, stop=None):
        return self.file_manager.iterover(contig, iterover, win_len, start, stop)
  
         