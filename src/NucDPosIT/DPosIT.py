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


class DPosIT:
    def __init__(self, input_path, file_format, out_dir=None, l0=147, reg_koef = 0.12, max_iter=2000):
        self.file_manager = AlignmentFileManager(input_path, file_format, out_dir)
        self.exo_model = ExoModel(l0, reg_koef, max_iter)
    
    @property
    def alignemnt(self):
        return self.file_manager.alignment_file
    
    def fit_exo_model(self):
        tlens = []
        for read in self.alignemnt:
            if read.is_paired and not read.is_unmapped and not read.mate_is_unmapped and read.tlen > 0:
                tlens.append(read.tlen)
        np.array(tlens).reshape(-1, 1)
        self.exo_model.fit(tlens)