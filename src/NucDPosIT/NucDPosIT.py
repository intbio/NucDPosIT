from abc import ABC, abstractmethod, abstractproperty


    
class AbstractNucDPosIT(ABC):
    
    @abstractproperty
    def file_format(self):
        pass
    
    @abstractproperty
    def input_path(self):
        pass
    
    
    @abstractmethod
    def run(self):
        pass
    
    
    @abstractmethod
    def set_formatter(self):
        pass





class NucDPosIT:
    def __init__(input_path, out_dir, file_format):
        self.__input_path = input_path
        self.__out_dir = out_dir
        self.__file_format = file_format
        
    def run():
        pass
        