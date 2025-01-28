from abc import ABC, abstractmethod
import warnings


class AbstractWindowIterator(ABC):
    def __init__(self, iterable):
        self.iterable = self.check_iterable(iterable)
        
    def check_iterable(self, iterable, key=None) -> 'iterable':
        if len(iterable) == 1:
            return iterable
        for i in range(1, len(iterable)):
            cur_item, next_item = iterable[i - 1], iterable[i]
            if next_item < cur_item:
                warnings.warn("elements are not sorted, sorting...")
                return sorted(iterable, key=key)
        return iterable
            
        
    @abstractmethod
    def __iter__(self):
        pass
    
    @abstractmethod
    def __next__(self):
        pass
    

class WindowTemplateIterator(AbstractWindowIterator):
    def __init__(self, templates, win_len, start=0):
        super().__init__(templates)
        self.__win_len = self.__check_win_len(win_len)
        self.start_cord = start
        self.cur_cord = start

    def __check_win_len(self, win_len):
        if win_len <= 0:
            raise ValueError("window_len can\'t be 0 or negative")
        return win_len
    
    @property
    def win_len(self):
        return self.__win_len
    
    @win_len.setter
    def win_len(self, new_winlen):
        try:
            self.__win_len = self.__check_win_len(new_winlen)
        except ValueError as e:
            raise e
        
    def __iter__(self):
        self.__reset_cord()
        return self
    
    def __next__(self):
        if self.cur_cord >= len(self.iterable):
            self.__reset_cord()
            raise StopIteration
        batch = self.iterable[self.cur_cord : self.cur_cord + self.win_len]
        self.cur_cord += self.win_len
        return batch
    
    def __reset_cord(self):
        self.cur_cord = self.start_cord
