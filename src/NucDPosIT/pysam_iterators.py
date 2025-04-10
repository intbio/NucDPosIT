from abc import ABC, abstractmethod
import numpy as np
import pysam


class AbstractWindowIterator(ABC):
    def __init__(self):
        self.__generator = self.template_generator()
        
    @property
    def generator(self):
        return self.__generator
        
    @abstractmethod
    def __iter__(self):
        pass
    
    @abstractmethod
    def __next__(self):
        pass
    
    @abstractmethod
    def template_generator(self):
        pass
    
    
# class BedWindowIterator
            
            
class BaseWindowIterator(AbstractWindowIterator):
    def __init__(self, pysam, contig, winlen=300, start=0, stop=None):
        super().__init__()
        self.pysam = pysam
        self.contig = contig
        self.winlen = winlen
        self.start = start
        self.stop = stop 
        self.__paired_reads = {}
        
    @property
    def paired_reads(self):
        return self.__paired_reads
        
    def __iter__(self):
        return self
    
    def __next__(self):
        batch = True
        while batch:
            batch = next(self.generator)
            return batch
        raise StopIteration
        
    def template_generator(self):
        self.find_pair_reads(self.start, self.stop)
        batch_records = []
        for qid, records in self.paired_reads.items():
            if len(records) == 2:
                record1, record2 = records
                start, stop = record1.reference_start, record2.reference_end
            else:
                record = records[0]
                if record.is_reverse:
                    start, stop = record.reference_end + record.tlen, record.reference_end
                else:
                    start, stop = record.reference_start, record.reference_start + record.tlen
            batch_records.append([start, stop, qid])
            if len(batch_records) == self.winlen:    
                batch = np.array(batch_records)
                batch_records.clear()
                yield batch
        yield np.array(batch_records)
        
    def find_pair_reads(self, start, stop):
        paired_reads = {}
        for read in self.pysam.fetch(self.contig, start, stop):
            if read.is_paired and read.is_proper_pair:
                pair_name = read.qname
                if pair_name not in paired_reads:
                    paired_reads[pair_name] = [read]
                else:
                    if read.is_reverse:
                        paired_reads[pair_name].append(read)
                    else:
                        paired_reads[pair_name].insert(0, read)
        self.__paired_reads = paired_reads.copy()
           

class PysamOverlapingWindowIterator(BaseWindowIterator):
    def __init__(self, pysam, contig, winlen, start, stop, step=1):
        super().__init__(pysam, contig, winlen, start, stop)
        self.step = step
        
    def template_generator(self):
        for start_cord in range(self.start, self.stop, self.step):
            if start_cord + self.winlen > self.stop:
                super().find_pair_reads(start_cord, self.stop)
            else:
                super().find_pair_reads(start_cord, start_cord + self.winlen)
            batch_records = []
            for qid, records in self.paired_reads.items():
                if len(records) == 2:
                    record1, record2 = records
                    start, stop = record1.reference_start, record2.reference_end
                else:
                    record = records[0]
                    if record.is_reverse:
                        start, stop = record.reference_end + record.tlen, record.reference_end
                    else:
                        start, stop = record.reference_start, record.reference_start + record.tlen
                batch_records.append([start, stop, qid])  
            batch = np.array(batch_records)
            yield batch
        yield np.array(batch_records)

