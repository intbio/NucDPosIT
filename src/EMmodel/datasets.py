from torch.utils.data import Dataset
from collections import defaultdict
import pysam
import torch


class BamFileIterator:
    def __init__(
        self, pysam_path, chromosome, window_size=4000, start=0, stop=None, step=3500, device="cpu"
    ):
        self.al_file = pysam.AlignmentFile(pysam_path)
        self.chromosome = chromosome
        self.window_size = window_size
        self.step = step
        self.__initial_start = start
        self.__stop = stop if stop else dict(zip(self.al_file.references, self.al_file.lengths))[chromosome]
        self.device = device 
        
    @property
    def initial_start(self):
        return self.__initial_start
    
    @property
    def stop(self):
        return self.__stop


        self.chromo_len = dict(zip(self.al_file.references, self.al_file.lengths))[
            chromosome
        ]

    def __del__(self):
        self.al_file.close()

    def __iter__(self):
        self.cur_start = self.initial_start
        self.cur_stop = self.cur_start + self.window_size
        self.stop_iter = False
        return self

    def __next__(self):
        if self.stop_iter == True:
            raise StopIteration
        
        if self.cur_stop >= self.stop:
            self.stop_iter = True
        
        records = list(self.al_file.fetch(self.chromosome, self.cur_start, self.cur_stop))
        paired_records = self._collect_paired_reads(records)
        items = {
            "start": torch.tensor(list(map(lambda x: x[0].reference_start, paired_records))).view(-1 ,1),
            "end": torch.tensor(list(map(lambda x: x[1].reference_end, paired_records))).view(-1, 1),
            "id": list(map(lambda x: x[0].qname, paired_records)),
        }
        self.cur_start += self.step
        self.cur_stop = self.cur_start + self.window_size
        return items

    def _collect_paired_reads(self, records):
        read_pairs = defaultdict(dict)
        valid_pairs = []

        for read in records:
            name = read.query_name
            if read.is_read1:
                read_pairs[name]["R1"] = read
            elif read.is_read2:
                read_pairs[name]["R2"] = read

        # Отбираем только полные пары
        for name, pair in read_pairs.items():
            if (
                "R1" in pair
                and "R2" in pair
                and pair["R2"].reference_end - pair["R1"].reference_start <= 195
            ):
                valid_pairs.append((pair["R1"], pair["R2"]))
        return valid_pairs