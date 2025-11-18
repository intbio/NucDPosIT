from torch.utils.data import Dataset
from collections import defaultdict
import pysam
import torch


class PysamRecordsDataset(Dataset):
    def __init__(self, pysam_records):
        self.records = self._collect_paired_reads(pysam_records)
        self.chromosome = pysam_records[0].reference_name

    def __getitem__(self, idx: int):
        record = self.records[idx]
        items = {
            "start": record[0].reference_start,
            "end": record[1].reference_end,
            "id": record[0].qname,
            "chromo": record[0].reference_name,
        }
        return items

    def __len__(self):
        return len(self.records)

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
            if "R1" in pair and "R2" in pair:
                valid_pairs.append((pair["R1"], pair["R2"]))
        return valid_pairs
    
    
class BamFileIterator:
    def __init__(self, pysam_path, chromosome, window_size=1000, step=200, start=0):
        self.al_file = pysam.AlignmentFile(pysam_path)
        self.chromosome = chromosome
        self.window_size = window_size
        self.step = step
        self.initial_start = start
        
        self.start = start
        self.stop = window_size
        self.chromo_len = dict(zip(self.al_file.references, self.al_file.lengths))[chromosome]

    def __del__(self):
        self.al_file.close()

    def __iter__(self):
        self.start = self.initial_start
        self.stop = self.start + self.window_size
        return self

    def __next__(self):
        records = list(self.al_file.fetch(self.chromosome, self.start, self.stop))
        paired_records = self._collect_paired_reads(records)
        items = {
            "start": torch.tensor(list(map(lambda x: x[0].reference_start, paired_records))).view(-1 ,1),
            "end": torch.tensor(list(map(lambda x: x[1].reference_end, paired_records))).view(-1, 1),
            "id": list(map(lambda x: x[0].qname, paired_records)),
        }
        if self.stop <= self.chromo_len:
            self.start += self.step
            self.stop = self.start + self.window_size
            return items
        else:
            return items
            raise StopIteration

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
            if "R1" in pair and "R2" in pair:
                valid_pairs.append((pair["R1"], pair["R2"]))
        return valid_pairs